#!/usr/bin/env python
"""Run many ``run_experiment.py`` jobs in one process, with per-stage timings.

``run_experiment.py`` fits exactly one (model, dataset, target, repeat, fold, config) per
process. On GPU pods most of the wall time then goes into work around the fit -- interpreter
start, imports, CUDA init, dataset loading, task building -- repeated for every line, so the GPU
sits idle. This runner calls the same :func:`run_experiment.run_one` for every line of a
jobspec inside one process, so that work is paid once per process (imports, CUDA) or once per
dataset (loading). Results are written exactly where ``run_experiment.py`` writes them, and
already-cached results are skipped exactly as there (``run_cached``). The per-spectrum
preprocessing steps go through :mod:`raman_bench.preprocessing.spectrum_cache` (``--spectrum-cache``,
in memory by default), so each spectrum is smoothed and baseline-corrected once per process and
recipe instead of once per bag child and call.

Jobspec: JSON Lines, one job per line, keys are ``run_one`` arguments; the command-line options
give the defaults for every key a line leaves out::

    {"dataset_name": "acetic_acid_species", "target_idx": 0, "model_key": "TABPFN-V3.5",
     "repeat": 0, "fold": 1, "recipe_config": "configs/snv.json"}

``--claim-dir``: several processes or pods may share one jobspec. Each job is claimed with an
atomic ``mkdir`` on the shared filesystem before it runs; a job claimed elsewhere is skipped, and
a failed job releases its claim. Use a fresh directory per submission round.

A failing job is logged with its traceback and the loop continues; the exit code is 1 if any job
failed. ``--timings-out`` writes one JSON record per job with the seconds spent in each stage
(data loading, task building, preprocessing fit/transform, checkpoint loading, the cached
experiment run as a whole), and a per-model summary is printed at the end.

    python scripts/run_experiment_batch.py --jobspec jobs.jsonl --results-dir results/v1/data \\
        --num-bag-folds 3 --use-gpu --claim-dir /data/claims/round1 --timings-out timings.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import logging
import os
import shutil
import socket
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rex  # noqa: E402

logger = logging.getLogger("run_experiment_batch")

# run_one arguments a jobspec line may set; everything else comes from the command line.
LINE_KEYS = {
    "dataset_name", "target_idx", "model_key", "repeat", "fold", "config_index", "recipe_config",
    "n_repeats", "n_splits", "num_random_configs", "num_bag_folds", "time_limit",
    "max_train_samples", "min_samples_per_class", "min_groups_per_class",
}
REQUIRED_KEYS = ("dataset_name", "model_key", "repeat", "fold")


class StageTimer:
    """Accumulates wall-clock seconds per stage for the job that is currently running.

    Nested calls of the same stage (e.g. ``_preprocess_fit`` inside
    ``_preprocess_fit_ensemble``) are counted once, at the outermost level.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._depth: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()

    @contextlib.contextmanager
    def stage(self, name: str):
        self._depth[name] += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._depth[name] -= 1
            if self._depth[name] == 0:
                self.seconds[name] += time.perf_counter() - t0
                self.calls[name] += 1

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with self.stage(name):
                return fn(*args, **kwargs)

        return wrapped


def instrument(timer: StageTimer) -> None:
    """Patch the stages run_one goes through with timers (module attributes are looked up at
    call time inside run_one, so patching them here is enough)."""
    import torch

    from raman_bench import experiment_utils as eu
    from raman_bench.preprocessing.mixin import RamanPreprocessingMixin as M

    eu.load_dataframe = timer.wrap("load_dataframe", eu.load_dataframe)
    eu.build_task = timer.wrap("build_task", eu.build_task)
    eu.run_cached = timer.wrap("run_cached", eu.run_cached)
    for name, stage in (("_preprocess_fit", "prep_fit"), ("_preprocess_fit_ensemble", "prep_fit"),
                        ("_preprocess_transform", "prep_transform"),
                        ("_preprocess_transform_ensemble", "prep_transform")):
        setattr(M, name, timer.wrap(stage, getattr(M, name)))
    torch.load = timer.wrap("torch_load", torch.load)


def memoize_load_dataframe() -> None:
    """Load each (dataset, target, row cap, source) once per process.

    Every call gets its own copy of the DataFrame, so a job that modifies it cannot leak into
    the next one; the dataset object is shared read-only.
    """
    from raman_bench import experiment_utils as eu

    inner = eu.load_dataframe
    memo: dict = {}

    @functools.wraps(inner)
    def load(dataset_name, target_idx, **kwargs):
        key = (dataset_name, target_idx, json.dumps(kwargs, sort_keys=True, default=str))
        if key not in memo:
            memo[key] = inner(dataset_name, target_idx, **kwargs)
        dataset, df, sample_idx, problem_type = memo[key]
        return dataset, df.copy(), sample_idx, problem_type

    eu.load_dataframe = load


def claim_name(job: dict) -> str:
    recipe = job.get("recipe_config")
    recipe = Path(recipe).stem if recipe else "default"
    return (f"{job['model_key']}/{recipe}__{job['dataset_name']}__{job.get('target_idx', 0)}"
            f"__{job['repeat']}_{job['fold']}__c{job.get('config_index', 0)}")


def try_claim(claim_dir: Path, job: dict) -> Path | None:
    path = claim_dir / claim_name(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        return None
    (path / "owner").write_text(f"{socket.gethostname()} {os.getpid()} {time.strftime('%FT%T')}\n")
    return path


def read_jobspec(path: Path) -> list[dict]:
    jobs = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        job = json.loads(line)
        unknown = set(job) - LINE_KEYS
        assert not unknown, f"{path}:{n}: unknown keys {sorted(unknown)}"
        missing = [k for k in REQUIRED_KEYS if k not in job]
        assert not missing, f"{path}:{n}: missing keys {missing}"
        jobs.append(job)
    return jobs


def summarise(records: list[dict]) -> str:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["status"] == "done":
            by_model[r["model_key"]].append(r)
    stages = sorted({s for r in records for s in r["seconds"]})
    lines = ["median seconds per job: " + " ".join(["model", "n", "wall", *stages])]
    for m, rs in sorted(by_model.items()):
        med = [statistics.median(r["seconds"].get(s, 0.0) for r in rs) for s in stages]
        lines.append(" ".join([m, str(len(rs)), f"{statistics.median(r['wall_s'] for r in rs):.1f}",
                               *(f"{v:.1f}" for v in med)]))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobspec", required=True, type=Path, help="JSON Lines, one run_one job per line")
    ap.add_argument("--claim-dir", type=Path, default=None)
    ap.add_argument("--timings-out", type=Path, default=None)
    ap.add_argument("--n-repeats", type=int, default=rex.DEFAULT_N_REPEATS)
    ap.add_argument("--n-splits", type=int, default=rex.DEFAULT_N_SPLITS)
    ap.add_argument("--config-index", type=int, default=0)
    ap.add_argument("--num-random-configs", type=int, default=0)
    ap.add_argument("--num-bag-folds", type=int, default=rex.DEFAULT_NUM_BAG_FOLDS)
    ap.add_argument("--time-limit", type=float, default=rex.DEFAULT_TIME_LIMIT)
    ap.add_argument("--results-dir", default="results/v1/data")
    ap.add_argument("--cache-dir", default=".cache_v1")
    ap.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    ap.add_argument("--use-mirror", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--scratch-root", default=None,
                    help="Each job gets its own scratch subdirectory here, removed after the job")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--min-samples-per-class", type=int, default=9)
    ap.add_argument("--min-groups-per-class", type=int, default=0)
    ap.add_argument("--spectrum-cache", default="memory",
                    help="Per-spectrum preprocessing cache (raman_bench.preprocessing.spectrum_cache): "
                    "'memory', a directory with precomputed rows (plus memory), or 'off'")
    ap.add_argument("--no-memo-data", action="store_true",
                    help="Reload the dataset for every job (as run_experiment.py does)")
    args = ap.parse_args()

    rex.load_env()
    from raman_bench.preprocessing import spectrum_cache

    os.environ[spectrum_cache.ENV_MODE] = args.spectrum_cache
    timer = StageTimer()
    instrument(timer)
    if not args.no_memo_data:
        memoize_load_dataframe()

    jobs = read_jobspec(args.jobspec)
    defaults = dict(
        target_idx=0, config_index=args.config_index, n_repeats=args.n_repeats,
        n_splits=args.n_splits, num_random_configs=args.num_random_configs,
        num_bag_folds=args.num_bag_folds, time_limit=args.time_limit,
        max_train_samples=args.max_train_samples, min_samples_per_class=args.min_samples_per_class,
        min_groups_per_class=args.min_groups_per_class,
    )
    fixed = dict(results_dir=args.results_dir, cache_dir=args.cache_dir, mirror_repo=args.mirror_repo,
                 use_mirror=args.use_mirror, use_gpu=args.use_gpu)
    timings = args.timings_out.open("a") if args.timings_out else None
    records: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    t_start = time.perf_counter()
    logger.info("Batch: %d jobs from %s on %s", len(jobs), args.jobspec, socket.gethostname())

    for i, line in enumerate(jobs, 1):
        job = {**defaults, **line}
        claim = None
        if args.claim_dir is not None:
            claim = try_claim(args.claim_dir, job)
            if claim is None:
                counts["claimed_elsewhere"] += 1
                continue
        scratch = None
        if args.scratch_root:
            scratch = Path(args.scratch_root) / f"{socket.gethostname()}_{os.getpid()}_{i}"
            scratch.mkdir(parents=True, exist_ok=True)
        timer.reset()
        t0 = time.perf_counter()
        status = "done"
        try:
            out = rex.run_one(**job, **fixed, scratch_dir=str(scratch) if scratch else None)
            if out is None:
                status = "skipped"
        except Exception:  # one bad job must not stop the pod; it is reported and counted
            status = "failed"
            logger.error("FAILED job %d/%d %s\n%s", i, len(jobs), json.dumps(line), traceback.format_exc())
            if claim is not None:
                shutil.rmtree(claim, ignore_errors=True)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
            gc.collect()
            if args.use_gpu:
                import torch

                torch.cuda.empty_cache()
        wall = time.perf_counter() - t0
        counts[status] += 1
        rec = {**line, "model_key": job["model_key"], "status": status, "wall_s": round(wall, 3),
               "seconds": {k: round(v, 3) for k, v in timer.seconds.items()},
               "calls": dict(timer.calls), "host": socket.gethostname()}
        records.append(rec)
        if timings:
            timings.write(json.dumps(rec) + "\n")
            timings.flush()
        done = sum(counts[k] for k in ("done", "skipped", "failed"))
        elapsed = time.perf_counter() - t_start
        eta = elapsed / max(done, 1) * (len(jobs) - i)
        print(f"[{i}/{len(jobs)}] {status} {job['model_key']} {job['dataset_name']} t{job['target_idx']} "
              f"r{job['repeat']} f{job['fold']} {Path(job.get('recipe_config') or 'default').stem}: "
              f"{wall:.1f}s (upper-bound ETA {eta / 60:.0f} min)", flush=True)

    print(summarise(records), flush=True)
    cache = spectrum_cache.get_cache()
    if cache is not None:
        print(f"Spectrum cache: {cache.hits} rows served, {cache.misses} computed", flush=True)
    print(f"Batch finished in {(time.perf_counter() - t_start) / 60:.1f} min: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
