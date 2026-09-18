#!/usr/bin/env python
"""One-time import of already-completed results.pkl files into wandb.

Walks ``--results-dir`` for every cached ``results.pkl`` (the same files the
opportunistic scheduler's backlog check and ``sync_results.py`` already treat
as the source of truth for "is this task done" -- see run_experiment.py's
module docstring and this repo's cluster/CRON.md for why tracking stays
file-based, not wandb-based), and logs each one as a wandb run using the
EXACT SAME field extraction as live per-task logging
(``run_experiment.build_wandb_metrics``), so backfilled and live runs are
indistinguishable in wandb except for the ``backfilled: true`` config tag
this script adds.

This is possible with full fidelity because every field the live path logs
(metric_error, metric_error_val, time_train_s, time_infer_s, memory_usage,
...) was ALREADY being written into results.pkl by tabarena's own
ExperimentRunner, long before wandb tracking existed -- nothing is
approximated or re-derived, it's a straight read of what's already on disk.

Task identity (model, dataset, target_idx, repeat, fold, config_index) is
recovered two ways:
    - dataset/target_idx/repeat/fold: directly from the pickle's own
      ``task_metadata`` field (confirmed present on a real cached result --
      see the ``name``/``repeat``/``fold`` keys).
    - model/config_index: parsed from the pickle's ``framework`` field (e.g.
      ``"RandomForest_c1_BAG_L1"`` -- confirmed empirically, see
      cluster/opportunistic_scheduler.py's ``_ag_name`` docstring) by
      reversing the registry's key->ag_name mapping and the ``_c1``
      (config_index 0, the default config) / ``_rN`` (config_index N, an HPO
      pool config) naming convention from tabarena's
      ``combine_manual_and_random_configs``.

Idempotent: each run's wandb ``id`` is a deterministic hash of its task
identity, with ``resume="allow"`` -- interrupting and re-running this script
only re-logs runs it didn't already reach, it never creates duplicates. That
matters here because logging tens of thousands of already-completed tasks
one wandb-API-round-trip at a time is inherently slow (network-bound, not
compute-bound) and realistically needs to run unattended (nohup/screen) over
a long stretch -- interruption should be a non-event, not a restart from
scratch.

Usage
-----
    # Preview what would be logged, no wandb calls made:
    python scripts/backfill_wandb.py --results-dir results/v1/data --dry-run --limit 20

    # The real thing -- run under nohup/screen/tmux, this can take hours for
    # a large results tree (see the module docstring above):
    export WANDB_API_KEY=...
    nohup python scripts/backfill_wandb.py --results-dir results/v1/data \\
        > backfill_wandb.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Imported at module level (not deferred into a function) so it's visible to
# worker processes too -- ProcessPoolExecutor workers re-import this module,
# and on a 'spawn' start method (default on macOS; Linux defaults to 'fork',
# which would've been safe either way) a name that only exists inside
# `if __name__ == "__main__":` wouldn't be defined in the reimported copy.
# run_experiment.py's own heavy imports (raman_bench/tabarena) are deferred
# INSIDE run_one(), not at its module level, so importing the module itself
# here is cheap -- confirmed by reading that file.
from run_experiment import build_wandb_metrics, _wandb_run_name, wandb_project_for_model  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Matches e.g. "RandomForest_c1_BAG_L1" (config_index 0) or
# "TabPFNMix_r7_BAG_L1" (config_index 7, the 7th HPO pool config) -- see
# module docstring for where this convention is confirmed.
_FRAMEWORK_RE = re.compile(r"^(?P<ag_name>.+)_(?P<kind>[cr])(?P<num>\d+)_BAG_L1$")


def _load_pickle(path: Path) -> dict:
    """results.pkl files are gzip-compressed pickles (CacheFunctionPickle's
    own format, confirmed via a real cached file's leading bytes: 0x1f 0x8b)."""
    with open(path, "rb") as f:
        head = f.read(2)
    opener = gzip.open if head == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def _build_ag_name_to_model_key(scope_path: Path) -> dict[str, str]:
    from raman_bench.models.registry import infer_model_cls

    scope = json.loads(scope_path.read_text())
    mapping: dict[str, str] = {}
    for key in scope["models"]:
        try:
            mapping[infer_model_cls(key).ag_name] = key
        except Exception as e:
            logger.warning("Could not resolve model %r via the registry (%s) -- "
                            "results for it won't be backfilled.", key, e)
    return mapping


def _parse_framework(framework: str, ag_name_to_key: dict[str, str]) -> tuple[str, int] | None:
    m = _FRAMEWORK_RE.match(framework or "")
    if not m:
        return None
    model_key = ag_name_to_key.get(m["ag_name"])
    if model_key is None:
        return None
    config_index = 0 if m["kind"] == "c" else int(m["num"])
    return model_key, config_index


def _parse_task_name(name: str) -> tuple[str, int]:
    """"alzheimer__0" -> ("alzheimer", 0). Dataset names never contain "__"
    themselves (confirmed: this is exactly the format RamanBenchTaskWrapper's
    own task_name construction uses, dataset + "__" + target_idx)."""
    dataset, _, target_idx = name.rpartition("__")
    return dataset, int(target_idx)


def _run_id(task_config: dict) -> str:
    key = "|".join(str(task_config[k]) for k in
                    ("model", "dataset", "target_idx", "repeat", "fold", "config_index"))
    return "bf-" + hashlib.sha1(key.encode()).hexdigest()[:16]


def iter_result_files(results_dir: Path):
    yield from sorted(results_dir.glob("*/*/*/results.pkl"))


def backfill_one(pkl_path: Path, results_dir: Path, ag_name_to_key: dict[str, str], *, dry_run: bool,
                  base_project: str, entity: str | None) -> str:
    """Returns one of "logged", "skipped_unreadable", "skipped_unrecognized"."""
    try:
        out = _load_pickle(pkl_path)
    except Exception as e:
        logger.warning("Unreadable %s: %s", pkl_path, e)
        return "skipped_unreadable"

    parsed = _parse_framework(out.get("framework", ""), ag_name_to_key)
    if parsed is None:
        logger.warning("Unrecognized framework %r for %s -- skipping.", out.get("framework"), pkl_path)
        return "skipped_unrecognized"
    model_key, config_index = parsed

    task_metadata = out.get("task_metadata") or {}
    try:
        dataset, target_idx = _parse_task_name(task_metadata["name"])
        repeat = int(task_metadata["repeat"])
        fold = int(task_metadata["fold"])
    except (KeyError, ValueError) as e:
        logger.warning("Missing/malformed task_metadata in %s: %s -- skipping.", pkl_path, e)
        return "skipped_unrecognized"

    task_config = {
        "model": model_key,
        "dataset": dataset,
        "target_idx": target_idx,
        "repeat": repeat,
        "fold": fold,
        "config_index": config_index,
        "problem_type": out.get("problem_type"),
        "backfilled": True,
        "source_path": str(pkl_path.relative_to(results_dir)),
    }
    metrics = build_wandb_metrics(out)
    # Original completion time (unix ts) from the pickle itself -- kept as a
    # logged field since wandb's own run-creation timestamp reflects when
    # THIS SCRIPT ran, not when the task actually finished; see module
    # docstring.
    experiment_metadata = out.get("experiment_metadata") or {}
    if experiment_metadata.get("time_end") is not None:
        metrics["original_time_end"] = float(experiment_metadata["time_end"])

    if dry_run:
        logger.info("[dry-run] %s -> %s", pkl_path, {**task_config, **metrics})
        return "logged"

    import wandb

    run = wandb.init(
        project=wandb_project_for_model(model_key, base_project=base_project),
        entity=entity,
        id=_run_id(task_config),
        resume="allow",
        group=f"{model_key}_{dataset}",
        job_type=task_config.get("problem_type"),
        name=_wandb_run_name(task_config),
        tags=[model_key, dataset, "backfilled"],
        config=task_config,
        reinit=True,
        # A live per-task run (run_experiment.py) benefits from wandb's
        # default system/git/code metadata capture -- for THIS use case
        # (importing a historical, already-finished scalar result) it's
        # pure overhead: confirmed live on htw, the default settings gave
        # ~8.4 runs/minute (34,584 results -> ~69 hours). Every one of these
        # also does nothing useful here -- there's no running process to
        # sample stats from, no code/git state relevant to a result computed
        # possibly weeks ago, and no console output to capture.
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            x_disable_stats=True,
            x_disable_meta=True,
            console="off",
        ),
    )
    run.log(metrics)
    run.finish()
    return "logged"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--scope", default=str(REPO_ROOT / "configs/v1/scope_default.json"),
                         help="Used only to resolve model registry keys -> ag_names (see module docstring)")
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "raman-bench"),
        help="BASE project name -- actual runs land in '<base>-<model>' (one project per model, "
             "see run_experiment.wandb_project_for_model's docstring for why).",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many results.pkl files (testing)")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes. Each wandb.init/log/finish round-trip is network-"
             "latency-bound, not CPU-bound (confirmed live: ~8.4 runs/min single-threaded, "
             "34,584 results -> ~69h) -- workers > 1 gives close to linear speedup. Each "
             "worker makes its own wandb API connection.",
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set -- export it first, or pass --dry-run to preview.")

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir {results_dir} does not exist.")

    ag_name_to_key = _build_ag_name_to_model_key(Path(args.scope))
    logger.info("Resolved %d model registry keys for framework-name parsing.", len(ag_name_to_key))

    pkl_paths = list(iter_result_files(results_dir))
    if args.limit:
        pkl_paths = pkl_paths[: args.limit]
    logger.info("Found %d results.pkl file(s) under %s (workers=%d)", len(pkl_paths), results_dir, args.workers)

    counts = {"logged": 0, "skipped_unreadable": 0, "skipped_unrecognized": 0}
    processed = 0

    if args.workers <= 1:
        for pkl_path in pkl_paths:
            status = backfill_one(
                pkl_path, results_dir, ag_name_to_key, dry_run=args.dry_run,
                base_project=args.wandb_project, entity=args.wandb_entity,
            )
            counts[status] += 1
            processed += 1
            if processed % args.progress_every == 0:
                logger.info("Progress: %d/%d processed (%s)", processed, len(pkl_paths), counts)
    else:
        # Each wandb.init/log/finish round-trip is network-latency-bound (see
        # --workers help text) -- separate PROCESSES, not threads, because
        # each needs its own independent wandb run/HTTP session; there's no
        # shared state to protect, so plain ProcessPoolExecutor.submit (no
        # initializer/global state) is enough.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    backfill_one, pkl_path, results_dir, ag_name_to_key,
                    dry_run=args.dry_run, base_project=args.wandb_project, entity=args.wandb_entity,
                ): pkl_path
                for pkl_path in pkl_paths
            }
            for future in as_completed(futures):
                pkl_path = futures[future]
                try:
                    status = future.result()
                except Exception as e:
                    logger.warning("Worker failed on %s: %s", pkl_path, e)
                    status = "skipped_unreadable"
                counts[status] += 1
                processed += 1
                if processed % args.progress_every == 0:
                    logger.info("Progress: %d/%d processed (%s)", processed, len(pkl_paths), counts)

    logger.info("Done: %s", counts)


if __name__ == "__main__":
    main()
