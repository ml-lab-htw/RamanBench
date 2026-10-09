#!/usr/bin/env python
"""Run a self-contained system (``raman_bench.systems.SYSTEMS``) on the v1 protocol.

A system does its own validation and ensembling (e.g. Chakra-Tab, a hosted API), so it is
fit once per outer fold through TabArena's ``ExternalSystemExperiment`` instead of
AutoGluon's bagging. Everything else is the v1 protocol: the tasks of ``protocol.json``,
the same data cleaning, row sample and folds as ``run_experiment.py`` (shared
``experiment_utils.load_dataframe``/``build_task``), each task's time budget
(``time_limit`` raised by ``time_limit_overrides``) and the same result cache
(``results_dir/<result_dir>/<task>/<repeat>_<fold>/results.pkl``), so
``aggregate_results.py`` and its fold check pick the results up unchanged. A cached
result is skipped, so an interrupted run resumes.

Usage:
    export CHAKRA_TAB_KEY=...                       # never put the key in a file
    python scripts/run_system.py --system CHAKRA-TAB --dry-run           # what would run
    python scripts/run_system.py --system CHAKRA-TAB --tasks alzheimer__0 --folds 0
    python scripts/run_system.py --system CHAKRA-TAB --workers 4         # all 405 folds

``--workers`` runs that many folds at once (separate processes); keep it within the
concurrency the API provider allows.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("run_system")

N_SPLITS = 3


def task_time_limit(protocol: dict, dataset: str) -> float:
    return max(protocol["time_limit"], protocol.get("time_limit_overrides", {}).get(dataset, 0))


def run_one(
    *,
    system_key: str,
    task: str,
    fold: int,
    repeat: int = 0,
    results_dir: str = "results/v1/data",
    cache_dir: str = ".cache_v1",
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    time_limit: float | None = None,
    force_recompute: bool = False,
) -> dict | None:
    """Fit *system_key* on one outer fold of one protocol task and cache the result."""
    from tabarena.benchmark.experiment.experiment_constructor import ExternalSystemExperiment

    from raman_bench.compare import load_protocol
    from raman_bench.experiment_utils import build_task, load_dataframe, run_cached
    from raman_bench.systems import get_system

    system = get_system(system_key)
    protocol = load_protocol()
    (spec,) = [t for t in protocol["tasks"] if t["task"] == task]
    dataset_name, target_idx = spec["dataset"], spec["target_idx"]

    dataset, df, sample_idx, problem_type = load_dataframe(
        dataset_name,
        target_idx,
        max_train_samples=protocol["max_train_samples_overrides"].get(dataset_name),
        cache_dir=cache_dir,
        mirror_repo=mirror_repo,
    )
    task_name, task_wrapper = build_task(
        dataset_name=dataset_name,
        target_idx=target_idx,
        df=df,
        raw_targets=dataset.targets,
        problem_type=problem_type,
        n_repeats=spec["n_repeats"],
        n_splits=spec["n_folds"],
        sample_idx=sample_idx,
        min_samples_per_class=protocol["min_samples_per_class"],
    )
    if task_wrapper is None:
        logger.warning("%s has no usable rows; skipped", task)
        return None

    time_limit = time_limit or task_time_limit(protocol, dataset_name)
    experiment = ExternalSystemExperiment(
        name=system.result_dir,
        system_cls=system.load_cls(),
        system_hyperparameters=dict(system.hyperparameters),
        method_kwargs={
            "fit_kwargs": {
                "time_limit": time_limit,
                "num_cpus": num_cpus or os.cpu_count(),
                "num_gpus": system.num_gpus if num_gpus is None else num_gpus,
            }
        },
        experiment_kwargs={"require_warmup": False},
    )
    return run_cached(
        experiment,
        task_name=task_name,
        task_wrapper=task_wrapper,
        repeat=repeat,
        fold=fold,
        results_dir=results_dir,
        force_recompute=force_recompute,
    )


def _run_job(kwargs: dict) -> tuple[str, int, str]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    start = time.monotonic()
    try:
        out = run_one(**kwargs)
    except Exception as e:  # one failed fold must not stop the others
        logger.exception("%s fold %d failed", kwargs["task"], kwargs["fold"])
        return kwargs["task"], kwargs["fold"], f"FAILED {type(e).__name__}: {str(e)[:200]}"
    if out is None:
        return kwargs["task"], kwargs["fold"], "skipped"
    return kwargs["task"], kwargs["fold"], f"metric_error={out.get('metric_error'):.4g} ({time.monotonic() - start:.0f} s)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, help="Key of raman_bench.systems.SYSTEMS, e.g. CHAKRA-TAB")
    parser.add_argument("--tasks", nargs="*", help="Protocol tasks (<dataset>__<target_idx>); default: all")
    parser.add_argument("--task-type", choices=["classification", "regression"])
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(N_SPLITS)))
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument("--workers", type=int, default=1, help="Folds run at once (separate processes)")
    parser.add_argument("--num-cpus", type=int, default=None, help="CPUs reported to the system per fold")
    parser.add_argument("--num-gpus", type=int, default=None, help="GPUs per fold (default: the system's)")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List the folds that would run, call nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from raman_bench.compare import load_protocol, protocol_tasks
    from raman_bench.systems import get_system

    system = get_system(args.system)
    protocol = load_protocol()
    tasks = protocol_tasks(task_type=args.task_type, tasks=args.tasks or None)
    jobs = []
    for spec in tasks:
        for fold in args.folds:
            done = os.path.exists(
                os.path.join(args.results_dir, system.result_dir, spec["task"], f"0_{fold}", "results.pkl")
            )
            if done and not args.force_recompute:
                continue
            jobs.append(
                {
                    "system_key": args.system,
                    "task": spec["task"],
                    "fold": fold,
                    "results_dir": args.results_dir,
                    "cache_dir": args.cache_dir,
                    "mirror_repo": args.mirror_repo,
                    "num_cpus": args.num_cpus,
                    "num_gpus": args.num_gpus,
                    "force_recompute": args.force_recompute,
                }
            )
    n_all = len(tasks) * len(args.folds)
    budget = sum(task_time_limit(protocol, s["dataset"]) for s in tasks) * len(args.folds)
    logger.info(
        "%s (%s, %s): %d of %d fold(s) to run, %d cached; time budgets sum to %.1f h",
        args.system, system.result_dir, system.hyperparameters, len(jobs), n_all, n_all - len(jobs), budget / 3600,
    )
    if system.notes:
        logger.info("Note: %s", system.notes)
    if args.dry_run:
        for j in jobs:
            print(j["task"], j["fold"])
        return
    missing = [v for v in system.env if not os.environ.get(v)]
    if missing:
        sys.exit(f"{args.system} needs the environment variable(s) {missing}; nothing was run.")

    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            task, fold, status = f.result()
            failed += status.startswith("FAILED")
            logger.info("[%d/%d] %s fold %d: %s", i, len(jobs), task, fold, status)
    logger.info("Done: %d fold(s), %d failed", len(jobs), failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
