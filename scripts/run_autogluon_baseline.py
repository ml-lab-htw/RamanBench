#!/usr/bin/env python
"""Whole-predictor AutoGluon preset baseline runner (sibling to ``run_experiment.py``).

Reproduces TabArena's own AutoGluon "extreme" preset baselines
(``tabarena.models.automl.generate_autogluon_extreme_experiments``:
``presets="extreme"`` at three time budgets -- 300s/3600s/14400s) against RamanBench's
own datasets, using RamanBench's currently-pinned AutoGluon version (not a specific
AutoGluon release -- TabArena's own experiment names like "AutoGluon_v140_eq_4h8c" are
just labels, not a version pin).

Architecturally different from ``run_experiment.py``: that script fits exactly ONE
model class per job via a per-model ``ConfigGenerator``-derived ``AGModelBagExperiment``
(``gen_<key>.generate_all_bag_experiments(...)[config_index]``). This script instead
builds a ``tabarena.benchmark.experiment.experiment_constructor.AGExperiment`` --
AutoGluon's own whole-predictor ``TabularPredictor.fit(presets=..., time_limit=...)``
auto-stacking/ensembling run across every model AutoGluon itself considers. There is no
``model_key``, no ``ConfigGenerator``, no ``config_index`` here -- the "model" IS the
AutoGluon preset itself.

Data loading/cleaning/splitting is shared verbatim with ``run_experiment.py`` via
:func:`raman_bench.experiment_utils.build_task` -- both need byte-identical splits for
the same (dataset, target_idx, repeat, fold, n_repeats, n_splits) for results to be
genuinely comparable. See that function's docstring.

Output is written to the exact same cache layout ``run_experiment.py`` uses
(``results_dir/{experiment.name}/{task_name}/{repeat}_{fold}/results.pkl``, via
TabArena's own ``CacheFunctionPickle`` + ``Experiment.run``), so
``scripts/aggregate_results.py`` picks up these results with zero changes -- confirmed
directly: ``AGExperiment`` and ``AGModelBagExperiment`` both route through the same
``OOFExperimentRunner``/``Experiment.run`` machinery and produce the identical
``task_metadata``/``problem_type``/``metric_error``/``framework`` output schema
``EndToEnd.from_raw`` expects.

Usage
-----
    python scripts/run_autogluon_baseline.py --dataset wheat_lines --target-idx 0 \\
        --repeat 0 --fold 0 --time-limit 300 --results-dir results/v1/data
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_N_REPEATS = 1
DEFAULT_N_SPLITS = 3
# TabArena's own three "extreme" preset budgets (tabarena.models.automl.
# generate_autogluon_extreme_experiments) -- 5 minutes / 1 hour / 4 hours.
EXTREME_TIME_BUDGETS = {"5m": 300, "1h": 3600, "4h": 14400}


def _experiment_name(budget_label: str) -> str:
    # Mirrors TabArena's own naming shape (AutoGluon_v140_eq_<budget>8c) closely enough
    # to be recognizable, without claiming a specific AutoGluon version -- this repo's
    # pin floats (see pyproject.toml's autogluon extra), so "v140" would be misleading.
    return f"AutoGluon_extreme_{budget_label}"


def run_one(
    *,
    dataset_name: str,
    target_idx: int,
    repeat: int,
    fold: int,
    time_limit: float,
    budget_label: str,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_splits: int = DEFAULT_N_SPLITS,
    results_dir: str = "results/v1/data",
    cache_dir: str = ".cache_v1",
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
    use_mirror: bool = True,
    use_gpu: bool = False,
    scratch_dir: str | None = None,
    min_samples_per_class: int = 9,
    max_train_samples: int | None = None,
    force_recompute: bool = False,
) -> dict | None:
    """Run exactly one (dataset, target, repeat, fold) AutoGluon-extreme baseline job.

    Returns ``None`` (a clean, expected skip, matching ``run_one``'s own semantics in
    ``run_experiment.py``) if this target fails rare-class filtering or every row has a
    missing label. Unlike ``run_experiment.py``, there are no model-specific
    compatibility skips here (classification-only/regression-only/max-features/VRAM
    caps) -- ``TabularPredictor.fit`` handles any problem type and any width/row count
    within its own time/memory budget.
    """
    from tabarena.benchmark.experiment.experiment_constructor import AGExperiment
    from tabarena.utils.cache import CacheFunctionPickle

    from raman_bench.benchmark import RamanBenchmark
    from raman_bench.experiment_utils import build_task

    bench = RamanBenchmark(
        dataset_names_classification=[],
        dataset_names_regression=[],
        cache_dir=cache_dir,
        use_mirror=use_mirror,
        mirror_repo=mirror_repo,
    )
    dataset = bench._load_raman_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name!r}")

    df = dataset.to_dataframe(target_idx)
    sample_idx = None
    if max_train_samples is not None and len(df) > max_train_samples:
        n_before = len(df)
        df = df.sample(n=max_train_samples, random_state=0)
        sample_idx = df.index.to_numpy()
        logger.info(
            "Subsampled %s: %d -> %d rows (max_train_samples=%d); this is a real change to "
            "what is measured for this dataset, not a performance-neutral optimisation.",
            dataset_name,
            n_before,
            len(df),
            max_train_samples,
        )

    from raman_data import TASK_TYPE

    problem_type = (
        "classification" if dataset.task_type == TASK_TYPE.Classification else "regression"
    )

    task_name, task_wrapper = build_task(
        dataset_name=dataset_name,
        target_idx=target_idx,
        df=df,
        raw_targets=dataset.targets,
        problem_type=problem_type,
        n_repeats=n_repeats,
        n_splits=n_splits,
        sample_idx=sample_idx,
        min_samples_per_class=min_samples_per_class,
        filter_unlabeled=True,
    )
    if task_wrapper is None:
        return None

    experiment_name = _experiment_name(budget_label)
    init_kwargs = None
    if scratch_dir is not None:
        # Same reasoning as run_experiment.py's own scratch_dir handling: a
        # deterministic predictor path (instead of AutoGluon's default relative
        # AutogluonModels/ag-<timestamp> under cwd) lets a cluster wrapper's cleanup
        # trap find and remove it reliably even if the job is killed mid-fit.
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        init_kwargs = {"path": scratch_dir}

    experiment = AGExperiment(
        name=experiment_name,
        init_kwargs=init_kwargs,
        fit_kwargs=dict(presets="extreme", time_limit=time_limit),
        experiment_kwargs={
            # Same reasoning as run_experiment.py's require_warmup=False: TabArena's
            # warm-up framework is audited against its own known model registry: not
            # re-verified here against the whole-predictor AGWrapper path, so disabled
            # for safety/consistency rather than assumed compatible.
            "require_warmup": False,
        },
    )

    cache_path = os.path.join(results_dir, experiment_name, task_name, f"{repeat}_{fold}")
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    cacher = CacheFunctionPickle(
        cache_name="results", cache_path=cache_path, include_self_in_call=True
    )

    # Same cache policy as run_experiment.py: an already-cached result stands as final
    # unless --force-recompute is explicitly given. See that script's own long comment
    # on why (docs/internal/invalidating-results.md).
    if cacher.exists and not force_recompute:
        out = cacher.load_cache()
        logger.info(
            "%s on %s repeat=%d fold=%d: using existing cached result at %s",
            experiment_name,
            task_name,
            repeat,
            fold,
            cache_path,
        )
        logger.info("Done: metric_error=%s", out.get("metric_error"))
        return out

    logger.info(
        "Running %s on %s repeat=%d fold=%d -> %s",
        experiment_name,
        task_name,
        repeat,
        fold,
        cache_path,
    )
    out = experiment.run(
        task=task_wrapper,
        fold=fold,
        repeat=repeat,
        task_name=task_name,
        cache_task_key=task_name,
        cacher=cacher,
        ignore_cache=force_recompute,
    )
    logger.info("Done: metric_error=%s", out.get("metric_error"))
    return out


def _wandb_enabled() -> bool:
    return bool(os.environ.get("WANDB_API_KEY"))


def _log_to_wandb(*, out: dict | None, task_config: dict) -> None:
    """Same wandb logging contract as run_experiment.py's own _log_to_wandb -- reuses
    its build_wandb_metrics/wandb_project_for_model helpers directly so the two
    scripts' wandb runs are structurally identical (same metric keys, same per-model
    project-sharding convention, just a different "model" slug)."""
    try:
        import wandb
    except ImportError:
        logger.warning(
            "WANDB_API_KEY is set but the `wandb` package isn't installed "
            "(pip install raman-bench[tracking]) -- skipping tracking for this task."
        )
        return

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_run_experiment_wandb", Path(__file__).resolve().parent / "run_experiment.py"
    )
    run_experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_experiment)

    try:
        run = wandb.init(
            project=run_experiment.wandb_project_for_model(task_config["model"]),
            entity=os.environ.get("WANDB_ENTITY"),
            group=f"{task_config['model']}_{task_config['dataset']}",
            job_type=task_config.get("problem_type"),
            name=(
                f"{task_config['model']}_{task_config['dataset']}"
                f"_t{task_config['target_idx']}_r{task_config['repeat']}"
                f"_f{task_config['fold']}"
            ),
            tags=[task_config["model"], task_config["dataset"]],
            config=task_config,
            reinit=True,
        )
        if out is None:
            run.summary["skipped"] = True
        else:
            run.log(run_experiment.build_wandb_metrics(out))
        run.finish()
    except Exception:
        logger.warning(
            "wandb logging failed for this task -- continuing (results.pkl is unaffected).",
            exc_info=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="raman_data dataset name")
    parser.add_argument("--target-idx", type=int, default=0)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--time-limit",
        type=float,
        required=True,
        help="AutoGluon TabularPredictor.fit's time_limit in seconds, e.g. 300/3600/14400 "
        "for TabArena's own 5m/1h/4h extreme-preset budgets (see EXTREME_TIME_BUDGETS).",
    )
    parser.add_argument(
        "--budget-label",
        default=None,
        help="Label for this time budget used in the experiment/cache name (e.g. '5m', '1h', "
        "'4h'). Defaults to '<time_limit>s' if not given -- pass one of EXTREME_TIME_BUDGETS' "
        "keys to match TabArena's own naming exactly.",
    )
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument(
        "--use-mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--min-samples-per-class", type=int, default=9)
    args = parser.parse_args()

    budget_label = args.budget_label or f"{int(args.time_limit)}s"

    out = run_one(
        dataset_name=args.dataset,
        target_idx=args.target_idx,
        repeat=args.repeat,
        fold=args.fold,
        time_limit=args.time_limit,
        budget_label=budget_label,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        mirror_repo=args.mirror_repo,
        use_mirror=args.use_mirror,
        use_gpu=args.use_gpu,
        scratch_dir=args.scratch_dir,
        min_samples_per_class=args.min_samples_per_class,
        max_train_samples=args.max_train_samples,
        force_recompute=args.force_recompute,
    )
    if out is None:
        logger.info("Target skipped (see the reason logged above) -- clean exit, not an error.")

    if _wandb_enabled():
        _log_to_wandb(
            out=out,
            task_config={
                "model": f"AUTOGLUON-EXTREME-{budget_label.upper()}",
                "dataset": args.dataset,
                "target_idx": args.target_idx,
                "repeat": args.repeat,
                "fold": args.fold,
                "n_repeats": args.n_repeats,
                "n_splits": args.n_splits,
                "time_limit": args.time_limit,
                "use_gpu": args.use_gpu,
                "problem_type": (out or {}).get("problem_type"),
            },
        )


if __name__ == "__main__":
    main()
