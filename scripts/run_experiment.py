#!/usr/bin/env python
"""Thin per-config-job cluster runner for RamanBench v1 (TabArena-based).

One process = one (model, dataset, target, repeat, fold, config-index)
experiment -- "every single evaluation gets a completely fresh job" (see the
v1 refactor plan). Mirrors TabArena's own
``tabflow_slurm/run_tabarena_experiment.py`` in spirit -- including its
``--repeat``/``--fold`` interface, not just an integer ``--seed``, now that
splitting uses real repeated k-fold CV (see ``raman_bench.splitting``) rather
than one holdout split per seed: resolve compute resources from the
environment, build the task, run exactly one experiment, cache the result to
disk, exit.

This is the only benchmark execution path in this package as of v2.0.0 -- the
earlier v0.1-era pipeline (``raman_bench.model.AutoGluonModel``,
``raman_bench.predictions``, ``scripts/run_benchmark.py``) has been removed;
see CHANGELOG.md for the v1.x release tag if you need to reproduce that
pipeline's exact behavior.

Job identity and reproducibility
---------------------------------
``--config-index 0`` is always the default config (``_c1``); indices
``1..num_random_configs`` are the HPO random-search pool (``_r1..rN``).
``--num-random-configs`` MUST be identical across every job for a given model
(it fixes the deterministic seed sequence the config pool is sampled from --
see ``raman_bench.models.generate``), even though a given job only needs and
runs a single one of those configs. Passing a different value in different
jobs for the same model would desynchronize which hyperparameters
``config-index N`` actually refers to. Likewise, ``--n-repeats``/
``--n-splits`` MUST be identical across every job for a given (dataset,
target) -- they define the entire repeated-k-fold split scheme that
``--repeat``/``--fold`` index into.

Usage
-----
    python scripts/run_experiment.py --dataset wheat_lines --target-idx 0 \\
        --model PLS --repeat 0 --fold 0 --config-index 0 \\
        --results-dir results/v1/data

With an explicit preprocessing recipe (``--recipe-config``, see ``run_one``'s docstring;
defaults to no recipe / model-class defaults, unrestricted, if omitted)::

    python scripts/run_experiment.py --dataset wheat_lines --target-idx 0 \\
        --model PLS --repeat 0 --fold 0 --config-index 0 \\
        --recipe-config /path/to/RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json \\
        --results-dir results/v1/data
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_NUM_RANDOM_CONFIGS = 50
DEFAULT_NUM_BAG_FOLDS = 8
# Raised from 3600 (2026-09-18): the largest datasets in RamanPreprocessing's k-fold corpus
# (mlrod ~130k rows, bacteria_identification ~78.5k, wheat_lines ~53k) reliably hit
# TimeLimitExceeded on even simple CPU models (PLS/PCR/RIDGE/PCALDA/KNN/SVM) at the old 1h
# budget once available memory was no longer the binding constraint -- raising this alone
# does not slow down small/fast datasets, since AutoGluon returns as soon as fitting
# finishes; it only raises the ceiling for genuinely slow fits. SVM at N~130k may still
# need more (exact-kernel SVM is O(N^2)+); if it keeps failing even at this budget, that is
# a scope question (cap N for SVM, or switch kernel approximation), not a bigger number.
DEFAULT_TIME_LIMIT = 18000
DEFAULT_N_REPEATS = 10
DEFAULT_N_SPLITS = 3

def _resolve_num_cpus() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            return int(val)
    return os.cpu_count() or 1


def _resolve_num_gpus(use_gpu: bool) -> int:
    if not use_gpu:
        return 0
    try:
        import torch

        return 1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def _load_recipe_config(recipe_config_path: str | None) -> tuple[dict | None, dict | None]:
    """Load a preprocessing recipe file and return ``(preprocessing_config, preprocessing_params)``.

    ``recipe_config_path`` uses the *same* JSON schema as the RamanPreprocessing
    repo's own recipe configs (e.g. ``RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json``):
    a top-level ``"preprocessing"`` dict/bool (step-key -> enabled) and an optional flat
    ``"preprocessing_params"`` override dict. Only those two keys are read here -- every
    other key in the file (``datasets_regression``, ``models``, ``autogluon_time_limit``,
    ``subsample``, ...) is ignored, so this script can point ``--recipe-config`` directly at
    an existing RamanPreprocessing recipe file with no duplication or new recipe format.

    Reuses ``raman_bench.config``'s own normalisation helpers
    (``_normalize_preprocessing_config`` / ``_normalize_preprocessing_params``) rather than
    re-deriving the ``True``/``False``/dict-shorthand handling. Returns ``(None, None)`` if ``recipe_config_path`` is
    ``None`` (no recipe -- every preprocessing step stays at the model class's own default,
    matching the current, pre-recipe-argument behaviour of this script).
    """
    if recipe_config_path is None:
        return None, None

    import json

    from raman_bench.config import _normalize_preprocessing_config, _normalize_preprocessing_params

    with open(recipe_config_path) as f:
        raw = json.load(f)
    raw = _normalize_preprocessing_config(raw)
    raw = _normalize_preprocessing_params(raw)
    return raw["preprocessing_config"], raw["preprocessing_params"]


def _import_generator(model_key: str):
    """Return the ``gen_<key>`` ``ConfigGenerator`` for a model key.

    Tries the new per-model-directory convention first
    (``raman_bench.models.custom.<key>.hpo``, see ``raman_bench.models.discover``),
    falling back to the older flat-file convention
    (``raman_bench.models.generate.<key>``) for models not yet migrated -- both
    conventions coexist (see ``RamanBench/.claude/agents/model-agent.md``). A package
    holding several models (``models/custom/aeon/``) is found through its
    ``ModelInfo.search_space`` instead.
    """
    module_key = model_key.lower().replace("-", "_").replace(".", "")
    gen_name = f"gen_{module_key}"
    try:
        module = importlib.import_module(f"raman_bench.models.custom.{module_key}.hpo")
    except ImportError:
        try:
            module = importlib.import_module(f"raman_bench.models.generate.{module_key}")
        except ImportError:
            from raman_bench.models.discover import discover_custom_models

            info = discover_custom_models().get(model_key.upper())
            if info is None:
                raise
            return info.search_space
    if not hasattr(module, gen_name):
        raise AttributeError(
            f"{module.__name__} has no attribute {gen_name!r}. "
            f"Add an hpo.py (or generate.py) module for model key {model_key!r} first."
        )
    return getattr(module, gen_name)


def run_one(
    *,
    dataset_name: str,
    target_idx: int,
    model_key: str,
    repeat: int,
    fold: int,
    config_index: int,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_splits: int = DEFAULT_N_SPLITS,
    num_random_configs: int = DEFAULT_NUM_RANDOM_CONFIGS,
    num_bag_folds: int = DEFAULT_NUM_BAG_FOLDS,
    time_limit: float = DEFAULT_TIME_LIMIT,
    results_dir: str = "results/v1/data",
    cache_dir: str = ".cache_v1",
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
    use_mirror: bool = True,
    use_gpu: bool = False,
    scratch_dir: str | None = None,
    min_samples_per_class: int = 9,
    filter_unlabeled: bool = True,
    recipe_config: str | None = None,
    max_train_samples: int | None = None,
    force_recompute: bool = False,
) -> dict | None:
    """Run exactly one (model, dataset, target, repeat, fold, config) job and cache the result.

    Returns ``None`` (a clean, expected skip -- not an error) if this target
    fails rare-class filtering (classification only, matching how the old
    ``RamanBenchmark`` silently excluded such keys from the benchmark), if
    every row for this target has a missing (NaN) label, if ``model_key`` only
    supports the other problem type (``wrapped_models.CLASSIFICATION_ONLY_MODELS``/
    ``REGRESSION_ONLY_MODELS``), if ``model_key`` is capped below this dataset's
    feature count (``wrapped_models.MAX_FEATURES_MODELS``, e.g. TABSTAR), or if
    ``model_key`` is predicted to exceed its VRAM budget on this dataset's
    feature-count/row-count combination (``wrapped_models.VRAM_CAPPED_MODELS``,
    e.g. ORIONMSP).

    ``filter_unlabeled`` (default ``True``) drops rows with a missing (NaN)
    target before splitting -- today's models all require a real training
    label, so this is the only currently-usable setting for a normal
    supervised run. Set ``False`` to keep unlabeled rows instead: they flow
    into ``splitting.py``'s NaN-aware ``_repeated_kfold_splits``, which keeps
    them out of every test fold (no ground truth to score against) but
    includes them in every train fold -- the data-layer half of
    semi-supervised benchmarking support. No model in this codebase yet
    consumes unlabeled training rows (that needs a model wired to something
    like AutoGluon's ``fit_pseudolabel``, deliberately out of scope here) --
    running a normal model with ``filter_unlabeled=False`` will still fail
    inside AutoGluon's own fit call, with AutoGluon's own clear "NaN in
    label" error, which is expected until such a model exists.

    ``recipe_config`` (default ``None``) names a preprocessing recipe JSON file, in the
    *same* schema as the RamanPreprocessing repo's own recipe configs (e.g.
    ``RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json``) -- a top-level
    ``"preprocessing"`` dict/bool and optional ``"preprocessing_params"`` override dict.
    Applied via :func:`raman_bench.model.build_prep_model_hyperparameters`, the
    restriction-application code path shared by this script and any other caller
    building ``Prep_*`` hyperparameters from a restriction dict. ``None``
    (the default) means "use the model class's own preprocessing defaults, unrestricted" --
    the only behaviour this script had before this argument existed. Only applies when
    ``model_key`` resolves to a ``RamanPreprocessingMixin`` subclass (every model in
    ``wrapped_models.PREPROCESSED_MODELS``, i.e. every model in this repo's curated grid);
    for any other model class the recipe is ignored with a warning, matching how
    ``create_preprocessed_hyperparameters`` silently passes such models through with no
    preprocessing hyperparameters at all.
    """
    from raman_bench.experiment_utils import (
        bag_experiment_kwargs,
        build_task,
        load_dataframe,
        run_cached,
    )
    from raman_bench.model import build_prep_model_hyperparameters
    from raman_bench.models.registry import infer_model_cls
    from raman_bench.preprocessing.mixin import RamanPreprocessingMixin
    from raman_bench.preprocessing.wrapped_models import (
        CLASSIFICATION_ONLY_MODELS,
        MAX_FEATURES_MODELS,
        REGRESSION_ONLY_MODELS,
        VRAM_CAPPED_MODELS,
    )

    num_cpus = _resolve_num_cpus()
    num_gpus = _resolve_num_gpus(use_gpu)
    logger.info("Resources: num_cpus=%d num_gpus=%d", num_cpus, num_gpus)

    # Mirror-first loading with fallback to the original source; use_mirror=False
    # forces the original source. Shared with raman_bench.evaluate.
    dataset, df, sample_idx, problem_type = load_dataframe(
        dataset_name,
        target_idx,
        max_train_samples=max_train_samples,
        cache_dir=cache_dir,
        use_mirror=use_mirror,
        mirror_repo=mirror_repo,
    )

    # Model/dataset compatibility checks that AutoGluon itself would also
    # reject (via a clean `ConstraintViolationError` skip -- see
    # `wrapped_models.CLASSIFICATION_ONLY_MODELS`/`REGRESSION_ONLY_MODELS`/
    # `MAX_FEATURES_MODELS`'s own docstrings), but RamanBench's own cluster
    # jobs always fit exactly one model per `TabularPredictor.fit()` call, with
    # no other model for AutoGluon to fall back on -- so its own
    # `raise_on_no_models_fitted=True` default turns that same clean skip into
    # a job-crashing `RuntimeError: No models were trained successfully during
    # fit()` instead (confirmed with a real local run: ORIONMSP, a
    # classification-only model, against a regression dataset). Checked here,
    # before any of the (comparatively expensive) split/task-building work
    # below, for a real clean exit -- same convention as the rare-class/
    # all-NaN-label skips further down (log a message, return None, no
    # results.pkl written, exit 0).
    model_key_upper = model_key.upper()
    if problem_type == "regression" and model_key_upper in CLASSIFICATION_ONLY_MODELS:
        logger.info(
            "Skipping %s target %d: %s only supports classification tasks",
            dataset_name,
            target_idx,
            model_key,
        )
        return None
    if problem_type == "classification" and model_key_upper in REGRESSION_ONLY_MODELS:
        logger.info(
            "Skipping %s target %d: %s only supports regression tasks",
            dataset_name,
            target_idx,
            model_key,
        )
        return None
    if model_key_upper in MAX_FEATURES_MODELS:
        max_features = MAX_FEATURES_MODELS[model_key_upper]
        n_features = df.shape[1] - 1  # exclude label_col; group_id (if any) isn't added yet here
        if n_features > max_features:
            logger.info(
                "Skipping %s target %d: %s is capped at max_features=%d, but this "
                "dataset has %d features",
                dataset_name,
                target_idx,
                model_key,
                max_features,
                n_features,
            )
            return None
    if model_key_upper in VRAM_CAPPED_MODELS:
        # Joint (features AND rows) counterpart to the MAX_FEATURES_MODELS check
        # above -- see wrapped_models.VRAM_CAPPED_MODELS's own docstring for why
        # a single max_features number can't express this model's real
        # constraint. `n_rows` mirrors the outer-CV train partition size this
        # job will actually fit on (n_splits-1/n_splits of the full dataset,
        # BEFORE any rare-class/NaN-label filtering below -- a slight
        # over-estimate, i.e. conservative in the skip direction).
        n_features = df.shape[1] - 1
        n_rows_estimate = round(len(df) * (n_splits - 1) / n_splits)
        if VRAM_CAPPED_MODELS[model_key_upper](n_features, n_rows_estimate):
            logger.info(
                "Skipping %s target %d: %s is predicted to exceed its VRAM "
                "budget at %d features x ~%d rows",
                dataset_name,
                target_idx,
                model_key,
                n_features,
                n_rows_estimate,
            )
            return None

    # Group-id inference, NaN-feature/NaN-label row drops, rare-class filtering, and
    # the actual repeated-k-fold split are all shared with scripts/run_autogluon_baseline.py
    # (the whole-predictor AutoGluon baseline runner) -- see
    # raman_bench.experiment_utils.build_task's own docstring for why this must stay
    # one shared implementation rather than two independently-written copies.
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
        filter_unlabeled=filter_unlabeled,
    )
    if task_wrapper is None:
        return None

    # 2026-10-02: reversed the 2026-09-25 "no small-dataset bag-fold scaling"
    # decision for genuinely tiny datasets -- confirmed concretely for
    # diabetes_skin_vein/diabetes_skin_ear_lobe (20 rows, 9/11 class split):
    # num_bag_folds=8 on a ~13-14 row training partition gives each internal
    # bag-fold only ~1.6-1.75 held-out rows on average, so individual bag-folds
    # routinely end up with a single, trivially-one-class holdout, crashing
    # AutoGluon's ROC AUC computation -- not a class-imbalance problem (9/11 is
    # nearly balanced), a structural mismatch between a fixed num_bag_folds and
    # a tiny train partition. Below 50 rows, use 3 bag-folds instead of
    # whatever was configured (still >=2, ValidationProtocol's own minimum).
    # Explicit instruction: inconsistency with already-cached 8-fold results
    # for these same tiny datasets is accepted, not reconciled -- a changed
    # num_bag_folds only applies to genuinely new (not-yet-cached) work anyway,
    # per the cache-key comment below.
    _TINY_DATASET_ROW_THRESHOLD = 50
    _TINY_DATASET_NUM_BAG_FOLDS = 3
    effective_num_bag_folds = num_bag_folds
    if len(df) < _TINY_DATASET_ROW_THRESHOLD:
        effective_num_bag_folds = _TINY_DATASET_NUM_BAG_FOLDS
        logger.info(
            "%s on %s: %d row(s) < %d -- using num_bag_folds=%d instead of the "
            "configured %d",
            model_key, dataset_name, len(df), _TINY_DATASET_ROW_THRESHOLD,
            effective_num_bag_folds, num_bag_folds,
        )

    model_cls = infer_model_cls(model_key)
    gen = _import_generator(model_key)

    # Apply the requested preprocessing recipe (§6.1 of
    # docs/kfold_priority_plan.md in the RamanPreprocessing repo -- this script previously had
    # no way to specify a recipe at all, only --config-index for model hyperparameters).
    # Uses raman_bench.model.build_prep_model_hyperparameters, the shared
    # restriction-application code path, so a given recipe file produces
    # identical prep_*_enabled/prep_* hyperparameters regardless of caller. `optimize=False`
    # is hardcoded here: the k-fold plan pins --config-index 0 (default model
    # hyperparameters, no HPO) for every job, so preprocessing HPO search-space injection
    # (only relevant when optimize=True) never applies.
    extra_model_hyperparameters = None
    if recipe_config is not None:
        if isinstance(model_cls, type) and issubclass(model_cls, RamanPreprocessingMixin):
            preprocessing_config, preprocessing_params = _load_recipe_config(recipe_config)
            base_cfg: dict = {}
            if preprocessing_config is not None:
                base_cfg["_prep_restriction"] = preprocessing_config
            prep_hyperparameters = build_prep_model_hyperparameters(
                model_cls,
                base_cfg,
                preprocessing_params=preprocessing_params,
                optimize=False,
                model_extra_params=None,
            )
            if prep_hyperparameters:
                extra_model_hyperparameters = prep_hyperparameters
                logger.info(
                    "%s: applying recipe %s -> %s",
                    model_key,
                    recipe_config,
                    prep_hyperparameters,
                )
        else:
            logger.warning(
                "recipe_config=%r ignored: %s (%s) is not a RamanPreprocessingMixin "
                "subclass -- no prep_*_enabled hyperparameters to restrict.",
                recipe_config,
                model_key,
                model_cls,
            )

    # Shared with raman_bench.evaluate, see bag_experiment_kwargs for why each setting
    # is what it is (ValidationProtocol, require_warmup=False, fold-config-wise seeds).
    generate_kwargs = bag_experiment_kwargs(
        num_random_configs=num_random_configs,
        time_limit=time_limit,
        num_bag_folds=effective_num_bag_folds,
        scratch_dir=scratch_dir,
    )
    if extra_model_hyperparameters:
        generate_kwargs["extra_model_hyperparameters"] = extra_model_hyperparameters
    experiments = gen.generate_all_bag_experiments(**generate_kwargs)
    if config_index >= len(experiments):
        raise IndexError(
            f"config_index={config_index} out of range for {model_key} "
            f"(only {len(experiments)} configs generated with num_random_configs={num_random_configs})"
        )
    experiment = experiments[config_index]
    assert experiment.method_kwargs["model_cls"] is model_cls, (
        f"generate.py for {model_key!r} produces a different model_cls "
        f"({experiment.method_kwargs['model_cls']}) than the registry ({model_cls})"
    )

    # Disambiguate the on-disk cache by recipe: experiment.name only encodes the model
    # class + HPO-config suffix (e.g. "PLS_c1_BAG_L1"), not which preprocessing recipe was
    # applied, so two different recipes for the same (model, dataset, repeat, fold) would
    # otherwise silently collide on the same cache_path and overwrite each other's
    # results.pkl. Only append a recipe segment when a recipe is actually given, so cache
    # entries from before this argument existed (recipe_config=None, the only mode this
    # script supported previously) resolve to the exact same path as before -- no cache
    # invalidation for already-completed no-recipe jobs.
    experiment_dir_name = experiment.name
    if recipe_config is not None:
        recipe_slug = os.path.splitext(os.path.basename(recipe_config))[0]
        experiment_dir_name = f"{experiment.name}__recipe_{recipe_slug}"
    # An already-cached result stands as final unless force_recompute is set (see
    # run_cached and docs/internal/invalidating-results.md).
    return run_cached(
        experiment,
        task_name=task_name,
        task_wrapper=task_wrapper,
        repeat=repeat,
        fold=fold,
        results_dir=results_dir,
        experiment_dir_name=experiment_dir_name,
        force_recompute=force_recompute,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="raman_data dataset name")
    parser.add_argument("--target-idx", type=int, default=0)
    parser.add_argument("--model", required=True, help="Model key, e.g. PLS, GBM, REZERONET")
    parser.add_argument(
        "--repeat", type=int, required=True, help="Which repeat (of --n-repeats) to run in this job"
    )
    parser.add_argument(
        "--fold", type=int, required=True, help="Which fold (of --n-splits) to run in this job"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Must be identical across every job for this (dataset, target) -- defines the repeated-kfold scheme",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Must be identical across every job for this (dataset, target) -- folds per repeat",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap a dataset to at most this many rows (fixed random_state=0 sample) before "
        "splitting/fitting. None (default) leaves every dataset at full size -- this is an "
        "explicit, opt-in change to what is measured for a dataset, not a performance tweak; "
        "added 2026-09-18 for mlrod (~130k rows), whose per-fold AsLS baseline-correction cost "
        "(a pure-Python per-spectrum loop in raman_preprocessing.py) dominated wall-clock time.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute and overwrite an already-cached result.pkl instead of using it as-is. "
        "Default policy is to never touch a cached result (see run_one's own docstring on why), "
        "so this is an explicit opt-in for the one case that should override it: a "
        "dataset-DEFINITION fix (bad task_type, mislabeled targets, ...) makes the cached "
        "result itself meaningless, not just out of date relative to a compute-budget change. "
        "See docs/internal/invalidating-results.md.",
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=0,
        help="0 = default config (_c1); 1..num-random-configs = HPO pool config (_rN)",
    )
    parser.add_argument(
        "--num-random-configs",
        type=int,
        default=DEFAULT_NUM_RANDOM_CONFIGS,
        help="Must be identical across every job for this model (fixes config-pool identity)",
    )
    parser.add_argument("--num-bag-folds", type=int, default=DEFAULT_NUM_BAG_FOLDS)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument(
        "--use-mirror", action=argparse.BooleanOptionalAction, default=True,
        help="Load via the HF mirror first, falling back to the original raman-data "
             "source on a miss (default: on -- much faster and more reliable). Pass "
             "--no-use-mirror to force direct raman-data access.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--recipe-config",
        default=None,
        help="Path to a preprocessing recipe JSON file, same schema as the RamanPreprocessing "
        "repo's own recipe configs (e.g. RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json): "
        "a top-level \"preprocessing\" dict/bool and optional \"preprocessing_params\" "
        "override dict. Only those two keys are read -- every other key (datasets, models, "
        "autogluon_*, subsample, ...) is ignored, so this can point directly at an existing "
        "RamanPreprocessing recipe file. Applied via "
        "raman_bench.model.build_prep_model_hyperparameters. Default: None (no recipe -- the model "
        "class's own preprocessing defaults, unrestricted; this was the only behaviour "
        "before this argument existed).",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Deterministic path for AutoGluon's predictor artifacts (for cluster cleanup); "
        "defaults to AutoGluon's own relative AutogluonModels/ag-<timestamp> under cwd",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=9,
        help="Classification only: drop classes with fewer rows than this; skip the target "
        "cleanly (exit 0, no error) if fewer than 2 classes remain. 0 disables filtering.",
    )
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep rows with a missing (NaN) target instead of dropping them before "
        "splitting, for semi-supervised benchmarking. They're excluded from every "
        "test fold but included in every train fold (see splitting.py's "
        "_repeated_kfold_splits). No model in this codebase yet fits on unlabeled "
        "training rows, so a normal model will still fail inside AutoGluon's own "
        "fit call -- this only affects the split, not model fitting. Default: off "
        "(unlabeled rows are dropped, matching every prior run).",
    )
    args = parser.parse_args()

    out = run_one(
        dataset_name=args.dataset,
        target_idx=args.target_idx,
        model_key=args.model,
        repeat=args.repeat,
        fold=args.fold,
        config_index=args.config_index,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        num_random_configs=args.num_random_configs,
        num_bag_folds=args.num_bag_folds,
        time_limit=args.time_limit,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        mirror_repo=args.mirror_repo,
        use_mirror=args.use_mirror,
        use_gpu=args.use_gpu,
        scratch_dir=args.scratch_dir,
        min_samples_per_class=args.min_samples_per_class,
        filter_unlabeled=not args.keep_unlabeled,
        recipe_config=args.recipe_config,
        max_train_samples=args.max_train_samples,
        force_recompute=args.force_recompute,
    )
    if out is None:
        logger.info("Target skipped (see the reason logged above) -- clean exit, not an error.")



if __name__ == "__main__":
    main()
