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

This is an ADDITIVE, parallel execution path. It does not replace or modify
``raman_bench.model.AutoGluonModel`` / ``raman_bench.predictions``, which the
paper repo's currently-published/in-flight v0/v1 results still depend on.

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
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_NUM_RANDOM_CONFIGS = 50
DEFAULT_NUM_BAG_FOLDS = 8
DEFAULT_TIME_LIMIT = 3600
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

    ``recipe_config_path`` uses the *same* JSON schema as Pipeline A's own recipe configs
    (e.g. ``RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json``): a top-level
    ``"preprocessing"`` dict/bool (step-key -> enabled) and an optional flat
    ``"preprocessing_params"`` override dict. Only those two keys are read here -- every
    other key in the file (``datasets_regression``, ``models``, ``autogluon_time_limit``,
    ``subsample``, ...) is ignored, so Pipeline B can point ``--recipe-config`` directly at
    an existing Pipeline A recipe file with no duplication or new recipe format.

    Reuses ``raman_bench.config``'s own normalisation helpers
    (``_normalize_preprocessing_config`` / ``_normalize_preprocessing_params``) rather than
    re-deriving the ``True``/``False``/dict-shorthand handling -- the exact same normalisation
    Pipeline A's ``load_config`` applies. Returns ``(None, None)`` if ``recipe_config_path`` is
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
    conventions coexist (see ``RamanBench/.claude/agents/model-agent.md``).
    """
    module_key = model_key.lower().replace("-", "_").replace(".", "")
    gen_name = f"gen_{module_key}"
    try:
        module = importlib.import_module(f"raman_bench.models.custom.{module_key}.hpo")
    except ImportError:
        module = importlib.import_module(f"raman_bench.models.generate.{module_key}")
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
    *same* schema as Pipeline A's own recipe configs (e.g.
    ``RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json``) -- a top-level
    ``"preprocessing"`` dict/bool and optional ``"preprocessing_params"`` override dict.
    Applied via the same restriction-application code path Pipeline A's
    ``AutoGluonModel._build_model_hyperparameters`` uses
    (:func:`raman_bench.model.build_prep_model_hyperparameters`), so a given recipe produces
    identical ``prep_*_enabled``/``prep_*`` hyperparameters under either pipeline. ``None``
    (the default) means "use the model class's own preprocessing defaults, unrestricted" --
    the only behaviour this script had before this argument existed. Only applies when
    ``model_key`` resolves to a ``RamanPreprocessingMixin`` subclass (every model in
    ``wrapped_models.PREPROCESSED_MODELS``, i.e. every model in this repo's curated grid);
    for any other model class the recipe is ignored with a warning, matching how Pipeline A's
    ``create_preprocessed_hyperparameters`` silently passes such models through with no
    preprocessing hyperparameters at all.
    """
    from raman_data import TASK_TYPE
    from tabarena.utils.cache import CacheFunctionPickle

    from raman_bench.benchmark import RamanBenchmark
    from raman_bench.model import build_prep_model_hyperparameters
    from raman_bench.models.registry import infer_model_cls
    from raman_bench.preprocessing.mixin import RamanPreprocessingMixin
    from raman_bench.preprocessing.wrapped_models import (
        CLASSIFICATION_ONLY_MODELS,
        MAX_FEATURES_MODELS,
        REGRESSION_ONLY_MODELS,
        VRAM_CAPPED_MODELS,
    )
    from raman_bench.splitting import (
        GROUP_COL,
        RamanBenchTaskWrapper,
        TooFewClassesError,
        build_user_task,
        filter_rare_classes,
        infer_group_ids_from_targets,
    )

    num_cpus = _resolve_num_cpus()
    num_gpus = _resolve_num_gpus(use_gpu)
    logger.info("Resources: num_cpus=%d num_gpus=%d", num_cpus, num_gpus)

    # Reuses RamanBenchmark's existing, tested mirror-first (with fallback)
    # dataset loading -- no need for the full init_datasets() index/cache
    # bookkeeping, just the one dataset this job needs. Mirror-first is the
    # default (much faster and more reliable than every original source);
    # use_mirror=False forces direct raman-data (original source) access.
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
    label_col = df.columns[-1]
    problem_type = (
        "classification" if dataset.task_type == TASK_TYPE.Classification else "regression"
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

    # Explicit per-spectrum group ids (from a loader that knows the dataset's
    # replicate structure, e.g. locust_phase_hemolymph's real Sample column)
    # take priority. Most datasets don't have one yet, so fall back to
    # inferring group structure from the dataset's own full target matrix
    # (regression only -- see infer_group_ids_from_targets's docstring for
    # why this is invalid for classification) rather than silently running
    # every dataset ungrouped.
    if GROUP_COL not in df.columns and problem_type == "regression":
        inferred = infer_group_ids_from_targets(dataset.targets)
        if inferred is not None:
            df[GROUP_COL] = inferred
            logger.info(
                "%s: no explicit group_ids -- inferred %d group(s) from matching target values",
                dataset_name,
                len(set(inferred.tolist())),
            )

    # Drop rows with any NaN in their *feature* (spectral) columns, before any
    # preprocessing recipe or split (but after the group-id inference above,
    # which must see one row per original sample to align positionally with
    # ``dataset.targets``). Confirmed isolated to exactly two datasets across
    # the full 77-dataset corpus (checked directly, not assumed):
    # adenine_colloidal_silver (45/630 rows) and adenine_solid_silver
    # (135/1851 rows) each carry a block of NaN columns for a subset of
    # samples -- a genuine measurement-range gap in the source data (those
    # samples were acquired over a narrower Raman-shift window than the
    # dataset's common resampled column grid, not a raman_data loader bug:
    # colloidal_silver's NaN block is a single contiguous trailing run
    # (indices 467-533 of 534 columns), solid_silver's is a leading+trailing
    # pair (indices 0-38 and 394-533) -- both consistent with "these samples
    # just weren't measured over the full range", not corruption. Sklearn's
    # own `fit()` for PLS/RIDGE/SVM rejects NaN outright (`ValueError: Input
    # X contains NaN`), while KNN's sklearn wrapper tolerates it -- so
    # without this fix, PLS/RIDGE/SVM failed 100% deterministically on these
    # two datasets regardless of model/recipe/repeat/fold (see
    # docs/kfold_priority_plan.md finding 3 in the RamanPreprocessing repo).
    # Applied here unconditionally for every dataset (not gated to these two
    # by name), so it's a single, traceable fix rather than a per-dataset
    # special case, and so sample counts stay identical across every model
    # for a given dataset (KNN included) -- this mirrors Pipeline A's
    # existing, older `RamanBenchmark._load_dataset_from_key`'s
    # `data_df.dropna()`, which already handled this silently for Pipeline
    # A; Pipeline B had no equivalent until now.
    feature_cols = [c for c in df.columns if c not in (label_col, GROUP_COL)]
    nan_feature_mask = df[feature_cols].isna().any(axis=1)
    n_nan_feature_rows = int(nan_feature_mask.sum())
    if n_nan_feature_rows:
        logger.info(
            "%s target %d: dropping %d/%d rows with a NaN value in their feature "
            "(spectral) columns before any split or preprocessing",
            dataset_name,
            target_idx,
            n_nan_feature_rows,
            len(df),
        )
        df = df[~nan_feature_mask]

    # Drop rows with a missing (NaN) label. Common in multi-target regression
    # datasets where not every sample was characterized for every analyte
    # (e.g. fuel_benchtop: target 0 has 0 NaNs, every other target has
    # 15-157 out of 179) -- AutoGluon refuses to fit on a NaN label, and
    # until this was added that crashed the whole job instead of just
    # excluding the unmeasured rows for this particular target.
    #
    # Optional (filter_unlabeled=False): keep unlabeled rows instead, for
    # semi-supervised benchmarking -- see this function's docstring.
    n_before = len(df)
    if filter_unlabeled:
        df = df[df[label_col].notna()]
        if len(df) < n_before:
            logger.info(
                "%s target %d: dropped %d/%d rows with a missing (NaN) label",
                dataset_name,
                target_idx,
                n_before - len(df),
                n_before,
            )
    else:
        n_unlabeled = int(df[label_col].isna().sum())
        if n_unlabeled:
            logger.info(
                "%s target %d: keeping %d/%d unlabeled (NaN-label) rows for "
                "semi-supervised splitting (filter_unlabeled=False)",
                dataset_name,
                target_idx,
                n_unlabeled,
                n_before,
            )
    if df.empty or df[label_col].notna().sum() == 0:
        logger.info(
            "Skipping %s target %d: every row has a missing label", dataset_name, target_idx
        )
        return None

    if problem_type == "classification":
        try:
            df = filter_rare_classes(
                df, label_col=label_col, min_samples_per_class=min_samples_per_class
            )
        except TooFewClassesError as e:
            logger.info("Skipping %s target %d: %s", dataset_name, target_idx, e)
            return None

    task_name = f"{dataset_name}__{target_idx}"
    _, task_obj = build_user_task(
        task_name=task_name,
        df=df,
        label_col=label_col,
        problem_type=problem_type,
        n_repeats=n_repeats,
        n_splits=n_splits,
        group_col=GROUP_COL if GROUP_COL in df.columns else None,
    )
    task_wrapper = RamanBenchTaskWrapper(task=task_obj)

    # AutoGluon's internal bagging splits the TRAINING portion of the data
    # (roughly len(df) * (n_splits - 1) / n_splits rows) into num_bag_folds
    # folds. For a tiny dataset the default of 8 folds can leave very few
    # samples per fold -- confirmed in practice on a real 20-row
    # classification dataset (diabetes_skin_ear_lobe, 9/11 class split): 8
    # folds over a ~16-row training set averages ~2 rows/fold, which is a
    # real chance of landing a validation fold that's entirely one class,
    # crashing AutoGluon's ROC AUC computation ("Only one class present in
    # y_true"). Scale bag folds down for small data rather than let that
    # crash the whole job.
    if problem_type == "classification":
        min_class_count = int(df[label_col].value_counts().min())
        effective_bag_folds = max(2, min(num_bag_folds, min_class_count // 2))
    else:
        n_train_est = max(1, int(len(df) * (n_splits - 1) / n_splits))
        effective_bag_folds = max(2, min(num_bag_folds, n_train_est // 4))
    if effective_bag_folds < num_bag_folds:
        logger.info(
            "%s target %d: reducing num_bag_folds %d -> %d for a small dataset (%d rows)",
            dataset_name,
            target_idx,
            num_bag_folds,
            effective_bag_folds,
            len(df),
        )
        num_bag_folds = effective_bag_folds

    model_cls = infer_model_cls(model_key)
    gen = _import_generator(model_key)

    # Apply the requested preprocessing recipe (§6.1 of
    # docs/kfold_priority_plan.md in the RamanPreprocessing repo -- Pipeline B previously had
    # no way to specify a recipe at all, only --config-index for model hyperparameters).
    # Reuses the exact same restriction-application code path as Pipeline A
    # (raman_bench.model.build_prep_model_hyperparameters, factored out of
    # AutoGluonModel._build_model_hyperparameters), so a given recipe file produces
    # identical prep_*_enabled/prep_* hyperparameters under either pipeline. `optimize=False`
    # is hardcoded here: Pipeline B's k-fold plan pins --config-index 0 (default model
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

    generate_kwargs = dict(
        num_random_configs=num_random_configs,
        time_limit=time_limit,
        num_bag_folds=num_bag_folds,
        fold_fitting_strategy="sequential_local",
        # Matches TabArena's own real production default (tabflow_slurm/
        # setup_slurm_base_v2.py's default_seed_config), not
        # generate_all_bag_experiments' bare "static" default -- gives each
        # of AutoGluon's internal bag-folds, and each HPO config once
        # HPO is opted in, a genuinely different internal random seed
        # instead of all sharing seed 0.
        add_seed="fold-config-wise",
    )
    if extra_model_hyperparameters:
        generate_kwargs["extra_model_hyperparameters"] = extra_model_hyperparameters
    if scratch_dir is not None:
        # Deterministic AutoGluon predictor path for this job (default is a
        # relative AutogluonModels/ag-<timestamp> under cwd) -- lets the
        # sbatch wrapper's cleanup trap find and remove it reliably, even if
        # the job is killed rather than finishing normally.
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        generate_kwargs["method_kwargs"] = {"init_kwargs": {"path": scratch_dir}}
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
    # invalidation for already-completed no-recipe Pipeline B jobs.
    experiment_dir_name = experiment.name
    if recipe_config is not None:
        recipe_slug = os.path.splitext(os.path.basename(recipe_config))[0]
        experiment_dir_name = f"{experiment.name}__recipe_{recipe_slug}"
    cache_path = os.path.join(results_dir, experiment_dir_name, task_name, f"{repeat}_{fold}")
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    cacher = CacheFunctionPickle(
        cache_name="results", cache_path=cache_path, include_self_in_call=True
    )

    logger.info(
        "Running %s on %s repeat=%d fold=%d -> %s",
        experiment.name,
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
        # The task's own canonical cache identifier (UserTask.cache_key == its slug,
        # here identical to task_name) -- a required kwarg as of a later tabarena
        # version than when this was first written; used to key its text-embedding
        # cache scope. We build our own results cache path independently (below),
        # so this only matters for that internal scoping, not for where results land.
        cache_task_key=task_name,
        cacher=cacher,
        ignore_cache=False,
    )
    logger.info("Done: metric_error=%s", out.get("metric_error"))
    return out


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
        help="Path to a preprocessing recipe JSON file, same schema as Pipeline A's own "
        "recipe configs (e.g. RamanPreprocessing/configs/preprocessing_ablation_dl/snv.json): "
        "a top-level \"preprocessing\" dict/bool and optional \"preprocessing_params\" "
        "override dict. Only those two keys are read -- every other key (datasets, models, "
        "autogluon_*, subsample, ...) is ignored, so this can point directly at an existing "
        "Pipeline A recipe file. Applied via the same restriction-application code path "
        "Pipeline A uses (raman_bench.model.build_prep_model_hyperparameters), so results "
        "are directly comparable across pipelines. Default: None (no recipe -- the model "
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
    )
    if out is None:
        logger.info("Target skipped (see the reason logged above) -- clean exit, not an error.")


if __name__ == "__main__":
    main()
