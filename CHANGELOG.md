# Changelog

All notable changes to RamanBench are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-04-14

### Added

- Initial public release of RamanBench.
- **74 datasets** across Chemical, Medical, Biological, and Material Science domains.
- **163 prediction targets** (regression and classification).
- **28 baseline models** evaluated with 3 seeds each:
  - Classical ML: PLS, KNN, LR, RF, XT, GBM, XGB, CatBoost, Dummy
  - Tabular DL: NN_TORCH, FastAI, RealMLP
  - Foundation models: TabPFN v2, TabPFN v2.5, MITRA, TabM, TabDPT, TabICL
  - Time-series: ROCKET, ARSENAL
  - Raman-specific: DeepCNN, RamanNet, SANet, RamanFormer, RamanTransformer,
    ReZeroNet, FC-ResNeXt, CoAtNet
  - AutoGluon ensemble (AUTOGLUON)
- **Leaderboard** with Elo, Score, Avg Rank, Improvability, and timing metrics.
- **Precomputed results** bundled in `data/precomputed/`.
- **`Leaderboard` class** for evaluating new models against baselines.
- **17 new datasets** released alongside the paper (see [NEW_DATASETS.md](NEW_DATASETS.md)).
- Raman preprocessing pipeline: cosmic-ray removal, baseline correction, MSC,
  denoising, SNV, standard scaling, spectral augmentation.
- AutoGluon HPO mixin (`RamanPreprocessingMixin`) for jointly optimising
  preprocessing and model hyperparameters.
- Grouped regression splits to prevent data leakage from replicate measurements.
- `raman-bench` CLI entry point.
- Example notebooks in `notebooks/`.

### Links

- raman-data: https://github.com/ml-lab-htw/raman_data | `pip install raman-data`
- raman-bench: https://github.com/ml-lab-htw/RamanBench | `pip install raman-bench`
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Paper (under review): https://arxiv.org/abs/2605.02003

---

## Unreleased

RamanBench v1: migrates the model/metrics/splitting layer onto `tabarena`/`bencheval`
directly, rather than reimplementing patterns "inspired by" them.

### Added

- Real repeated k-fold cross-validation (`RepeatedStratifiedKFold`/`RepeatedKFold` for
  ungrouped datasets; a manually-repeated `StratifiedGroupKFold`/`GroupKFold` for datasets
  with physical-replicate structure, since sklearn has no repeated wrapper for those),
  replacing the previous scheme of 3 independent holdout splits with no folds.
  `StratifiedGroupKFold` also lifts the old scheme's limitation that a grouped
  classification split couldn't additionally be class-balanced.
- `raman_bench.splitting.get_n_repeats`: dataset-size-adaptive repeat count (10 repeats
  under 2,500 instances, 3 up to 250,000, 1 above), ported directly from
  `tabarena.nips2025_utils.fetch_metadata._get_n_repeats` and confirmed against TabArena's
  own real 51-dataset curated metadata (`num_folds` is a fixed 3 for every dataset there;
  only `n_repeats` varies by size) -- a uniform repeat count for every dataset regardless
  of size is not what the actual protocol does.
- `scripts/build_target_list.py`: builds a (dataset, target) list from dataset-name JSON
  files, recording each target's real name, instance count, and size-adaptive `n_repeats`;
  `cluster/submit_full_benchmark.py` now reads each target's own `n_repeats` from this
  list instead of applying one value to every dataset.
- `add_seed="fold-config-wise"` on every generated config, matching TabArena's own real
  production default (`tabflow_slurm/setup_slurm_base_v2.py`'s `default_seed_config`) --
  the bare library default is `"static"` (every bag-fold and every HPO config shares
  internal seed 0); this gives each of AutoGluon's internal bag-folds, and each HPO config
  once HPO is opted in, a genuinely different internal random seed.
- 2 new datasets: `chlorinated_samples` (binary classification, chloroform detection in
  Raman spectra) and `locust_phase_hemolymph` (binary classification, density-dependent
  phase state in desert locusts -- the first dataset in `raman-data` with `group_ids`
  populated from real source metadata rather than inferred from target values).
- `DatasetInfo.is_grouped` (in `raman-data`): a filterable field marking whether a dataset
  has confirmed physical-replicate structure.
- Public, cluster-agnostic `cluster/` job-submission tooling: `submit_job.py`,
  `run_experiment.sbatch`, `detect_cluster.py`, `janitor.py`, `refresh_deps.py` (refreshes
  the released `raman-data` version and the `RamanBench` checkout once before a submission
  batch, rather than each job pulling for itself).
- `raman_bench.models.registry` (`raman_bench_model_registry`), built on TabArena's own
  model registry.
- Six Claude Code agents (dataset/model/cluster/leaderboard/frontend/docs) under
  `.claude/agents/`.
- Optional semi-supervised benchmarking support: `raman_bench.splitting`'s repeated
  k-fold splitting is now NaN-target-aware -- unlabeled (NaN-label) rows are never placed
  in a test fold (no ground truth to score against) but are included in every train fold
  (for grouped datasets, only when they don't share a fold's test group). `scripts/
  run_experiment.py --keep-unlabeled` (default: off, preserving every prior run's
  behavior) opts a job into this instead of the default unconditional NaN-drop. No model
  in this codebase yet fits on unlabeled training rows; this is data-layer plumbing for
  a future semi-supervised-aware model.
- `DatasetInfo.has_missing_labels` (in `raman-data`): a filterable field marking whether
  a dataset has confirmed NaN values in its target column(s), mirroring `is_grouped`'s
  True/False/None semantics.

### Changed

- `tabarena`/`bencheval` dependencies now point at upstream `autogluon/tabarena`
  (`packages/{tabarena,bencheval}`, since upstream's monorepo restructure) instead of a
  personal fork. Verified end-to-end by installing `packages/tabarena[benchmark]` fresh
  (per upstream's own quickstart) and running its full quickstart script.
- The `models` extra now depends on `tabarena[tabicl,ebm,search_spaces,realmlp,tabdpt,tabm]`
  instead of hand-listing a subset of the same per-model packages a second time -- that
  hand-listed subset had drifted out of sync with upstream and was missing `tabicl`/`tabm`
  (both import fine from `autogluon.tabular.models` -- the model class is always
  discoverable -- but actually fitting either failed on a missing package, since
  AutoGluon's own model classes import their backing library lazily inside `_fit()`, not
  at class-definition time). Deliberately not tabarena's full `[benchmark]` union: that
  also pulls `[tabpfn]` (`tabpfn>=8.0.8`), which conflicts with `tabpfnwide` (every
  release exact-pins an older `tabpfn`) -- confirmed via a real pip resolve
  (`ResolutionImpossible`). RamanBench keeps its own, more permissive `tabpfn`/
  `tabpfn-extensions` floor instead, which every `tabpfnwide` release's exact pin already
  satisfies. Installing `bencheval` needs `uv pip install` (resolves its
  `[tool.uv.workspace]`-only reference to its sibling package automatically) or a manual
  `pip install bencheval @ git+...#subdirectory=packages/bencheval` before `pip install
  raman-bench[...]` -- plain pip cannot resolve tabarena's own bare `bencheval`
  dependency in one shot (confirmed via a real pip resolve: "No matching distribution
  found for bencheval").
- Model onboarding: new models should follow the `models/custom/<key>/{model.py,hpo.py,
  info.py}` per-directory convention (auto-discovered via
  `raman_bench.models.discover.discover_custom_models`), matching upstream TabArena's own
  post-restructure onboarding pattern (`ModelInfo`/`discover_models`). `ridge` migrated as
  the reference implementation; other already-integrated models are unaffected and keep
  working via the previous flat-file convention.

- Metrics now use TabArena/AutoGluon's own problem-type-appropriate defaults (ROC AUC for
  binary classification, log loss for multiclass, RMSE for regression) instead of
  RamanBench's earlier bespoke metric computation.
- The job unit changed from (model, dataset, target, seed, config) to (model, dataset,
  target, repeat, fold, config), matching TabArena's own `tabflow_slurm`
  `--repeat`/`--fold` interface rather than a single `--seed`.
- `microgel_size_*` reduced from 14 curated entries to 1 (`microgel_size_raw_global`) --
  the other 13 were preprocessing-variant or spectral-range-subset duplicates (different
  baseline-correction/normalization already applied upstream, or a fingerprint-region
  subset) of the same 235 samples, redundant given RamanBench's own preprocessing
  pipeline.

### Fixed

- `run_experiment.sbatch` now propagates the real Python exit code -- previously a crashed
  job could still report SLURM/`sacct` success, hiding real failures.
- Regression targets with missing (NaN) values are now dropped before fitting instead of
  crashing the whole job (found on datasets where not every sample was characterized for
  every analyte, e.g. `fuel_benchtop`'s target 1: 157/179 NaN).
- `num_bag_folds` now scales down for small datasets/classes, avoiding AutoGluon's internal
  bagging producing a single-class validation fold and crashing ROC AUC computation.
- Fixed 6 real breaks against current upstream `tabarena` (written against the personal
  fork, never updated as upstream's post-fork restructure moved on 224 commits) --
  found by actually installing from upstream fresh and running the real pipeline, not
  just reading source:
  - `models/registry.py`: `tabarena.benchmark.models.model_registry` no longer exists ->
    `tabarena.benchmark.exec_models.registry`.
  - `splitting.py`: `GroupLabelTypes` moved from `tabarena.benchmark.task.user_task`
    (now a type-checking-only import there, invisible at runtime) to
    `tabarena.benchmark.task.metadata`.
  - `models/generate/gbm.py`, `models/generate/ta_tabpfn_3.py`: TabArena's own per-model
    `generate.py` modules were renamed to `hpo.py` (same function names).
  - `scripts/run_experiment.py`: `Experiment.run()` gained a new required `cache_task_key`
    keyword argument (the task's canonical cache identifier); now passed as `task_name`
    (matching `UserTask.cache_key`'s semantics for our tasks).
  - `scripts/aggregate_results.py`: three separate breaks -- `CacheFunctionPickle` now
    gzip-compresses cache writes by default (a raw `pickle.load` no longer works; use
    `tabarena.utils.pickle_utils.load_pickle`, which handles both transparently);
    `EndToEnd.from_raw` no longer accepts a legacy `task_metadata` DataFrame directly
    (wrap it in `TaskMetadataCollection.from_legacy_df`, which needs a fuller column set
    than this repo's frame used to carry -- `n_folds`/`n_repeats` are now derived
    accurately from what was actually run per task, other per-dataset stats not tracked
    in the results cache are explicit placeholders); `EndToEnd.from_raw` now returns
    `EndToEndResults` directly, with `get_results(use_model_results=True/False)`
    replacing the old `.to_results().model_results`/`.hpo_results`.
  - Verified end-to-end, not just import-checked: a real `run_experiment.py` ->
    `aggregate_results.py` round trip reproduced the same `metric_error` through both
    the raw cache and the aggregated `hpo_results` row.

### Still planned

- Sphinx documentation site
- Contribution templates for new datasets and models
