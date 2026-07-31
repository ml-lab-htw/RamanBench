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
  with physical-replicate structure, since sklearn has no repeated wrapper for those) --
  matches TabArena's own documented convention for custom datasets
  (`RepeatedStratifiedKFold(n_repeats=10, n_splits=3)`), replacing the previous scheme of
  3 independent holdout splits with no folds. `StratifiedGroupKFold` also lifts the old
  scheme's limitation that a grouped classification split couldn't additionally be
  class-balanced.
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

### Changed

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

### Still planned

- Sphinx documentation site
- Contribution templates for new datasets and models
