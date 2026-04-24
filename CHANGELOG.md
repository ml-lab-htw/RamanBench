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
- Leaderboard: https://huggingface.co/spaces/ml-lab-htw/RamanBench
- Paper (NeurIPS 2026): https://arxiv.org/abs/TBD

---

## Unreleased

- Sphinx documentation site
- Contribution templates for new datasets and models
