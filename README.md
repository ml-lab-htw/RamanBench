# RamanBench

[![PyPI](https://img.shields.io/pypi/v/raman-bench)](https://pypi.org/project/raman-bench/)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org)
[![CI](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b)](https://arxiv.org/abs/TBD)
[![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-HuggingFace-orange)](https://huggingface.co/spaces/ml-lab-htw/RamanBench)

**A large-scale benchmark for machine learning on Raman spectroscopy data.**

> 74 datasets · 163 prediction targets · 28 baseline models · 4 application domains

RamanBench provides a reproducible evaluation protocol and a curated collection
of public Raman spectroscopy datasets spanning Material Science, Biological,
Medical, and Chemical applications.  Researchers can rank new models against
28 pre-evaluated baselines — from classical PLS to tabular foundation models
and Raman-specific deep learning architectures — without re-running all experiments.

---

## Ecosystem

```
raman-data   ──▶  raman-bench  ──▶  Live Leaderboard
(datasets)        (this package)     HuggingFace Space
PyPI / GitHub     PyPI / GitHub
```

| Resource | Link |
|---|---|
| **raman-data** (dataset loader) | [GitHub](https://github.com/ml-lab-htw/raman_data) · [PyPI](https://pypi.org/project/raman-data/) |
| **raman-bench** (this package) | [GitHub](https://github.com/ml-lab-htw/RamanBench) · [PyPI](https://pypi.org/project/raman-bench/) |
| **Live Leaderboard** | [huggingface.co/spaces/ml-lab-htw/RamanBench](https://huggingface.co/spaces/ml-lab-htw/RamanBench) |
| **Paper** (NeurIPS 2026) | [arXiv TBD](https://arxiv.org/abs/TBD) |

---

## Quick Start

### Installation

```bash
# Core package (leaderboard + dataset loading, no heavy dependencies)
pip install raman-bench
```

**For running the full benchmark** (AutoGluon + deep learning models), RamanBench
requires a patched AutoGluon fork.  The official AutoGluon release caps tabular
foundation models (TabPFN v2, TabICL, TabDPT, MITRA, …) at 500 features and
silently skips them on larger datasets; Raman spectra typically have 500–4000
wavenumber points.  The fork removes this cap.  Install it first:

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
pip install -r requirements-autogluon-fork.txt
pip install "raman-bench[deep]"
```

### Explore the precomputed leaderboard

```python
from raman_bench import Leaderboard

# Load v0.1 results: 28 models × 74 datasets
lb = Leaderboard.from_precomputed()
print(lb.rank())          # ranked DataFrame
lb.plot()                 # horizontal bar chart
```

### Evaluate a new model

```python
from raman_bench import Leaderboard
from sklearn.cross_decomposition import PLSRegression

lb = Leaderboard.from_precomputed()

# Evaluates your model on all 74 datasets (3 seeds) and adds it to the ranking
results = lb.evaluate_and_add(
    model_name="My-PLS-10",
    model=PLSRegression(n_components=10),
)
print(lb.rank())
lb.plot()
```

### Run the full benchmark pipeline

```bash
# 1. Clone, install the AutoGluon fork, then install in development mode
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
pip install -r requirements-autogluon-fork.txt
pip install -e ".[deep]"

# 2. Pre-cache all dataset splits (optional, speeds up the run)
python scripts/prepare_datasets.py --config configs/benchmark_v0.1.json

# 3. Run predictions → metrics
raman-bench run --config configs/benchmark_v0.1.json

# 4. Run a single step
raman-bench run --config configs/benchmark_v0.1.json --step predictions
raman-bench run --config configs/benchmark_v0.1.json --step metrics
```

### Notebooks

| Notebook | Description |
|---|---|
| [`01_quick_start.ipynb`](notebooks/01_quick_start.ipynb) | Load a dataset, explore the precomputed leaderboard, plot rankings |
| [`02_benchmark_new_model.ipynb`](notebooks/02_benchmark_new_model.ipynb) | Evaluate your own model and add it to the leaderboard |
| [`03_explore_results.ipynb`](notebooks/03_explore_results.ipynb) | Deep dive into per-dataset and per-domain results |
| [`04_contribute_dataset.ipynb`](notebooks/04_contribute_dataset.ipynb) | Step-by-step guide to contributing a new dataset |

---

## Benchmark Composition

### Datasets

74 public Raman spectroscopy datasets from four application domains:

| Domain | Datasets | Task | Sources |
|---|---|---|---|
| Chemical | 37 | Regression | Zenodo, HuggingFace |
| Medical | 11 | Classification | Kaggle, Zenodo |
| Biological | 8 | Regression | HuggingFace, Zenodo |
| Material Science | 4 | Classification | RRUFF, Zenodo |

All datasets are accessible via `pip install raman-data`:

```python
from raman_data import raman_data

dataset = raman_data("amino_acids_glycine")
X = dataset.spectra          # (n_samples, n_wavenumbers)
y = dataset.targets          # regression targets or class labels
w = dataset.raman_shifts     # wavenumber axis in cm⁻¹
```

**Dataset catalog:** [raman-data on GitHub](https://github.com/ml-lab-htw/raman_data)

### Models (v0.1 — 28 baselines)

**Classical ML / Spectroscopy**
- PLS (partial least squares)
- KNN, LR, RF, XT, GBM (LightGBM), XGB (XGBoost), CatBoost

**Tabular Deep Learning**
- NN_TORCH, FastAI, RealMLP

**Tabular Foundation Models**
- TabPFN v2, TabPFN v2.5, MITRA, TabM, TabDPT, TabICL

**Time-Series / Spectral Classifiers**
- ROCKET, ARSENAL

**Raman-Specific Neural Networks**
- DeepCNN (Liu et al., 2017)
- RamanNet (Ibtehaz et al., 2023)
- SANet (Deng et al., 2021)
- RamanFormer (Koyun et al., 2024)
- RamanTransformer (Liu et al., 2023)
- ReZeroNet, FC-ResNeXt, CoAtNet (Lange et al., 2025)

**AutoGluon ensemble** (AUTOGLUON)

---

## Ranking Protocol

Models are evaluated under three complementary metrics:

| Metric | Description |
|---|---|
| **Elo** | Pairwise win-rate Elo calibrated to RF = 1000 (200-round bootstrap) |
| **Score** | Normalised per-dataset score: best model = 1, median model = 0 |
| **Avg Rank** | Average rank across all datasets and targets |
| **Improvability** | % gap to the best model, averaged across datasets |

See the [live leaderboard](https://huggingface.co/spaces/ml-lab-htw/RamanBench) for
interactive filtering by model category, task type, and dataset domain.

---

## Repository Structure

```
RamanBench/
├── src/raman_bench/       # Python package (install via pip)
│   ├── benchmark.py       # Dataset loading and caching
│   ├── model.py           # AutoGluon wrapper
│   ├── evaluation.py      # Metric computation (Step 2)
│   ├── predictions.py     # Prediction generation (Step 1)
│   ├── leaderboard.py     # Leaderboard + model evaluation
│   ├── config.py          # JSON config loader
│   ├── preprocessing/     # Raman preprocessing pipeline
│   ├── metrics/           # Classification + regression metrics
│   └── models/custom/     # 9 Raman-specific architectures
├── configs/               # Benchmark configuration files
│   ├── benchmark_v0.1.json
│   ├── models/            # Model lists (all, raman, traditional, foundation)
│   └── datasets/          # Dataset lists (regression_all, classification_all)
├── data/precomputed/      # Bundled v0.1 results (CSVs + dataset_stats.json)
├── notebooks/             # Example Jupyter notebooks
├── scripts/               # CLI scripts (run_benchmark.py, prepare_datasets.py)
├── tests/                 # pytest test suite
└── docs/                  # Sphinx documentation
```

---

## Contributing

We welcome contributions of new models and datasets!

### Adding a New Model

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-model).

Quick summary:
1. Implement your model as an AutoGluon `AbstractModel` subclass (or use the
   `BaseCustomModel` shared training loop).
2. Register it in `configs/models/`.
3. Add tests in `tests/models/`.

### Adding a New Dataset

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-dataset) and
[NEW_DATASETS.md](NEW_DATASETS.md) for detailed instructions and examples.

Quick summary:
1. Upload your dataset to HuggingFace Datasets or Zenodo under CC BY 4.0.
2. Add a loader to the [raman-data](https://github.com/ml-lab-htw/raman_data) package
   (open a PR there).
3. Open an issue here linking to the raman-data PR.

The [live leaderboard](https://huggingface.co/spaces/ml-lab-htw/RamanBench)
also has a "How to Contribute" section with step-by-step instructions.

---

## Citation

If you use RamanBench in your research, please cite:

```bibtex
@inproceedings{koddenbrock2026ramanbench,
  title     = {RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy Data},
  author    = {Koddenbrock, Mario and Lange, Christoph and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/TBD}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Dataset licenses vary; see the [dataset catalog](https://huggingface.co/spaces/ml-lab-htw/RamanBench)
or [raman-data](https://github.com/ml-lab-htw/raman_data) for per-dataset license information.
Most datasets are released under CC BY 4.0.