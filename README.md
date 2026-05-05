# RamanBench

[![PyPI](https://img.shields.io/pypi/v/raman-bench)](https://pypi.org/project/raman-bench/)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org)
[![CI](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.02003-b31b1b)](https://arxiv.org/abs/2605.02003)
[![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-HuggingFace-orange)](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)

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

| Resource                        | Link                                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------------------|
| **raman-data** (dataset loader) | [GitHub](https://github.com/ml-lab-htw/raman_data) · [PyPI](https://pypi.org/project/raman-data/)  |
| **raman-bench** (this package)  | [GitHub](https://github.com/ml-lab-htw/RamanBench) · [PyPI](https://pypi.org/project/raman-bench/) |
| **Live Leaderboard**            | [huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) |
| **Paper**                       | [arXiv:2605.02003](https://arxiv.org/abs/2605.02003)                                                             |

---

## Installation

### Option 1 — Datasets + leaderboard (recommended starting point)

```bash
pip install raman-bench
```

This gives you:

- **All 74 datasets** with standardised train/test splits via `raman-data`
- **Precomputed results** for 28 baseline models (bundled CSVs, no internet needed)
- **Leaderboard API** — rank, plot, and compare against baselines
- **Evaluation API** — `lb.evaluate_and_add(model)` works with *any* sklearn-compatible model

You can use any ML library you already have installed — scikit-learn, LightGBM,
XGBoost, PyTorch, JAX, or anything else — against a large-scale, curated data
foundation without installing a single additional dependency.

### Option 2 — With all built-in models

Adds all Raman-specific architectures and standalone tabular foundation models,
all with a standard `fit(X, y)` / `predict(X)` interface:

```bash
pip install "raman-bench[models]"
```

This installs `torch`, `tabpfn`, `pytabkit`, `tabdpt`, `sktime`, and
`ramanspy` on top of the core package.  **No AutoGluon required.**

### Option 3 — Full benchmark reproducibility (AutoGluon fork)

The paper's benchmark runs all models through AutoGluon's automated
preprocessing and HPO pipeline.  The fork addresses two limitations of
standard AutoGluon 1.5:

1. **Feature cap** — AutoGluon caps tabular foundation models (TabPFN v2,
   TabICL, TabDPT, MITRA) at 500 features; Raman spectra typically have
   500–4000 wavenumber points.  The fork removes this cap.
2. **TabICL v2 regression** — AutoGluon 1.5 ships TabICL v1, which supports
   classification only.  The fork upgrades to TabICL v2, adding regression
   support.  This limitation is expected to be resolved in AutoGluon 1.6.

A [patched fork](https://github.com/ml-lab-htw/autogluon) incorporates both fixes.

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
pip install -r requirements-autogluon-fork.txt
pip install -e ".[models]"
```

> **The fork is only needed to reproduce the exact paper benchmark.**
> Options 1 and 2 work with a standard `pip install` and give full access to
> all datasets, splits, and built-in models.

---

## Quick Start

### Load a dataset (Option 1 — core install only)

```python
from raman_data import raman_data

ds = raman_data("amino_acids_glycine")
print(ds.spectra.shape)      # (n_samples, n_wavenumbers)
print(ds.targets.shape)      # (n_samples,)
print(ds.raman_shifts[:5])   # wavenumber axis in cm⁻¹
```

All 74 datasets are available this way.  Each comes with a fixed train/test
split so results are directly comparable to the precomputed baselines.

### Evaluate your model against 28 baselines (Option 1)

Any scikit-learn–compatible estimator works:

```python
from raman_bench import Leaderboard
from sklearn.cross_decomposition import PLSRegression

lb = Leaderboard.from_precomputed()   # loads bundled v0.1 results

# Evaluates on all 74 datasets (3 seeds) and inserts into the ranking
results = lb.evaluate_and_add(
    model_name="My-PLS-10",
    model=PLSRegression(n_components=10),
)
print(lb.rank())
lb.plot()
```

Bring any library — LightGBM, XGBoost, a PyTorch model, a JAX model — and it
will be scored on the same protocol as the 28 precomputed baselines.

### Explore the precomputed leaderboard (Option 1)

```python
from raman_bench import Leaderboard

lb = Leaderboard.from_precomputed()
print(lb.rank())          # ranked DataFrame
lb.plot()                 # horizontal bar chart
```

### Use a built-in Raman model directly

All built-in models expose a standard sklearn `fit` / `predict` API:

```python
import numpy as np
from raman_bench.models.custom import DeepCNNModel, TabPFNModel, RocketModel

X = np.random.randn(200, 512).astype("float32")  # 200 spectra, 512 wavenumbers
y = np.random.randn(200)                          # regression targets

# Raman-specific deep learning model
model = DeepCNNModel(n_epochs=50)
model.fit(X, y)
predictions = model.predict(X)

# Tabular foundation model (no feature-count limit)
tfm = TabPFNModel()
tfm.fit(X, y)
predictions = tfm.predict(X)
```

### Run the full benchmark pipeline (fork required)

```bash
# Pre-cache all dataset splits (optional, speeds up the run)
python scripts/prepare_datasets.py --config configs/benchmark_v0.1.json

# Run predictions → metrics
raman-bench run --config configs/benchmark_v0.1.json

# Run individual steps
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

## Models

### Paper baselines (28 models)

All results in the paper were produced through the AutoGluon pipeline (Option 3 install).

| Category | Models |

| Category | Models |
|---|---|
| Classical spectroscopy | PLS, KNN, LR |
| Tree ensembles | GBM (LightGBM), XGB, CatBoost, RF, XT |
| Tabular deep learning | NN_TORCH, FastAI, RealMLP |
| Tabular foundation models | TabPFN v2, TabPFN v2.5, TabM, TabDPT, TabICL, MITRA |
| Time-series classifiers | ROCKET, Arsenal |
| Raman-specific DL | DeepCNN, RamanNet, SANet, RamanFormer, RamanTransformer, ReZeroNet, FC-ResNeXt, CoAtNet |
| AutoGluon ensemble | AUTOGLUON |

### Standalone sklearn wrappers (`raman-bench[models]`)

`raman-bench[models]` provides sklearn-compatible (`fit` / `predict`) wrappers
for many of the same algorithm families, usable directly without AutoGluon or
the fork.  These are **not** the exact pipeline configurations from the paper
(no AutoGluon preprocessing or HPO), but they use the same underlying
algorithms and are well-suited for building and evaluating new models.

| Class | Algorithm | Requires |
|---|---|---|
| `PLSModel` | Partial Least Squares | — |
| `DeepCNNModel` | Raman-specific CNN | `torch` |
| `RamanNetModel` | Raman-specific CNN | `torch` |
| `SANetModel` | Spectral attention net | `torch` |
| `RamanFormerModel` | Raman transformer | `torch` |
| `RamanTransformerModel` | Raman transformer | `torch` |
| `ReZeroNetModel` | ReZero CNN | `torch` |
| `FCResNeXtModel` | FC-ResNeXt | `torch` |
| `CoAtNetModel` | Conv + attention | `torch` |
| `RocketModel` | ROCKET classifier | `sktime` |
| `ArsenalModel` | Arsenal classifier | `sktime` |
| `TabPFNModel` | TabPFN v2 | `tabpfn` |
| `RealMLPModel` | RealMLP-TD | `pytabkit` |
| `TabMModel` | TabM-D | `pytabkit` |
| `TabDPTModel` | TabDPT | `tabdpt` |

All classes support classification and regression and auto-detect the task from
`y`.  All package dependencies are included in `raman-bench[models]`.

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

---

## Ranking Protocol

Models are evaluated under three complementary metrics:

| Metric | Description |
|---|---|
| **Elo** | Pairwise win-rate Elo calibrated to RF = 1000 (200-round bootstrap) |
| **Score** | Normalised per-dataset score: best model = 1, median model = 0 |
| **Avg Rank** | Average rank across all datasets and targets |
| **Improvability** | % gap to the best model, averaged across datasets |

See the [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) for
interactive filtering by model category, task type, and dataset domain.

---

## Repository Structure

```
RamanBench/
├── src/raman_bench/
│   ├── leaderboard.py          # Leaderboard + model evaluation API
│   ├── benchmark.py            # Dataset loading and cross-validation
│   ├── predictions.py          # Prediction generation (benchmark step 1)
│   ├── evaluation.py           # Metric computation (benchmark step 2)
│   ├── model.py                # AutoGluon pipeline wrapper (fork required)
│   ├── config.py               # JSON config loader
│   ├── models/custom/          # All built-in Raman models (sklearn API)
│   │   ├── base.py             #   BaseRamanEstimator (shared training loop)
│   │   ├── deepcnn.py          #   DeepCNNModel
│   │   ├── ramannet.py         #   RamanNetModel
│   │   ├── sanet.py            #   SANetModel
│   │   ├── ramanformer.py      #   RamanFormerModel
│   │   ├── ramantransformer.py #   RamanTransformerModel
│   │   ├── rezeronet.py        #   ReZeroNetModel
│   │   ├── fcresnext.py        #   FCResNeXtModel
│   │   ├── coatnet.py          #   CoAtNetModel
│   │   ├── pls.py              #   PLSModel
│   │   ├── sktime_models.py    #   RocketModel, ArsenalModel
│   │   └── tabular_foundation.py # TabPFNModel, RealMLPModel, TabMModel, TabDPTModel
│   └── preprocessing/
│       ├── mixin.py            #   RamanPreprocessingMixin (AutoGluon HPO)
│       └── wrapped_models.py   #   Prep_* classes + SklearnAutoGluonBridge
├── configs/                    # Benchmark configuration files
├── data/precomputed/           # Bundled v0.1 results
├── notebooks/                  # Example Jupyter notebooks
├── scripts/                    # CLI scripts
└── tests/                      # pytest test suite
```

### Architecture: two paths, one set of model classes

Custom models are implemented once as plain scikit-learn `BaseEstimator`
subclasses.  The same classes are used in both usage modes:

```
  Custom model (e.g. DeepCNNModel)
  BaseEstimator — no AutoGluon dependency
  fit(X, y) / predict(X)
        │
        ├─── Standalone path (pip install "raman-bench[models]")
        │      CUSTOM_MODELS["DEEPCNN"] → DeepCNNModel().fit(X, y)
        │
        └─── AutoGluon pipeline path (fork required)
               SklearnAutoGluonBridge._fit() → DeepCNNModel(**params).fit(X_np, y_np)
               Prep_DEEPCNN(_RamanDLBase, _DeepCNNBridge)
```

`SklearnAutoGluonBridge` (in `preprocessing/wrapped_models.py`) is the only
file that imports AutoGluon.  All model source files are AutoGluon-free.

---

## Contributing

We welcome contributions of new models and datasets!

### Adding a New Model

The simplest way to add a model is to implement it as a scikit-learn–compatible
estimator and submit a pull request.  No AutoGluon knowledge is required.

1. Create `src/raman_bench/models/custom/my_model.py`:

```python
import numpy as np
from sklearn.base import BaseEstimator

class MyModel(BaseEstimator):

    def __init__(self, n_components=10, lr=1e-3):
        self.n_components = n_components
        self.lr = lr

    def fit(self, X, y):
        # X: np.ndarray (n_samples, n_features)
        # y: np.ndarray — float → regression, int/str → classification
        ...
        return self

    def predict(self, X):
        ...  # return np.ndarray (n_samples,)

    def predict_proba(self, X):
        ...  # classification only, return (n_samples, n_classes)
```

For PyTorch-based models, inherit from `BaseRamanEstimator` in
`models/custom/base.py` which provides a complete training loop with early
stopping, cosine LR schedule, mixed-class augmentation, and batched inference.

2. Register in `src/raman_bench/models/custom/__init__.py`:

```python
from raman_bench.models.custom.my_model import MyModel

CUSTOM_MODELS["MYMODEL"] = MyModel
```

3. Add tests in `tests/models/test_my_model.py` following the patterns in
   `tests/models/test_sanet.py`.

4. Open a pull request — CI will run the full test suite automatically.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including how to
optionally wire your model into the AutoGluon benchmark pipeline for full
reproducibility.

### Adding a New Dataset

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-dataset) and
[NEW_DATASETS.md](NEW_DATASETS.md) for detailed instructions and examples.

Quick summary:
1. Upload your dataset to HuggingFace Datasets or Zenodo under CC BY 4.0.
2. Add a loader to the [raman-data](https://github.com/ml-lab-htw/raman_data)
   package (open a PR there).
3. Open an issue here linking to the raman-data PR.

The [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
also has a "How to Contribute" section with step-by-step instructions.

---

## Citation

If you use RamanBench in your research, please cite:

```bibtex
@misc{koddenbrock2026ramanbench,
  title         = {RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy},
  author        = {Koddenbrock, Mario and Lange, Christoph and Legner, Robin and Jaeger, Martin
                   and K{\"o}gler, Martin and Cruz Bournazou, Mariano N. and Neubauer, Peter
                   and Bie{\ss}mann, Felix and Rodner, Erik},
  year          = {2026},
  eprint        = {2605.02003},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.02003}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Dataset licenses vary; see the [dataset catalog](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
or [raman-data](https://github.com/ml-lab-htw/raman_data) for per-dataset license information.
Most datasets are released under CC BY 4.0.
