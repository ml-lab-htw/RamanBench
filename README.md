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

### Option 3 — Full benchmark reproducibility

The paper's benchmark runs all models through AutoGluon's automated
preprocessing and HPO pipeline, on plain upstream AutoGluon (>=1.6.1, no fork):

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
pip install -e ".[models]"
```

RamanBench previously depended on a patched AutoGluon fork here to work around
two limitations of AutoGluon 1.5:

1. **Feature cap** — AutoGluon caps tabular foundation models (TabPFN v2,
   TabICL, TabDPT, Mitra) at 500–2000 features (varies by model); Raman
   spectra typically have 500–4000 wavenumber points. RamanBench now lifts
   this cap itself, per model, using AutoGluon's own supported
   `_default_auxiliary_params_extra` subclass extension point — see
   `Prep_MITRA` / `Prep_TABDPT` / `Prep_TABICL` / `Prep_REALTABPFN_V2` /
   `Prep_REALTABPFN_V25` in `preprocessing/wrapped_models.py`. No fork needed.
   Accepted tradeoff: the fork additionally routed >10-class datasets on
   Mitra/TabPFN through an ECOC many-class wrapper (`tabpfn-extensions`'
   `ManyClassClassifier`); that extra is not reproduced, so such datasets may
   now fail on those two models instead of falling back to the wrapper.
2. **TabICL v2 regression** — AutoGluon 1.5 shipped TabICL v1 (classification
   only). AutoGluon 1.6 upgraded to TabICL v2 natively, adding regression
   support — no fork or override needed for this part anymore.

---

## RamanBench v1

RamanBench's benchmark-running layer is being migrated onto
[TabArena](https://github.com/autogluon/tabarena) (`tabarena`/`bencheval`) directly, rather
than reimplementing patterns "inspired by" it. Concretely: each (dataset, target) pair
becomes a TabArena `UserTask`; splitting is real repeated k-fold cross-validation
(`raman_bench.splitting`, dataset-size-adaptive `n_repeats`, matching TabArena's own
documented convention); each model's HPO search space is a TabArena `ConfigGenerator`
config pool, and the default/tuned/tuned+ensemble triad is recycled from that pool with
zero extra fitting (`EndToEnd.from_raw(...).get_results(...)`, no re-training). See
TabArena's own docs for the underlying mechanism — RamanBench's layer on top is thin by
design: datasets come from the raman-data HF mirror instead of OpenML, and ~13 additional
Raman-specific model architectures are wired in alongside TabArena's own registry.

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
uv pip install --prerelease=allow -e ".[models]"   # uv resolves bencheval automatically;
                                                    # plain pip needs it installed first --
                                                    # see the note in pyproject.toml
```

Key entry points:
- `scripts/run_experiment.py` — thin per-(model, dataset, target, repeat, fold,
  config-index) job runner; reads datasets mirror-first by default
  (`--use-mirror`/`--no-use-mirror` to opt out).
- `raman_bench.models.registry.raman_bench_model_registry` — TabArena's full model
  registry plus RamanBench's own architectures and Raman-preprocessing overrides.
- `cluster/` — cluster-agnostic SLURM job submission (`submit_job.py`,
  `run_experiment.sbatch`, `detect_cluster.py`, `janitor.py`) driven by a profile YAML;
  works locally too when no cluster is available.
- `.claude/agents/` — six Claude Code agents for routine maintenance (see
  **Contributor Agents** below).

New models are onboarded via a per-model directory —
`raman_bench/models/custom/<key>/{model.py,hpo.py,info.py}`, auto-discovered into the
registry — see `models/custom/ridge/` for the reference implementation and
`.claude/agents/model-agent.md` for the full workflow.

### What's changed since v0.1

Full details in [CHANGELOG.md](CHANGELOG.md); short version:

**Models**
- Onboarded 30 TabArena-native models directly from TabArena's own registry (TabFM,
  TabPFN-3, TabSwift, ModernNCA, EBM, PerpetualBooster, xRFM, ChimeraBoost, NORI,
  SAP-RPT-OSS, OrionMSP, ILTM, LIMIX, TabSTAR, and more)
- Added TabPFN v2.6, v3, and v3-Thinking
- Added TabPFN-Wide (wide, few-sample classification) and RamanPFN (Pan et al., 2025)
- Verified all 10 pre-existing custom Raman architectures against the new v1 pipeline

**Benchmark methodology (v1)**
- Migrated the model/metrics/splitting layer to depend directly on TabArena/`bencheval`
  instead of reimplementing patterns "inspired by" them
- Switched to real repeated k-fold CV with dataset-size-adaptive repeat counts,
  replacing the old 3-independent-holdout-split scheme
- Fixed AutoGluon bagging to a genuine, TabArena-matching `num_bag_folds=8` (v0.1 had
  effectively no bagging for 27 of 28 models)
- Added semi-supervised-aware splitting (unlabeled rows kept in train, never in test)
- Ported TabArena's trivial-dataset filter into RamanBench itself as a first-class feature

**Preprocessing**
- 8 new steps: airPLS, arPLS, rubberband, EMSC, Savitzky-Golay derivative, wavelet
  denoising, fingerprint-region crop, L2 vector normalization
- New preprocessing-ensemble mechanism (parallel recipe blocks, concatenated) and
  config-level `preprocessing_params` overrides

**Datasets**
- +4 datasets: `chlorinated_samples`, `locust_phase_hemolymph`, `cspp_serum_metabolites`,
  `ait_glucose_blood_sers`
- New `is_grouped` / `has_missing_labels` fields on `raman-data`'s `DatasetInfo`

**Infrastructure**
- Public, cluster-agnostic job-submission tooling plus a new opportunistic,
  capacity-aware scheduler for routine full-benchmark sweeps
- Dropped the patched AutoGluon fork; moved to upstream AutoGluon 1.6.1 with
  RamanBench-local cap overrides
- Automatic `.env` credential loading; `main` now requires PRs (no direct pushes)

**Reliability fixes**
- Atomic prediction/index writes to avoid concurrent-job races
- Skip (not delete) mismatched predictions during metric computation
- Guard R² against degenerate near-constant test folds

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
│   ├── leaderboard.py          # Leaderboard + model evaluation API (v0)
│   ├── benchmark.py            # Dataset loading (mirror-first) and cross-validation
│   ├── predictions.py          # Prediction generation (v0 benchmark step 1)
│   ├── evaluation.py           # Metric computation (v0 benchmark step 2)
│   ├── model.py                # v0 AutoGluon pipeline wrapper (fork required)
│   ├── config.py               # JSON config loader
│   ├── splitting.py            # v1: repeated k-fold CV + TabArena UserTask construction
│   ├── models/
│   │   ├── registry.py         #   raman_bench_model_registry (TabArena's + ours)
│   │   ├── discover.py         #   auto-discovery for models/custom/<key>/info.py
│   │   ├── _model_info.py      #   lightweight per-model ModelInfo dataclass
│   │   ├── generate/           #   per-model ConfigGenerator (HPO search space) modules
│   │   └── custom/             # All built-in Raman models (sklearn API)
│   │       ├── base.py         #   BaseRamanEstimator (shared training loop)
│   │       ├── ridge/           #   reference implementation of the new per-directory
│   │       │                    #   {model.py,hpo.py,info.py} onboarding convention
│   │       ├── deepcnn.py, ramannet.py, sanet.py, ramanformer.py, ...
│   │       │                    #   (older flat-file convention, still supported)
│   │       └── tabular_foundation.py
│   └── preprocessing/
│       ├── mixin.py            #   RamanPreprocessingMixin (AutoGluon HPO)
│       ├── bridge_bases.py     #   SklearnAutoGluonBridge + shared bases
│       └── wrapped_models.py   #   Prep_* classes, PREPROCESSED_MODELS registry
├── cluster/                    # v1: cluster-agnostic SLURM submission
│   ├── detect_cluster.py, submit_job.py, run_experiment.sbatch, janitor.py
│   └── profiles/                #   generic + example cluster profiles (no secrets)
├── scripts/
│   ├── run_experiment.py       # v1: per-(model,dataset,target,repeat,fold,config) job runner
│   ├── build_target_list.py    # v1: builds the full-benchmark target list (mirror-first)
│   └── aggregate_results.py    # v1: recycles cached results into default/tuned/tuned+ensemble
├── .claude/agents/              # model-agent, cluster-agent (see Contributor Agents)
├── configs/                    # Benchmark configuration files
├── data/precomputed/           # Bundled v0.1 results
├── notebooks/                  # Example Jupyter notebooks
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

The steps above cover the standalone sklearn-compatible path (Option 2). To also wire
your model into the full RamanBench v1 benchmark pipeline (Raman preprocessing HPO,
default/tuned/tuned+ensemble recycling, cluster submission), follow the per-model
directory convention instead — `raman_bench/models/custom/<key>/{model.py,hpo.py,info.py}`,
auto-discovered into `raman_bench_model_registry`. `models/custom/ridge/` is the reference
implementation; `.claude/agents/model-agent.md` documents the full workflow end to end
(implement → test → run across the benchmark, cluster or local).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

### Adding a New Dataset

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-dataset) and
[NEW_DATASETS.md](NEW_DATASETS.md) for detailed instructions and examples.
`.claude/agents/dataset-agent.md` (in the `raman-data` repo) documents the full
onboarding workflow, including the HF mirror sync new datasets need to be discoverable
through `run_experiment.py`'s mirror-first loading.

---

## Contributor Agents

Six Claude Code agents cover routine maintenance so contributors don't need any private
tooling. Each is a `.claude/agents/*.md` file in the repo it operates on:

| Agent | Repo | Responsibility |
|---|---|---|
| `dataset-agent` | `raman_data` | Onboard a new dataset: pick the right loader, add a `DatasetInfo` entry, populate `group_ids`/`has_missing_labels` if applicable, sync to the HF mirror. |
| `model-agent` | `RamanBench` (this repo) | Add a new model via the per-directory convention, test it, ask whether to also propose it upstream to TabArena, then run it across the benchmark (cluster or local). |
| `cluster-agent` | `RamanBench` (public) + `raman_bench_paper` (private profiles) | Fleet management: submit job arrays, detect stalled/cancelled tasks and resubmit, run `cluster/janitor.py`'s disk-cleanup sweep. |
| `leaderboard-agent` | `raman_bench_paper` | Regenerate leaderboard CSVs/figures from completed results; always shows a diff and asks permission before publishing to the live HF Space. |
| `hf-frontend-agent` | `HF_spaces/RamanBench` | Frontend work on the public Gradio leaderboard Space. |
| `docs-agent` | `raman_bench_paper` (cross-repo aware) | Keeps README/CONTRIBUTING in sync across all repos as things change. |

Typical handoff: dataset-agent → model-agent (optional) → cluster-agent → leaderboard-agent
(with explicit permission before publishing) → hf-frontend-agent. docs-agent runs
independently after structural changes. None of these agents ever add a Claude/Anthropic
co-author trailer to a commit.

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
@article{koddenbrock2026ramanbench,
  title={RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy},
  author={Koddenbrock, Mario and Lange, Christoph and Legner, Robin and J{\"a}ger, Martin and K{\"o}gler, Martin and Bournazou, Mariano N Cruz and Neubauer, Peter and Biessmann, Felix and Rodner, Erik},
  journal={arXiv preprint arXiv:2605.02003},
  year={2026}
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ml-lab-htw/RamanBench&type=Date&...)](
https://star-history.com/#ml-lab-htw/RamanBench&Date
)

---

## License

MIT — see [LICENSE](LICENSE).

Dataset licenses vary; see the [dataset catalog](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
or [raman-data](https://github.com/ml-lab-htw/raman_data) for per-dataset license information.
Most datasets are released under CC BY 4.0.
