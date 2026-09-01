# RamanBench

[![PyPI](https://img.shields.io/pypi/v/raman-bench)](https://pypi.org/project/raman-bench/)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org)
[![CI](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.02003-b31b1b)](https://arxiv.org/abs/2605.02003)
[![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-HuggingFace-orange)](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)

**A machine-learning benchmark for Raman spectroscopy.**

RamanBench collects 74 public Raman datasets (163 prediction targets, four
application domains: Material Science, Biological, Medical, Chemical) behind one
evaluation protocol, and ships the results of 28 baseline models run through it.
You can score a new model against those baselines without re-running the
experiments. The baselines cover classical chemometrics, gradient boosting,
tabular foundation models, and deep networks built for spectra.

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

This is enough to:

- load all 74 datasets with fixed train/test splits (via `raman-data`)
- read the precomputed results for the 28 baselines (bundled CSVs, works offline)
- score your own model against them: `lb.evaluate_and_add(model)` takes any
  sklearn-compatible estimator

Your model can come from any library you already have — scikit-learn, LightGBM,
XGBoost, PyTorch, JAX. The core install adds no heavy dependencies of its own.

### Option 2 — With all built-in models

Adds all Raman-specific architectures and standalone tabular foundation models,
all with a standard `fit(X, y)` / `predict(X)` interface:

```bash
pip install "raman-bench[models]"
```

This adds `torch`, `tabpfn`, `pytabkit`, `tabdpt`, `sktime`, and `ramanspy` to
the core package. AutoGluon is not needed for this path.

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

The benchmark-running layer is being rebuilt directly on
[TabArena](https://github.com/autogluon/tabarena) (`tabarena` / `bencheval`)
instead of on its own reimplementation of the same ideas.

Each (dataset, target) pair becomes a TabArena `UserTask`. Splitting is real
repeated k-fold cross-validation (`raman_bench.splitting`, with `n_repeats`
scaled to dataset size, as TabArena documents). Each model's HPO search space is
a TabArena `ConfigGenerator` config pool, and the default / tuned /
tuned+ensemble results are read back from that pool without any re-training
(`EndToEnd.from_raw(...).get_results(...)`).

What RamanBench adds on top is small: datasets come from the raman-data HF
mirror rather than OpenML, and about 13 Raman-specific architectures are
registered alongside TabArena's own models.

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
- `.claude/agents/` — Claude Code agents for routine maintenance, spread across
  this repo and its siblings (see **Contributor Agents** below).

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

Every dataset loads this way, each with the same fixed train/test split the
precomputed baselines used.

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

### Run the full v0 benchmark pipeline

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
| [`03_explore_results.ipynb`](notebooks/03_explore_results.ipynb) | Per-dataset and per-domain results |
| [`04_contribute_dataset.ipynb`](notebooks/04_contribute_dataset.ipynb) | Adding a new dataset, step by step |
| [`05_reproduce_benchmark.ipynb`](notebooks/05_reproduce_benchmark.ipynb) | Re-running the v0 benchmark from configs |
| [`06_hpo_ensemble_ablation.ipynb`](notebooks/06_hpo_ensemble_ablation.ipynb) | HPO and ensembling ablation |

---

## Models

### Paper baselines (28 models)

All results in the paper were produced through the AutoGluon pipeline (Option 3 install).

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
algorithms, and are a convenient starting point for building a new model.

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
| `RocketModel` | ROCKET regression/classification | `sktime` |
| `HydraModel` | Hydra + closed-form GPU ridge, regression/classification | `torch` |
| `TabPFNModel` | TabPFN v2 | `tabpfn` |
| `RealMLPModel` | RealMLP-TD | `pytabkit` |
| `TabMModel` | TabM-D | `pytabkit` |
| `TabDPTModel` | TabDPT | `tabdpt` |

All classes support classification and regression and auto-detect the task from
`y`.  All package dependencies are included in `raman-bench[models]`.

---

## Benchmark Composition

### Datasets

The 74 datasets span four application domains (Material Science, Biological,
Medical, Chemical) and both task types. They range from a few dozen spectra to
over 100,000, and from roughly 100 to 12,000 wavenumber points. The
[raman-data catalog](https://github.com/ml-lab-htw/raman_data) lists every one
with its source, task, size, and license.

All datasets load via `pip install raman-data`:

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

Models are ranked on four metrics:

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

### How the two paths share model code

Custom models are written once as plain scikit-learn `BaseEstimator`
subclasses, and the same classes serve both usage modes:

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

`SklearnAutoGluonBridge` (in `preprocessing/wrapped_models.py`) is the only file
that imports AutoGluon; the model source files never do.

---

## Contributing

New models and datasets are welcome.

### Adding a New Model

Open this repo in Claude Code and say:

```
Add my model to RamanBench, test it, and run it across the benchmark.
```

The `model-agent` implements it (or wires up an existing TabArena model if one
already fits), tests it locally, asks whether to propose it upstream to
TabArena, and runs it across the benchmark on a cluster or locally.
`.claude/agents/model-agent.md` has the full workflow.

<details>
<summary>Manual steps (no agent)</summary>

The simplest manual way to add a model is to implement it as a scikit-learn–compatible
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

</details>

### Adding a New Dataset

Same idea, in Claude Code:

```
Add my dataset to RamanBench and make it benchmarkable.
```

The `dataset-agent` bootstraps a `raman_data` checkout if needed, picks the right
loader, syncs the dataset to the HF mirror the benchmark reads from, and opens a
`raman_data` PR. See `.claude/agents/dataset-agent.md` for the full workflow.

<details>
<summary>Manual steps (no agent)</summary>

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-dataset) and
[NEW_DATASETS.md](NEW_DATASETS.md) for detailed instructions and examples.
`.claude/agents/dataset-agent.md` (in the `raman-data` repo) documents the full
onboarding workflow, including the HF mirror sync new datasets need to be discoverable
through `run_experiment.py`'s mirror-first loading.

</details>

---

## Contributor Agents

Six Claude Code agents handle routine maintenance across the three repos, so a
contributor doesn't need any private tooling. Each is a `.claude/agents/*.md`
file in the repo it works on:

| Agent | Repo | Responsibility |
|---|---|---|
| `dataset-agent` | `raman_data` | Onboard a new dataset: pick the right loader, add a `DatasetInfo` entry, populate `group_ids`/`has_missing_labels` if applicable, sync to the HF mirror. |
| `model-agent` | `RamanBench` (this repo) | Add a new model via the per-directory convention, test it, ask whether to also propose it upstream to TabArena, then run it across the benchmark (cluster or local). |
| `cluster-agent` | `RamanBench` (public) + `raman_bench_paper` (private profiles) | Fleet management: submit job arrays, detect stalled/cancelled tasks and resubmit, run `cluster/janitor.py`'s disk-cleanup sweep. |
| `leaderboard-agent` | `raman_bench_paper` | Regenerate leaderboard CSVs/figures from completed results; always shows a diff and asks permission before publishing to the live HF Space. |
| `hf-frontend-agent` | `HF_spaces/RamanBench` | Frontend work on the public Gradio leaderboard Space. |
| `docs-agent` | `raman_bench_paper` (cross-repo aware) | Keeps README/CONTRIBUTING in sync across all repos as things change. |

A typical handoff runs dataset-agent → model-agent (optional) → cluster-agent →
leaderboard-agent (which asks before publishing) → hf-frontend-agent. docs-agent
runs on its own after structural changes.

The [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
has its own "How to Contribute" section.

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ml-lab-htw/RamanBench&type=Date)](https://star-history.com/#ml-lab-htw/RamanBench&Date)

---

## License

MIT — see [LICENSE](LICENSE).

Dataset licenses vary; see the [dataset catalog](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
or [raman-data](https://github.com/ml-lab-htw/raman_data) for per-dataset license information.
Most datasets are released under CC BY 4.0.
