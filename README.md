# RamanBench

[![PyPI](https://img.shields.io/pypi/v/raman-bench)](https://pypi.org/project/raman-bench/)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org)
[![CI](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-lab-htw/RamanBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.02003-b31b1b)](https://arxiv.org/abs/2605.02003)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-9370DB)](https://arxiv.org/abs/2605.02003)
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
| **Paper**                       | Accepted at NeurIPS 2026 · [arXiv:2605.02003](https://arxiv.org/abs/2605.02003)                                  |

---

## Installation

### Option 1 — Datasets + leaderboard (recommended starting point)

```bash
pip install raman-bench
```

This is enough to:

- load every dataset (via `raman-data`)
- read the per-fold v1 results of all 55 leaderboard models (and the two AutoGluon reference systems), which ship with the
  package (`raman_bench.compare.load_reference()`, works offline)

With `pip install "raman-bench[plots]"` you can also score them and rank your own
results against them (`raman_bench.compare`, see [Compare your model](#compare-your-model-against-the-v1-leaderboard)).
Running a model on the benchmark protocol needs TabArena and AutoGluon (Option 3).

### Option 2 — With all built-in models

Adds all Raman-specific architectures and standalone tabular foundation models,
all with a standard `fit(X, y)` / `predict(X)` interface:

```bash
pip install "raman-bench[models]"
```

This adds `torch`, `tabpfn`, `pytabkit`, `tabdpt`, `sktime`, `aeon`, and `ramanspy` to
the core package. AutoGluon is not needed for this path.

`tabarena` itself, and three wrapped models built on it — `Prep_TABFM`,
`Prep_SAP_RPT_OSS`, `Prep_ORIONMSP` — have git-only upstreams and cannot ship
in a PyPI package. Install them separately if you need them:

```bash
pip install -r requirements-tabarena-git.txt
pip install -r requirements-models-git.txt
```

Until then, every `Prep_*` class that wraps a `tabarena.models.*` model
(which is most of them) raises a clear "not available" error; the
Raman-specific standalone architectures work without this step.

### Option 3 — Full benchmark reproducibility

The paper's benchmark runs all models through AutoGluon's automated
preprocessing and HPO pipeline, on plain upstream AutoGluon (>=1.6.1, no fork):

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
pip install -e ".[models]"
pip install -r requirements-tabarena-git.txt
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
pip install -r requirements-tabarena-git.txt       # tabarena is git-pinned, not on PyPI --
                                                    # see requirements-tabarena-git.txt
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

### Browse the v1 leaderboard

The per-fold results of all 55 leaderboard models (and the two AutoGluon reference systems) on all 135 tasks ship with the
package (`src/raman_bench/data/precomputed/v1/`). `leaderboard()` scores them like the
[live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) does
(needs `pip install "raman-bench[plots]"`):

```python
from raman_bench.compare import leaderboard, load_reference

scores = leaderboard()                       # {"all", "classification", "regression"}
scores["all"].leaderboard.head(10)           # Elo, win rate, times, preprocessing, ...
load_reference()                             # one row per (task, fold, model)
```

### Compare your model against the v1 leaderboard

You only run your own model: on the same 135 tasks, the same 3 outer folds, with
3-fold bagging, the same 600 s budget and the same row caps
(`raman_bench.compare.load_protocol()`). Then `compare` ranks it among the 55
leaderboard models.

**A scikit-learn estimator** (needs the Option 3 install):

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from raman_bench.evaluate import evaluate_estimator
from raman_bench.compare import compare

evaluate_estimator(
    "MY-PLS",
    classifier=LogisticRegression(max_iter=2000),    # needs predict_proba
    regressor=PLSRegression(n_components=10),
    results_dir="results/my_model",
    # hyperparameters={"prep_snv_enabled": True},   # optional RamanBench preprocessing
)
scores = compare("results/my_model", out_dir="results/my_model_figures")
scores["all"].leaderboard.head(15)
```

`evaluate_estimator` runs your estimator through the same TabArena/AutoGluon bagged
experiment as the leaderboard models, caching one result per fold, so an interrupted run
picks up where it stopped. Pass `tasks=[...]` for a quick look at a few tasks, and
`compare(..., tasks="own")` to score every model on just those. That subset is not
comparable with the published leaderboard.

**A model registered in RamanBench** (`models/custom/<key>/`, see
[Adding a New Model](#adding-a-new-model)) runs through `scripts/run_experiment.py`;
`raman-bench protocol` prints one call per task and fold with the protocol settings:

```bash
raman-bench protocol MY_MODEL --results-dir results/my_model > jobs.txt && bash jobs.txt
raman-bench compare results/my_model --output-dir results/my_model_figures --top 10
```

A task your model has no result for gets Random Forest's result and counts as
imputed (`imputed_pct`); a model more than 50% imputed is not ranked. The
`preprocessing` column lists the RamanBench preprocessing steps each model ran with
(`none` for most). Steps inside your own scikit-learn `Pipeline` don't appear there.
`notebooks/02_benchmark_new_model.ipynb` walks through all of it.

### Use the v1 folds in another framework

To run the protocol outside RamanBench (in aeon, scikit-learn, or your own harness),
take the outer cross-validation folds as plain spectrum ids:

```bash
raman-bench folds raman_bench_v1_folds.parquet                      # all 135 tasks
raman-bench folds clf_folds.csv --task-type classification          # the 21 classification tasks
```

```python
from raman_bench.experiment_utils import load_dataframe
from raman_bench.folds import export_folds, train_test_ids

folds = export_folds(task_type="classification")
_, df, _, _ = load_dataframe("alzheimer", 0)   # the mirror's data; index = spectrum_id
train, test = train_test_ids(folds, "alzheimer__0", fold=0)
X = df.drop(columns=["target", "_group_id"], errors="ignore")
X_train, y_train = X.loc[train], df.loc[train, "target"]
X_test, y_test = X.loc[test], df.loc[test, "target"]
```

One row per task and spectrum: `task`, `dataset`, `target_idx`, `problem_type`,
`repeat`, `spectrum_id` (the spectrum's row in the dataset as `raman_data` and the
Hugging Face mirror store it) and `fold` (the outer fold whose test set holds it). The
training set of fold `f` is every other listed spectrum of that task. Spectra a task
doesn't use (a NaN in the spectrum, no label, a class with fewer than 9 spectra, or not
drawn into the 10,000-row sample on mlrod, wheat_lines and bacteria_identification)
aren't listed. Drop the `_group_id` column, if present, before fitting: it marks
replicate groups and isn't a feature.

These are the folds `scripts/run_experiment.py` builds. Score each fold with the task's
metric from `load_protocol()["tasks"]` (ROC AUC for binary, log loss for multiclass,
RMSE for regression) to compare with the leaderboard models' per-fold results.

<details>
<summary>The v0.1 leaderboard</summary>

The v0.1 results (29 models, one holdout split per seed) are still bundled and scored by
the `Leaderboard` class. They use a different protocol, so don't compare them with v1
numbers:

```python
from raman_bench import Leaderboard

lb = Leaderboard.from_precomputed()
print(lb.rank())
```

</details>

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

### Run the full benchmark pipeline

Single-experiment runs use `scripts/run_experiment.py` (one process per
`(model, dataset, target, repeat, fold, config-index)`), typically submitted
via `cluster/submit_job.py`/`cluster/submit_full_benchmark.py` for a real
cluster (SLURM or Kubernetes) sweep:

```bash
python scripts/run_experiment.py --dataset wheat_lines --target-idx 0 \
    --model PLS --repeat 0 --fold 0 --config-index 0 \
    --results-dir results/v1/data
```

See `cluster/submit_job.py --help` for submitting a real array/sweep.

### Leaderboard figures

Once results exist, aggregate them and draw the leaderboard figures (needs
`pip install "raman-bench[plots]"`):

```bash
python scripts/aggregate_results.py --results-dir results/v1/data --output-dir results/v1/aggregated
python scripts/plot_results.py --input results/v1/aggregated/hpo_results.csv --output-dir results/v1/figures
```

Scores come from TabArena's own evaluator (`bencheval`): Elo with bootstrap
CIs (Random Forest = 1000), win rate, improvability and the pairwise win-rate
matrix. Every figure (benchmark overview, example spectra, composition, Elo
ranking, score and improvability vs. time per 1K spectra, model progress over
release date, win rates, efficiency, critical difference diagrams) is written as PNG and PDF
under `static/` and as an interactive HTML page under `interactive/`, with an
`index.html` linking them all. By default only the top 2 models per category
are drawn in colour and the rest in grey (`--focus-top-k`, `0` colours all);
the HTML pages can switch between both views. The dataset figures read
metadata from raman_data and the RamanBench mirror on the Hugging Face Hub;
`--no-dataset-figures` skips them offline.

**Known contamination.** RamanPFN (Pan et al., 2026, arXiv:2608.02157) was
developed and evaluated on the RamanBench v0.1 datasets and task splits, with
no separate development data reported. Its TabPFN weights were not trained on
RamanBench, but its fixed design choices may be tuned to these datasets, so its
scores may be optimistic. It is ranked like every other model and marked with
† in figures and tables (`raman_bench.plotting.models.KNOWN_CONTAMINATION`).

### Notebooks

| Notebook | Description |
|---|---|
| [`01_quick_start.ipynb`](notebooks/01_quick_start.ipynb) | Load a dataset, look at the v1 leaderboard |
| [`02_benchmark_new_model.ipynb`](notebooks/02_benchmark_new_model.ipynb) | Run your own model on the benchmark protocol and rank it against the 55 leaderboard models |
| [`03_explore_results.ipynb`](notebooks/03_explore_results.ipynb) | Per-task results, task types, speed against accuracy, pairwise win rates |
| [`04_contribute_dataset.ipynb`](notebooks/04_contribute_dataset.ipynb) | Adding a new dataset, step by step |

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
| `AeonClassifierModel` | HIVE-COTE 2 or one of its components (STC, DrCIF, Arsenal, TDE), classification only | `aeon` |
| `TabPFNModel` | TabPFN v2 | `tabpfn` |
| `RealMLPModel` | RealMLP-TD | `pytabkit` |
| `TabMModel` | TabM-D | `pytabkit` |
| `TabDPTModel` | TabDPT | `tabdpt` |

All classes except `AeonClassifierModel` support classification and regression and
auto-detect the task from `y`.  All package dependencies are included in `raman-bench[models]`.

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

Every v1 model runs the same protocol: 3-fold outer cross-validation per task
(group-aware where a dataset has replicate groups), 3-fold bagging inside each training
split, 600 s per fit (`mlrod`: 10,800 s), at most 10,000 training rows on the three
largest datasets, and each model's default configuration. Scores come from TabArena's
evaluator (`bencheval`). The protocol ships with the package
(`raman_bench.compare.load_protocol()`).

See the [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) for
interactive filtering by model category, task type, and dataset domain.

---

## Repository Structure

```
RamanBench/
├── src/raman_bench/
│   ├── compare.py              # v1 reference results + protocol; rank your results against them
│   ├── evaluate.py             # run a scikit-learn estimator (or print run_experiment calls) on the v1 protocol
│   ├── aggregation.py          # cached results.pkl -> TabArena result tables (EndToEnd)
│   ├── experiment_utils.py     # shared task building / experiment running for every v1 runner
│   ├── leaderboard.py          # v0.1 Leaderboard (bundled v0.1 results)
│   ├── benchmark.py            # Dataset loading (mirror-first) and cross-validation
│   ├── model.py                # build_prep_model_hyperparameters (Prep_* hyperparameter builder)
│   ├── config.py               # JSON config loader
│   ├── splitting.py            # Repeated k-fold CV + TabArena UserTask construction
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
│   ├── aggregate_results.py    # v1: recycles cached results into default/tuned/tuned+ensemble
│   ├── build_reference_results.py  # v1: bundles the leaderboard results + protocol with the package
│   └── plot_results.py         # v1: leaderboard figures (static PNG/PDF + interactive HTML)
├── .claude/agents/              # model-agent, cluster-agent (see Contributor Agents)
├── configs/                    # Benchmark configuration files
├── src/raman_bench/data/precomputed/   # v1/: reference results + protocol; *.csv: v0.1 results
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

Only want to see where your model lands? You don't need to add it to the
repository: run it with `evaluate_estimator` and `compare` (see
[Compare your model](#compare-your-model-against-the-v1-leaderboard)). To put it on the
leaderboard, open an issue or pull request with the model and its results directory.

To add the model to RamanBench itself, open this repo in Claude Code and say:

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

RamanBench has been accepted at NeurIPS 2026. If you use RamanBench in your research, please cite:

```bibtex
@inproceedings{koddenbrock2026ramanbench,
  title={RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy},
  author={Koddenbrock, Mario and Lange, Christoph and Legner, Robin and J{\"a}ger, Martin and K{\"o}gler, Martin and Bournazou, Mariano N Cruz and Neubauer, Peter and Biessmann, Felix and Rodner, Erik},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  note={arXiv preprint arXiv:2605.02003}
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
