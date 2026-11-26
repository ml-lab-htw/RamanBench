# Benchmark composition and ranking protocol

## Datasets

The 74 datasets (163 prediction targets) span four application domains (Material Science,
Biological, Medical, Chemical) and both task types. They range from a few dozen spectra to over
100,000, and from roughly 100 to 12,000 wavenumber points. The
[raman-data catalog](https://github.com/ml-lab-htw/raman_data) lists every one with its source,
task, size and licence. Dataset licences vary; most are CC BY 4.0.

All datasets load via `pip install raman-data`:

```python
from raman_data import raman_data

dataset = raman_data("amino_acids_glycine")
X = dataset.spectra          # (n_samples, n_wavenumbers)
y = dataset.targets          # regression targets or class labels
w = dataset.raman_shifts     # wavenumber axis in cm⁻¹
```

Every dataset loads this way, each with the same fixed train/test split the precomputed
baselines used. The benchmark itself reads the Parquet copy of the datasets on the Hugging Face
mirror (`HTW-KI-Werkstatt/RamanBench`).

The benchmark has 135 tasks (one per usable dataset and target): 21 classification and 114
regression tasks. Targets and classes that are too small to split are left out (a class needs at
least 9 spectra). Benchmark v2 adds more datasets; see [configs/v2/README.md](../configs/v2/README.md).

## Ranking protocol

Models are ranked on four metrics:

| Metric | Description |
|---|---|
| **Elo** | Pairwise win-rate Elo calibrated to RF = 1000 (200-round bootstrap) |
| **Score** | Normalised per-dataset score: best model = 1, median model = 0 |
| **Avg Rank** | Average rank across all datasets and targets |
| **Improvability** | % gap to the best model, averaged across datasets |

Every v1 model runs the same protocol: 3-fold outer cross-validation per task (group-aware where
a dataset has replicate groups), 3-fold bagging inside each training split, 600 s per fit
(`mlrod`: 10,800 s) and each model's default configuration. The three largest datasets (mlrod,
wheat_lines, bacteria_identification) are cut to a random sample of 10,000 spectra before
splitting. A few models run on a smaller sample of some datasets because of memory limits
(`model_max_train_samples_overrides`, e.g. PerpetualBooster and Mitra at 3,000 rows). The protocol
ships with the package (`raman_bench.compare.load_protocol()`).

Each fold is scored with the task's metric: ROC AUC for binary classification, log loss for
multiclass classification, RMSE for regression (TabArena's evaluator, `bencheval`). Elo, score,
rank and improvability are computed from those per-fold errors. To rank by accuracy, F1, R² or
another metric instead, see [Rank by another metric](compare-your-model.md#rank-by-another-metric).

See the [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) for
interactive filtering by model category, task type and dataset domain.

## Benchmark v2

Benchmark v2 adds datasets to v1 and keeps all of its tasks: 187 tasks, the 135 of v1 and 52 new ones.
Its protocol is `configs/v2/protocol.json`, the one protocol file of the version, and every model runs under
the same settings, with no per-model exceptions:

- 3-fold outer cross-validation, group-aware, and 3-fold bagging inside each training split. A class needs
  at least 3 groups, and the inner folds of grouped tasks are group-disjoint.
- A 600 s optimization budget per fit, as in v1: the time a model sees and adapts to. Separately, a fold is
  cancelled and labelled `timeout` once it has run `max_run_time` = 10800 s. The budget caps each bagged model at
  480 s (0.8 of the bag limit, what AutoGluon gave it in v1); the cancel limit only lets slow models finish.
  Both are the same for every model on every dataset.
- One row cap, `max_train_samples = 30000`, for every dataset and every model: a dataset with more spectra is
  cut to a random sample of 30000 before splitting. That applies to `mlrod`, `wheat_lines`,
  `bacteria_identification`, `glioma_idh` and `soft_tissue_sarcoma`; every other dataset runs on all its rows.
  There are no per-dataset or per-model row caps.

A model that runs out of memory or time on a large dataset fails there; the fold is reported with its failure
label and imputed like any missing task. A result produced under v1's row caps (10,000 rows, per-model caps) is rerun for v2 where a cap applied to it;
the optimization budget is unchanged from v1, so a longer cancel limit alone does not invalidate a result.

## Departures from the protocol

- Systems (Chakra-Tab, TabFM+, TabLDM+, TabPFN Thinking, LightGBM-Prep-FTE25) do their own validation
  and ensembling, so they are fit once per outer fold instead of through AutoGluon's bagging (see
  [Models and systems](#models-and-systems)).
- HIVE-COTE 2 gets 21,600 s per fit instead of 600 s, because aeon turns the budget into its own
  time contract; it has no result on `mlrod`, which is imputed.
- A task a model has no result for is imputed with Random Forest's result and counted in
  `imputed_pct`; a model more than 50% imputed is not ranked.

## Models and systems

RamanBench takes two kinds of entrant, and they are fitted differently.

| | Model | System |
|---|---|---|
| What it is | One method with a fixed configuration | A pipeline that does its own validation, tuning and ensembling |
| Fit | 3-fold bagging inside each training split, 600 s | Once per outer fold on the whole training split |
| Run with | `scripts/run_experiment.py`, `evaluate_estimator` | `scripts/run_system.py` |

The two are not interchangeable. A model fitted once on the whole training split sees more rows per
fit than a bagged one, and in-context models gain from that: Nums AI's fit-once Causilo 1.5 scored
Elo 1570 on the v1 folds where the bagged run scores about 1525. The leaderboard's reference rows are
bagged fits (and systems), so a single fit of a plain model is not comparable with them. Run it
bagged, or submit it as a system and say so. `compare()` warns when it cannot check how results were
fitted.

## Failures and incomplete models

A fold that raises leaves a `failure.json` next to where its `results.pkl` would be, with one uniform
label:

| Label | Meaning |
|---|---|
| `oom_gpu` | CUDA out of memory |
| `oom_cpu` | out of host memory |
| `timeout` | the fit exceeded its time limit |
| `error` | any other exception |

A fold with neither result nor record (a pod that was killed) is reported as `missing`.
`scripts/aggregate_results.py` collects them into `failures.csv` and logs the counts per model. The
ranking is unchanged: a missing task is imputed with Random Forest's result, counted in
`imputed_pct`, and a model more than 50% imputed is not ranked.

## What is recorded with each result

Next to every `results.pkl` the runners write a `setup.json`: the package versions (`raman-bench`,
`tabarena`, `autogluon.tabular`, `torch`, `scikit-learn`, `numpy`), the Python version, the hardware,
whether the run was `bagged` or a `system`, and the settings it ran under (time limit, bag folds, row
cap, rows after cleaning, CPUs and GPUs). A result run before this was recorded has none.

## One protocol file per benchmark version

Each benchmark version has one protocol file that lists all of its tasks and settings; for v1 it is
`src/raman_bench/data/precomputed/v1/protocol.json`. The scope file used to run the sweeps must agree
with it on the folds, bagging, time limit and row caps, and a test checks that.

Known contamination and closed-API markers are in [Models](models.md#known-contamination-and-closed-apis).
