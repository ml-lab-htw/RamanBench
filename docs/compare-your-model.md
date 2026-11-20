# Compare your model with the v1 leaderboard

Everything here works from the PyPI package plus the results that ship with it. Running your
own model on the protocol needs the full install (Option 3 in [Installation](installation.md)).

## Browse the v1 leaderboard

The per-fold results of all 69 leaderboard models (and the two AutoGluon reference systems) on
all 135 tasks ship with the package (`src/raman_bench/data/precomputed/v1/`). `leaderboard()`
scores them like the [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
does (needs `pip install "raman-bench[plots]"`):

```python
from raman_bench.compare import leaderboard, load_reference

scores = leaderboard()                       # {"all", "classification", "regression"}
scores["all"].leaderboard.head(10)           # Elo, win rate, times, preprocessing, ...
load_reference()                             # one row per (task, fold, model)
```

## Score your own model

You only run your own model: on the same 135 tasks, the same 3 outer folds, with 3-fold
bagging, the same 600 s budget and the same 10,000-spectrum sample of the largest datasets
(`raman_bench.compare.load_protocol()`). Then `compare` ranks it among the leaderboard models.

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

To use a [named preprocessing recipe](preprocessing.md) with your estimator, pass
`hyperparameters=recipe_hyperparameters("<name>")`.

**A model registered in RamanBench** (`models/custom/<key>/`, see
[Adding a new model](../CONTRIBUTING.md#adding-a-new-model)) runs through
`scripts/run_experiment.py`; `raman-bench protocol` prints one call per task and fold with the
protocol settings:

```bash
raman-bench protocol MY_MODEL --results-dir results/my_model > jobs.txt && bash jobs.txt
raman-bench compare results/my_model --output-dir results/my_model_figures --top 10
```

A task your model has no result for gets Random Forest's result and counts as imputed
(`imputed_pct`); a model more than 50% imputed is not ranked. The `preprocessing` column lists
the RamanBench preprocessing steps each model ran with (`none` for most). Steps inside your own
scikit-learn `Pipeline` don't appear there.
[`notebooks/02_benchmark_new_model.ipynb`](../notebooks/02_benchmark_new_model.ipynb) walks
through all of it.

## Rank by another metric

The leaderboard ranks classification by ROC AUC (binary) or log loss (multiclass) and
regression by RMSE. Every result also stores its test predictions, so the bundled reference
results carry more metrics per fold, and `compare` and `leaderboard` can rank by any of them:

```python
from raman_bench.compare import compare, leaderboard

leaderboard(classification_metric="accuracy", regression_metric="r2")["classification"].leaderboard
compare("results/my_model", classification_metric="balanced_accuracy", regression_metric="mae")
```

```bash
raman-bench metrics                                   # what there is
raman-bench compare results/my_model --classification-metric f1_macro --regression-metric rpd
```

| Classification | Regression |
|---|---|
| `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `mcc`, `cohen_kappa`, `roc_auc`, `log_loss`, `brier` | `rmse`, `mae`, `median_ae`, `max_error`, `r2`, `explained_variance`, `pearson_r`, `spearman_r`, `rpd` |

The predicted class is the most probable one. `rpd` is the standard deviation of the test
targets over the RMSE. Your own results get these metrics when you pass a results directory
(or a tidy DataFrame with the metric columns). Elo, win rate and rank are then computed on
that metric exactly as for the default one.

## Use the v1 folds in another framework

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

These are the folds `scripts/run_experiment.py` builds, the same whatever scikit-learn
version is installed. Score each fold with the task's metric from
`load_protocol()["tasks"]` (ROC AUC for binary, log loss for multiclass, RMSE for
regression) to compare with the leaderboard models' per-fold results. A few leaderboard
models run on a smaller random sample of some datasets because of memory limits
(`load_protocol()["model_max_train_samples_overrides"]`, e.g. PerpetualBooster and Mitra
at 3,000 rows); their results on those datasets come from that sample, not these folds.

## The v0.1 leaderboard

The v0.1 results (29 models, one holdout split per seed) are still bundled and scored by
the `Leaderboard` class. They use a different protocol, so don't compare them with v1
numbers:

```python
from raman_bench import Leaderboard

lb = Leaderboard.from_precomputed()
print(lb.rank())
```
