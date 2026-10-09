# v1 reference results

The per-fold results of every model on the RamanBench v1 leaderboard, and the protocol
they were produced with. They ship with the package, so a new model can be ranked against
the leaderboard without rerunning it (`raman_bench.compare`).

| File | Contents |
|---|---|
| `reference_results.parquet` | One row per (task, fold, model): 58 models (incl. the `DUMMY` baseline and two AutoGluon reference systems) x 135 tasks x 3 folds, 22,574 rows |
| `protocol.json` | Folds, bagging, time budget, row caps, the task list and each model's enabled preprocessing |

## `reference_results.parquet`

| Column | Meaning |
|---|---|
| `dataset` | Task key `<dataset>__<target_idx>` |
| `fold` | Outer fold (0-2) |
| `model` | Model key, e.g. `PLS`, `KUMO-TABULAR` |
| `metric_error` | Lower is better: 1 - ROC AUC (binary), log loss (multiclass), RMSE in the target's units (regression) |
| `metric` | `roc_auc`, `log_loss` or `rmse` |
| `task` | `classification` or `regression` |
| `time_train_s`, `time_infer_s` | Wall-clock training (incl. bagging) and inference time of the fold |
| `reference` | `True` for the AutoGluon reference systems, which the figures draw as lines and do not rank |
| `num_instances` | Spectra in the task |
| `ta_name` | The model's result directory name (without `_c1_BAG_L1`) |
| `accuracy`, `balanced_accuracy`, `f1_macro`, ..., `rmse`, `mae`, `r2`, ... | The 20 metrics of `raman_bench.fold_metrics.METRICS`, computed from the fold's stored test predictions (`scripts/compute_fold_metrics.py`); NaN where a metric doesn't apply to the task type, and for the AutoGluon reference systems, which store no predictions |

## `protocol.json`

- `n_splits`: outer folds per task (3). `n_repeats` per task is 1.
- `num_bag_folds`: AutoGluon bagging folds inside each training split (3).
- `time_limit`: seconds per fit (600). `time_limit_overrides` raises it per dataset
  (`mlrod`: 10,800).
- `max_train_samples_overrides`: per dataset, a random sample of this many spectra is drawn before splitting (10,000 on the three largest).
- `model_max_train_samples_overrides`: smaller samples for a few models on some datasets, because of memory limits.
- `num_random_configs`: 0, i.e. each model's default configuration.
- `min_samples_per_class`: classes with fewer spectra are dropped (9).
- `reference_model`: the Elo anchor (`RF` = 1000), also used to impute missing results.
- `preprocessing`: the RamanBench preprocessing steps each model was fit with (`none`:
  raw spectra).
- `tasks`: one entry per task with `dataset`, `target_idx`, `problem_type`, `metric`,
  `num_instances`, `n_repeats` and `n_folds`.

Model-specific overrides from the sweep's scope file (e.g. a longer budget or a lower row
cap for one slow model) are not part of the protocol; a new model gets the defaults.

## Using it

```python
from raman_bench.compare import compare, leaderboard, load_protocol, load_reference

load_reference()            # the table above, without DUMMY
leaderboard()               # scores, as on the live leaderboard
compare("results/my_model") # your results ranked among them
```

See the README's "Compare your model against the v1 leaderboard" and
`notebooks/02_benchmark_new_model.ipynb`.

## Rebuilding

After a new sweep, from the aggregated results and the raw cached results:

```bash
python scripts/aggregate_results.py --results-dir results/v1/data --output-dir results/v1/aggregated
python scripts/build_reference_results.py --input results/v1/aggregated/hpo_results.csv \
    --results-dir results/v1/data --output-dir src/raman_bench/data/precomputed/v1
```

`--scope` picks the models (default `configs/v1/scope_default.json`), `--target-list` the
tasks. When the raw results live elsewhere (e.g. on a cluster volume), write
`raman_bench.aggregation.preprocessing_by_model(results_dir)` to JSON there and pass it as
`--preprocessing-json`.

This build (2026-10-05): the 2026-10-05 sweep including Causilo, scope
`scope_default.json` + `CAUSILO`, without TabSTAR (later release).
