# Version comparison figures

Figures that compare two RamanBench versions (first used for v0.1 vs. v1), and a few that look
at the newer one in depth.

| Script | Figures |
|---|---|
| `wins_by_family.py` | who wins, per model category and family: all tasks and models of each version (`wins_by_family_full`), and only the tasks and models in both (`wins_by_family_shared`) |
| `analyses.py` | `01` / `01b` winner flow (same models / with the new models), `02` Elo rank changes, `03` progress by model release date, `04` headroom per task, `05` winner by dataset shape, `06` winner by domain, `07` ties within fold noise, `08` cost of the best result, `09` regret of always using one model, `10` licences |
| `common.py` | loading, categories, model families and their fixed colours, normalized score |

## Inputs

- `--old`: the older version, either a reference-results parquet (v1 and later), or v0.1's
  `precomputed_metrics` directory (`classification_metrics.csv`, `regression_metrics.csv`).
- `--new`: the newer version's reference-results parquet, e.g.
  `src/raman_bench/data/precomputed/v1/reference_results.parquet`.
- `--leaderboard-dir` (`analyses.py` only): `leaderboard_all.csv` and `datasets.csv` of the
  newer version, as written by `scripts/plot_results.py`.
- `--old-leaderboard` (figure 02): the older version's Elo leaderboard CSV. For a v0.1
  directory it defaults to its `leaderboard_overall.csv`. Without one, figure 02 is skipped.

Each version is scored on its own metric:

- **v1 and later:** the leaderboard's `metric_error` (1 − ROC AUC for binary tasks, log loss
  for multiclass, RMSE for regression), plus the normalized score derived from it.
- **v0.1:** F1 and RMSE, because v0.1 stored no probabilities.

Thresholds are relative to the error (for example "within 5 % of the winner's error"), so they
mean the same for every metric. DUMMY, AutoGluon, TabSTAR and RamanPFN are not ranked (see
`common.EXCLUDE`).

## Usage

```bash
python scripts/version_comparison/wins_by_family.py \
    --old <v0_1>/precomputed_metrics \
    --new src/raman_bench/data/precomputed/v1/reference_results.parquet \
    --out results/v1/figures_v01_vs_v1/more
python scripts/version_comparison/analyses.py \
    --old <v0_1>/precomputed_metrics \
    --new src/raman_bench/data/precomputed/v1/reference_results.parquet \
    --leaderboard-dir <plot_results.py output> \
    --out results/v1/figures_v01_vs_v1/more --only 01,03
```

- `--old-label` / `--new-label` set the version names (defaults `v0.1`, `v1`).
- `--no-footnote` writes clean versions for posts.

## When reusing

- **Colours:** a family keeps one colour in every figure and both versions
  (`common.FAMILY_COLOURS`). A winner that has no colour yet is drawn in its category's colour,
  and `wins_by_family.py` prints it, so add it to the table.
- **Licences:** `common.NON_COMMERCIAL` / `HOSTED` were checked on 2026-10-16. Re-check them
  before reusing figure 10.
- **Figure 03:** set `--since` to change the first release date shown. Set `--regret-models`
  to change the models figure 09 compares with the leaderboard's #1.
