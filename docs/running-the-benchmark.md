# Running the benchmark

For maintainers and anyone who wants to reproduce the leaderboard or add a model's results. It
needs the full install (Option 3 in [Installation](installation.md)). To score a single model
against the published results, [Compare your model](compare-your-model.md) is shorter.

## One experiment

`scripts/run_experiment.py` runs one process per `(model, dataset, target, repeat, fold,
config-index)`:

```bash
python scripts/run_experiment.py --dataset wheat_lines --target-idx 0 \
    --model PLS --repeat 0 --fold 0 --config-index 0 \
    --results-dir results/v1/data
```

A real sweep goes through `cluster/submit_job.py` or `cluster/submit_full_benchmark.py`, on
SLURM or Kubernetes (`python cluster/submit_job.py --help`). Finished folds are skipped on a
rerun.

## Systems

A *system* does its own validation and ensembling, so it isn't wrapped in AutoGluon's bagging:
`scripts/run_system.py` fits it once per outer fold on the same tasks, folds, row sample and time
budgets, into the same result cache (`raman_bench.systems` lists them):

```bash
python scripts/run_system.py --system TABFM-PLUS --tasks alzheimer__0 --folds 0
export CHAKRA_TAB_KEY=...        # hosted API: every fold's data is sent to the provider
python scripts/run_system.py --system CHAKRA-TAB --workers 4
```

## Leaderboard figures

Once results exist, aggregate them and draw the leaderboard figures (needs
`pip install "raman-bench[plots]"`):

```bash
python scripts/aggregate_results.py --results-dir results/v1/data --output-dir results/v1/aggregated
python scripts/plot_results.py --input results/v1/aggregated/hpo_results.csv --output-dir results/v1/figures
```

Scores come from TabArena's own evaluator (`bencheval`): Elo with bootstrap CIs (Random Forest =
1000), win rate, improvability and the pairwise win-rate matrix. Every figure (benchmark
overview, example spectra, composition, Elo ranking, score and improvability against time per 1K
spectra, model progress over release date, win rates, efficiency, critical difference diagrams)
is written as PNG and PDF under `static/` and as an interactive HTML page under `interactive/`,
with an `index.html` linking them all. By default only the top 2 models per category are drawn
in colour and the rest in grey (`--focus-top-k`, `0` colours all); the HTML pages can switch
between both views. The dataset figures read metadata from raman_data and the RamanBench mirror
on the Hugging Face Hub; `--no-dataset-figures` skips them offline.

`scripts/plot_tradeoff_100k.py` draws Elo against median time per 100K spectra for a chosen set
of models from a `leaderboard_*.csv`.

## Rebuilding the bundled reference results

After a sweep, `scripts/build_reference_results.py` rebuilds the results and protocol that ship
with the package; see `src/raman_bench/data/precomputed/v1/README.md`. The release steps are in
[RELEASING.md](../RELEASING.md).

## Notebooks

| Notebook | Description |
|---|---|
| [`01_quick_start.ipynb`](../notebooks/01_quick_start.ipynb) | Load a dataset, look at the v1 leaderboard |
| [`02_benchmark_new_model.ipynb`](../notebooks/02_benchmark_new_model.ipynb) | Run your own model on the benchmark protocol and rank it against the leaderboard models |
| [`03_explore_results.ipynb`](../notebooks/03_explore_results.ipynb) | Per-task results, task types, speed against accuracy, pairwise win rates |
| [`04_contribute_dataset.ipynb`](../notebooks/04_contribute_dataset.ipynb) | Adding a new dataset, step by step |
