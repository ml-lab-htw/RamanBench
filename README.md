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
evaluation protocol, and ships the v1 results of 69 models on its 135 benchmark tasks.
You can score a new model against those baselines without re-running the
experiments. The baselines cover classical chemometrics, gradient boosting,
tabular foundation models, and deep networks built for spectra.

## Ecosystem

| Resource                        | Link                                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------------------|
| **raman-data** (dataset loader) | [GitHub](https://github.com/ml-lab-htw/raman_data) · [PyPI](https://pypi.org/project/raman-data/)  |
| **raman-bench** (this package)  | [GitHub](https://github.com/ml-lab-htw/RamanBench) · [PyPI](https://pypi.org/project/raman-bench/) |
| **Live Leaderboard**            | [huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) |
| **Paper**                       | Accepted at NeurIPS 2026 · [arXiv:2605.02003](https://arxiv.org/abs/2605.02003)                    |

## Install

```bash
pip install raman-bench                # datasets, v1 results, preprocessing; works offline
pip install "raman-bench[plots]"       # also score and rank results
pip install "raman-bench[models]"      # also the built-in models
```

Running a model on the benchmark protocol needs TabArena and AutoGluon. Versions, the git-only
requirements and the full install are in [docs/installation.md](docs/installation.md). The
package's major version is the benchmark version: every `raman-bench 1.x` release scores
benchmark v1.

## Quick start

Load a dataset:

```python
from raman_data import raman_data

ds = raman_data("amino_acids_glycine")
print(ds.spectra.shape)      # (n_samples, n_wavenumbers)
print(ds.raman_shifts[:5])   # wavenumber axis in cm⁻¹
```

Browse the v1 leaderboard (the per-fold results ship with the package):

```python
from raman_bench.compare import leaderboard

leaderboard()["all"].leaderboard.head(10)    # Elo, win rate, times, ...
```

Score your own scikit-learn model on the same 135 tasks and rank it among the baselines (needs the
full install):

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from raman_bench.evaluate import evaluate_estimator
from raman_bench.compare import compare

evaluate_estimator("MY-PLS", classifier=LogisticRegression(max_iter=2000),
                   regressor=PLSRegression(n_components=10), results_dir="results/my_model")
compare("results/my_model")["all"].leaderboard.head(15)
```

More: other metrics, the v1 folds as plain spectrum ids for use in another framework, and the
v0.1 leaderboard are in [docs/compare-your-model.md](docs/compare-your-model.md).

## Preprocessing

The benchmark's preprocessing is a scikit-learn interface: one transformer per step, a
`RamanPreprocessor` that takes a recipe, and 26 named recipes from the RamanPreprocessing study.
It needs only the core install, and a recipe gives the same output as a transformer and as the
`prep_*` hyperparameters of a benchmark model.

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from raman_bench.preprocessing import recipe

model = make_pipeline(recipe("bc_snv_avgpool"), PLSRegression(n_components=10))
model.fit(X_train, y_train)
```

The interface, the recipes and how to add a step are documented in
[docs/preprocessing.md](docs/preprocessing.md).

## Documentation

| Page | What it covers |
|---|---|
| [Installation](docs/installation.md) | The three install levels, extra requirements files |
| [Compare your model](docs/compare-your-model.md) | Score a model against the leaderboard, other metrics, the v1 folds |
| [Preprocessing interface](docs/preprocessing.md) | Transformers and 26 named recipes |
| [Models](docs/models.md) | The leaderboard models and the standalone scikit-learn wrappers |
| [Protocol](docs/protocol.md) | Datasets, ranking protocol, departures from it |
| [Architecture](docs/architecture.md) | How v1 is built on TabArena, repository structure |
| [Running the benchmark](docs/running-the-benchmark.md) | Experiments, systems, cluster sweeps, figures |
| [Contributor agents](docs/contributor-agents.md) | The Claude Code agents for model, dataset and cluster work |

Example notebooks are in [`notebooks/`](notebooks/). Release notes are in
[CHANGELOG.md](CHANGELOG.md).

## Contributing

New models, systems and datasets are welcome. Only want to see where your method lands? You don't
need to add it to the repository: run it with `evaluate_estimator` and `compare`. To put it on the
leaderboard, open an issue or pull request with the method and its results directory.

### Model or system?

RamanBench takes two kinds of entrant, and they are run differently.

| | Model | System |
|---|---|---|
| What it is | One method with a fixed configuration, scored under the shared protocol | A pipeline that does its own validation, tuning and ensembling |
| How it runs | Through the protocol's bagged experiment: 3-fold bagging, a 600 s budget, the row caps of `protocol.json` | Once per outer fold on the whole training split, on the same folds and budget |
| Code | `src/raman_bench/models/custom/<key>/` | `src/raman_bench/systems/` |
| Run it | `scripts/run_experiment.py`, or `evaluate_estimator` | `scripts/run_system.py` |

If your method is a plain `fit` / `predict_proba` estimator, it is a model. If it searches, validates
and ensembles by itself (an AutoML framework, a hosted API), it is a system. Results of the two are
not interchangeable: a model fitted once on the full training split sees more rows per fit than a
bagged one, so it cannot be compared with bagged leaderboard models. Run it as a model, bagged, or
submit it as a system, and say which.

### Submitting

1. Say how your method differs from the models already on the leaderboard. A new version of an
   existing method goes next to the old one under its own key.
2. Run it on the v1 protocol: all 405 folds of `data/precomputed/v1/protocol.json`, with the bagging,
   time budget and row caps the protocol sets for models, or as a system. Do not change the folds,
   the budget or the row caps for your method.
3. Score it with `compare`, and keep the results directory (the `results.pkl` files).
4. Open an issue or pull request with the method (or the code to run it), the results directory, the
   package versions and the hardware.
5. We check the results against the protocol and re-run the method on our hardware for the final
   entry, then add it to the leaderboard.

To add a model or dataset to RamanBench itself, open this repo in Claude Code and say
`Add my model to RamanBench, test it, and run it across the benchmark.` (or the same for a
dataset); the [agents](docs/contributor-agents.md) do the work. The manual route is in
[CONTRIBUTING.md](CONTRIBUTING.md) and [NEW_DATASETS.md](NEW_DATASETS.md).

## Citation

RamanBench has been accepted at NeurIPS 2026. If you use RamanBench in your research, please cite (also in [`CITATION.cff`](CITATION.cff), which GitHub shows as "Cite this repository"):

```bibtex
@inproceedings{koddenbrock2026ramanbench,
  title={RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy},
  author={Koddenbrock, Mario and Lange, Christoph and Legner, Robin and J{\"a}ger, Martin and K{\"o}gler, Martin and Bournazou, Mariano N Cruz and Neubauer, Peter and Biessmann, Felix and Rodner, Erik},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  note={arXiv preprint arXiv:2605.02003}
}
```

## License

MIT, see [LICENSE](LICENSE).

Dataset licenses vary; see the [raman-data catalog](https://github.com/ml-lab-htw/raman_data) for
per-dataset license information. Most datasets are released under CC BY 4.0.

[![Star History Chart](https://api.star-history.com/svg?repos=ml-lab-htw/RamanBench&type=Date)](https://star-history.com/#ml-lab-htw/RamanBench&Date)
