# Precomputed Results — RamanBench v0.1

This directory contains the precomputed benchmark results published alongside the paper (under review).
They represent 28 baseline models evaluated on 74 datasets (163 prediction targets)
across 3 random seeds.

## Files

| File | Description |
|---|---|
| `leaderboard_overall.csv` | Combined leaderboard (all task types): Elo, Score, Avg Rank, Improvability, timing |
| `leaderboard_clf.csv` | Classification-only leaderboard |
| `leaderboard_reg.csv` | Regression-only leaderboard |
| `datasets.csv` | Dataset catalog: name, domain, task, samples, features, classes/targets, license |
| `dataset_stats.json` | Machine-readable dataset metadata (used by the benchmark pipeline) |

## Columns

### Leaderboard CSVs

| Column | Description |
|---|---|
| `Model` | Model identifier |
| `Elo` | Elo rating calibrated to RF = 1000 (200-round bootstrap) |
| `Score` | Normalised per-dataset score: best model = 1, median = 0; averaged across datasets |
| `Avg Rank` | Average rank across all datasets/targets |
| `Improvability` | % gap to the best model per dataset, averaged across datasets |
| `Train Time (s)` | Mean training time in seconds |
| `Infer. s/1K` | Mean inference time per 1,000 samples in seconds |

## Version

**v0.1.0** — released 2026-04-14
- 74 datasets, 163 targets, 28 models, 3 seeds each
- Config: `configs/benchmark_v0.1.json` equivalent (`extreme_quality` preset, no HPO, no preprocessing)

## Using with the Leaderboard class

```python
from raman_bench import Leaderboard

lb = Leaderboard.from_precomputed()
print(lb.rank())
```

## Live Leaderboard

The interactive leaderboard with figures and dataset explorer is hosted at:
https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench

## Links

- raman-data (datasets): https://github.com/ml-lab-htw/raman_data | `pip install raman-data`
- raman-bench (package): https://github.com/ml-lab-htw/RamanBench | `pip install raman-bench`
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Paper (under review): https://arxiv.org/abs/2605.02003

## Citation

If you use these results, please cite:

```bibtex
@article{koddenbrock2026ramanbench,
  title={RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy},
  author={Koddenbrock, Mario and Lange, Christoph and Legner, Robin and J{\"a}ger, Martin and K{\"o}gler, Martin and Bournazou, Mariano N Cruz and Neubauer, Peter and Biessmann, Felix and Rodner, Erik},
  journal={arXiv preprint arXiv:2605.02003},
  year={2026}
}
```
