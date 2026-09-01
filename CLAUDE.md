# CLAUDE.md — RamanBench

Guidance for Claude Code when working with this repository.

## Current State (as of v1.0.0 release)

**Status**: v1.0.0 released (Sept 1, 2026)
- Main branch contains the completed v1 refactor (TabArena integration, k-fold CV, new preprocessing)
- All commits from `refactor/v1-tabarena` have been merged to main
- There are no active feature branches for major work; v1 is complete and released

**Important**: The v1 refactor is NOT a separate branch anymore. Do not look for work-in-progress on `refactor/v1-tabarena` — it is merged. If you see references to "mid-refactor," they are stale.

## What This Repo Is

**RamanBench** is the public benchmark package for machine learning on Raman spectroscopy data.

- **Public**: Published on GitHub (`github.com/ml-lab-htw/RamanBench`) and PyPI (`pip install raman-bench`)
- **Core components**:
  - **Models** (`src/raman_bench/models/`): 28+ baseline architectures + new models (RamanPFN, etc.)
  - **Metrics** (`src/raman_bench/metrics/`): Classification and regression evaluation
  - **Preprocessing** (`src/raman_bench/preprocessing/`): Tunable Raman-specific pipeline (airPLS, arPLS, EMSC, GCU, LVSE, etc.)
  - **Splitting** (`src/raman_bench/splitting.py`): Repeated k-fold CV with group-aware handling
  - **Benchmark orchestration** (`src/raman_bench/benchmark.py`): `RamanBenchmark` class for loading datasets and running evaluations
  - **Leaderboard** (`src/raman_bench/leaderboard.py`): Elo rankings, scores, timing metrics

## Architecture: Depends on TabArena (v1)

RamanBench v1 now depends directly on **TabArena/bencheval** (`tabarena` package) for:
- **Model registry & configuration** (`tabarena.models.registry`, `ConfigGenerator`)
- **Metrics computation** (`tabarena.metrics`)
- **Splitting/UserTask** (`tabarena.benchmark.experiment`, `tabarena.splits`)

This is a real architectural dependency, not a style reference. If you need to:
- Add a new model type → Check if TabArena already has it; if not, add to TabArena first
- Change how metrics are computed → That logic now lives in TabArena
- Understand the splitting protocol → See `tabarena.splits` + `raman_bench.splitting` (which wraps it for Raman-specific group handling)

## Key Files

| File | Purpose |
|------|---------|
| `src/raman_bench/benchmark.py` | Core `RamanBenchmark` class; loads datasets (raw or HF mirror) and orchestrates train/test splits |
| `src/raman_bench/preprocessing/mixin.py` | `RamanPreprocessingMixin`: joint AutoGluon preprocessing + model HPO (8 core steps + ensemble + GCU/LVSE) |
| `src/raman_bench/models/custom/` | Raman-specific architectures (DeepCNN, RamanNet, RamanPFN, etc.); custom models go here |
| `src/raman_bench/splitting.py` | Repeated k-fold logic with group-aware fallbacks for replicate structure |
| `pyproject.toml` | Version (currently 1.0.0), dependencies (including `tabarena`), package metadata |
| `CHANGELOG.md` | Release notes; update this for every release (move "Unreleased" → version section) |

## Datasets & The Mirror

RamanBench's benchmark loads datasets from **raman_data** (`raman_data` package):
- **Raw**: `raman_data("dataset_key")` downloads from original sources (Kaggle, HuggingFace, Zenodo, Figshare, GitHub)
- **Mirror**: HuggingFace mirror (`HTW-KI-Werkstatt/RamanBench` Space) caches datasets as Parquet for reliability
- **Group IDs**: If a dataset has `is_grouped=True` and explicit `group_ids` array, that's serialized as `_group_id` column in mirror parquet and read back by `_load_from_mirror()`

Default is `use_mirror=True`. Tests and integration scripts that need raw source access pass `use_mirror=False` explicitly.

## Versions & Releases

**Version numbering**: Semantic versioning. v1.0.0 was the TabArena+k-fold release.

**Release process**:
1. Update `pyproject.toml` version field
2. Move "Unreleased" section in CHANGELOG.md to a dated release section (e.g., `## [1.0.0] — 2026-09-01`)
3. Add empty "Unreleased" section at top
4. Commit: `git commit -m "Release vX.Y.Z: <summary>"`
5. Tag: `git tag -a vX.Y.Z -m "<release notes>"`
6. Push: `git push origin main vX.Y.Z` (or create PR if main is protected)
7. Tag will trigger PyPI publish via CI (if configured)

**Current version**: 1.0.0 (released Sept 1, 2026)

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_smoke.py -v       # Smoke tests (fast, no downloads)
pytest tests/ --cov=raman_bench     # With coverage
```

Tests that download from raw sources (not mirror) carry `@pytest.mark.skip` or conditional `@pytest.mark.skipif` decorators to prevent CI flakiness from external timeouts.

## Common Tasks

### Adding a new model
1. Implement in `src/raman_bench/models/custom/<model_name>.py`
2. Register in `src/raman_bench/models/__init__.py` or the model registry
3. If it depends on TabArena, check if TabArena already has it first
4. Add test in `tests/`
5. Update CHANGELOG.md under "Added"

### Adding a new preprocessing step
1. Implement the fit/transform functions in `src/raman_bench/preprocessing/raman_preprocessing.py`
2. Add a `_PREP_STEP_DEFINITIONS` entry in `src/raman_bench/preprocessing/mixin.py`
3. Register in `_ALL_PREPROCESSING_STEPS`
4. Add HPO search space to `AutoGluonModel._build_model_hyperparameters()`
5. Add test (especially for shape-changing steps)
6. Update CHANGELOG.md

### Running the full benchmark locally
```bash
python scripts/run_benchmark.py --config configs/v1_default.json --step predictions
python scripts/run_benchmark.py --config configs/v1_default.json --step metrics
python scripts/run_benchmark.py --config configs/v1_default.json --step plots
```

Cluster runs (SLURM) use `cluster/submit_job.py` and `cluster/opportunistic_scheduler.py`.

## Important: No Half-Done State

This codebase is currently in a released state (v1.0.0). There are no major in-flight refactors or experimental branches. If you see a branch that looks like active development:
- `refactor/v1-tabarena` — This was merged to main long ago; ignore it
- Feature branches (e.g., `feat/new-preprocessing`) — These are rare; check git log to see if they're stale

If you're about to start significant work, create a feature branch, but assume main is stable.

## Troubleshooting

**"I can't find where metric X is computed"**
→ Check `src/raman_bench/metrics/` first. If not there, it's in TabArena (`tabarena.metrics`). Look at how `BenchmarkResult` or `Leaderboard` calls into TabArena.

**"The preprocessing pipeline is slow"**
→ Check if you're running ensemble preprocessing (`prep_ensemble_enabled`). Each block is independent and fold-safe, but concatenating many blocks creates a wide feature matrix. Start with a single recipe.

**"Group-aware splitting isn't working"**
→ Verify the dataset has `is_grouped=True` AND an actual `group_ids` array (not None). If the array is missing, the fallback is `infer_group_ids_from_targets()`, which only works for regression. Classification datasets with no group_ids get regular (ungrouped) splits.

**"CI is failing on dataset download"**
→ Probably a network timeout from Zenodo/Kaggle/HuggingFace, not a code bug. The test should have `@pytest.mark.skip` or `@pytest.mark.skipif` already; if it doesn't, that test file needs the decorator added.

## Links

- Public repo: https://github.com/ml-lab-htw/RamanBench
- PyPI: https://pypi.org/project/raman-bench/
- Paper: https://arxiv.org/abs/2605.02003
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Upstream TabArena: https://github.com/ml-lab-htw/tabarena (our fork; original at https://github.com/automl/tabarena)
- raman_data: https://github.com/ml-lab-htw/raman_data