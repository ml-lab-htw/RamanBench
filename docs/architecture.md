# Architecture

How RamanBench v1 is built, and where things are in the repository.

## RamanBench v1 on TabArena

The benchmark-running layer is built directly on
[TabArena](https://github.com/autogluon/tabarena) (`tabarena` / `bencheval`) instead of on its
own reimplementation of the same ideas.

Each (dataset, target) pair becomes a TabArena `UserTask`. Splitting is real repeated k-fold
cross-validation (`raman_bench.splitting`, with `n_repeats` scaled to dataset size, as TabArena
documents). Each model's HPO search space is a TabArena `ConfigGenerator` config pool, and the
default / tuned / tuned+ensemble results are read back from that pool without any re-training
(`EndToEnd.from_raw(...).get_results(...)`).

What RamanBench adds on top is small: datasets come from the raman-data HF mirror rather than
OpenML, and about 13 Raman-specific architectures are registered alongside TabArena's own models.

`tabarena` is pinned to a commit of upstream `autogluon/tabarena` main, not a PyPI release, and
installed from `requirements-tabarena-git.txt`, whose comments give the pin history and why.

### Key entry points

- `scripts/run_experiment.py`: thin per-(model, dataset, target, repeat, fold, config-index) job
  runner; reads datasets mirror-first by default (`--use-mirror` / `--no-use-mirror` to opt out).
- `raman_bench.models.registry.raman_bench_model_registry`: TabArena's full model registry plus
  RamanBench's own architectures and Raman-preprocessing overrides.
- `scripts/run_system.py`: runs a self-contained *system* (`raman_bench.systems`: Chakra-Tab's hosted
  API, TabFM+) once per outer fold, on the same tasks, folds and budgets; see
  [Running the benchmark](running-the-benchmark.md).
- `cluster/`: cluster-agnostic job submission for SLURM and Kubernetes (`submit_job.py`,
  `run_experiment.sbatch`, `k8s_entrypoint.sh`, `detect_cluster.py`, `janitor.py`) driven by a
  profile YAML; works locally too when no cluster is available.
- `.claude/agents/`: Claude Code agents for routine maintenance, see
  [Contributor agents](contributor-agents.md).

New models are onboarded via a per-model directory,
`raman_bench/models/custom/<key>/{model.py,hpo.py,info.py}`, auto-discovered into the registry; see
`models/custom/ridge/` for the reference implementation and `.claude/agents/model-agent.md` for the
full workflow. Models that already exist in TabArena are wrapped with the Raman preprocessing in
`preprocessing/wrapped_models.py` and get a generator in `models/generate/`.

## Repository structure

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
│   ├── folds.py                # the v1 outer folds as plain spectrum ids (raman-bench folds)
│   ├── fold_metrics.py         # per-fold metrics from stored test predictions
│   ├── systems/                # self-contained systems (Chakra-Tab, TabFM+) for run_system.py
│   ├── models/
│   │   ├── registry.py         #   raman_bench_model_registry (TabArena's + ours)
│   │   ├── discover.py         #   auto-discovery for models/custom/<key>/info.py
│   │   ├── _model_info.py      #   lightweight per-model ModelInfo dataclass
│   │   ├── generate/           #   per-model ConfigGenerator (HPO search space) modules
│   │   └── custom/             #   all built-in Raman models (scikit-learn API)
│   │       ├── base.py         #     BaseRamanEstimator (shared training loop)
│   │       ├── ridge/          #     reference for the per-directory {model.py,hpo.py,info.py} convention
│   │       └── deepcnn.py, ramannet.py, sanet.py, ...   (older flat-file convention, still supported)
│   └── preprocessing/          # see preprocessing.md
│       ├── transformers.py     #   RamanPreprocessor and the one-step transformers (public)
│       ├── recipes.py          #   the 26 named recipes (public)
│       ├── mixin.py            #   RamanPreprocessingMixin, shared by transformers and Prep_* models
│       ├── bridge_bases.py     #   SklearnAutoGluonBridge + shared bases
│       └── wrapped_models.py   #   Prep_* classes, PREPROCESSED_MODELS registry
├── cluster/                    # cluster-agnostic SLURM and Kubernetes submission
│   ├── detect_cluster.py, submit_job.py, run_experiment.sbatch, k8s_entrypoint.sh, janitor.py
│   └── profiles/               #   generic + example cluster profiles (no secrets)
├── scripts/
│   ├── run_experiment.py       # per-(model,dataset,target,repeat,fold,config) job runner
│   ├── run_system.py           # one self-contained system per outer fold
│   ├── compute_fold_metrics.py # per-fold metrics for the reference results
│   ├── build_target_list.py    # builds the full-benchmark target list (mirror-first)
│   ├── aggregate_results.py    # recycles cached results into default/tuned/tuned+ensemble
│   ├── build_reference_results.py  # bundles the leaderboard results + protocol with the package
│   └── plot_results.py         # leaderboard figures (static PNG/PDF + interactive HTML)
├── .claude/agents/             # model-agent, cluster-agent, ... (see contributor-agents.md)
├── configs/                    # benchmark configuration files (configs/v1, configs/v2)
├── docs/                       # this documentation
├── src/raman_bench/data/precomputed/   # v1/: reference results + protocol; *.csv: v0.1 results
├── notebooks/                  # example Jupyter notebooks
└── tests/                      # pytest test suite
```

## What changed since v0.1

Full details are in [CHANGELOG.md](../CHANGELOG.md); the short version:

**Models**
- Onboarded 30 TabArena-native models directly from TabArena's own registry (TabFM, TabPFN-3,
  TabSwift, ModernNCA, EBM, PerpetualBooster, xRFM, ChimeraBoost, NORI, SAP-RPT-OSS, OrionMSP,
  ILTM, LIMIX, TabSTAR, and more)
- Added TabPFN v2.6, v3, and v3-Thinking
- Added TabPFN-Wide (wide, few-sample classification) and RamanPFN (Pan et al., 2025)
- Verified all 10 pre-existing custom Raman architectures against the new v1 pipeline

**Benchmark methodology (v1)**
- Migrated the model/metrics/splitting layer to depend directly on TabArena/`bencheval` instead
  of reimplementing patterns "inspired by" them
- Switched to real repeated k-fold CV with dataset-size-adaptive repeat counts, replacing the old
  3-independent-holdout-split scheme
- Fixed AutoGluon bagging to a genuine, TabArena-style bag (v0.1 had effectively no bagging for 27
  of 28 models); the v1 protocol uses 3 bagging folds, see [Protocol](protocol.md)
- Added semi-supervised-aware splitting (unlabeled rows kept in train, never in test)
- Ported TabArena's trivial-dataset filter into RamanBench itself as a first-class feature

**Preprocessing**
- 8 new steps: airPLS, arPLS, rubberband, EMSC, Savitzky-Golay derivative, wavelet denoising,
  fingerprint-region crop, L2 vector normalization
- New preprocessing-ensemble mechanism (parallel recipe blocks, concatenated) and config-level
  `preprocessing_params` overrides
- Since 1.6.0: the steps as scikit-learn transformers and 26 named recipes, see
  [Preprocessing interface](preprocessing.md)

**Datasets**
- +4 datasets: `chlorinated_samples`, `locust_phase_hemolymph`, `cspp_serum_metabolites`,
  `ait_glucose_blood_sers`
- New `is_grouped` / `has_missing_labels` fields on `raman-data`'s `DatasetInfo`

**Infrastructure**
- Public, cluster-agnostic job-submission tooling plus an opportunistic, capacity-aware scheduler
  for routine full-benchmark sweeps
- Dropped the patched AutoGluon fork; moved to upstream AutoGluon 1.6 with RamanBench-local cap overrides
- Automatic `.env` credential loading; `main` requires PRs (no direct pushes)

**Reliability fixes**
- Atomic prediction/index writes to avoid concurrent-job races
- Skip (not delete) mismatched predictions during metric computation
- Guard R² against degenerate near-constant test folds
