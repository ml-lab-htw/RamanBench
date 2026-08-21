# Changelog

All notable changes to RamanBench are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-04-14

### Added

- Initial public release of RamanBench.
- **74 datasets** across Chemical, Medical, Biological, and Material Science domains.
- **163 prediction targets** (regression and classification).
- **28 baseline models** evaluated with 3 seeds each:
  - Classical ML: PLS, KNN, LR, RF, XT, GBM, XGB, CatBoost, Dummy
  - Tabular DL: NN_TORCH, FastAI, RealMLP
  - Foundation models: TabPFN v2, TabPFN v2.5, MITRA, TabM, TabDPT, TabICL
  - Time-series: ROCKET, ARSENAL
  - Raman-specific: DeepCNN, RamanNet, SANet, RamanFormer, RamanTransformer,
    ReZeroNet, FC-ResNeXt, CoAtNet
  - AutoGluon ensemble (AUTOGLUON)
- **Leaderboard** with Elo, Score, Avg Rank, Improvability, and timing metrics.
- **Precomputed results** bundled in `data/precomputed/`.
- **`Leaderboard` class** for evaluating new models against baselines.
- **17 new datasets** released alongside the paper (see [NEW_DATASETS.md](NEW_DATASETS.md)).
- Raman preprocessing pipeline: cosmic-ray removal, baseline correction, MSC,
  denoising, SNV, standard scaling, spectral augmentation.
- AutoGluon HPO mixin (`RamanPreprocessingMixin`) for jointly optimising
  preprocessing and model hyperparameters.
- Grouped regression splits to prevent data leakage from replicate measurements.
- `raman-bench` CLI entry point.
- Example notebooks in `notebooks/`.

### Links

- raman-data: https://github.com/ml-lab-htw/raman_data | `pip install raman-data`
- raman-bench: https://github.com/ml-lab-htw/RamanBench | `pip install raman-bench`
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Paper (under review): https://arxiv.org/abs/2605.02003

---

## Unreleased

RamanBench v1: migrates the model/metrics/splitting layer onto `tabarena`/`bencheval`
directly, rather than reimplementing patterns "inspired by" them.

### Added

- `configs/v1/`: the public, canonical definition of what the v1 benchmark runs --
  `datasets/{classification,regression}_all.json` (the curated 68-dataset scope),
  `target_list.json` (one row per (dataset, target), built by
  `scripts/build_target_list.py`; currently 156 runnable targets), `models.json` (the
  roster of models validated end-to-end on the new pipeline; started as just `PLS`, now
  also the 10 pre-existing custom architectures and TABPFN-WIDE -- see "Fixed" below), and
  `scope_default.json` (the opportunistic scheduler's default scope -- see below). No
  institution-specific values live here; only the SLURM profile
  (`raman_bench_paper/cluster/profiles/{htw,tu}.yaml`) stays private.
- `cluster/opportunistic_scheduler.py`: one tick of a capacity-aware, opportunistic
  scheduler for the routine full-benchmark sweep. Computes the backlog of not-yet-cached
  `(model, dataset, target, repeat, fold)` tasks fresh every tick (no separate state file
  to get out of sync -- a completed task just stops appearing), checks real idle cluster
  capacity (`sinfo`) and this user's own resident task count (`squeue`) against a courtesy
  ceiling, and submits exactly one modest-sized chunk if there's room, otherwise skips --
  logging every decision. Meant to be driven by a cron entry (see the private
  `raman_bench_paper/cluster/submit_v1_opportunistic.sh`), never by an agent call, since
  the routine decision itself needs no judgment. Verified end-to-end against the real HTW
  cluster in `--dry-run`: computed a real 4151-task backlog, correctly read 192 idle CPUs
  / 2 resident tasks off `sinfo`/`squeue`, and produced a correct `sbatch` command for a
  300-task chunk spanning many (dataset, target) pairs for one model in a single array.
  `.claude/agents/cluster-agent.md` extended with the human-facing control surface for
  this scheduler (out-of-cycle submit, backlog/progress "watch", pause/resume/adjust the
  cron trigger, and -- unlike the autonomous layer, which never touches submitted work --
  cancel/requeue a genuinely stuck array on explicit request).
- Real repeated k-fold cross-validation (`RepeatedStratifiedKFold`/`RepeatedKFold` for
  ungrouped datasets; a manually-repeated `StratifiedGroupKFold`/`GroupKFold` for datasets
  with physical-replicate structure, since sklearn has no repeated wrapper for those),
  replacing the previous scheme of 3 independent holdout splits with no folds.
  `StratifiedGroupKFold` also lifts the old scheme's limitation that a grouped
  classification split couldn't additionally be class-balanced.
- `num_bag_folds` (AutoGluon's bagging knob -- fit N models on N different folds of the
  training data and ensemble them, independent of the outer k-fold CV above) is now a
  fixed `8` for every model: `DEFAULT_NUM_BAG_FOLDS` in `scripts/run_experiment.py:51`,
  threaded through `cluster/opportunistic_scheduler.py`'s
  `num_bag_folds=scope.get("num_bag_folds", 8)` and `cluster/submit_job.py` (downscaled
  per-target only for very small datasets/classes -- see the `effective_bag_folds` fix
  below). This matches TabArena's own real default: `AGModelBagExperiment.__init__`
  (`tabarena/benchmark/experiment/experiment_constructor.py`, confirmed by reading the
  class directly under this repo's `ramanbench_1` env) declares `num_bag_folds: int = 8`
  as its own default.
- v0.1's bagging behavior, by contrast, was never a fixed number -- and for most models it
  was not used at all. `AutoGluonModel` (`src/raman_bench/model.py`) had
  `ensemble: bool = True` as its constructor default (docstring: "Enable AutoGluon
  bagging/stacking"), but `fit()` only ever sets an explicit fold count in the
  `ensemble=False` branch (`fit_args["num_bag_folds"] = 0`, `num_stack_levels = 0`); an
  `ensemble=True` fit passed no `num_bag_folds` at all, deferring entirely to AutoGluon's
  own `best_quality`-preset default (`presets_configs.py`:
  `{"auto_stack": True, "dynamic_stacking": "auto", "hyperparameters": "zeroshot"}`, no
  `num_bag_folds` key) -- which is itself not a fixed number but a data-size curve,
  confirmed in the currently-installed AutoGluon 1.6.1's `get_validation_and_stacking_method`
  (`autogluon/tabular/configs/pipeline_presets.py`):
  `DEFAULT_VALIDATION_SIZE_CURVES["num_bag_folds"] = [[59, 5], [69, 6], [79, 7], 8]`, i.e. 5
  folds at <=59 training rows, rising to 8 above 79. Critically, the config that actually
  produced the published v0.1.0 leaderboard, `configs/benchmark_v0.1.json` (present at the
  `v0.1.0` git tag, commit `3884c01`), sets `"ensemble": false` explicitly (`"optimize":
  false` too) -- so bagging was fully *disabled*, `num_bag_folds=0`, not any
  AutoGluon-chosen count, for 27 of the 28 published baselines. The one exception is the
  native `AUTOGLUON` baseline (`models=["AUTOGLUON"]`): `fit()` only applies the
  `ensemble=False` override inside `if not self._autogluon_native:`, so that one model's
  fit was never forced to `num_bag_folds=0` and did get `best_quality`'s auto_stack-driven
  bagging, at whichever fold count the AutoGluon version installed at the time resolved to
  -- the `pyproject.toml` pin at the `v0.1.0` tag is a floor (`autogluon.tabular>=1.5`),
  not an exact version, so that one historical count is not independently recoverable from
  this repo.
- "3 seeds each" (the v0.1.0 entry above) is a separate, unrelated mechanism from bagging,
  not a second stacked layer of ensembling on top of it: `n_repetitions: 3` in the same
  `configs/benchmark_v0.1.json` resolves via `raman_bench.seeds.get_seeds` to `[0, 1, 2]`,
  and `predictions.py`'s per-seed loop sets `config["random_state"] = seed` and rebuilds
  the benchmark from scratch each iteration -- `configure_benchmark`/`RamanBenchmark`
  (`src/raman_bench/benchmark.py`) draws one `train_test_split`/`GroupShuffleSplit` 80/20
  holdout per seed (`test_size: 0.2`), not a k-fold split. So v0.1's "3 seeds" meant 3
  independent single train/test holdouts, each running one un-bagged, un-tuned
  (`"optimize": false`) `TabularPredictor.fit()` per model -- not "3 seeds x AutoGluon
  bag-folds" for every model except `AUTOGLUON`. This is a real protocol difference, not a
  default that "used to be implicit and is now explicit": v1's per-model scores now come
  from a genuinely bagged, out-of-fold-aggregated fit (8-way, or downscaled) on every
  repeated-k-fold split, while v0.1's published numbers for 27/28 models came from single
  non-bagged fits repeated across 3 holdout splits. It is part of why v1 job durations for
  the same model differ from v0.1's, and it means v0.1-vs-v1 comparisons for any model
  other than `AUTOGLUON` are not "same bagging, different outer splitting" -- v0.1 had no
  bagging there at all. It also explains why "AutoGluon ensemble (AUTOGLUON)" was called
  out as its own distinct baseline in the v0.1.0 "Added" list above: it was the only model
  in that lineup whose published numbers ever touched AutoGluon's bagging machinery.
- `raman_bench.splitting.get_n_repeats`: dataset-size-adaptive repeat count (10 repeats
  under 2,500 instances, 3 up to 250,000, 1 above), ported directly from
  `tabarena.nips2025_utils.fetch_metadata._get_n_repeats` and confirmed against TabArena's
  own real 51-dataset curated metadata (`num_folds` is a fixed 3 for every dataset there;
  only `n_repeats` varies by size) -- a uniform repeat count for every dataset regardless
  of size is not what the actual protocol does.
- `scripts/build_target_list.py`: builds a (dataset, target) list from dataset-name JSON
  files, recording each target's real name, instance count, and size-adaptive `n_repeats`;
  `cluster/submit_full_benchmark.py` now reads each target's own `n_repeats` from this
  list instead of applying one value to every dataset.
- `add_seed="fold-config-wise"` on every generated config, matching TabArena's own real
  production default (`tabflow_slurm/setup_slurm_base_v2.py`'s `default_seed_config`) --
  the bare library default is `"static"` (every bag-fold and every HPO config shares
  internal seed 0); this gives each of AutoGluon's internal bag-folds, and each HPO config
  once HPO is opted in, a genuinely different internal random seed.
- 2 new datasets: `chlorinated_samples` (binary classification, chloroform detection in
  Raman spectra) and `locust_phase_hemolymph` (binary classification, density-dependent
  phase state in desert locusts -- the first dataset in `raman-data` with `group_ids`
  populated from real source metadata rather than inferred from target values).
- `cspp_serum_metabolites` (3-class classification, Zenodo 5644790): a serum metabolite
  SERS subset, `raman-data` v1.5.0.
- `ait_glucose_blood_sers` (regression, GitHub `AIT-brainlab/raman-for-glucose-measurement`,
  `bloodSERs/5x` subset): 35 spectra, 6 distinct glucose concentrations, `raman-data`
  v1.6.0. **Provenance caveat, please read before reusing**: no informed-consent, IRB, or
  ethics-review documentation was found anywhere associated with this real human
  blood/glucose data (source repo README, wiki, predecessor repo, or lab project pages);
  the source repo's predecessor states no journal paper was ever produced from this work,
  so this dataset does not meet `raman-data`'s own stated citable-reference inclusion
  criterion. A paper citing data matching this description (arXiv:2608.14227) attributes it
  to "AIT brainlab and MIT" -- the "MIT" half could not be verified and is likely erroneous
  (the source repo's actual copyright holder is "Future Lab", an AIT x BUPT joint facility,
  not Massachusetts Institute of Technology). Included at the explicit direction of the
  RamanBench maintainer, who intends to follow up directly with the originating lab
  regarding consent/ethics documentation -- not a routine inclusion, not a template for
  skipping the citable-reference or consent checks on future datasets. Also has a real
  scientific limitation independent of the above: a PLS smoke test returns RMSE ~230
  against a target range of only 80, a consequence of only 6 distinct concentration levels
  combined with group-aware splitting forcing extrapolation to unseen concentrations every
  fold.
- `DatasetInfo.is_grouped` (in `raman-data`): a filterable field marking whether a dataset
  has confirmed physical-replicate structure.
- Public, cluster-agnostic `cluster/` job-submission tooling: `submit_job.py`,
  `run_experiment.sbatch`, `detect_cluster.py`, `janitor.py`, `refresh_deps.py` (refreshes
  the released `raman-data` version and the `RamanBench` checkout once before a submission
  batch, rather than each job pulling for itself).
- `raman_bench.models.registry` (`raman_bench_model_registry`), built on TabArena's own
  model registry.
- Six Claude Code agents (dataset/model/cluster/leaderboard/frontend/docs) under
  `.claude/agents/`.
- Optional semi-supervised benchmarking support: `raman_bench.splitting`'s repeated
  k-fold splitting is now NaN-target-aware -- unlabeled (NaN-label) rows are never placed
  in a test fold (no ground truth to score against) but are included in every train fold
  (for grouped datasets, only when they don't share a fold's test group). `scripts/
  run_experiment.py --keep-unlabeled` (default: off, preserving every prior run's
  behavior) opts a job into this instead of the default unconditional NaN-drop. No model
  in this codebase yet fits on unlabeled training rows; this is data-layer plumbing for
  a future semi-supervised-aware model.
- `DatasetInfo.has_missing_labels` (in `raman-data`): a filterable field marking whether
  a dataset has confirmed NaN values in its target column(s), mirroring `is_grouped`'s
  True/False/None semantics.
- `raman_bench.filters` (`TrivialFilterConfig`, `compute_trivial_keys`, `get_trivial_keys`,
  `get_trivial_keys_from_dir`, `filter_trivial_keys`; `tests/test_filters.py`, 29 cases):
  a v1-native port of the trivial-dataset filter previously only implemented in the private
  `raman_bench_paper` repo (`raman_bench_paper/filters.py`), now a real feature of the
  RamanBench package itself -- usable by anyone running the benchmark against `results.pkl`
  output, not just paper plotting scripts. Flags a (dataset, target) *key* as "trivial" --
  carrying no real discriminative signal between models -- using the same two-criterion
  definition TabArena (NeurIPS 2025, arXiv:2506.16791) uses to curate its own benchmark
  suite. Appendix B.1, "Dataset Selection Criteria", quoted verbatim: "We exclude datasets
  that are trivial to solve... We define trivial datasets as datasets where one of the
  following criteria applies: (1) at least one of the models in our scope is consistently
  able to achieve perfect performance; (2) multiple models achieve exactly the same highest
  performance." (This is TabArena's own two-criterion version, confirmed against the real
  paper PDF; TabRepo, arXiv:2311.02971, has an earlier, simpler single-numeric-threshold
  ancestor of this idea -- AUC>0.999/logloss<0.001/r2>0.999 against one default Random
  Forest -- that is deliberately *not* what got ported.) TabArena applies this once, at
  dataset-curation time, deciding which datasets enter its fixed suite at all, before any
  leaderboard numbers exist. RamanBench's port applies it differently and more narrowly:
  post-hoc, per (dataset, target) key, against already-computed `hpo_results`, opt-in and
  off by default (`TrivialFilterConfig.enabled=False`). Nothing in this module excludes a
  dataset from being *run* -- it only flags keys whose results a downstream
  leaderboard/plot/table may want to drop.
  - **Data shape, not a copy-paste port.** `raman_bench_paper.filters` reads v0.1's
    `metrics/{classification,regression}_metrics.csv` and iterates per-*seed* rows. v1 has
    no such CSV or "seed" column -- `scripts/run_experiment.py` caches one `results.pkl`
    per `(model, dataset, target, repeat, fold)`, and `scripts/aggregate_results.py` turns
    every cached result into `hpo_results`/`model_results` via TabArena's own
    `EndToEnd.from_raw`. Read `tabarena.benchmark.result.baseline_result.BaselineResult`
    and `tabarena.benchmark.task.utils.get_split_idx` directly (installed under this repo's
    `ramanbench_1` env) to confirm the right mapping: TabArena's own pipeline already
    flattens `(repeat, fold)` into a single `split_idx = n_folds * n_samples * repeat +
    n_samples * fold + sample`, and writes that out as `hpo_results`/`model_results`'s
    `fold` column (`BaselineResult.compute_df_result`: `"fold": self.split_idx`) -- so that
    column is already exactly the old per-seed granularity, independently corroborated by
    `scripts/plot_v1_results.py`'s own comment ("Each (dataset, method) has one row per
    (repeat, fold)"). `compute_trivial_keys` defaults to `hpo_results` (one row per
    (key, model, fold), recycled default/tuned/tuned+ensemble) over the more granular
    `model_results` (one row per raw hyperparameter config) for the same reason
    `raman_bench_paper` iterated per-*model*, not per-config: criterion 2 needs to count
    distinct architectures, not HPO variants of the same one. `method_subtype="default"`
    (this module's own default) enforces that by restricting to one row per architecture
    per key/fold before either criterion runs.
  - **`perfect_clf`/`perfect_reg` default `0.0`/`0.0` here, not the paper config's
    `1.0`/`0.0`.** v0.1's per-seed metrics were F1 (higher-is-better, ceiling 1.0) for
    classification and RMSE (lower-is-better, floor 0.0) for regression -- opposite
    directions, hence the old asymmetric defaults. v1's `metric_error` column is *always*
    an error (TabArena/AutoGluon's own problem-type-appropriate metric -- ROC AUC for
    binary, log loss for multiclass, RMSE for regression -- via `Scorer.error()`, e.g.
    `1 - roc_auc`), uniformly lower-is-better with `0.0` as the shared theoretical perfect
    floor for both problem types. Copying the old `perfect_clf=1.0` forward would have
    silently matched nothing, since a v1 classification error essentially never sits at or
    above `1.0`. Kept as two separate config keys anyway (not one shared `perfect_error`)
    for config-shape continuity with `raman_bench_paper/configs/*.json`'s `trivial_filter`
    block and to let classification/regression thresholds still be tuned independently.
    `problem_type` itself is read using AutoGluon's own stored convention (`"binary"` /
    `"multiclass"` / `"regression"`, confirmed by reading
    `tabarena.benchmark.task.wrapper.RamanBenchTaskWrapper`'s docstring and its
    `self.problem_type = metadata.problem_type` assignment directly), not the
    `"classification"`/`"regression"` strings `run_experiment.py` uses internally before
    task construction.
  - **Tie detection (criterion 2) reuses the `tie_decimals` rounding convention from
    `autogluon/tabarena#311`, but not its code.** That PR -- by this project's own
    maintainer, already merged upstream -- added an opt-in `tie_decimals: int | None` to
    `bencheval.winrate_utils.compute_winrate_matrix`/`compute_winrate`, motivated (per the
    PR's own description) by exactly this porting effort: "While porting the
    trivial-dataset criterion to our benchmark, we noticed [strict `==` tie detection was
    too conservative for floating-point noise]". `bencheval` is already a RamanBench
    dependency (`pyproject.toml`'s `autogluon` extra), so it was evaluated as the natural
    home for tie detection here -- and rejected, deliberately: `compute_winrate_matrix`
    computes a pairwise win-rate matrix (or its per-method average) aggregated *across*
    every task in its input, a single leaderboard-style summary over many datasets at once;
    this module needs the opposite shape, an independent True/False decision *per key*,
    gated on a condition holding on *every single fold* for that key (an AND across folds,
    not an average). Reusing it would mean calling it once per key just to reverse-engineer
    "was every fold tied" out of one blended win-rate number. More fundamentally, criterion
    1 (a model reaching a perfect *absolute* score) has no equivalent in
    `compute_winrate_matrix` at all -- win-rate is purely relative, never compared against
    an absolute threshold -- so at least half of this filter needed an independent
    implementation regardless. `raman_bench.filters` therefore keeps a small,
    self-contained, directly testable groupby-round-compare implementation (mirroring
    `raman_bench_paper.filters.compute_trivial_keys`'s own structure), reusing only the
    `tie_decimals` name and round-before-compare technique from the upstream PR. Full
    reasoning recorded in the module's own "Why not bencheval" docstring section.
  - **Wired into `scripts/aggregate_results.py`** as the natural post-aggregation step
    (`--trivial-filter`, plus `--trivial-filter-perfect-clf/-perfect-reg/-min-tie-models/
    -tie-decimals/-exclude-model`): when enabled, writes `trivial_keys.csv` (key, reason)
    unconditionally, and, when any keys are flagged, additionally writes
    `model_results_nontrivial.csv`/`hpo_results_nontrivial.csv` (the same tables with
    flagged keys dropped via `filter_trivial_keys`) alongside the always-written, unfiltered
    `model_results.csv`/`hpo_results.csv` -- purely additive, default `aggregate_results.py`
    runs are byte-identical to before. Deliberately does not carry forward
    `raman_bench_paper.filters`'s process-wide `activate()`/`filter_keys()` global-state
    pattern (built for many paper plotting scripts implicitly sharing one active filter);
    explicit function calls fit a library used by arbitrary external callers better.
  - **Verified against real and realistic v1 data, not just synthetic unit fixtures.** Ran
    the real `scripts/aggregate_results.py` (unmodified default path) against the one real
    cached result in this repo (`results/v1/smoke_resource_fixes/data/TabSTAR_c1_BAG_L1/
    kaiser_ecoli_fermentation__0/0_0/results.pkl`) with `--trivial-filter` on: correctly
    wrote an empty `trivial_keys.csv` (only one model exists in that fixture, so neither
    criterion can fire). Then built a second, realistic multi-model fixture by deep-copying
    that same real `results.pkl` (preserving its real `simulation_artifacts`/`method_metadata`
    shape, rekeying the `pred_proba_dict_{val,test}` entries to each synthetic framework
    name so `ConfigResult._align_result_input_format` accepts them) into three synthetic
    (model, task) combinations -- a "tie" task (2 models at identical `metric_error` on
    every fold, 1 clearly worse), a "perfect" task (1 model at `metric_error=0.0` on every
    fold), and a "normal" task (no perfect score, no tie) -- and ran the real, unmodified
    `EndToEnd.from_raw` pipeline over all of it. Confirmed the CLI correctly flagged exactly
    `tie_task__0` (`tie:2`) and `perfect_task__0` (`perfect:ModelA`), left `normal_task__0`
    unflagged, and that `hpo_results_nontrivial.csv`/`model_results_nontrivial.csv`
    contained only `normal_task__0`'s 6 rows -- end to end through real TabArena machinery,
    not a mocked shortcut. All 29 `tests/test_filters.py` cases pass (both criteria, the
    off-by-default gate, `tie_decimals` floating-point-noise absorption, `method_subtype`
    filtering, `exclude_models`, empty/`None`/single-model/missing-column/mixed-problem-type
    edge cases), and the full existing `tests/` suite was re-run afterward to confirm no
    regressions.

### Changed

- `cluster/submit_job.py`'s jobspec format extended: each line now carries
  `DATASET TARGET_IDX REPEAT FOLD CONFIG_INDEX N_REPEATS` instead of just
  `REPEAT FOLD CONFIG_INDEX` (with dataset/target/n_repeats fixed `--export` env vars for
  the whole array). This lets one SLURM array span many (dataset, target) pairs for the
  same model, not just many (repeat, fold, config) tuples for one fixed target -- the
  mechanism the opportunistic scheduler needs to "combine as much as possible into
  arrays." `MODEL` (and thus GPU/memory tier) stays array-wide, since a single `sbatch`
  call's resource flags apply to every task in it -- an array never mixes models.
  `run_experiment.sbatch` reads the extra fields from its jobspec line instead of env
  vars. The single-(dataset,target) CLI path (`submit_job.py`'s own `submit()`, used by
  model-agent/cluster-agent for one-off submissions) is unaffected -- it just repeats the
  same three values on every line now. New shared `submit_jobs()` helper factors out the
  actual chunking/sbatch-invocation logic so both that path and the opportunistic
  scheduler's multi-target case go through one code path.
- `tabarena`/`bencheval` dependencies now point at upstream `autogluon/tabarena`
  (`packages/{tabarena,bencheval}`, since upstream's monorepo restructure) instead of a
  personal fork. Verified end-to-end by installing `packages/tabarena[benchmark]` fresh
  (per upstream's own quickstart) and running its full quickstart script.
- The `models` extra now depends on `tabarena[tabicl,ebm,search_spaces,realmlp,tabdpt,tabm]`
  instead of hand-listing a subset of the same per-model packages a second time -- that
  hand-listed subset had drifted out of sync with upstream and was missing `tabicl`/`tabm`
  (both import fine from `autogluon.tabular.models` -- the model class is always
  discoverable -- but actually fitting either failed on a missing package, since
  AutoGluon's own model classes import their backing library lazily inside `_fit()`, not
  at class-definition time). Deliberately not tabarena's full `[benchmark]` union: that
  also pulls `[tabpfn]` (`tabpfn>=8.0.8`), which conflicts with `tabpfnwide` (every
  release exact-pins an older `tabpfn`) -- confirmed via a real pip resolve
  (`ResolutionImpossible`). RamanBench keeps its own, more permissive `tabpfn`/
  `tabpfn-extensions` floor instead, which every `tabpfnwide` release's exact pin already
  satisfies. Installing `bencheval` needs `uv pip install` (resolves its
  `[tool.uv.workspace]`-only reference to its sibling package automatically) or a manual
  `pip install bencheval @ git+...#subdirectory=packages/bencheval` before `pip install
  raman-bench[...]` -- plain pip cannot resolve tabarena's own bare `bencheval`
  dependency in one shot (confirmed via a real pip resolve: "No matching distribution
  found for bencheval").
- Model onboarding: new models should follow the `models/custom/<key>/{model.py,hpo.py,
  info.py}` per-directory convention (auto-discovered via
  `raman_bench.models.discover.discover_custom_models`), matching upstream TabArena's own
  post-restructure onboarding pattern (`ModelInfo`/`discover_models`). `ridge` migrated as
  the reference implementation; other already-integrated models are unaffected and keep
  working via the previous flat-file convention.
- All remaining models migrated to the per-directory convention: PLS, DeepCNN, RamanNet,
  SANet, RamanFormer, RamanTransformer, ReZeroNet, FC-ResNeXt, CoAtNet, ROCKET, Arsenal,
  TabPFN-Wide, GBM, TA-TABPFN-3. `wrapped_models.py` now only hand-lists the built-in-
  AutoGluon-backed `Prep_*` classes that have no separate "pure model" class of their own;
  every Raman-specific architecture's `Prep_*` class, search space, and metadata now lives
  next to its model code. `models/generate/` is kept as an empty package purely as a
  fallback import path for any future model added the old way before being migrated --
  `scripts/run_experiment.py::_import_generator` tries the new per-model `hpo.py` location
  first, falling back to the old `generate/<key>.py` location. Verified via the full test
  suite (152 passed, same 3 pre-existing unrelated TabPFN-license failures) and by
  confirming `PREPROCESSED_MODELS` resolves all 33 keys, including migrated ones, to the
  exact same classes as before.

- Metrics now use TabArena/AutoGluon's own problem-type-appropriate defaults (ROC AUC for
  binary classification, log loss for multiclass, RMSE for regression) instead of
  RamanBench's earlier bespoke metric computation.
- The job unit changed from (model, dataset, target, seed, config) to (model, dataset,
  target, repeat, fold, config), matching TabArena's own `tabflow_slurm`
  `--repeat`/`--fold` interface rather than a single `--seed`.
- `microgel_size_*` reduced from 14 curated entries to 1 (`microgel_size_raw_global`) --
  the other 13 were preprocessing-variant or spectral-range-subset duplicates (different
  baseline-correction/normalization already applied upstream, or a fingerprint-region
  subset) of the same 235 samples, redundant given RamanBench's own preprocessing
  pipeline.

### Fixed

- `cluster/opportunistic_scheduler.py`'s `compute_backlog()` only excluded targets with an
  on-disk `results.pkl`, never targets that already had a queued/running SLURM task for
  them. Since `pick_chunk()` deterministically takes the first `chunk_size` items, any tick
  firing before the previous array finished (an out-of-cycle tick, or an array that takes
  close to an hour against the hourly cron cadence) recomputed and resubmitted the
  identical slice. Confirmed in practice: three duplicate 300-task PLS arrays (26454,
  26527, 26671) got submitted within about 40 minutes for the same targets. Now also
  excludes targets with a live SLURM task, found by matching `squeue`/`sacct` job names
  (`submit_jobs` always names a job `RB_{model}_{part_slug}`) back to that submission's
  jobspec file and reading the tasks it covers -- no new state file, same
  recomputed-fresh-every-tick design as the disk-cache check. Verified against the live
  HTW cluster: backlog dropped from the stale 4158 to 3858, with zero overlap between the
  newly picked chunk and the 300 targets already covered by the three duplicate arrays.
- `cluster/opportunistic_scheduler.py`'s `load_scope()` left `results_dir`/`cache_dir` as
  given -- relative paths from `scope_default.json`, resolved against whatever cwd the
  calling process happened to have. Fine for a manual invocation from inside the workspace,
  but the routine cron tick invokes this script from cron's default cwd (`$HOME`), never
  the workspace, so every on-disk `results.pkl` check in `compute_backlog()` silently
  looked in the wrong place and found nothing -- the backlog on the real cron path never
  shrank even as completions piled up. This is why the in-flight dedup fix above didn't
  fully stop duplicate submissions there: that fix only excludes still-queued targets, and
  once an array finishes and drops out of `squeue`, the cache check was the only remaining
  signal, and it was cwd-blind. Both paths now resolve relative to `submit_job.REPO_ROOT`,
  the same anchor `submit_job.py` already uses for `JOBSPEC_DIR` and the `--chdir` it
  passes every SLURM submission -- no new workspace-root concept. Verified: a dry-run from
  `$HOME` (matching cron's actual invocation cwd) against the unpatched script reported the
  stale backlog of 4158; the patched script reports 3855 from both `$HOME` and from inside
  the workspace.
- `cluster/submit_job.py` now passes an explicit `--chdir <workspace>` to every `sbatch`
  call. `run_experiment.sbatch`'s `#SBATCH --output=.logs/...`/`--error=.logs/...` are
  relative paths, resolved by SLURM against the directory `sbatch` was invoked *from* --
  not the benchmark workspace the script itself later `cd`s into. This was invisible as
  long as every submission happened to run from the workspace root (the normal case for a
  developer manually calling `submit_job.py`), but the opportunistic scheduler's first real
  submission (run from a login shell's home directory, matching what a cron job's default
  cwd would also be) failed **every single task** immediately (exit code 2, no log file
  ever written -- SLURM couldn't create the output file) until this fix. Re-verified after
  the fix with a real submission: tasks reached `RUNNING` and then `COMPLETED`, with a real
  `results.pkl` landing in the expected cache path.
- `run_experiment.sbatch` now propagates the real Python exit code -- previously a crashed
  job could still report SLURM/`sacct` success, hiding real failures.
- Regression targets with missing (NaN) values are now dropped before fitting instead of
  crashing the whole job (found on datasets where not every sample was characterized for
  every analyte, e.g. `fuel_benchtop`'s target 1: 157/179 NaN).
- `num_bag_folds` now scales down for small datasets/classes, avoiding AutoGluon's internal
  bagging producing a single-class validation fold and crashing ROC AUC computation.
- `_build_foundation_hyperparameters()` in `src/raman_bench/model.py` pinned CatBoost to
  CPU via `ag_args_fit={"ag.num_gpus": 0}` alone. Upstream AutoGluon 1.6.1 (and current
  `autogluon/autogluon` master) has a real, unguarded `ZeroDivisionError` in
  `autogluon/core/hpo/executors.py`'s `HpoExecutor.register_resources()`: when only
  `num_gpus` is set in `ag_args_fit` and it's `0`, it unconditionally computes
  `num_gpus // user_specified_trial_num_gpus` to infer the per-trial cpu count, dividing
  by zero -- reachable any time the AUTOGLUON meta-preset runs with HPO enabled (the
  default). Sidestepped locally by also setting `"ag.num_cpus": 1`, which makes both
  user-specified fields non-`None` and skips the buggy inference branch entirely.
  Reproduced directly against `HpoExecutor.register_resources()` before/after the change
  (crashes pre-fix, succeeds post-fix); a proper zero-guard fix is being prepared for
  upstream `autogluon/autogluon` (not yet released, so the local sidestep stays).
- LIMIX crashed on every run, right after training completed, with `AttributeError: Can't
  pickle local object '_nan_clean_encoder_cls.<locals>._NaNCleanEncoder'` -- confirmed on
  4/4 real cluster runs (classification and regression alike). Real bug in upstream
  `tabarena`: `tabarena.models.limix.model._nan_clean_encoder_cls()` builds its NaN-
  sanitizing `nn.Module` wrapper as a class local to the factory function (deliberately, to
  keep `torch` off that module's import path for lightweight consumers), which gets the
  default qualname `_nan_clean_encoder_cls.<locals>._NaNCleanEncoder` -- unresolvable by
  `pickle`, which AutoGluon's bagged-ensemble `save_child()` triggers on every fold right
  after it finishes training. Already reported and fixed upstream in
  https://github.com/autogluon/tabarena/pull/468 (open as of 2026-08-10). Sidestepped
  locally by `raman_bench.preprocessing.wrapped_models._patch_limix_pickle_bug`, which
  reproduces that exact upstream fix at runtime (rewrites the produced class's
  `__qualname__` and adds a module-level `__getattr__` (PEP 562) that rebuilds/returns the
  same, `functools.cache`-stable class on demand) and is applied automatically at import
  time. Verified end-to-end with a real `TabularPredictor.fit()`/`.save()`/`.predict()`
  run (bagged, 2 folds, CPU): crashes pre-patch at exactly `save_child()`, succeeds
  post-patch, and a saved predictor loads and predicts correctly in a cold process that
  never called the factory. Remove the patch call once RamanBench's `tabarena` pin
  includes the merged upstream fix.
- Fixed 6 real breaks against current upstream `tabarena` (written against the personal
  fork, never updated as upstream's post-fork restructure moved on 224 commits) --
  found by actually installing from upstream fresh and running the real pipeline, not
  just reading source:
  - `models/registry.py`: `tabarena.benchmark.models.model_registry` no longer exists ->
    `tabarena.benchmark.exec_models.registry`.
  - `splitting.py`: `GroupLabelTypes` moved from `tabarena.benchmark.task.user_task`
    (now a type-checking-only import there, invisible at runtime) to
    `tabarena.benchmark.task.metadata`.
  - `models/generate/gbm.py`, `models/generate/ta_tabpfn_3.py`: TabArena's own per-model
    `generate.py` modules were renamed to `hpo.py` (same function names).
  - `scripts/run_experiment.py`: `Experiment.run()` gained a new required `cache_task_key`
    keyword argument (the task's canonical cache identifier); now passed as `task_name`
    (matching `UserTask.cache_key`'s semantics for our tasks).
  - `scripts/aggregate_results.py`: three separate breaks -- `CacheFunctionPickle` now
    gzip-compresses cache writes by default (a raw `pickle.load` no longer works; use
    `tabarena.utils.pickle_utils.load_pickle`, which handles both transparently);
    `EndToEnd.from_raw` no longer accepts a legacy `task_metadata` DataFrame directly
    (wrap it in `TaskMetadataCollection.from_legacy_df`, which needs a fuller column set
    than this repo's frame used to carry -- `n_folds`/`n_repeats` are now derived
    accurately from what was actually run per task, other per-dataset stats not tracked
    in the results cache are explicit placeholders); `EndToEnd.from_raw` now returns
    `EndToEndResults` directly, with `get_results(use_model_results=True/False)`
    replacing the old `.to_results().model_results`/`.hpo_results`.
  - Verified end-to-end, not just import-checked: a real `run_experiment.py` ->
    `aggregate_results.py` round trip reproduced the same `metric_error` through both
    the raw cache and the aggregated `hpo_results` row.
- None of the 16 "built-in" TabArena-native models listed in `configs/models/all.json`
  (RF, CAT, XGB, XT, KNN, LR, NN_TORCH, FASTAI, DUMMY, REALMLP, MITRA, TABM, TABDPT,
  TABICL, REALTABPFN-V2, REALTABPFN-V2.5) could actually be submitted through the real v1
  pipeline: `scripts/run_experiment.py::_import_generator` requires every model key to
  resolve to a `ConfigGenerator` under `models/custom/<key>/hpo.py` or
  `models/generate/<key>.py`, and neither existed for any of them -- the "Migrate remaining
  14 models" per-directory migration only ever covered RamanBench's own bespoke
  architectures, never the plain AutoGluon-backed `Prep_*` classes already hand-listed in
  `preprocessing/wrapped_models.py`. Fixed by populating `models/generate/<key>.py` for all
  16 (a new `raman_bench_model_registry` key, `XT`, was pulled in by the exact same gap and
  fixed alongside the reported 15) as thin one-liners around a new shared
  `models/generate/_tabarena_adapter.py::rebind_tabarena_generator()`, which reuses TabArena's
  own maintained search space (`tabarena.models.<key>.hpo.gen_<key>`, installed via the
  `tabarena[...,search_spaces,...]` extra) and just retargets it at RamanBench's `Prep_*`
  class instead of hand-writing 16 near-duplicate HPO configs. Two models needed bespoke
  handling instead of a straight rebind: `KNN`'s upstream generator turned out to be bound to
  a different, incompatible TabArena-only model subclass (`KNNNewModel`, with `scaler`/
  `cat_threshold` knobs plain AutoGluon `KNNModel` doesn't accept) -- reuses only the subset
  of the search space genuinely shared with plain `KNNModel`; `DUMMY` and `REALTABPFN-V2`
  have no upstream TabArena generator to reuse at all (`DummyModel` isn't part of TabArena's
  benchmarked roster; TabArena's own package has moved past plain TabPFN v2 in favor of
  v2.5/v2.6) and get an empty (default-config-only) search space instead, matching the same
  convention TabArena itself uses for its own HPO-less models (e.g. Mitra). Verified
  end-to-end, not just import-checked: real `run_experiment.py` runs against a small real
  dataset for KNN, XGB, TABICL, and DUMMY (one simple/classical, one boosted-tree, one
  foundation model, one bespoke-empty-search-space case), plus a `_import_generator()` +
  `generate_all_bag_experiments()` smoke check for all 16 keys confirming each resolves the
  exact same `Prep_*` class the registry does.
- 3 resource-tuning issues found during batch-3 verification of the 14-model
  onboarding effort, before rolling those models into the opportunistic
  scheduler's routine rotation (rollout itself tracked separately):
  - **TabSTAR memory scaling.** Confirmed OOM-killed (`RuntimeError("OOM even
    with batch size 1!")`) on `microgel_synthesis` (11,084 features) -- it
    builds a per-column LM text embedding, so memory scales with feature
    count, not row count, matching upstream's own documented warning about
    >200-column datasets. Characterized against RamanBench's real, current
    dataset distribution (`configs/v1/target_list.json` cross-referenced with
    `data/precomputed/dataset_stats.json`): there's a completely dataset-free
    gap between the widest confirmed-safe target
    (`pharmaceutical_ingredients`, 3,276 features) and the next-widest target
    (`bioprocess_analytes_kaiser`, 5,472 features), above which sits the
    acid-species/microgel cluster (9 targets, 11,084-11,689 features) that
    produced the OOM. Fixed both ways: (a) `Prep_TABSTAR` now carries
    `max_features=4,000` (`wrapped_models._TABSTAR_MAX_FEATURES`, sitting in
    that real gap) via AutoGluon's own constraint mechanism -- the opposite
    direction from `_NO_FOUNDATION_MODEL_FEATURE_CAP`, which LIFTS this same
    cap for Mitra/TabDPT/TabICL/RealTabPFN -- plus a job-level clean skip
    (`wrapped_models.MAX_FEATURES_MODELS`, consumed by
    `scripts/run_experiment.py::run_one()`) since RamanBench's cluster jobs
    fit exactly one model at a time, so AutoGluon's own
    `raise_on_no_models_fitted=True` default would otherwise turn a clean
    `ConstraintViolationError` skip into a job-crashing `RuntimeError`
    (confirmed with a real local run); (b) `cluster/profiles/htw.yaml`'s
    `mem_tiers` gets a `TABSTAR: "128G"` entry (matching
    MITRA/TABDPT/TABFM/TABSWIFT) for headroom on the sub-cap-but-still-large
    remainder (`pharmaceutical_ingredients`, the `diabetes_skin_*` family).
    Verified: a genuinely-too-wide dataset (`microgel_synthesis`) now skips
    cleanly in 3.6s (exit 0, no results.pkl, no model load attempted) instead
    of OOM-crashing.
  - **EBM interaction-detection blowup.** `interpret`'s own default
    (`interactions="3x"`) triggers a FAST interaction-ranking pre-scan over
    essentially every candidate feature-pair combination -- confirmed via a
    real local timed run (`microgel_synthesis`, 11,084 features): the
    pre-scan alone produced millions of per-pair log lines and hadn't
    finished after 9+ minutes, well before boosting itself even starts, and
    -- unlike the boosting rounds, which do respect AutoGluon's own
    `time_limit`/`EbmCallback` -- this pre-scan isn't bounded by the time
    budget at all, so a bigger `time_limit` alone can't fix it (this is what
    was producing the reported `TimeLimitExceeded`, even under a reduced
    smoke-test budget). `Prep_EBM._fit` now forces `interactions=0` above
    4,000 features (same threshold/gap as TabSTAR's cap) -- confirmed a small
    positive count (e.g. `20`) does NOT avoid this: `interpret`'s own
    `rank_interactions` has to rank every candidate pair before it can keep
    only the top-N, and only literal `interactions=0` skips the ranking loop
    entirely (`interpret.glassbox._ebm._ebm.py`: `if interactions == 0:
    break`, before the ranking call). Verified: a real post-fix run on
    `microgel_synthesis` (11,084 features, 2 bag folds) now finishes in 426s
    wall time (400s training) with `metric_error` a sane finite number, well
    inside even the unmodified 3600s default budget -- no flooding, no
    timeout.
  - **Opportunistic scheduler time-budget mechanism.**
    `cluster/opportunistic_scheduler.py`'s `time_limit_overrides` was
    dataset-keyed only, unable to express "this model needs more time on this
    dataset, but other models sharing it are fine" (EBM's fix above; also
    real evidence that ORIONMSP -- classification-only, so it never actually
    reaches the ultra-wide regression cluster -- OOM-crashed on
    `pharmaceutical_ingredients`, its own widest reachable dataset: fitting
    finished in ~175s but the process crashed right after with ~25-27GB
    peak RSS on a 36GB machine, i.e. more a memory problem than a time one
    there). `effective_time_limit()` now also accepts
    `scope["model_time_limit_overrides"]` (model -> dataset -> seconds),
    composed with the existing flat dataset-keyed overrides (max of whatever
    applies). `configs/v1/scope_default.json` carries real,
    evidence-calibrated values (EBM: 5,400s headroom on the 10-target wide
    cluster -- comfortably above the ~1,700s extrapolated full-8-bag-fold
    worst case; ORIONMSP: 7,200s on `pharmaceutical_ingredients`, mostly
    insurance given the real risk there is memory) even though neither model
    is in the routine rotation yet (`"models": ["PLS"]`, unchanged --
    rollout is separate). `cluster/profiles/htw.yaml` also gets an
    `ORIONMSP: "128G"` mem tier for the same reason.
- **10 pre-existing custom architectures verified against the v1 pipeline.** RamanBench's
  own bespoke models (as opposed to the TabArena-native ones onboarded above) predate
  `scripts/run_experiment.py` and had never actually been exercised through it -- only
  through the older `raman_bench.predictions`/v0.1 path. Real local smoke tests
  (`diabetes_skin_ear_lobe`, classification, 20 samples/2,803-3,160 features after
  feature pruning) confirmed all 10 run cleanly end-to-end with a sane finite
  `metric_error`: ARSENAL, ROCKET (sktime-based, classification-only --
  `wrapped_models.CLASSIFICATION_ONLY_MODELS`), DEEPCNN, FCRESNEXT, RAMANFORMER,
  RAMANNET, RAMANTRANSFORMER, REZERONET, SANET, COATNET (from-scratch PyTorch, GPU-tagged
  but auto-fall back to CPU on this dev machine via `BaseRamanEstimator._setup_device()`
  -- correctly so, unlike TABPFN-WIDE below). No code changes needed; all 10 added to
  `configs/v1/models.json`. On this CPU-only dev machine, several of the transformer-family
  models (RAMANTRANSFORMER, COATNET, SANET) only completed 1-10 epochs within a 60s smoke
  budget before AutoGluon's own `time_limit` cut them off -- expected on CPU, not a bug;
  real cluster runs get the GPU node these are tagged for.
  - Investigated (not integrated) two side-leads on ARSENAL/ROCKET, at the user's request,
    to check for a faster/regression-capable alternative to the current sktime-based
    wrapper: (a) `aeon` (sktime's actively-maintained TSC/TSER fork) ships
    `RocketRegressor` but no Arsenal or HIVE-COTE regressor -- sktime already has the
    identical `RocketRegressor` (`sktime.regression.kernel_based`), so `aeon` adds a new
    dependency without adding regression coverage sktime doesn't already offer; (b)
    `sktime.classification.hybrid.HIVECOTEV2` (the full 4-component meta-ensemble Arsenal
    is one part of) is real and already installed, but confirmed genuinely heavy even at
    trivial scale -- a real local timed run (20 synthetic samples, 500 features, a 1-minute
    `time_limit_in_minutes` budget) still took 42s, and its ShapeletTransformClassifier
    component alone defaults to a 2-hour contract with no budget set; classification-only,
    no regression variant in sktime or aeon either. Neither integrated; recommendation
    (aeon: skip, no net new capability; HIVE-COTE: skip for routine rotation, cluster-budget
    heavy for a 4x-the-cost-of-Arsenal-alone ensemble) reported back rather than applied
    unilaterally, since it's an architecture-replacement decision, not a wiring fix.
- **TABPFN-WIDE re-added after fixing a real CPU-fallback bug** (previously removed in
  `e726176` for being "extremely slow", no root cause recorded at the time). Root-caused
  via `cProfile` on a real local fit/predict (`diabetes_skin_ear_lobe`): wall time is
  genuinely dominated by `torch._C._nn.scaled_dot_product_attention` inside the
  transformer's attention-between-features layers (3.7s of a 6.3s predict call) --
  legitimate model compute, not a Python-level bug (no accidental O(n^2) loop, no
  redundant recomputation/reloading found). The actual bug: unlike every other GPU-tagged
  model in this codebase (from-scratch DL models auto-detect CUDA via
  `BaseRamanEstimator._setup_device()`; TabArena-native foundation models thread
  AutoGluon's allocated `num_gpus` through `AbstractTorchModel`), both
  `custom/tabpfn_wide/model.py` classes hardcoded `device="cpu"` as their constructor
  default -- and `SklearnAutoGluonBridge._fit()` (this model's bridge) never reads or
  forwards AutoGluon's `num_gpus` resource kwarg at all. So despite `info.py` tagging
  `compute="gpu"` (reserving a GPU node on the cluster), the model silently ran on CPU
  every time. Fixed: `device` now defaults to `None` ("auto"), resolved at fit time via
  the same CUDA-availability check `_setup_device()` uses elsewhere
  (`raman_bench.models.custom.tabpfn_wide.model._resolve_device`); explicit
  `"cpu"`/`"cuda"` still accepted. Verified no regression on this CPU-only dev machine
  (before: 1.01s fit/4.49s predict; after, same data: 1.08s fit/4.63s predict -- within
  noise, both resolve to `device="cpu"` since there's no CUDA here) and a real
  `run_experiment.py` end-to-end run (`metric_error=0.5`, sane). A synthetic scaling probe
  (still CPU-only, no accidental blowup found: 20 total rows -> 5.7s, 120 rows -> 28.9s,
  350 rows -> 88.2s, roughly linear in this range) combined with real dataset sizes in
  `data/precomputed/datasets.csv` (three classification datasets -- `mlrod` 130,061 rows,
  `bacteria_identification` 78,500 rows, `wheat_lines` 53,134 rows -- far larger than any
  smoke test) makes CPU-forced fitting on one of those a plausible real-world explanation
  for "extremely slow", on top of simply never getting the GPU it's tagged for. The actual
  GPU speedup this fix should produce could not be verified locally (this dev machine has
  no CUDA GPU) -- flagged as a follow-up to confirm on an actual HTW/TU GPU node. Re-added
  to `configs/models/all.json` and `configs/v1/models.json`; new
  `tests/models/test_tabpfn_wide.py` covers `_resolve_device()` plus fit/predict on
  synthetic data.

### Still planned

- Sphinx documentation site
- Contribution templates for new datasets and models
