# RamanBench v1 configs

Everything needed to define *what* the v1 (tabarena-based) benchmark runs lives here,
in the public repo, so external contributors can reproduce or extend the curated
sweep without needing the private `raman_bench_paper` repo at all. Only genuinely
institution-specific values -- HTW/TU account, partition, mail, workspace path --
stay private, in `raman_bench_paper/cluster/profiles/{htw,tu}.yaml`.

- `datasets/classification_all.json`, `datasets/regression_all.json` -- the curated
  68-dataset v1 scope (25 classification + 43 regression), by `raman_data` key.
  Mirrors (and is the canonical source for) the same lists historically kept in
  `raman_bench_paper/configs/datasets/`; that copy stays in place since the paper
  repo's own already-published results were produced by the older `run_benchmark.py`/
  `raman_bench.predictions` execution path (removed from this package in v2.0.0 --
  see CHANGELOG.md), kept deliberately separate so neither can silently affect the
  other (see the paper repo's `rebuttal-scope-rule`).
- `target_list.json` -- one row per (dataset, target), built by
  `scripts/build_target_list.py` from the two dataset lists above (mirror-first
  loading, dataset-size-adaptive `n_repeats`). Model-agnostic: every model in scope
  runs against the same target list. Regenerate after adding/removing a dataset:
  ```
  python scripts/build_target_list.py \
      --dataset-list configs/v1/datasets/classification_all.json \
      --dataset-list configs/v1/datasets/regression_all.json \
      --output configs/v1/target_list.json
  ```
  A handful of targets are marked `excluded` for quality reasons (not just the
  raw `time_h`-style name exclusions) -- see `quality_exclusions.json` and
  `EXCLUDED_TARGETS.md` below.

  **Current state (2026-09-24): every non-excluded target is pinned to
  `n_repeats=1`** via `--force-n-repeats 1`, overriding TabArena's own
  dataset-size-adaptive 10/3/1 schedule -- a deliberate compute-scaling
  decision (cuts tasks/model from 3,609 to 405, an 88.8% reduction), not a
  change to what the split protocol itself means. This does not touch or
  invalidate any already-completed `results.pkl` for `repeat >= 1` -- those
  stay on disk and remain usable if repeats are ever raised again. To restore
  the original adaptive schedule, regenerate without `--force-n-repeats`.
- `quality_exclusions.json` / `EXCLUDED_TARGETS.md` -- the benchmark-scope "trivial"
  and "not learnable" target exclusions (TabArena's own dataset-curation criteria,
  and the paper's baseline-check ablation), merged into `target_list.json` by
  `build_target_list.py --quality-exclusions`. Read `EXCLUDED_TARGETS.md` before
  touching either file -- it covers both criteria's exact definitions, how the
  current list was derived, the periodic re-check process, and the compute impact.
  Datasets/targets excluded here stay fully available via `raman_data`/`RamanBench`
  itself -- this only controls what the benchmark's own sweep runs against.
- `models.json` -- the curated roster of models considered "real-benchmark ready"
  under the new pipeline (currently just `PLS`; grows as more models are validated
  end-to-end against `scripts/run_experiment.py`, see `.claude/agents/model-agent.md`).
- `scope_default.json` -- the opportunistic scheduler's (`cluster/opportunistic_scheduler.py`)
  default scope: which models, which target list, chunk size, and capacity
  thresholds. No institution-specific values -- a private wrapper
  (`raman_bench_paper/cluster/submit_v1_opportunistic.sh`) supplies the actual
  cluster profile (`--profile cluster/profiles/htw.yaml`) alongside this scope.
