# RamanBench v1 configs

Everything needed to define *what* the v1 (tabarena-based) benchmark runs lives here,
in the public repo, so external contributors can reproduce or extend the curated
sweep without needing the private `raman_bench_paper` repo at all. Only genuinely
institution-specific values -- HTW/TU account, partition, mail, workspace path --
stay private, in `raman_bench_paper/cluster/profiles/{htw,tu}.yaml`.

- `datasets/classification_all.json`, `datasets/regression_all.json` -- the curated
  66-dataset v1 scope (23 classification + 43 regression), by `raman_data` key.
  Mirrors (and is the canonical source for) the same lists historically kept in
  `raman_bench_paper/configs/datasets/`; that copy stays in place too since it still
  feeds the older, currently-published/in-flight `v0_default`/`v1_default` paper
  results pipeline (`scripts/run_benchmark.py` -> `raman_bench.predictions`) -- a
  different execution path from this one, kept deliberately separate so neither can
  silently affect the other (see the paper repo's `rebuttal-scope-rule`).
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
- `models.json` -- the curated roster of models considered "real-benchmark ready"
  under the new pipeline (currently just `PLS`; grows as more models are validated
  end-to-end against `scripts/run_experiment.py`, see `.claude/agents/model-agent.md`).
- `scope_default.json` -- the opportunistic scheduler's (`cluster/opportunistic_scheduler.py`)
  default scope: which models, which target list, chunk size, and capacity
  thresholds. No institution-specific values -- a private wrapper
  (`raman_bench_paper/cluster/submit_v1_opportunistic.sh`) supplies the actual
  cluster profile (`--profile cluster/profiles/htw.yaml`) alongside this scope.
