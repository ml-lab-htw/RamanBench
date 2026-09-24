# Benchmark-scope quality exclusions

RamanBench's routine model-comparison sweep skips a small set of `(dataset,
target)` keys that carry little or no signal for finding a state-of-the-art
model. This document is the policy: what gets excluded, why, how the list was
built, and how it gets revisited over time.

**This is a benchmark-scope decision only.** Every excluded dataset and
target stays fully available through `raman_data(...)`/`RamanBench(...)` and
the published package — nothing here removes anything from RamanBench as a
*data source*. It only controls which `(dataset, target)` keys
`cluster/submit_full_benchmark.py`'s routine sweep runs models against, via
`configs/v1/target_list.json`'s `excluded`/`exclusion_reason` fields.

## The two criteria

### 1. Trivial

A key is **trivial** if either:

- one model scores a perfect result on every seed, or
- two or more models tie for the best result on every seed.

This is TabArena's own dataset-curation rule (NeurIPS 2025, arXiv:2506.16791,
Appendix B.1) — RamanBench applies the identical definition, just post-hoc
per key instead of at dataset-intake time. Implementation:
`raman_bench.filters.compute_trivial_keys`.

Trivial keys are **permanently** excluded (barring a manual re-review): once
a key is provably perfect-or-tied for every model that's been tried, no
future model can make it *less* solved. There is no periodic re-check for
this category.

### 2. Not learnable

A key is **not learnable** if no in-scope model meaningfully beats the
`DUMMY` baseline (predicts the training mean for regression, the majority
class for classification) — the model comparison at this key carries no real
information about *which* model is better, because none of them are actually
modeling the signal.

This ports `raman_bench_paper/scripts/ablation_baseline_check.py`'s ablation
check. That script used two different-shaped formulas (classification:
`best F1 − Dummy F1 > 0.05`; regression: `best R² > 0`). RamanBench's v1
metrics only carry a single `metric_error` column (always lower-is-better,
zero-is-perfect — see `raman_bench/filters.py`'s module docstring), not raw
F1/R², so `raman_bench.filters.compute_unlearnable_keys` generalizes both
into one rule: **does the best in-scope model's mean error beat Dummy's mean
error by more than a margin (default `0.05`)?** This is not a cosmetic
rewrite — `DUMMY` predicts the training mean for regression, so "beats the
mean predictor" (R²>0) and "beats Dummy" are the same test; the two paper
formulas really were one idea in two different units all along.

Unlike "trivial", **not-learnable is not permanent** — a new or improved
model can beat Dummy on a key that previously failed. See the re-check
process below.

## How the current list was built

Both criteria need results from *every* in-scope model to mean anything, and
v1's own cluster sweep did not have full model coverage at the time this list
was built (2026-09-24). So the current
[`quality_exclusions.json`](quality_exclusions.json) registry was computed
against the most complete single snapshot available:
`raman_bench_paper/results/v0_default_backup_splitfix` — 77 datasets, 171
`(dataset, target)` keys, 31 models, every model's results present. All 171
v0.1-era keys map cleanly onto v1's current 162-key scope (v1 dropped 13
`microgel_size_*` fine-grained variants and added `marine_pathogens`,
`marine_pathogens_binary`, and 4 others — see `configs/v1/README.md`'s
dataset diff), so this snapshot is a valid, if temporary, stand-in for v1's
own eventual complete coverage.

Result: **25 keys excluded** (8 trivial + 17 not-learnable, no overlap
between the two categories), on top of 2 pre-existing exclusions (raw
`time_h` elapsed-time columns, excluded for an unrelated reason — 2 of the
17 not-learnable keys happen to be `time_h` columns too, which makes sense:
elapsed time isn't predictable from a spectrum).

| Category | Count |
|---|---|
| Trivial | 8 |
| Not learnable | 17 |
| Pre-existing (`time_h`, unrelated) | 2 |
| **Total excluded** | **27** |
| Remaining, in the routine sweep | 135 |

See `quality_exclusions.json` for the exact keys and per-key reasons.

## The periodic "learnability sweep"

Because "not learnable" can change as better models are added, it needs a
re-check process — but re-running *every* model against *every* excluded key
on *every* routine sweep would defeat the point of excluding them (see
"Compute impact" below). Instead:

1. **Routine sweeps** (the normal `cluster/submit_full_benchmark.py` runs
   against `configs/v1/target_list.json`) always skip excluded keys, full
   stop. This is already wired in — `target_list.json`'s `excluded` field is
   what `submit_full_benchmark.py` reads.
2. **From time to time** (no fixed cadence yet — a judgment call, e.g. after
   onboarding a new class of model, or every few months), run a **second,
   separate sweep**: only the current **top models** (the leaderboard's
   current best performers, not the full ~44-model roster) against only the
   currently-excluded not-learnable keys.
3. Aggregate that sweep's results and run:
   ```
   python scripts/aggregate_results.py --results-dir <that sweep's results dir> \
       --output-dir <...> --learnability-filter
   ```
   This writes `unlearnable_keys.csv` — compare it against
   `quality_exclusions.json`'s current `not_learnable` list. Any key that no
   longer appears has been beaten by a top model and should be removed from
   `quality_exclusions.json` (with a note of which model and when), then
   `target_list.json` regenerated:
   ```
   python scripts/build_target_list.py \
       --dataset-list configs/v1/datasets/classification_all.json \
       --dataset-list configs/v1/datasets/regression_all.json \
       --output configs/v1/target_list.json
   ```
4. Trivial keys are *not* part of this re-check (see "permanently excluded"
   above) — only `not_learnable` entries are ever candidates for removal.

Once v1's own sweep reaches full model coverage, the whole registry should be
**re-derived from v1's own `hpo_results.csv`** (via `compute_trivial_keys`
and `compute_unlearnable_keys` directly, not the v0.1 backup) as the
authoritative source going forward — the v0.1-derived list here is a
bootstrap, not a permanent oracle.

## Compute impact

Removing 25 keys from the routine sweep's 160 previously-in-scope targets
(160 → 135, a **15.6% reduction**) cuts the total task count *per model* from
**4,278 to 3,609** (`n_repeats × n_splits(3)` summed over targets, at
`config_indices=0` / no HPO — the routine sweep's own settings), i.e. **669
fewer (repeat, fold) fits per model**.

Scaled across the ~44 models in the current v1 scope
(`configs/v1/scope_default.json`), that is roughly **29,400 fewer task
submissions** for one full routine sweep of every model. In wall-clock terms
this varies enormously by model (a LIMIX-class in-context model finishes a
task in ~10 minutes; a MITRA-class fine-tuned foundation model can take
~55-60 minutes per task, see the cluster-throughput discussion elsewhere in
this session) — the exact time saved depends on which models are actually
run, but as a rough floor: even at LIMIX's ~10 min/task rate, 669 fewer tasks
is over 100 GPU-hours saved per model; at MITRA's rate it's closer to
600+ GPU-hours per model.

This is why the periodic-sweep design in the previous section matters: doing
a full not-learnable re-check with all ~44 models on every routine run would
give back most of this saving for no benefit (a trivial key can't stop being
trivial; a not-learnable key rarely flips between routine sweeps) — the
"top-models-only, occasional" re-check keeps the saving while still allowing
the exclusion list to improve over time.

## Files

| File | Role |
|---|---|
| `configs/v1/quality_exclusions.json` | The exclusion registry itself: `{dataset}_{target_idx}` → reason, split into `trivial`/`not_learnable` |
| `scripts/build_target_list.py --quality-exclusions` | Merges the registry into `target_list.json` (`excluded`/`exclusion_reason` fields) |
| `configs/v1/target_list.json` | The generated target list `cluster/submit_full_benchmark.py` reads; excluded keys are skipped automatically |
| `src/raman_bench/filters.py` | `compute_trivial_keys`/`compute_unlearnable_keys` — the functions that (re-)derive this list from real `hpo_results` |
| `scripts/aggregate_results.py --trivial-filter` / `--learnability-filter` | CLI entry points for (re-)computing either criterion against a results directory |
