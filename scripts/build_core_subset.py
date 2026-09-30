#!/usr/bin/env python
"""Derive RamanBench's "core" subset -- a small, hardware-friendly slice of the full
v1 target list, meant so people without cluster-scale GPUs/time can still run (and
compare against) a real RamanBench sweep.

Unlike the paper's Study 1 100-target subset (stratified for statistical
comparability across a fixed budget), the core subset's criterion is purely
practical: exclude anything too *heavy* to fit on modest hardware, exclude anything
too *small* to trust, then cap per-dataset target count so no single multi-target
dataset dominates the sweep. Concretely, starting from every non-excluded row in
``target_list.json``:

1. Drop any target whose dataset is in ``scope_default.json``'s ``large_datasets``
   (the same list this repo already uses to route heavy datasets to bigger-memory
   GPUs on the cluster -- these are real, previously-confirmed OOM/TimeLimitExceeded
   datasets, not a guess).
2. Drop any target with ``num_instances < 50`` (``TINY_THRESHOLD``, matching
   ``raman_bench_paper``'s existing tiny-vs-medium ablation cutoff -- below this,
   per-fold sample counts get unreliable, see that repo's
   ``scripts/ablation_tiny_vs_medium.py``).
3. Cap each remaining dataset at ``--max-targets-per-dataset`` (default 2), keeping
   the lowest ``target_idx`` values (deterministic, not random) so a dataset with
   many regression targets doesn't crowd out dataset-level diversity.

At the default cutoffs (50 / 50 / cap=2) this yields exactly 53 targets across 37
datasets, confirmed live against the real target_list.json on 2026-09-30 -- if a
future `target_list.json` regeneration changes that number, re-run this script
rather than hand-editing the output.

Writes two things:

- Back into ``target_list.json`` itself: an ``is_core`` bool on every entry (source
  of truth for "is this target in the core subset", usable by anything that already
  reads target_list.json, e.g. the leaderboard's core-vs-main delta view).
- ``core_target_list.json``: the same schema as ``target_list.json``, containing
  ONLY the selected targets (``excluded`` always ``false`` here) -- pass this
  directly as ``--targets-file`` to ``cluster/submit_full_benchmark.py`` or
  ``cluster/opportunistic_scheduler.py --scope configs/v1/scope_core.json`` to run
  (or resume) a core-only sweep.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TINY_THRESHOLD = 50  # matches raman_bench_paper/scripts/ablation_tiny_vs_medium.py


def build_core_subset(
    targets: list[dict],
    large_datasets: set[str],
    *,
    tiny_threshold: int = TINY_THRESHOLD,
    max_targets_per_dataset: int = 2,
) -> set[tuple[str, int]]:
    """Return the ``{(dataset, target_idx)}`` keys selected for the core subset."""
    active = [t for t in targets if not t.get("excluded")]
    eligible = [
        t for t in active
        if t["dataset"] not in large_datasets and t.get("num_instances", 0) >= tiny_threshold
    ]

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for t in eligible:
        by_dataset[t["dataset"]].append(t)

    selected: set[tuple[str, int]] = set()
    for dataset, ts in by_dataset.items():
        ts_sorted = sorted(ts, key=lambda t: t["target_idx"])
        for t in ts_sorted[:max_targets_per_dataset]:
            selected.add((dataset, t["target_idx"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-list", default="configs/v1/target_list.json")
    parser.add_argument("--scope", default="configs/v1/scope_default.json")
    parser.add_argument("--output", default="configs/v1/core_target_list.json")
    parser.add_argument("--tiny-threshold", type=int, default=TINY_THRESHOLD)
    parser.add_argument("--max-targets-per-dataset", type=int, default=2)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resulting counts without writing anything.",
    )
    args = parser.parse_args()

    target_list_path = Path(args.target_list)
    targets = json.loads(target_list_path.read_text())
    scope = json.loads(Path(args.scope).read_text())
    large_datasets = set(scope.get("large_datasets", []))

    selected = build_core_subset(
        targets, large_datasets,
        tiny_threshold=args.tiny_threshold,
        max_targets_per_dataset=args.max_targets_per_dataset,
    )

    n_datasets = len({ds for ds, _ in selected})
    print(f"Core subset: {len(selected)} target(s) across {n_datasets} dataset(s)")
    print(f"  (from {len(targets)} total, {sum(1 for t in targets if not t.get('excluded'))} active, "
          f"excluding {len(large_datasets)} large dataset(s) and N<{args.tiny_threshold} targets, "
          f"capped at {args.max_targets_per_dataset}/dataset)")

    if args.dry_run:
        return

    for t in targets:
        t["is_core"] = (t["dataset"], t["target_idx"]) in selected
    target_list_path.write_text(json.dumps(targets, indent=2) + "\n")
    print(f"Wrote is_core flags into {target_list_path}")

    core_targets = [t for t in targets if t["is_core"]]
    Path(args.output).write_text(json.dumps(core_targets, indent=2) + "\n")
    print(f"Wrote {len(core_targets)} target(s) to {args.output}")


if __name__ == "__main__":
    main()
