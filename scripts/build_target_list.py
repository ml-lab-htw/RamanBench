#!/usr/bin/env python
"""Build a target list for a full-benchmark submission, one entry per (dataset, target).

Reads dataset-name lists (JSON arrays of raman_data keys, e.g. the paper repo's
``configs/datasets/{classification,regression}_all.json``), loads each dataset once to
determine its real target names and instance count, and writes one row per target with
the dataset-size-adaptive ``n_repeats`` TabArena's own real protocol uses (see
``raman_bench.splitting.get_n_repeats`` -- ported directly from
``tabarena.nips2025_utils.fetch_metadata._get_n_repeats``, confirmed against their real
51-dataset curated metadata). A target whose name is in ``--exclude-targets`` (e.g.
"time_h") is marked ``excluded`` rather than dropped, so the full list stays a complete,
auditable record of what was and wasn't run.

``--force-n-repeats`` overrides that dataset-size-adaptive default with one uniform
value for every dataset -- a deliberate compute-scaling knob, not TabArena's own
protocol. Existing cached ``results.pkl`` files (keyed by (model, dataset, repeat,
fold), see ``scripts/run_experiment.py``) for repeat-indices at or above the new
value are simply not resubmitted going forward -- they stay on disk untouched and
still count as real completed evaluations if this is ever raised again later; this
flag never deletes or invalidates anything, it only changes what gets requested
next.

A target whose ``{dataset}_{target_idx}`` key appears in ``--quality-exclusions`` (a JSON
registry shaped like ``configs/v1/quality_exclusions.json``, with top-level ``"trivial"``/
``"not_learnable"`` maps of ``{key: reason}``) is likewise marked ``excluded``, with
``exclusion_reason`` set to ``"trivial"``/``"not_learnable"`` (``None`` for targets excluded
by ``--exclude-targets`` or not excluded at all). See ``configs/v1/EXCLUDED_TARGETS.md`` for
what these two quality criteria mean and how/when they get re-derived.

Loads each dataset via ``RamanBenchmark._load_raman_dataset``, mirror-first by default
(falling back to the original raman-data source only on a mirror miss) -- the same path
``scripts/run_experiment.py`` uses. The mirror is much faster and more reliable than
hitting original sources directly: several datasets whose original source is currently
unreachable (an RWTH bot-detection wall, a network path to Zenodo that hangs from
certain hosts) load fine from the mirror, so calling ``raman_data()`` directly would
incorrectly mark them unloadable even though they're already mirrored and every other
part of the pipeline can read them fine. Pass ``--no-use-mirror`` to force direct
raman-data (original source) access instead -- e.g. to verify the mirror itself is
faithfully in sync, or when a specific dataset's mirror entry is suspected stale.

A classification target where fewer than 2 classes survive the class filters of
``scripts/run_experiment.py`` (``--min-samples-per-class``, default 9, and
``--min-groups-per-class``, default 0; pass the values the sweep uses) is marked
``excluded`` with ``exclusion_reason`` ``"too_few_classes"`` and a warning, instead of
being listed as runnable and then skipped by every job without a result.

A dataset that fails to load from *both* the mirror and the original source (network
hiccup, a genuinely broken source) is skipped with a warning rather than aborting the
whole build -- rerun for just that dataset separately once fixed.

Usage:
    python scripts/build_target_list.py \\
        --dataset-list classification_all.json --dataset-list regression_all.json \\
        --output targets.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from raman_bench.benchmark import RamanBenchmark
from raman_bench.splitting import GROUP_COL, TooFewClassesError, filter_rare_classes, get_n_repeats


def load_quality_exclusions(path: str | None) -> dict[str, str]:
    """Flatten a ``configs/v1/quality_exclusions.json``-shaped registry into
    ``{"{dataset}_{target_idx}": reason_category}``, where ``reason_category`` is
    ``"trivial"`` or ``"not_learnable"`` (the registry's two top-level keys; any
    ``_``-prefixed key, e.g. ``"_comment"``/``"_criterion"``, is metadata and skipped).
    Returns ``{}`` if ``path`` is ``None``.
    """
    if path is None:
        return {}
    with open(path) as f:
        registry = json.load(f)
    flat: dict[str, str] = {}
    for category, entries in registry.items():
        if category.startswith("_") or not isinstance(entries, dict):
            continue
        for key in entries:
            if key.startswith("_"):
                continue
            flat[key] = category
    return flat


def build_target_list(
    dataset_lists: list[str],
    exclude_targets: set[str],
    quality_exclusions: dict[str, str] | None = None,
    cache_dir: str | None = None,
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
    use_mirror: bool = True,
    force_n_repeats: int | None = None,
    min_samples_per_class: int = 9,
    min_groups_per_class: int = 0,
) -> tuple[list[dict], list[str]]:
    quality_exclusions = quality_exclusions or {}
    names: list[str] = []
    for path in dataset_lists:
        with open(path) as f:
            names.extend(json.load(f))

    bench = RamanBenchmark(
        dataset_names_classification=[], dataset_names_regression=[],
        cache_dir=cache_dir, use_mirror=use_mirror, mirror_repo=mirror_repo,
    )

    targets = []
    failed = []
    for name in names:
        try:
            ds = bench._load_raman_dataset(name)
        except Exception as e:
            print(f"FAILED to load {name}: {e}")
            failed.append(name)
            continue
        if ds is None:
            print(f"FAILED to load {name}: not on the mirror and the original source failed")
            failed.append(name)
            continue
        num_instances = ds.spectra.shape[0]
        n_repeats = force_n_repeats if force_n_repeats is not None else get_n_repeats(num_instances)

        # ``target_names`` means two different things depending on target
        # dimensionality: for a 1D ``targets`` array (always true for
        # classification, and true for any single-target regression dataset),
        # it's the list of CLASS LABELS of the one target -- not separate
        # targets. RamanDataset.to_dataframe's target_idx is documented as
        # "ignored for single-target datasets", so target_idx>0 there would
        # silently re-run the identical target. Only a genuinely 2D targets
        # array (multi-target regression) has one independently-benchmarkable
        # target per column.
        if ds.targets.ndim == 1:
            target_entries = [(0, "target")]
        else:
            target_names = ds.target_names if isinstance(ds.target_names, list) else [ds.target_names]
            target_entries = list(enumerate(target_names))

        too_few_classes = False
        if "class" in str(getattr(ds, "task_type", "")).lower() and ds.targets.ndim == 1:
            labels = pd.DataFrame({"label": np.asarray(ds.targets)})
            if getattr(ds, "group_ids", None) is not None:
                labels[GROUP_COL] = np.asarray(ds.group_ids)
            try:
                filter_rare_classes(
                    labels,
                    label_col="label",
                    min_samples_per_class=min_samples_per_class,
                    min_groups_per_class=min_groups_per_class,
                )
            except TooFewClassesError as e:
                too_few_classes = True
                print(f"WARNING: {name}: excluded, {e}")

        for idx, tname in target_entries:
            quality_reason = quality_exclusions.get(f"{name}_{idx}")
            if quality_reason is None and too_few_classes:
                quality_reason = "too_few_classes"
            targets.append({
                "dataset": name,
                "target_idx": idx,
                "target_name": tname,
                "num_instances": num_instances,
                "n_repeats": n_repeats,
                "excluded": tname in exclude_targets or quality_reason is not None,
                "exclusion_reason": quality_reason,
            })
    return targets, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-list", action="append", required=True, dest="dataset_lists",
        help="Path to a JSON array of raman_data dataset names; pass multiple times to combine",
    )
    parser.add_argument(
        "--exclude-targets", nargs="+", default=["time_h"],
        help="Target names to mark excluded rather than run (e.g. a raw elapsed-time column)",
    )
    parser.add_argument(
        "--quality-exclusions", default="configs/v1/quality_exclusions.json",
        help="Path to a trivial/not_learnable exclusion registry (see "
             "configs/v1/quality_exclusions.json and EXCLUDED_TARGETS.md); pass an "
             "empty string to disable.",
    )
    parser.add_argument(
        "--force-n-repeats", type=int, default=None,
        help="Override TabArena's own dataset-size-adaptive n_repeats "
             "(raman_bench.splitting.get_n_repeats) with one uniform value for "
             "every dataset -- a deliberate compute-scaling knob, not a protocol "
             "change to reproduce. Does not affect already-cached results.pkl "
             "files, only what gets resubmitted going forward.",
    )
    parser.add_argument(
        "--min-samples-per-class", type=int, default=9,
        help="As in scripts/run_experiment.py: targets where fewer than 2 classes keep "
             "this many spectra are marked excluded (too_few_classes).",
    )
    parser.add_argument(
        "--min-groups-per-class", type=int, default=0,
        help="As in scripts/run_experiment.py (0 = off, benchmark v1; v2 uses 3).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument(
        "--use-mirror", action=argparse.BooleanOptionalAction, default=True,
        help="Load via the HF mirror first, falling back to the original raman-data "
             "source on a miss (default: on -- much faster and more reliable). Pass "
             "--no-use-mirror to force direct raman-data access for every dataset.",
    )
    args = parser.parse_args()

    quality_exclusions = load_quality_exclusions(args.quality_exclusions or None)

    targets, failed = build_target_list(
        args.dataset_lists, set(args.exclude_targets), quality_exclusions,
        cache_dir=args.cache_dir, mirror_repo=args.mirror_repo, use_mirror=args.use_mirror,
        force_n_repeats=args.force_n_repeats,
        min_samples_per_class=args.min_samples_per_class,
        min_groups_per_class=args.min_groups_per_class,
    )

    with open(args.output, "w") as f:
        json.dump(targets, f, indent=2)

    n_total = len(targets)
    n_excluded = sum(t["excluded"] for t in targets)
    n_repeats_counts = {}
    reason_counts: dict[str, int] = {}
    for t in targets:
        if not t["excluded"]:
            n_repeats_counts[t["n_repeats"]] = n_repeats_counts.get(t["n_repeats"], 0) + 1
        else:
            reason = t["exclusion_reason"] or "name_match"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    print(f"Wrote {n_total} target(s) to {args.output}")
    print(f"  excluded: {n_excluded} ({reason_counts}), to run: {n_total - n_excluded}")
    print(f"  n_repeats distribution among targets to run: {n_repeats_counts}")
    if failed:
        print(f"  FAILED to load ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
