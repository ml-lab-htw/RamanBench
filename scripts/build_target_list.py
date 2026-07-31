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

from raman_bench.benchmark import RamanBenchmark
from raman_bench.splitting import get_n_repeats


def build_target_list(
    dataset_lists: list[str],
    exclude_targets: set[str],
    cache_dir: str | None = None,
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
    use_mirror: bool = True,
) -> tuple[list[dict], list[str]]:
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
        n_repeats = get_n_repeats(num_instances)

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

        for idx, tname in target_entries:
            targets.append({
                "dataset": name,
                "target_idx": idx,
                "target_name": tname,
                "num_instances": num_instances,
                "n_repeats": n_repeats,
                "excluded": tname in exclude_targets,
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

    targets, failed = build_target_list(
        args.dataset_lists, set(args.exclude_targets),
        cache_dir=args.cache_dir, mirror_repo=args.mirror_repo, use_mirror=args.use_mirror,
    )

    with open(args.output, "w") as f:
        json.dump(targets, f, indent=2)

    n_total = len(targets)
    n_excluded = sum(t["excluded"] for t in targets)
    n_repeats_counts = {}
    for t in targets:
        if not t["excluded"]:
            n_repeats_counts[t["n_repeats"]] = n_repeats_counts.get(t["n_repeats"], 0) + 1
    print(f"Wrote {n_total} target(s) to {args.output}")
    print(f"  excluded: {n_excluded}, to run: {n_total - n_excluded}")
    print(f"  n_repeats distribution among targets to run: {n_repeats_counts}")
    if failed:
        print(f"  FAILED to load ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
