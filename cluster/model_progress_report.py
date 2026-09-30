#!/usr/bin/env python
"""Generate a per-model BHT k8s progress report as CSV, from real result files.

Counts actual completed tasks (one `results.pkl` per (dataset, target, repeat, fold)
on the shared PVC) rather than parsing live pod logs -- results persist regardless
of whether the submitting Job/pod still exists (pods get cleaned up once finished;
results don't), and a completed results.pkl is real, resolved ground truth, unlike
a log line, which conflates "attempted" with "succeeded" and gets tail-truncated on
long-running pods.

Requires at least one Running pod in the namespace to `kubectl exec` into (any one
will do -- they all share the same PVC mount and the same installed raman_bench
registry). Total task counts per partition (full vs large) are computed from
configs/v1/target_list.json + configs/v1/scope_default.json's own
n_repeats/n_splits/large_datasets logic -- the same arithmetic
cluster/submit_full_benchmark.py uses when submitting -- rather than hardcoded, so
this stays correct if the target list changes.

Usage:
    python cluster/model_progress_report.py [--out PATH] [--namespace NS]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAMESPACE = "mkoddenbrock-ext"
DEFAULT_OUT = Path("/Users/koddenbrock/Repository/raman_bench_paper/docs/model_progress.csv")
RESULTS_ROOT = "/data/RamanBench_v1/results/v1/data"

_REGISTRY_DUMP_SCRIPT = f"""
import json
from raman_bench.models.registry import raman_bench_model_registry
out = {{}}
for key, cls in raman_bench_model_registry.key_to_cls_map().items():
    name = getattr(cls, "ag_name", None)
    if name:
        out[name] = key
print(json.dumps(out))
"""

_ONLY_MODELS_DUMP_SCRIPT = """
import json
from raman_bench.preprocessing.wrapped_models import (
    CLASSIFICATION_ONLY_MODELS,
    REGRESSION_ONLY_MODELS,
)
print(json.dumps({
    "classification_only": sorted(CLASSIFICATION_ONLY_MODELS),
    "regression_only": sorted(REGRESSION_ONLY_MODELS),
}))
"""

_FIND_RESULTS_SCRIPT = f"""
import subprocess
out = subprocess.run(
    ["find", "{RESULTS_ROOT}", "-mindepth", "4", "-maxdepth", "4", "-name", "results.pkl"],
    capture_output=True, text=True,
)
print(out.stdout)
"""


def _kubectl(*args: str) -> str:
    result = subprocess.run(["kubectl", *args], capture_output=True, text=True, check=True)
    return result.stdout


def _pick_any_running_pod(namespace: str) -> str:
    pods = json.loads(_kubectl("get", "pods", "-n", namespace, "-o", "json"))
    for pod in pods["items"]:
        if pod["status"].get("phase") == "Running":
            return pod["metadata"]["name"]
    raise RuntimeError(f"No Running pod found in namespace {namespace!r} to exec into.")


def _exec_python(namespace: str, pod: str, script: str) -> str:
    return _kubectl("exec", pod, "-n", namespace, "--", "python3", "-c", script)


def _compute_partition_totals(
    scope_path: Path,
    targets_path: Path,
    classification_datasets_path: Path,
    regression_datasets_path: Path,
) -> tuple[dict, dict, int, int, set[str], dict[str, int], set[str], dict[str, int], dict[str, int]]:
    """Per-dataset expected task counts (n_repeats * n_splits), split into the full-
    and large-partition dicts {dataset: expected_tasks}, plus their sums, plus the
    set of excluded ``"{dataset}__{target_idx}"`` keys (same on-disk naming as
    ``main()``'s ``dataset_target``) -- so completed results.pkl files for a target
    that's since been quality-excluded (``configs/v1/quality_exclusions.json``,
    see ``EXCLUDED_TARGETS.md``) can be excluded from the numerator too, not just
    the denominator. Without this, a model that already ran before the exclusion
    (or, see the last return value's docstring, before a lowered n_repeats) reports
    >100%.

    Also returns ``dataset_target_n_repeats``: ``{"{dataset}__{target_idx}":
    n_repeats}`` for every non-excluded target, so ``main()`` can drop a completed
    ``results.pkl`` whose own ``repeat`` index is at or beyond the target's
    *current* ``n_repeats`` from the numerator too. Real bug hit in practice: after
    ``n_repeats`` was cut from TabArena's adaptive 10/3/1 schedule to a flat 1
    (2026-09-25 compute-scaling decision), models that had already completed
    ``repeat=1..9`` before that change kept counting those old results toward
    "done" while the denominator shrank to just ``repeat=0``'s tasks -- same
    >100% failure mode as the quality-exclusions case, different trigger.

    Also returns ``active_models``: ``scope["models"]`` as a set, so ``main()``
    can drop rows for models that have since been removed from the routine sweep
    (e.g. TABSTAR, TABPFN-WIDE) but still have historical results.pkl files on
    disk from before their removal -- without this, they'd keep showing up
    stuck at their last-attempted %, indistinguishable from a real stalled model.

    Finally returns ``full_total_by_tasktype``/``large_total_by_tasktype``:
    ``{"classification": N, "regression": M}`` per partition, computed by
    cross-referencing each target's dataset against ``configs/v1/datasets/
    {classification,regression}_all.json``. A ``CLASSIFICATION_ONLY_MODELS``/
    ``REGRESSION_ONLY_MODELS`` model (``wrapped_models.py`` -- e.g. ORIONMSP,
    NORI) has a real, lower ceiling than the full task count: it cleanly skips
    every task of the opposite problem type (see ``run_experiment.py``'s
    "only supports classification/regression tasks" skip), so counting those
    skips as "not done" against the FULL denominator produces a misleadingly
    low percentage that looks like a stalled/broken model when it's actually
    already at its true 100% (confirmed in practice for ORIONMSP: 45/303=14.9%
    against the full denominator vs. its real ceiling of 45/48 once regression
    targets are correctly excluded from ITS denominator).
    """
    scope = json.loads(scope_path.read_text())
    n_splits = scope["n_splits"]
    large_datasets = set(scope.get("large_datasets", []))
    active_models = set(scope["models"])
    targets = json.loads(targets_path.read_text())
    classification_datasets = set(json.loads(classification_datasets_path.read_text()))
    regression_datasets = set(json.loads(regression_datasets_path.read_text()))

    full_by_dataset: dict = defaultdict(int)
    large_by_dataset: dict = defaultdict(int)
    full_total_by_tasktype = {"classification": 0, "regression": 0}
    large_total_by_tasktype = {"classification": 0, "regression": 0}
    excluded_dataset_targets: set[str] = set()
    dataset_target_n_repeats: dict[str, int] = {}
    for t in targets:
        key = f"{t['dataset']}__{t['target_idx']}"
        if t.get("excluded"):
            excluded_dataset_targets.add(key)
            continue
        n_repeats = t.get("n_repeats", 10)
        dataset_target_n_repeats[key] = n_repeats
        tasks = n_repeats * n_splits
        is_large = t["dataset"] in large_datasets
        if is_large:
            large_by_dataset[t["dataset"]] += tasks
        else:
            full_by_dataset[t["dataset"]] += tasks
        totals = large_total_by_tasktype if is_large else full_total_by_tasktype
        if t["dataset"] in classification_datasets:
            totals["classification"] += tasks
        elif t["dataset"] in regression_datasets:
            totals["regression"] += tasks
    return (
        full_by_dataset,
        large_by_dataset,
        sum(full_by_dataset.values()),
        sum(large_by_dataset.values()),
        excluded_dataset_targets,
        dataset_target_n_repeats,
        active_models,
        full_total_by_tasktype,
        large_total_by_tasktype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--scope", type=Path, default=REPO_ROOT / "configs" / "v1" / "scope_default.json",
        help="Scope JSON to read n_splits/large_datasets from.",
    )
    parser.add_argument(
        "--targets", type=Path, default=REPO_ROOT / "configs" / "v1" / "target_list.json",
        help="Target list JSON (as written by scripts/build_target_list.py).",
    )
    parser.add_argument(
        "--classification-datasets", type=Path,
        default=REPO_ROOT / "configs" / "v1" / "datasets" / "classification_all.json",
    )
    parser.add_argument(
        "--regression-datasets", type=Path,
        default=REPO_ROOT / "configs" / "v1" / "datasets" / "regression_all.json",
    )
    args = parser.parse_args()

    (
        full_by_dataset,
        large_by_dataset,
        full_total_all,
        large_total_all,
        excluded_dataset_targets,
        dataset_target_n_repeats,
        active_models,
        full_total_by_tasktype,
        large_total_by_tasktype,
    ) = _compute_partition_totals(
        args.scope, args.targets, args.classification_datasets, args.regression_datasets
    )

    pod = _pick_any_running_pod(args.namespace)

    ag_name_to_key = json.loads(_exec_python(args.namespace, pod, _REGISTRY_DUMP_SCRIPT))
    only_models = json.loads(_exec_python(args.namespace, pod, _ONLY_MODELS_DUMP_SCRIPT))
    classification_only_models = set(only_models["classification_only"])
    regression_only_models = set(only_models["regression_only"])

    find_output = _exec_python(args.namespace, pod, _FIND_RESULTS_SCRIPT)

    # Path shape: <RESULTS_ROOT>/<ag_name>_c1_BAG_L1/<dataset>__<target_idx>/<repeat>_<fold>/results.pkl
    done_full: dict = defaultdict(lambda: defaultdict(int))
    done_large: dict = defaultdict(lambda: defaultdict(int))
    for line in find_output.splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line[len(RESULTS_ROOT) + 1:]
        parts = rel.split("/")
        if len(parts) != 4:
            continue
        model_dir, dataset_target, repeat_fold, _ = parts
        if not model_dir.endswith("_c1_BAG_L1"):
            continue
        if dataset_target in excluded_dataset_targets:
            # A completed result for a target that's since been quality-excluded
            # (see _compute_partition_totals's docstring) -- don't count it, or
            # tasks_done can exceed tasks_total for a model that ran before the
            # exclusion existed.
            continue
        n_repeats = dataset_target_n_repeats.get(dataset_target)
        if n_repeats is not None:
            repeat_str, _, _fold_str = repeat_fold.partition("_")
            try:
                repeat = int(repeat_str)
            except ValueError:
                repeat = None
            if repeat is not None and repeat >= n_repeats:
                # A completed result whose repeat index is at or beyond this
                # target's CURRENT n_repeats (see _compute_partition_totals's
                # docstring) -- e.g. a repeat=3 result from before n_repeats was
                # cut to 1. Don't count it, same >100% failure mode as the
                # quality-exclusions case above, different trigger.
                continue
        ag_name = model_dir[: -len("_c1_BAG_L1")]
        key = ag_name_to_key.get(ag_name)
        if key is None:
            continue
        dataset = dataset_target.rsplit("__", 1)[0]
        if dataset in large_by_dataset:
            done_large[key][dataset] += 1
        else:
            done_full[key][dataset] += 1

    all_keys = sorted((set(done_full) | set(done_large)) & active_models)
    rows = []
    for key in all_keys:
        full_done = sum(done_full[key].values())
        large_done = sum(done_large[key].values())
        # A CLASSIFICATION_ONLY_MODELS/REGRESSION_ONLY_MODELS model has a real,
        # lower ceiling than the full task count -- it cleanly skips every task
        # of the opposite problem type (see run_experiment.py's "only supports
        # classification/regression tasks" skip), so its denominator must be
        # just its own problem type's totals, not everything (see
        # _compute_partition_totals's docstring for the ORIONMSP false-alarm
        # this fixes: 45/303=14.9% against the full denominator vs. its real
        # ceiling of 45/48 once regression targets are excluded).
        if key in classification_only_models:
            full_total = full_total_by_tasktype["classification"]
            large_total = large_total_by_tasktype["classification"]
        elif key in regression_only_models:
            full_total = full_total_by_tasktype["regression"]
            large_total = large_total_by_tasktype["regression"]
        else:
            full_total = full_total_all
            large_total = large_total_all
        if full_total:
            rows.append(
                {
                    "model": key,
                    "partition": "full",
                    "tasks_done": full_done,
                    "tasks_total": full_total,
                    "percent_complete": f"{100 * full_done / full_total:.1f}",
                }
            )
        if large_total:
            rows.append(
                {
                    "model": key,
                    "partition": "large",
                    "tasks_done": large_done,
                    "tasks_total": large_total,
                    "percent_complete": f"{100 * large_done / large_total:.1f}",
                }
            )

    rows.sort(key=lambda r: float(r["percent_complete"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "partition", "tasks_done", "tasks_total", "percent_complete"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {args.out} (via pod {pod!r})")


if __name__ == "__main__":
    main()
