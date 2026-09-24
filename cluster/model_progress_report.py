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
    scope_path: Path, targets_path: Path
) -> tuple[dict, dict, int, int, set[str]]:
    """Per-dataset expected task counts (n_repeats * n_splits), split into the full-
    and large-partition dicts {dataset: expected_tasks}, plus their sums, plus the
    set of excluded ``"{dataset}__{target_idx}"`` keys (same on-disk naming as
    ``main()``'s ``dataset_target``) -- so completed results.pkl files for a target
    that's since been quality-excluded (``configs/v1/quality_exclusions.json``,
    see ``EXCLUDED_TARGETS.md``) can be excluded from the numerator too, not just
    the denominator. Without this, a model that already ran against a target
    before it was excluded reports >100% (real bug, hit in practice: v1's own
    quality-exclusions rollout dropped the denominator by 25 targets while
    already-completed results.pkl files for those targets stayed on disk).
    """
    scope = json.loads(scope_path.read_text())
    n_splits = scope["n_splits"]
    large_datasets = set(scope.get("large_datasets", []))
    targets = json.loads(targets_path.read_text())

    full_by_dataset: dict = defaultdict(int)
    large_by_dataset: dict = defaultdict(int)
    excluded_dataset_targets: set[str] = set()
    for t in targets:
        if t.get("excluded"):
            excluded_dataset_targets.add(f"{t['dataset']}__{t['target_idx']}")
            continue
        tasks = t.get("n_repeats", 10) * n_splits
        if t["dataset"] in large_datasets:
            large_by_dataset[t["dataset"]] += tasks
        else:
            full_by_dataset[t["dataset"]] += tasks
    return (
        full_by_dataset,
        large_by_dataset,
        sum(full_by_dataset.values()),
        sum(large_by_dataset.values()),
        excluded_dataset_targets,
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
    args = parser.parse_args()

    full_by_dataset, large_by_dataset, full_total_all, large_total_all, excluded_dataset_targets = (
        _compute_partition_totals(args.scope, args.targets)
    )

    pod = _pick_any_running_pod(args.namespace)

    ag_name_to_key = json.loads(_exec_python(args.namespace, pod, _REGISTRY_DUMP_SCRIPT))

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
        model_dir, dataset_target, _repeat_fold, _ = parts
        if not model_dir.endswith("_c1_BAG_L1"):
            continue
        if dataset_target in excluded_dataset_targets:
            # A completed result for a target that's since been quality-excluded
            # (see _compute_partition_totals's docstring) -- don't count it, or
            # tasks_done can exceed tasks_total for a model that ran before the
            # exclusion existed.
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

    all_keys = sorted(set(done_full) | set(done_large))
    rows = []
    for key in all_keys:
        full_done = sum(done_full[key].values())
        large_done = sum(done_large[key].values())
        if full_total_all:
            rows.append(
                {
                    "model": key,
                    "partition": "full",
                    "tasks_done": full_done,
                    "tasks_total": full_total_all,
                    "percent_complete": f"{100 * full_done / full_total_all:.1f}",
                }
            )
        if large_total_all:
            rows.append(
                {
                    "model": key,
                    "partition": "large",
                    "tasks_done": large_done,
                    "tasks_total": large_total_all,
                    "percent_complete": f"{100 * large_done / large_total_all:.1f}",
                }
            )

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
