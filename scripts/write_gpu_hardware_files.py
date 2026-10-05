#!/usr/bin/env python
"""Write a small ``gpu.json`` sidecar next to each completed task's ``results.pkl``,
recording which GPU model actually produced it (via wandb's own auto-collected
system metadata) -- so later speed/timing analysis can group results by
comparable hardware instead of silently mixing e.g. A100-SXM4-40GB and
A100-PCIE-40GB timings together.

Idempotent/resumable by design: a leaf result directory that already has a
``gpu.json`` is skipped entirely (no wandb API call for it), so running this
again later only processes results that have completed since the last run.

Must run where the results tree is mounted (a cluster pod with the PVC
mounted) -- this is why it's submitted as a one-off CPU pod rather than run
locally: the actual ``results.pkl`` files only exist on the cluster PVC.

Usage (inside a pod with the repo installed and WANDB_API_KEY set)::

    python scripts/write_gpu_hardware_files.py --results-dir /data/RamanBench_v1/results/v1/data
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Same mapping as cluster/model_progress_report.py's _BASELINE_EXPERIMENT_NAME_TO_KEY --
# the AutoGluon-extreme whole-predictor baseline's result-dir name isn't a registry
# ag_name (no Prep_* class backs it), so it needs its own literal lookup.
_BASELINE_EXPERIMENT_NAME_TO_KEY = {
    "AutoGluon_extreme_5m": "AUTOGLUON-EXTREME-5M",
    "AutoGluon_extreme_1h": "AUTOGLUON-EXTREME-1H",
    "AutoGluon_extreme_4h": "AUTOGLUON-EXTREME-4H",
}


def _build_ag_name_to_key() -> dict[str, str]:
    from raman_bench.models.registry import raman_bench_model_registry

    out = {}
    for key, cls in raman_bench_model_registry.key_to_cls_map().items():
        name = getattr(cls, "ag_name", None)
        if name:
            out[name] = key
    return out


def _wandb_project_for_model(model_key: str, base: str = "raman-bench-v1") -> str:
    # Must match scripts/run_experiment.py's wandb_project_for_model exactly.
    slug = model_key.lower().replace("_", "-").replace(".", "-").replace("(", "").replace(")", "")
    return f"{base}-{slug}"


def _wandb_run_name(model: str, dataset: str, target_idx: str, repeat: str, fold: str, config_index: str = "0") -> str:
    # Must match scripts/run_experiment.py's _wandb_run_name exactly.
    return f"{model}_{dataset}_t{target_idx}_r{repeat}_f{fold}_c{config_index}"


def _find_needed_tasks(results_root: Path, ag_name_to_key: dict[str, str]) -> dict[str, list[tuple[Path, str, str, str, str]]]:
    """Walk the results tree; return {model_key: [(leaf_dir, dataset, target_idx, repeat, fold), ...]}
    for every completed result that doesn't already have a gpu.json sidecar."""
    needed: dict[str, list[tuple[Path, str, str, str, str]]] = {}
    n_total = 0
    n_skipped_has_gpu = 0
    n_skipped_unknown_model = 0

    for pkl in results_root.glob("*/*/*/results.pkl"):
        n_total += 1
        leaf_dir = pkl.parent
        if (leaf_dir / "gpu.json").exists():
            n_skipped_has_gpu += 1
            continue

        rel = pkl.relative_to(results_root)
        model_dir, dataset_target, repeat_fold = rel.parts[0], rel.parts[1], rel.parts[2]

        if model_dir in _BASELINE_EXPERIMENT_NAME_TO_KEY:
            model_key = _BASELINE_EXPERIMENT_NAME_TO_KEY[model_dir]
        elif model_dir.endswith("_c1_BAG_L1"):
            ag_name = model_dir[: -len("_c1_BAG_L1")]
            model_key = ag_name_to_key.get(ag_name)
        else:
            model_key = None

        if model_key is None:
            n_skipped_unknown_model += 1
            continue

        if "__" not in dataset_target or "_" not in repeat_fold:
            n_skipped_unknown_model += 1
            continue
        dataset, target_idx = dataset_target.rsplit("__", 1)
        repeat, fold = repeat_fold.split("_", 1)

        needed.setdefault(model_key, []).append((leaf_dir, dataset, target_idx, repeat, fold))

    logger.info(
        "Scanned %d results.pkl: %d already have gpu.json, %d had an unrecognized "
        "model_dir/path shape, %d need a lookup across %d model(s).",
        n_total, n_skipped_has_gpu, n_skipped_unknown_model,
        sum(len(v) for v in needed.values()), len(needed),
    )
    return needed


def _process_model(api, model_key: str, tasks: list[tuple[Path, str, str, str, str]], max_workers: int) -> tuple[int, int]:
    import wandb  # noqa: F401  (ensures wandb is importable before use below)

    project = _wandb_project_for_model(model_key)
    try:
        runs = list(api.runs(f"bht/{project}", per_page=500))
    except Exception:
        logger.warning("Could not list wandb project %r for model %r -- writing a sentinel "
                        "gpu.json for its %d task(s) so they aren't rescanned every run.",
                        project, model_key, len(tasks))
        for leaf_dir, _dataset, _target_idx, _repeat, _fold in tasks:
            (leaf_dir / "gpu.json").write_text(json.dumps(
                {"gpu": None, "note": f"wandb project {project!r} not found for this model"}, indent=2
            ))
        return 0, len(tasks)

    # A run name can have multiple wandb runs (resubmissions/smoke tests) -- keep the
    # most recently created one per name as the best guess for "what actually produced
    # the currently-cached results.pkl".
    by_name: dict[str, object] = {}
    for r in runs:
        existing = by_name.get(r.name)
        if existing is None or r.created_at > existing.created_at:
            by_name[r.name] = r

    matched = []
    unmatched = []
    for leaf_dir, dataset, target_idx, repeat, fold in tasks:
        name = _wandb_run_name(model_key, dataset, target_idx, repeat, fold)
        run = by_name.get(name)
        if run is None:
            unmatched.append(leaf_dir)
        else:
            matched.append((leaf_dir, run))

    def fetch_and_write(leaf_dir: Path, run) -> None:
        meta = run.metadata or {}
        payload = {
            "gpu": meta.get("gpu"),
            "gpu_count": meta.get("gpu_count"),
            "host": meta.get("host"),
            "wandb_run_id": run.id,
            "wandb_project": project,
            "wandb_created_at": run.created_at,
        }
        (leaf_dir / "gpu.json").write_text(json.dumps(payload, indent=2))

    n_written = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(fetch_and_write, leaf_dir, run) for leaf_dir, run in matched]
        for fut in as_completed(futs):
            fut.result()
            n_written += 1

    for leaf_dir in unmatched:
        (leaf_dir / "gpu.json").write_text(json.dumps(
            {"gpu": None, "note": "no matching wandb run found for this result"}, indent=2
        ))

    return n_written, len(unmatched)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/RamanBench_v1/results/v1/data")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()

    results_root = Path(args.results_dir)
    ag_name_to_key = _build_ag_name_to_key()
    needed = _find_needed_tasks(results_root, ag_name_to_key)

    overall_start = time.time()
    total_written = 0
    total_unmatched = 0
    for i, (model_key, tasks) in enumerate(sorted(needed.items()), start=1):
        t0 = time.time()
        n_written, n_unmatched = _process_model(api, model_key, tasks, args.max_workers)
        total_written += n_written
        total_unmatched += n_unmatched
        logger.info(
            "[%d/%d] %s: %d task(s), %d gpu.json written, %d unmatched (no wandb run found) in %.1fs",
            i, len(needed), model_key, len(tasks), n_written, n_unmatched, time.time() - t0,
        )

    elapsed = time.time() - overall_start
    logger.info(
        "ALL DONE: %d gpu.json written, %d unmatched, in %.1fs (%.1f min)",
        total_written, total_unmatched, elapsed, elapsed / 60,
    )


if __name__ == "__main__":
    main()
