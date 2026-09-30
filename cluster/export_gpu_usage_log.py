#!/usr/bin/env python
"""Export a per-model CSV log of which GPU (and bag-fold count) computed each
real cluster task, backfilled from wandb -- the only place this was ever
recorded (results.pkl itself carries no GPU/hardware info at all, see
scripts/run_experiment.py's _log_to_wandb: wandb.init's own automatic system
telemetry captures the real GPU name, e.g. "NVIDIA A100-SXM4-40GB", as
run.metadata['gpu'], while task_config already logs num_bag_folds/model/
dataset/repeat/fold/config_index as run.config).

For the later train/inference-time analysis: comparisons should be
restricted to datasets that actually share the same (gpu, num_bag_folds)
combination across the models being compared -- this log is what makes that
restriction checkable, since the same model has in practice run under
different GPU types and bag-fold counts (8 vs 3) at different points in this
project's history (a 2026-09-25 compute-scaling decision cut num_bag_folds
8->3 for the routine sweep; large/wide datasets get routed to H100/H200
instead of the default A100 node pool).

Writes one CSV per model (one row per real task run logged to wandb) to
--out-dir, default raman_bench_paper/docs/gpu_usage/<MODEL>.csv -- same
convention as model_progress_report.py's DEFAULT_OUT.

Usage:
    export WANDB_API_KEY=...  # or already logged in
    python cluster/export_gpu_usage_log.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_OUT_DIR = Path("/Users/koddenbrock/Repository/raman_bench_paper/docs/gpu_usage")
DEFAULT_ENTITY = "bht"
# Every v1-pipeline model gets its own wandb project named exactly this way
# (scripts/run_experiment.py's wandb_project_for_model) -- excludes the
# legacy v0.1-era "raman-bench-<model>" projects (no "-v1-") and every
# unrelated project in the same wandb team.
PROJECT_PREFIX = "raman-bench-v1-"

FIELDNAMES = [
    "dataset", "target_idx", "repeat", "fold", "config_index",
    "num_bag_folds", "gpu", "gpu_count", "cuda_version",
    "run_state", "started_at", "wandb_run_id",
]


def _model_key_from_project(project_name: str) -> str:
    """wandb project names are the model's ag_key lowercased with '.'/'_'
    turned into '-' (see wandb_project_for_model) -- not perfectly invertible
    back to the exact ag_key casing, so this reads the real model key from
    each run's own config instead of guessing from the project name."""
    return project_name[len(PROJECT_PREFIX):]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    args = parser.parse_args()

    import wandb
    api = wandb.Api()

    projects = [p for p in api.projects(args.entity) if p.name.startswith(PROJECT_PREFIX)]
    print(f"Found {len(projects)} v1-pipeline project(s) under {args.entity!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for project in projects:
        rows = []
        model_key = None
        for run in api.runs(f"{args.entity}/{project.name}"):
            cfg = run.config
            meta = run.metadata or {}
            if model_key is None and cfg.get("model"):
                model_key = cfg["model"]
            gpu_list = meta.get("gpu_nvidia") or []
            rows.append({
                "dataset": cfg.get("dataset"),
                "target_idx": cfg.get("target_idx"),
                "repeat": cfg.get("repeat"),
                "fold": cfg.get("fold"),
                "config_index": cfg.get("config_index"),
                "num_bag_folds": cfg.get("num_bag_folds"),
                "gpu": meta.get("gpu") or (gpu_list[0]["name"] if gpu_list else None),
                "gpu_count": meta.get("gpu_count"),
                "cuda_version": meta.get("cudaVersion"),
                "run_state": run.state,
                "started_at": meta.get("startedAt"),
                "wandb_run_id": run.id,
            })
        if not rows:
            continue
        model_key = model_key or _model_key_from_project(project.name).upper()
        out_path = args.out_dir / f"{model_key}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {project.name} -> {out_path} ({len(rows)} row(s))")
        total_rows += len(rows)

    print(f"Done: {total_rows} row(s) across {len(projects)} model(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
