#!/usr/bin/env python
"""Reclassify "unknown" gpu.json sentinels as confirmed "cpu" for the handful
of classic sklearn/lightgbm-family models (PLS, CAT, DUMMY, KNN, LR, RF, XGB,
XT) whose results predate wandb tracking entirely -- bulk-synced from the HTW
SLURM cluster's CPU partition (see cluster/profiles/k8s.yaml's own sync notes).
These never needed a GPU, so "no matching wandb run found" really does mean
"ran on CPU", not "hardware unknown" -- unlike the general case
write_gpu_hardware_files.py's sentinel covers, where the hardware genuinely
isn't known.

Only touches gpu.json files that still have the "no matching wandb run found"
sentinel (gpu is None, no gpu_count key) -- never overwrites a real GPU
attribution or an already-confirmed cpu entry.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CPU_LEGACY_MODELS = {"PLS", "CAT", "DUMMY", "KNN", "LR", "RF", "XGB", "XT"}


def _build_ag_name_to_key() -> dict[str, str]:
    from raman_bench.models.registry import raman_bench_model_registry

    out = {}
    for key, cls in raman_bench_model_registry.key_to_cls_map().items():
        name = getattr(cls, "ag_name", None)
        if name:
            out[name] = key
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/RamanBench_v1/results/v1/data")
    args = parser.parse_args()

    ag_name_to_key = _build_ag_name_to_key()
    results_root = Path(args.results_dir)

    n_updated = 0
    n_skipped_other = 0
    n_scanned = 0
    models_touched: dict[str, int] = {}

    for gpu_path in results_root.glob("*/*/*/gpu.json"):
        n_scanned += 1
        model_dir = gpu_path.parts[-4]
        if not model_dir.endswith("_c1_BAG_L1"):
            continue
        ag_name = model_dir[: -len("_c1_BAG_L1")]
        key = ag_name_to_key.get(ag_name)
        if key not in CPU_LEGACY_MODELS:
            continue
        try:
            info = json.loads(gpu_path.read_text())
        except Exception:
            continue
        if info.get("gpu") is not None or info.get("gpu_count") == 0:
            n_skipped_other += 1
            continue  # already has real info or already confirmed cpu
        payload = {
            "gpu": None,
            "gpu_count": 0,
            "note": "manually attributed -- known CPU-only historical run bulk-synced "
                    "from the htw SLURM cluster (pre-dates wandb tracking); confirmed "
                    "as cpu, not unknown",
        }
        gpu_path.write_text(json.dumps(payload, indent=2))
        n_updated += 1
        models_touched[key] = models_touched.get(key, 0) + 1
        if n_updated % 5000 == 0:
            logger.info("  ...%d updated so far", n_updated)

    logger.info(
        "Scanned %d gpu.json. Updated %d to confirmed cpu, skipped %d (already had info).",
        n_scanned, n_updated, n_skipped_other,
    )
    logger.info("Per-model counts: %s", models_touched)


if __name__ == "__main__":
    main()
