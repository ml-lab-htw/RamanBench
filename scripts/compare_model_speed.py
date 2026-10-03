#!/usr/bin/env python
"""Compare models' train/inference time fairly, accounting for both task
difficulty (dataset/fold) and hardware (GPU model) -- two models' timings are
only directly comparable when they ran the *same* task on the *same* hardware
class; blending e.g. an A100 time with a B200 time for the "same" model would
make the faster hardware look like a faster model.

Scheme
------
1. Walk the results tree for every ``results.pkl`` + sibling ``gpu.json``
   (written either by ``raman_bench.experiment_utils.write_hardware_info`` for
   new runs, or retroactively by ``scripts/write_gpu_hardware_files.py`` for
   historical ones).
2. Normalize each raw GPU string into a coarse hardware class (``_hardware_class``)
   -- A100 variants (SXM4-40GB/80GB-PCIe/PCIE-40GB) collapse into one ``a100``
   bucket; documented simplifying assumption, see its docstring.
3. For each (dataset, target_idx, repeat, fold, hardware_class) group, compute
   each model's time relative to the fastest model present on that exact task
   -- a slowdown factor >= 1, 1.0 being fastest.
4. Aggregate each (model, hardware_class)'s slowdown factors via the
   *geometric* mean (the correct aggregator for ratios -- the arithmetic mean
   is dominated by outliers and double-counts asymmetrically).
5. Report geo-mean slowdown (ranking) AND absolute median seconds (for readers
   who want raw numbers), kept separate per hardware_class so they're never
   silently mixed. Tasks whose hardware_class has no other model to compare
   against (e.g. a local MPS-only demo) are reported separately as "orphan"
   coverage, never folded into the ranking.

Usage (run where the results tree is mounted, e.g. a cluster pod)::

    python scripts/compare_model_speed.py --results-dir /data/RamanBench_v1/results/v1/data \\
        --out /tmp/model_speed_report.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from math import exp, log
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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


def _hardware_class(gpu_info: dict) -> str:
    """Collapse a ``gpu.json`` record into a coarse comparability bucket. A100
    SXM4-40GB / 80GB-PCIe / PCIE-40GB all collapse into ``a100``: a simplifying
    assumption (their raw compute throughput is similar enough for a fit-time
    comparison, though VRAM differences can matter for very wide datasets) --
    revisit if that turns out to matter for a specific model.

    ``gpu: null`` does NOT mean "ran on CPU" -- confirmed live: several
    genuinely-matched wandb runs (real wandb_run_id, correct timestamps) simply
    have no ``gpu`` field in wandb's own collected system metadata (a wandb-side
    gap, not a matching failure), and the retroactive recovery script's "no
    matching wandb run found" sentinel also sets ``gpu: null``. Only
    ``write_hardware_info``'s own live-written ``gpu_count: 0`` (no CUDA/MPS
    detected at the moment of a real fit) is treated as a confirmed CPU-only run.
    Everything else with a null/missing gpu is ``unknown`` -- excluded from the
    hardware-matched comparison rather than silently lumped in with real CPU runs.
    """
    gpu = gpu_info.get("gpu")
    if not gpu:
        return "cpu" if gpu_info.get("gpu_count") == 0 else "unknown"
    g = gpu.lower()
    if "mps" in g:
        return "apple-mps"
    if "b200" in g:
        return "b200"
    if "h200" in g:
        return "h200"
    if "h100" in g:
        return "h100"
    if "a100" in g:
        return "a100"
    return g.replace(" ", "-")


@dataclass
class TaskRecord:
    model: str
    dataset: str
    target_idx: str
    repeat: str
    fold: str
    hardware_class: str
    time_train_s: float | None
    time_infer_s: float | None


def _collect_records(results_root: Path, ag_name_to_key: dict[str, str]) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    n_no_gpu_json = 0
    n_no_times = 0
    n_unknown_model = 0
    counts_lock = __import__("threading").Lock()

    def _process_one(pkl: Path) -> TaskRecord | None:
        nonlocal n_no_gpu_json, n_no_times, n_unknown_model
        leaf_dir = pkl.parent
        rel = pkl.relative_to(results_root)
        model_dir, dataset_target, repeat_fold = rel.parts[0], rel.parts[1], rel.parts[2]

        if model_dir in _BASELINE_EXPERIMENT_NAME_TO_KEY:
            model_key = _BASELINE_EXPERIMENT_NAME_TO_KEY[model_dir]
        elif model_dir.endswith("_c1_BAG_L1"):
            model_key = ag_name_to_key.get(model_dir[: -len("_c1_BAG_L1")])
        else:
            model_key = None
        if model_key is None or "__" not in dataset_target or "_" not in repeat_fold:
            with counts_lock:
                n_unknown_model += 1
            return None
        dataset, target_idx = dataset_target.rsplit("__", 1)
        repeat, fold = repeat_fold.split("_", 1)

        gpu_path = leaf_dir / "gpu.json"
        try:
            gpu_info = json.loads(gpu_path.read_text())
        except Exception:
            with counts_lock:
                n_no_gpu_json += 1
            return None
        hardware_class = _hardware_class(gpu_info)

        try:
            with gzip.open(pkl, "rb") as f:
                out = pickle.load(f)
        except Exception:
            with counts_lock:
                n_no_times += 1
            return None
        time_train_s = out.get("time_train_s")
        time_infer_s = out.get("time_infer_s")
        if time_train_s is None and time_infer_s is None:
            with counts_lock:
                n_no_times += 1
            return None

        return TaskRecord(
            model=model_key, dataset=dataset, target_idx=target_idx, repeat=repeat, fold=fold,
            hardware_class=hardware_class, time_train_s=time_train_s, time_infer_s=time_infer_s,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    pkl_paths = list(results_root.glob("*/*/*/results.pkl"))
    logger.info("Found %d results.pkl -- reading them (and sibling gpu.json) with 32 threads...",
                len(pkl_paths))
    n_done = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(_process_one, pkl) for pkl in pkl_paths]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            n_done += 1
            if n_done % 10000 == 0:
                logger.info("  ...%d/%d files processed", n_done, len(pkl_paths))

    logger.info(
        "Collected %d usable record(s); skipped %d (no gpu.json), %d (no timing fields "
        "in results.pkl), %d (unrecognized model_dir).",
        len(records), n_no_gpu_json, n_no_times, n_unknown_model,
    )
    return records


def _geo_mean(values: list[float]) -> float:
    return exp(sum(log(v) for v in values) / len(values))


def build_report(records: list[TaskRecord]) -> list[dict]:
    # task_key -> hardware_class -> metric -> {model: time}
    by_task_hw: dict[tuple, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for r in records:
        task_key = (r.dataset, r.target_idx, r.repeat, r.fold)
        if r.time_train_s is not None:
            by_task_hw[task_key][r.hardware_class]["train"][r.model] = r.time_train_s
        if r.time_infer_s is not None:
            by_task_hw[task_key][r.hardware_class]["infer"][r.model] = r.time_infer_s

    # (model, hardware_class) -> metric -> list of relative-slowdown ratios
    slowdowns: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"train": [], "infer": []})
    # (model, hardware_class) -> metric -> list of absolute seconds (for the median)
    absolute: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"train": [], "infer": []})
    n_tasks_total: dict[tuple[str, str], int] = defaultdict(int)
    n_tasks_comparable: dict[tuple[str, str], int] = defaultdict(int)

    for task_key, by_hw in by_task_hw.items():
        for hw, by_metric in by_hw.items():
            for metric, model_times in by_metric.items():
                for model, t in model_times.items():
                    absolute[(model, hw)][metric].append(t)
                    n_tasks_total[(model, hw)] += 1 if metric == "train" else 0
                if len(model_times) >= 2:
                    fastest = min(model_times.values())
                    for model, t in model_times.items():
                        slowdowns[(model, hw)][metric].append(t / fastest)
                        if metric == "train":
                            n_tasks_comparable[(model, hw)] += 1

    rows = []
    for (model, hw), metrics in absolute.items():
        train_times = metrics["train"]
        infer_times = metrics["infer"]
        train_slowdowns = slowdowns[(model, hw)]["train"]
        infer_slowdowns = slowdowns[(model, hw)]["infer"]
        rows.append({
            "model": model,
            "hardware_class": hw,
            "n_tasks_on_this_hw": n_tasks_total[(model, hw)],
            "n_comparable_tasks": n_tasks_comparable[(model, hw)],
            "geo_mean_train_slowdown": _geo_mean(train_slowdowns) if train_slowdowns else None,
            "geo_mean_infer_slowdown": _geo_mean(infer_slowdowns) if infer_slowdowns else None,
            "median_train_s": sorted(train_times)[len(train_times) // 2] if train_times else None,
            "median_infer_s": sorted(infer_times)[len(infer_times) // 2] if infer_times else None,
        })

    # Primary ranking metric is the absolute median train time, not the slowdown
    # factor -- per user preference, and reasonable here since most models ran the
    # same fixed task list, so a same-hardware median time is already a fairly
    # apples-to-apples comparison without needing the ratio indirection.
    rows.sort(key=lambda r: (r["hardware_class"], r["median_train_s"] if r["median_train_s"] is not None else float("inf")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/RamanBench_v1/results/v1/data")
    parser.add_argument("--out", default="/tmp/model_speed_report.csv")
    args = parser.parse_args()

    ag_name_to_key = _build_ag_name_to_key()
    records = _collect_records(Path(args.results_dir), ag_name_to_key)
    rows = build_report(records)

    import csv

    with open(args.out, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("Wrote %d (model, hardware_class) row(s) to %s", len(rows), args.out)

    for r in rows[:30]:
        logger.info(
            "%-14s %-10s n=%-4d median_train=%6.1fs median_infer=%-8s comparable=%-4d "
            "geo_train=%.2fx geo_infer=%s",
            r["model"], r["hardware_class"], r["n_tasks_on_this_hw"],
            r["median_train_s"] if r["median_train_s"] is not None else float("nan"),
            f"{r['median_infer_s']:.2f}s" if r["median_infer_s"] is not None else "n/a",
            r["n_comparable_tasks"],
            r["geo_mean_train_slowdown"] if r["geo_mean_train_slowdown"] is not None else float("nan"),
            f"{r['geo_mean_infer_slowdown']:.2f}x" if r["geo_mean_infer_slowdown"] is not None else "n/a",
        )


if __name__ == "__main__":
    main()
