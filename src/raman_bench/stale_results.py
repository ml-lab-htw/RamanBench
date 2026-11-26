"""Which cached results were produced under settings that differ from a protocol's, and so must be rerun.

A result is stale when the old settings changed what the model saw or how long it had:

- ``rows``: the dataset was cut to another number of rows (the old scope's dataset cap or the model's own cap against
  the protocol's ``max_train_samples``);
- ``time``: the protocol's time limit is higher and the fit used at least *threshold* of the old limit, so it was
  bound by its budget (a fit that stopped well before its limit would not change with more time).

``old_scope`` is the scope file the results ran under (``time_limit``, ``time_limit_overrides``,
``model_time_limit_overrides``, ``max_train_samples_overrides``, ``model_max_train_samples_overrides``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from raman_bench.compare import row_cap

THRESHOLD = 0.8


def _model_value(overrides: dict, model: str | None, dataset: str):
    value = (overrides or {}).get(model) if model else None
    return value.get(dataset) if isinstance(value, dict) else value


def old_row_cap(old_scope: dict, model: str | None, dataset: str) -> int | None:
    """The row cap a result of *model* on *dataset* ran under (the smaller of the dataset cap and the model's own)."""
    caps = [
        c
        for c in (
            (old_scope.get("max_train_samples_overrides") or {}).get(dataset),
            _model_value(old_scope.get("model_max_train_samples_overrides"), model, dataset),
        )
        if c
    ]
    return min(caps) if caps else None


def old_time_limit(old_scope: dict, model: str | None, dataset: str) -> float:
    """The time limit a result ran under: the default, raised by a dataset or model entry (the largest applies)."""
    candidates = [
        old_scope["time_limit"],
        (old_scope.get("time_limit_overrides") or {}).get(dataset),
        _model_value(old_scope.get("model_time_limit_overrides"), model, dataset),
    ]
    return max(c for c in candidates if c)


def stale_reason(
    model: str | None, task: dict, time_train_s: float, old_scope: dict, protocol: dict, threshold: float = THRESHOLD
) -> str:
    """``""`` for a result that is still valid under *protocol*, else ``"rows"``, ``"time"`` or ``"rows+time"``."""
    dataset, n = task["dataset"], task["num_instances"]
    old_cap, new_cap = old_row_cap(old_scope, model, dataset), row_cap(protocol, dataset)
    reasons = []
    if min(n, old_cap or n) != min(n, new_cap or n):
        reasons.append("rows")
    old_limit = old_time_limit(old_scope, model, dataset)
    if model is not None and protocol["time_limit"] > old_limit and time_train_s >= threshold * old_limit:
        reasons.append("time")
    return "+".join(reasons)


def classify(results: list[dict], old_scope: dict, protocol: dict, threshold: float = THRESHOLD) -> pd.DataFrame:
    """One row per loaded result (``raman_bench.aggregation.scan_cached_results``) with its ``reason`` for being stale."""
    tasks = {t["task"]: t for t in protocol["tasks"]}
    rows = []
    for r in results:
        meta = r["task_metadata"]
        task = tasks.get(meta["name"])
        if task is None:
            continue
        model = (r.get("method_metadata") or {}).get("model_type")
        rows.append(
            {
                "framework": r["framework"], "task": meta["name"], "repeat": meta["repeat"], "fold": meta["fold"],
                "model": model, "time_train_s": r.get("time_train_s"),
                "reason": stale_reason(model, task, r.get("time_train_s") or 0.0, old_scope, protocol, threshold),
            }
        )
    return pd.DataFrame(rows, columns=["framework", "task", "repeat", "fold", "model", "time_train_s", "reason"])


def quarantine(results_dir: str | Path, stale: pd.DataFrame, quarantine_dir: str | Path) -> int:
    """Move the fold directory of every stale result to *quarantine_dir*, keeping its relative path. Returns the count."""
    n = 0
    for r in stale.itertuples():
        rel = Path(r.framework) / r.task / f"{r.repeat}_{r.fold}"
        src = Path(results_dir) / rel
        if not src.is_dir():
            continue
        dst = Path(quarantine_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(os.fspath(src), os.fspath(dst))
        n += 1
    return n
