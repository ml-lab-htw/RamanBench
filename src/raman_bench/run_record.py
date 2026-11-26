"""What a run leaves next to its result: the setup it ran under, and why it failed if it did.

Every fold lives in ``<results_dir>/<experiment>/<task>/<repeat>_<fold>/``. A finished fold has a
``results.pkl`` and a ``setup.json``; a fold that raised has a ``failure.json`` instead. A fold with
neither failed without a trace (the pod was killed, for example) and is reported as ``missing``.

``failure.json`` carries one uniform label, so a model's failures can be counted by reason:

- ``oom_gpu``: CUDA out of memory.
- ``oom_cpu``: out of host memory (``MemoryError``, AutoGluon's ``NotEnoughMemoryError``).
- ``timeout``: the fit exceeded its time limit.
- ``error``: any other exception.

``setup.json`` records what the result was produced with: package versions, hardware, and the
protocol settings the runner passed (time limit, bag folds, row cap, ...).
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

SETUP_FILE = "setup.json"
FAILURE_FILE = "failure.json"
LABELS = ("oom_gpu", "oom_cpu", "timeout", "error")
MISSING = "missing"
PACKAGES = ("raman-bench", "tabarena", "autogluon.tabular", "torch", "scikit-learn", "numpy")
MESSAGE_CHARS = 500


def classify_failure(exc: BaseException) -> str:
    """The uniform label of *exc* (see the module docstring), looking through its ``__cause__`` / ``__context__`` chain."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name, text = type(exc).__name__, str(exc).lower()
        if "outofmemoryerror" in name.lower() or "cuda out of memory" in text or name == "NotEnoughCudaMemoryError":
            return "oom_gpu"
        if isinstance(exc, MemoryError) or name == "NotEnoughMemoryError":
            return "oom_cpu"
        if isinstance(exc, TimeoutError) or name == "TimeLimitExceeded":
            return "timeout"
        exc = exc.__cause__ or exc.__context__
    return "error"


def hardware_info() -> dict:
    """The device a fit runs on, read from torch."""
    import torch

    if torch.cuda.is_available():
        return {"gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()}
    if torch.backends.mps.is_available():
        return {"gpu": "Apple MPS", "gpu_count": 1}
    return {"gpu": None, "gpu_count": 0}


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def fit_mode(experiment) -> str:
    """``"system"`` for a self-contained system (fitted once per outer fold), ``"bagged"`` for a model run through the protocol's bagging."""
    return "system" if type(experiment).__name__ == "ExternalSystemExperiment" else "bagged"


def write_setup(cache_path: str | Path, experiment, setup: dict | None = None) -> None:
    """Write ``setup.json`` next to a finished fold's ``results.pkl``."""
    record = {
        "experiment": experiment.name,
        "fit_mode": fit_mode(experiment),
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": package_versions(),
        "hardware": hardware_info(),
        "setup": setup or {},
    }
    (Path(cache_path) / SETUP_FILE).write_text(json.dumps(record, indent=2, default=str))
    (Path(cache_path) / FAILURE_FILE).unlink(missing_ok=True)


def write_failure(cache_path: str | Path, experiment, exc: BaseException, setup: dict | None = None) -> None:
    """Write ``failure.json`` for a fold whose run raised *exc*."""
    record = {
        "experiment": experiment.name,
        "fit_mode": fit_mode(experiment),
        "label": classify_failure(exc),
        "exception": type(exc).__name__,
        "message": str(exc)[:MESSAGE_CHARS],
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "setup": setup or {},
    }
    (Path(cache_path) / FAILURE_FILE).write_text(json.dumps(record, indent=2, default=str))


def collect_folds(results_dir: str | Path) -> pd.DataFrame:
    """One row per fold directory under *results_dir*: ``status`` is ``ok``, ``failed`` or ``missing``.

    ``label`` is the failure label of a failed fold, ``missing`` for a fold with neither result nor record,
    and empty for an ok one. ``fit_mode`` comes from ``setup.json`` where there is one.
    """
    rows = []
    for fold_dir in sorted(Path(results_dir).glob("*/*/*_*")):
        if not fold_dir.is_dir():
            continue
        framework, task, rf = fold_dir.parent.parent.name, fold_dir.parent.name, fold_dir.name
        repeat, _, fold = rf.partition("_")
        row = {"framework": framework, "task": task, "repeat": int(repeat), "fold": int(fold)}
        if (fold_dir / "results.pkl").exists():
            record = _read_json(fold_dir / SETUP_FILE)
            row |= {"status": "ok", "label": "", "exception": "", "message": "", "fit_mode": record.get("fit_mode", "")}
        elif (fold_dir / FAILURE_FILE).exists():
            record = _read_json(fold_dir / FAILURE_FILE)
            row |= {
                "status": "failed", "label": record.get("label", "error"), "exception": record.get("exception", ""),
                "message": record.get("message", ""), "fit_mode": record.get("fit_mode", ""),
            }
        else:
            row |= {"status": "missing", "label": MISSING, "exception": "", "message": "", "fit_mode": ""}
        rows.append(row)
    return pd.DataFrame(rows, columns=["framework", "task", "repeat", "fold", "status", "label", "exception", "message", "fit_mode"])


def failure_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Per framework: the number of folds by outcome (``ok``, ``oom_gpu``, ``oom_cpu``, ``timeout``, ``error``, ``missing``)."""
    if folds.empty:
        return pd.DataFrame(columns=["framework", "ok", *LABELS, MISSING])
    outcome = folds["label"].where(folds["status"] != "ok", "ok")
    table = pd.crosstab(folds["framework"], outcome)
    for column in ("ok", *LABELS, MISSING):
        if column not in table:
            table[column] = 0
    return table[["ok", *LABELS, MISSING]].reset_index()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
