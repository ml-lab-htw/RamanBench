"""Prediction generation step for the benchmark pipeline (Step 1).

Trains each (model, dataset, seed) combination and saves predictions as CSV
files.  Supports resume: existing prediction files are skipped unless
``overwrite=True``.  Measures train/inference time, peak memory, GPU power
(pynvml), and CPU energy (Intel RAPL).

Usage
-----
::

    from raman_bench.config import load_config
    from raman_bench.predictions import compute_predictions

    config = load_config("configs/benchmark_v0.1.json")
    compute_predictions(config)
"""

import gc
import json
import logging
import os
import random
import threading
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from raman_data import TASK_TYPE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from raman_bench.benchmark import configure_benchmark
from raman_bench.logging_utils import LOG_FORMAT, run_file_logger
from raman_bench.preprocessing.wrapped_models import CLASSIFICATION_ONLY_MODELS
from raman_bench.seeds import get_seeds

try:
    from raman_bench.model import AutoGluonModel
except ImportError:
    AutoGluonModel = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data hash (staleness detection)
# ---------------------------------------------------------------------------


def _data_hash(df: pd.DataFrame) -> str:
    import hashlib

    label_col = df.columns[-1]
    h = hashlib.md5()
    h.update(str(len(df)).encode())
    h.update("|".join(sorted(df[label_col].astype(str))).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False


class _PsutilMemoryTracker:
    _POLL_S = 0.05

    def __init__(self):
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        proc = _psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / (1024**2)
                if rss > self._peak_mb:
                    self._peak_mb = rss
            except Exception:
                break
            self._stop.wait(self._POLL_S)

    def __enter__(self):
        self._stop.clear()
        self._peak_mb = _psutil.Process().memory_info().rss / (1024**2)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


class _TracemallocMemoryTracker:
    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *_):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_mb = peak / (1024**2)

    @property
    def peak_mb(self) -> float:
        return round(getattr(self, "_peak_mb", 0.0), 2)


def _memory_tracker():
    return _PsutilMemoryTracker() if _HAS_PSUTIL else _TracemallocMemoryTracker()


# ---------------------------------------------------------------------------
# Power / energy tracking
# ---------------------------------------------------------------------------

try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _pynvml = None
    _HAS_PYNVML = False

_RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
_RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"
_HAS_RAPL = os.path.isfile(_RAPL_PATH)


def _read_rapl():
    try:
        with open(_RAPL_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return None


class _PowerTracker:
    _POLL_S = 0.1

    def __init__(self):
        self._gpu_samples: list[float] = []
        self._gpu_handle = None
        self._elapsed_s = 0.0
        self._cpu_energy_j: float | None = None
        self._stop = threading.Event()
        self._thread = None

    def _poll_gpu(self):
        while not self._stop.is_set():
            try:
                mw = _pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                self._gpu_samples.append(mw / 1000.0)
            except Exception:
                pass
            self._stop.wait(self._POLL_S)

    def __enter__(self):
        self._start = time.perf_counter()
        self._gpu_samples = []
        if _HAS_PYNVML:
            try:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                self._stop.clear()
                self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
                self._thread.start()
            except Exception:
                self._gpu_handle = None
        self._cpu_start = _read_rapl() if _HAS_RAPL else None
        return self

    def __exit__(self, *_):
        self._elapsed_s = time.perf_counter() - self._start
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if _HAS_RAPL and self._cpu_start is not None:
            end = _read_rapl()
            if end is not None:
                delta = end - self._cpu_start
                if delta < 0:
                    try:
                        with open(_RAPL_MAX_PATH) as f:
                            delta += int(f.read().strip())
                    except Exception:
                        pass
                self._cpu_energy_j = delta / 1e6

    @property
    def gpu_mean_power_w(self):
        return (
            round(sum(self._gpu_samples) / len(self._gpu_samples), 2) if self._gpu_samples else None
        )

    @property
    def gpu_energy_j(self):
        if self._gpu_samples and self._elapsed_s > 0:
            return round(sum(self._gpu_samples) / len(self._gpu_samples) * self._elapsed_s, 2)
        return None

    @property
    def cpu_energy_j(self):
        return round(self._cpu_energy_j, 2) if self._cpu_energy_j is not None else None


# ---------------------------------------------------------------------------
# Excluded-key resolution (ablation-only keys/targets)
# ---------------------------------------------------------------------------


def _resolve_excluded_keys(config, benchmark) -> set[str]:
    """Resolve ``exclude_keys`` + ``exclude_targets`` to a set of dataset keys.

    Works during the prediction step without needing ``dataset_stats.json``:

    * ``exclude_keys``    — exact keys (e.g. ``"amino_acids_leucine_0"``), used directly.
    * ``exclude_targets`` — regression target column names (e.g. ``"time_h"``),
      resolved to ``"{dataset}_{target_idx}"`` via the dataset's target names
      (read from the mirror metadata by :meth:`RamanBenchmark.get_target_names`).

    Only regression dataset names are scanned for target-name matches:
    classification ``target_names`` are class labels, not target columns.
    """
    excluded: set[str] = set(config.get("exclude_keys", []))

    names = set(config.get("exclude_targets", []))
    if names:
        for ds in benchmark.dataset_names_regression or []:
            for idx, tname in enumerate(benchmark.get_target_names(ds)):
                if tname in names:
                    excluded.add(f"{ds}_{idx}")

    return excluded


# ---------------------------------------------------------------------------
# Subsampling (OOM guard for specific model/dataset combinations)
# ---------------------------------------------------------------------------


def _maybe_subsample(
    data_train: pd.DataFrame,
    model_name: str,
    key: str,
    task_type,
    subsample_config: dict | None,
    seed: int,
) -> pd.DataFrame:
    """Subsample *data_train* for listed (model, dataset) pairs.

    Applies uniform spectral downsampling (``max_features``) and/or stratified
    sample subsampling (``max_samples``) as an OOM guard for large datasets.
    A model whose dataset list contains ``"*"`` is subsampled on every dataset.
    Returns *data_train* unchanged if the pair is not listed.
    """
    if subsample_config is None:
        return data_train

    combos = subsample_config.get("combinations", {})
    model_keys = combos.get(model_name, [])
    # "*" subsamples the model on every dataset (OOM guard for memory-heavy models).
    if "*" not in model_keys and key not in model_keys:
        return data_train

    label_col = data_train.columns[-1]
    feature_cols = [c for c in data_train.columns if c != label_col]
    result = data_train

    max_f = subsample_config.get("max_features")
    if max_f and len(feature_cols) > max_f:
        step = len(feature_cols) / max_f
        selected = [feature_cols[round(i * step)] for i in range(max_f)]
        result = result[selected + [label_col]]

    max_n = subsample_config.get("max_samples", 10_000)
    if len(result) > max_n:
        if task_type == TASK_TYPE.Classification:
            _, result = train_test_split(
                result, test_size=max_n, random_state=seed, stratify=result[label_col]
            )
        else:
            result = result.sample(n=max_n, random_state=seed)

    return result


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def _timed():
    t = [0.0]
    start = time.perf_counter()
    try:
        yield t
    finally:
        t[0] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_predictions(
    config,
    seed_index: int | None = None,
    overwrite: bool = False,
    reverse: bool = False,
):
    """Train all models and save predictions to CSV.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration.
    seed_index : int | None
        Run only this zero-based seed index (for parallel seed execution).
    overwrite : bool
        Re-run even when a prediction file already exists.
    reverse : bool
        Iterate datasets in reverse order (useful for forward+reverse parallel jobs).
    """
    logger.info("=" * 60 + "\nSTEP 1: Computing Predictions")

    mem_backend = "psutil" if _HAS_PSUTIL else "tracemalloc"
    output_dir = config["output_dir"]

    cache_dir = config.get("cache_dir") or None
    autogluon_path = (
        os.path.join(cache_dir, "autogluon") if cache_dir else os.path.join(".cache", "autogluon")
    )
    os.makedirs(autogluon_path, exist_ok=True)

    seeds = get_seeds(config)
    if seed_index is not None:
        seeds = [seeds[seed_index]]

    model_names = config["models"]
    autogluon_time_limit = config["autogluon_time_limit"]
    autogluon_presets = config["autogluon_presets"]
    preprocessing_config = config.get("preprocessing_config")
    optimize = config.get("optimize", True)
    ensemble = config.get("ensemble", True)
    num_hpo_trials = config.get("num_hpo_trials", 0)
    subsample_config = config.get("subsample")

    # Per-epoch augmentation flag lives in preprocessing dict or at top level
    model_extra_params = {}
    raw_prep = config.get("preprocessing", {})
    key_src = raw_prep if isinstance(raw_prep, dict) else config
    if "per_epoch_augmentation" in key_src:
        model_extra_params["per_epoch_augmentation"] = key_src["per_epoch_augmentation"]

    for seed in seeds:
        logger.info("--- Seed %s ---", seed)
        config["random_state"] = seed
        _set_global_seeds(seed)
        benchmark = configure_benchmark(config)

        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        predictions_dir = os.path.join(seed_dir, "predictions")
        logs_dir = os.path.join(seed_dir, "logs")
        stats_dir = os.path.join(seed_dir, "stats")
        for d in (predictions_dir, logs_dir, stats_dir):
            os.makedirs(d, exist_ok=True)

        # Ablation-only keys/targets (exclude_keys/exclude_targets) are computed
        # by default. When compute_excluded_keys=False (e.g. HPO runs) drop them
        # from the key list so they are neither loaded nor predicted.
        if not config.get("compute_excluded_keys", True):
            excluded = _resolve_excluded_keys(config, benchmark)
            if excluded:
                pairs = [
                    (k, t)
                    for k, t in zip(benchmark._key_list, benchmark._task_type_list)
                    if k not in excluded
                ]
                n_dropped = len(benchmark._key_list) - len(pairs)
                benchmark._key_list = [k for k, _ in pairs]
                benchmark._task_type_list = [t for _, t in pairs]
                logger.info(
                    "compute_excluded_keys=False: skipping %d excluded key(s) in predictions.",
                    n_dropped,
                )

        if reverse:
            benchmark._key_list = list(reversed(benchmark._key_list))

        pbar = tqdm(total=len(benchmark) * len(model_names), desc=f"Seed {seed}")
        results = []

        for model_name in model_names:
            for data_train, data_test, key, task_type in benchmark:
                pbar.set_description(f"Seed {seed} | {key} | {model_name}")

                if data_train is None or data_test is None:
                    pbar.update(1)
                    continue

                # Skip classification-only models for regression datasets
                if task_type == TASK_TYPE.Regression and model_name in CLASSIFICATION_ONLY_MODELS:
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        f"{model_name} only supports classification tasks",
                    )
                    pbar.update(1)
                    continue

                # Skip constant-target datasets for regression
                if task_type == TASK_TYPE.Regression:
                    label_col = data_train.columns[-1]
                    if data_train[label_col].std() == 0:
                        _write_skip_record(
                            stats_dir,
                            key,
                            model_name,
                            len(data_train),
                            len(data_test),
                            "constant_target: all training target values are equal",
                        )
                        pbar.update(1)
                        continue

                pred_path = os.path.join(predictions_dir, f"{key}_{model_name}_predictions.csv")
                proba_path = os.path.join(predictions_dir, f"{key}_{model_name}_proba.csv")
                # For classification, both files must exist to skip — otherwise
                # rerun so the missing proba file gets written and downstream
                # log_loss / ROC-AUC become available.
                if task_type == TASK_TYPE.Classification:
                    already_complete = os.path.exists(pred_path) and os.path.exists(proba_path)
                else:
                    already_complete = os.path.exists(pred_path)
                if already_complete and not overwrite:
                    pbar.update(1)
                    continue

                log_path = os.path.join(logs_dir, f"{key}_{model_name}.log")
                with run_file_logger(log_path):
                    model = AutoGluonModel(
                        ensemble=ensemble,
                        optimize=optimize,
                        models=[model_name],
                        task_type=task_type,
                        autogluon_time_limit=autogluon_time_limit,
                        autogluon_presets=autogluon_presets,
                        autogluon_path=os.path.join(autogluon_path, key),
                        preprocessing_config=preprocessing_config,
                        model_extra_params=model_extra_params or None,
                        num_hpo_trials=num_hpo_trials,
                    )

                    record = {
                        "dataset": key,
                        "model": model_name,
                        "n_train_samples": len(data_train),
                        "n_test_samples": len(data_test),
                        "status": "pass",
                        "error": "",
                        "timestamp": datetime.now().isoformat(),
                        "train_time_s": None,
                        "inference_time_s": None,
                        "inference_time_per_sample_ms": None,
                        "train_peak_memory_mb": None,
                        "inference_peak_memory_mb": None,
                        "memory_backend": mem_backend,
                        "n_models_trained": None,
                        "n_base_models": None,
                        "ag_total_fit_time_s": None,
                        "ag_time_per_model_s": None,
                        "train_gpu_power_w": None,
                        "train_gpu_energy_j": None,
                        "train_cpu_energy_j": None,
                        "inference_gpu_power_w": None,
                        "inference_gpu_energy_j": None,
                        "inference_cpu_energy_j": None,
                    }

                    try:
                        data_train_fit = _maybe_subsample(
                            data_train, model_name, key, task_type, subsample_config, seed
                        )

                        with _timed() as tt, _memory_tracker() as tm, _PowerTracker() as tp:
                            model.fit(data_train_fit)
                        record["train_time_s"] = round(tt[0], 3)
                        record["train_peak_memory_mb"] = tm.peak_mb
                        record["train_gpu_power_w"] = tp.gpu_mean_power_w
                        record["train_gpu_energy_j"] = tp.gpu_energy_j
                        record["train_cpu_energy_j"] = tp.cpu_energy_j
                        record.update(model.get_fit_stats())

                        with _timed() as it, _memory_tracker() as im, _PowerTracker() as ip:
                            y_pred = model.predict(data_test)
                        record["inference_time_s"] = round(it[0], 3)
                        record["inference_peak_memory_mb"] = im.peak_mb
                        record["inference_time_per_sample_ms"] = round(
                            it[0] / len(data_test) * 1000, 4
                        )
                        record["inference_gpu_power_w"] = ip.gpu_mean_power_w
                        record["inference_gpu_energy_j"] = ip.gpu_energy_j
                        record["inference_cpu_energy_j"] = ip.cpu_energy_j

                        y_pred.sort_index().to_csv(pred_path, index=True)

                        # For classification, also persist class probabilities
                        # so downstream metrics (log_loss, ROC-AUC) can be
                        # computed without rerunning the model.
                        if task_type == TASK_TYPE.Classification:
                            try:
                                y_proba = model.predict_proba(data_test)
                                proba_path = os.path.join(
                                    predictions_dir, f"{key}_{model_name}_proba.csv"
                                )
                                y_proba.sort_index().to_csv(proba_path, index=True)
                            except Exception as e:
                                logger.warning(
                                    "predict_proba failed for %s / %s: %s",
                                    key,
                                    model_name,
                                    e,
                                )

                        truth_path = os.path.join(predictions_dir, f"{key}_ground_truth.csv")
                        if not os.path.exists(truth_path):
                            data_test[["target"]].sort_index().to_csv(truth_path, index=True)

                    except Exception as e:
                        logger.error("Error for %s / %s: %s", key, model_name, e, exc_info=True)
                        record["status"] = "fail"
                        record["error"] = str(e)

                    with open(os.path.join(stats_dir, f"{key}_{model_name}.json"), "w") as f:
                        json.dump(record, f, indent=2)

                # Free GPU memory between runs
                _cleanup_model(model)
                del model
                gc.collect()
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                results.append(record)
                pbar.update(1)

        pbar.close()
        _log_seed_summary(seed, results, stats_dir)


def _cleanup_model(model):
    """Clear AutoGluon internals before deleting to break reference cycles."""
    try:
        predictor = getattr(model, "predictor", None)
        if predictor is not None:
            learner = getattr(predictor, "_learner", None)
            trainer = getattr(learner, "_trainer", None) if learner else None
            if trainer is not None and hasattr(trainer, "models"):
                trainer.models.clear()
            model.predictor = None
    except Exception:
        pass


def _write_skip_record(stats_dir, key, model_name, n_train, n_test, error):
    record = {
        "dataset": key,
        "model": model_name,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "status": "fail",
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(stats_dir, f"{key}_{model_name}.json"), "w") as f:
        json.dump(record, f, indent=2)


def _log_seed_summary(seed, results, stats_dir):
    all_records = []
    for fname in sorted(os.listdir(stats_dir)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(stats_dir, fname)) as f:
                    all_records.append(json.load(f))
            except Exception:
                pass

    n_total = len(all_records)
    n_failed = sum(r.get("status") == "fail" for r in all_records)
    n_this = len(results)
    n_this_failed = sum(r.get("status") == "fail" for r in results)

    if n_total:
        logger.info(
            "Seed %s: %d this run (%d passed, %d failed) | %d pre-existing (%d failed). Stats: %s",
            seed,
            n_this,
            n_this - n_this_failed,
            n_this_failed,
            n_total - n_this,
            n_failed - n_this_failed,
            stats_dir,
        )
