"""Assemble HIVE-COTE 2 from its four components' cached results.

HC2 (Middlehurst et al., 2021) weights each component's class probabilities by its
train-set accuracy estimate raised to ``alpha`` (4; CAWPE) and normalises the sum.
aeon's ``HIVECOTEV2`` gets that estimate from each component's own internal
cross-validation / out-of-bag predictions. Here it comes from the out-of-fold
predictions of the 3-fold bagging every RamanBench result already stores, so HC2 is
built without fitting anything again.

For each (task, repeat, fold) where all of STC, DRCIF, ARSENAL and TDE have a result,
:func:`assemble` writes a ``results.pkl`` for the model ``HIVECOTEV2-ASSEMBLED`` next
to them, in the same format, so ``raman_bench.compare`` and the aggregation scripts
treat it as any other model. Its training and inference times are the sums of the
components'.
"""

from __future__ import annotations

import copy
import glob
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

COMPONENTS = ("STC", "DRCIF", "ARSENAL", "TDE")
ASSEMBLED_KEY = "HIVECOTEV2-ASSEMBLED"
ALPHA = 4


def _framework(key: str) -> str:
    return f"{key}_c1_BAG_L1"


def _as_matrix(pred: np.ndarray) -> np.ndarray:
    """Binary results store the positive-class column only."""
    pred = np.asarray(pred, dtype=np.float64)
    return np.column_stack([1 - pred, pred]) if pred.ndim == 1 else pred


def _accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((_as_matrix(pred).argmax(axis=1) == np.asarray(y)).mean())


def _combine(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted sum of probabilities, normalised (aeon ``_BaseHIVECOTE._predict_proba``)."""
    total = sum(w * _as_matrix(p) for w, p in zip(weights, preds))
    sums = total.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    out = total / sums
    return out[:, 1] if np.asarray(preds[0]).ndim == 1 else out


def assemble_result(results: dict[str, dict], alpha: float = ALPHA) -> dict:
    """HC2 result for one (task, repeat, fold) from ``{component: result}``."""
    from autogluon.core.metrics import get_metric

    first = results[COMPONENTS[0]]
    sa0 = first["simulation_artifacts"]
    y_val, y_test = sa0["y_val"], sa0["y_test"]
    for name, res in results.items():
        sa = res["simulation_artifacts"]
        if not (
            np.array_equal(sa["y_val_idx"], sa0["y_val_idx"])
            and np.array_equal(sa["y_test_idx"], sa0["y_test_idx"])
        ):
            raise ValueError(f"{name} was fit on different rows than {COMPONENTS[0]}")

    val = [
        res["simulation_artifacts"]["pred_proba_dict_val"][_framework(c)]
        for c, res in results.items()
    ]
    test = [
        res["simulation_artifacts"]["pred_proba_dict_test"][_framework(c)]
        for c, res in results.items()
    ]
    accuracies = np.array([_accuracy(y_val, p) for p in val])
    weights = accuracies**alpha
    if not weights.any():  # every component wrong on every row: fall back to equal weights
        weights = np.ones_like(weights)
    pred_val, pred_test = _combine(val, weights), _combine(test, weights)
    test_per_child = [
        _combine(list(children), weights)
        for children in zip(
            *(
                res["simulation_artifacts"]["bag_info"]["pred_proba_test_per_child"]
                for res in results.values()
            )
        )
    ]

    metric = get_metric(first["metric"], problem_type=first["problem_type"])
    framework = _framework(ASSEMBLED_KEY)
    out = copy.deepcopy(first)
    out["framework"] = framework
    out["metric_error"] = float(metric.error(y_test, pred_test))
    out["metric_error_val"] = float(metric.error(y_val, pred_val))
    out["time_train_s"] = float(sum(r["time_train_s"] for r in results.values()))
    out["time_infer_s"] = float(sum(r["time_infer_s"] for r in results.values()))
    meta = out["method_metadata"]
    meta.update(model_cls=None, model_type=ASSEMBLED_KEY, name_prefix=ASSEMBLED_KEY)
    meta["assembled_from"] = {c: float(w) for c, w in zip(results, weights)}
    sa = out["simulation_artifacts"]
    sa["pred_proba_dict_val"] = {framework: pred_val}
    sa["pred_proba_dict_test"] = {framework: pred_test}
    sa["bag_info"]["pred_proba_test_per_child"] = test_per_child
    return out


def assemble(results_dir: str, *, alpha: float = ALPHA, overwrite: bool = False) -> list[str]:
    """Write ``HIVECOTEV2-ASSEMBLED`` results for every split all four components finished.

    Returns the written paths. Splits missing a component are skipped (logged), so the
    assembled model is imputed there like any failed fold.
    """
    from tabarena.utils.pickle_utils import dumps_pickle, load_pickle

    splits = {
        os.path.relpath(os.path.dirname(p), os.path.join(results_dir, _framework(COMPONENTS[0])))
        for p in glob.glob(
            os.path.join(results_dir, _framework(COMPONENTS[0]), "*", "*", "results.pkl")
        )
    }
    written = []
    for split in sorted(splits):
        paths = {
            c: os.path.join(results_dir, _framework(c), split, "results.pkl") for c in COMPONENTS
        }
        missing = [c for c, p in paths.items() if not os.path.exists(p)]
        if missing:
            logger.info("Skipping %s: no result for %s", split, ", ".join(missing))
            continue
        target = os.path.join(results_dir, _framework(ASSEMBLED_KEY), split, "results.pkl")
        if os.path.exists(target) and not overwrite:
            continue
        out = assemble_result({c: load_pickle(p) for c, p in paths.items()}, alpha=alpha)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(dumps_pickle(out, compress=True))
        written.append(target)
    return written
