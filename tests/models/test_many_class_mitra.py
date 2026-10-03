"""Tests for the eager many-class (ECOC) wrapper used by Prep_MITRA/Prep_MITRA_V2.

The real Mitra checkpoints are GPU-only and downloaded on first fit, so the wrapper is
exercised against a CPU stand-in that mimics the two Mitra quirks the wrapper must
handle: no ``classes_`` attribute, and labels that must be exactly ``0..k-1`` (output
column j is label value j). End-to-end verification is via ``scripts/run_experiment.py``
on a GPU machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabpfn_extensions")
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from raman_bench.preprocessing.many_class_mitra import (  # noqa: E402
    MITRA_MAX_CLASSES,
    ManyClassMitra,
)


class _MitraLike:
    """Fixed 10-class head, no classes_, labels must be contiguous 0..k-1."""

    def __init__(self, **hyp):
        self.hyp = hyp
        self.trainers = ["trainer"]

    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        labels = np.unique(y)
        assert len(labels) <= MITRA_MAX_CLASSES
        assert (labels == np.arange(len(labels))).all(), labels
        if y_val is not None:
            assert np.isin(y_val, labels).all()
        self.m_ = LogisticRegression(max_iter=2000).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


def _data(n_classes, seed=0):
    X, y = make_classification(
        1500, 20, n_informative=15, n_classes=n_classes, n_clusters_per_class=1, random_state=seed
    )
    return pd.DataFrame(X), y


def test_few_classes_fits_base_directly():
    X, y = _data(5)
    m = ManyClassMitra(_MitraLike, seed=0).fit(X[:1000], y[:1000])
    assert len(m._models) == 1
    assert m.predict_proba(X[1000:]).shape == (500, 5)


@pytest.mark.parametrize("n_classes", [12, 30])
def test_many_classes_decodes_full_probabilities(n_classes):
    X, y = _data(n_classes)
    tr, va, te = slice(0, 1000), slice(1000, 1200), slice(1200, None)
    m = ManyClassMitra(_MitraLike, seed=0).fit(X[tr], y[tr], X_val=X[va], y_val=y[va], time_limit=60)
    proba = m.predict_proba(X[te])
    assert len(m._models) > 1
    assert proba.shape == (300, n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # ECOC should stay close to a direct fit of all classes (no label scrambling).
    direct = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).score(X[te], y[te])
    assert (m.predict(X[te]) == y[te]).mean() >= direct - 0.05
    # Hooks MitraModel reads (save/load, device moves) see every sub-model's trainers.
    assert len(m.trainers) == len(m._models)
