"""Eager many-class (ECOC) wrapper for Mitra and Mitra-v2.

Both Mitra checkpoints have a fixed 10-class head: ``MitraClassifier``'s preprocessor
asserts ``y < n_classes`` with ``n_classes = 10``, so any RamanBench dataset with more
classes (bacteria_identification 30, rruff_mineral_raw 79, mlrod 16, cancer_cell_* 12)
crashed with ``AssertionError: y contains class values that are not in the range of
n_classes``. The installed AutoGluon pin has no many-class fallback for Mitra (unlike
the TabPFN family, see ``wrapped_models._ManyClassTabPFNProxy``).

Why not reuse ``tabpfn_extensions.many_class.ManyClassClassifier`` directly, the way
``_ManyClassTabPFNProxy`` does: it is *lazy* -- ``fit`` only stores the codebook and the
training table, and every ``predict_proba`` call re-fits one base estimator per codebook
row. That is free for an in-context learner like TabPFN, but Mitra fine-tunes on every
fit (``DEFAULT_FINE_TUNE = True``), and a bagged AutoGluon fit calls ``predict_proba``
several times per child (OOF, then test) -- dozens of fine-tunes per task. This wrapper
uses ``ManyClassClassifier`` only to build the codebook (its ``fit`` is cheap), then
fine-tunes each per-row sub-model ONCE at fit time and reuses them at predict time. The
per-row weighting and the decoding (``row_weighter``/``make_aggregator``) are the
library's own, so the combination step is identical to the TabPFN path.

Hooks both Mitra wrappers rely on are forwarded to every sub-model: ``trainers`` (read by
``MitraModel``'s save/load/device/``post_fit_optimize`` code), ``configure_recipe`` and
``activate_heldout_in_support`` (Mitra-v2 only). Each sub-model is fit on the row's
re-coded labels for both the fit fold and, when present, the held-out fold, since
Mitra-v2 fine-tunes against ``X_val``/``y_val``.
"""

from __future__ import annotations

import functools
import time

import numpy as np
import pandas as pd

MITRA_MAX_CLASSES = 10  # both checkpoints' head width


def many_class_mitra_get_model_cls(self):
    """``get_model_cls`` override for ``Prep_MITRA``/``Prep_MITRA_V2``.

    Resolves the real classifier class from the next ``get_model_cls`` up the MRO and,
    for multiclass problems only, returns it wrapped in :class:`ManyClassMitra` (which
    behaves exactly like the real class up to 10 classes). Binary and regression are
    untouched.
    """
    for klass in type(self).__mro__:
        f = klass.__dict__.get("get_model_cls")
        if f is not None and f is not many_class_mitra_get_model_cls:
            real_cls = f(self)
            break
    else:  # pragma: no cover - every Mitra class defines get_model_cls
        raise AttributeError("no get_model_cls found in MRO")
    if self.problem_type == "multiclass":
        return functools.partial(ManyClassMitra, real_cls)
    return real_cls


class ManyClassMitra:
    """Constructor-compatible stand-in for a Mitra(-v2) sklearn classifier class.

    ``get_model_cls`` returns ``functools.partial(ManyClassMitra, real_cls)``; Mitra's
    ``_fit`` then calls it exactly like the real class (``model_cls(**hyp)``).
    """

    def __init__(self, real_cls, **hyp):
        self._real_cls = real_cls
        self._hyp = hyp
        self._recipe = None
        self._models: list = []
        self._sub_classes: list = []  # per sub-model: original row code of each output column
        self._direct = True
        self.classes_ = None
        # ECOC state (only set when n_classes > MITRA_MAX_CLASSES)
        self._code_book = None
        self._row_weights = None
        self._alphabet_size = None
        self._has_rest = False
        self._rest_mask = None
        self._use_log = True
        self._mask_rest_log_agg = False

    # -- construction hooks forwarded to every sub-model -------------------------------
    def configure_recipe(self, settings):
        self._recipe = settings
        for m in self._models:
            m.configure_recipe(settings)
        return self

    def _new_base(self):
        base = self._real_cls(**self._hyp)
        if self._recipe is not None:
            base.configure_recipe(self._recipe)
        return base

    # -- fit -------------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        from tabpfn_extensions.many_class import ManyClassClassifier
        from tabpfn_extensions.many_class._utils import align_probabilities

        y = pd.Series(np.asarray(y), index=getattr(X, "index", None))
        self.classes_ = np.unique(y.to_numpy())
        if len(self.classes_) <= MITRA_MAX_CLASSES:
            base = self._new_base()
            self._models = [base.fit(X, y, X_val=X_val, y_val=y_val, time_limit=time_limit)]
            self._direct = True
            return self

        start = time.monotonic()
        self._direct = False
        # Codebook only: with a real codebook, ManyClassClassifier.fit fits no estimator.
        mc = ManyClassClassifier(
            estimator=self._real_cls(**self._hyp),
            alphabet_size=MITRA_MAX_CLASSES,
            random_state=self._hyp.get("seed", self._hyp.get("random_state")),
        )
        mc.fit(np.zeros((len(y), 1)), y.to_numpy())
        stats = mc.codebook_stats_
        self._code_book = mc.code_book_
        self._alphabet_size = mc.alphabet_size_
        self._has_rest = bool(stats.get("has_rest_symbol", False))
        rest_code = stats.get("rest_class_code") if self._has_rest else None
        self._rest_mask = mc._row_class_mask_.astype(float) if self._has_rest else None
        # Same rule as ManyClassClassifier.predict_proba: without a rest symbol, decoding
        # always uses log-likelihood.
        self._use_log = mc._aggregation_config.log_likelihood or not self._has_rest
        self._mask_rest_log_agg = mc._aggregation_config.legacy_mask_rest_log_agg
        filter_rest = self._has_rest and mc._codebook_config.legacy_filter_rest_train
        class_index = mc.classes_index_
        train_codes = mc.Y_train_per_estimator  # (n_rows, n_train)
        val_idx = (
            np.array([class_index[c] for c in np.asarray(y_val)]) if y_val is not None else None
        )

        n_rows = self._code_book.shape[0]
        self._models, self._sub_classes, raw_weights = [], [], []
        for r in range(n_rows):
            codes = train_codes[r]
            keep = codes != rest_code if filter_rest else np.ones(len(codes), dtype=bool)
            # Mitra keeps no classes_: its output column j is label value j for the first
            # len(unique(y)) columns, so labels must be 0..k-1. A codebook row need not use
            # every code, hence the re-encoding; sub_classes maps columns back to codes.
            sub_classes = np.unique(codes[keep])
            y_r = pd.Series(np.searchsorted(sub_classes, codes[keep]), index=y.index[keep])
            X_r = X[keep]
            X_val_r = y_val_r = None
            if X_val is not None:
                val_codes = self._code_book[r, val_idx]
                vkeep = val_codes != rest_code if filter_rest else np.ones(len(val_codes), dtype=bool)
                # Mitra can't score a held-out label it never saw in training.
                vkeep &= np.isin(val_codes, sub_classes)
                if vkeep.any():
                    X_val_r = X_val[vkeep]
                    y_val_r = pd.Series(np.searchsorted(sub_classes, val_codes[vkeep]), index=X_val.index[vkeep])
            row_limit = None
            if time_limit is not None:
                row_limit = max(1.0, (time_limit - (time.monotonic() - start)) / (n_rows - r))
            sub = self._new_base().fit(X_r, y_r, X_val=X_val_r, y_val=y_val_r, time_limit=row_limit)
            self._models.append(sub)
            self._sub_classes.append(sub_classes)
            proba_train = align_probabilities(sub.predict_proba(X_r), sub_classes, self._alphabet_size)
            weight, _ = mc._row_weighter.weight(proba_train, sub_classes[y_r.to_numpy()], self._alphabet_size)
            raw_weights.append(weight)

        from tabpfn_extensions.many_class._utils import normalize_weights

        self._row_weights = normalize_weights(np.asarray(raw_weights, dtype=float))
        return self

    # -- predict ---------------------------------------------------------------------
    def predict_proba(self, X):
        if self._direct:
            return self._models[0].predict_proba(X)
        from tabpfn_extensions.many_class._strategies import make_aggregator
        from tabpfn_extensions.many_class._utils import align_probabilities

        proba_rows = np.stack(
            [
                align_probabilities(m.predict_proba(X), c, self._alphabet_size)
                for m, c in zip(self._models, self._sub_classes)
            ],
            axis=0,
        )
        aggregator = make_aggregator(self._use_log, mask_rest=self._has_rest and self._mask_rest_log_agg)
        return aggregator.aggregate(proba_rows, self._code_book, self._row_weights, rest_mask=self._rest_mask)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # -- hooks read by MitraModel / MitraV2Model -------------------------------------
    @property
    def trainers(self):
        return [t for m in self._models for t in m.trainers]

    def activate_heldout_in_support(self):
        for m in self._models:
            m.activate_heldout_in_support()

    def __getattr__(self, name):
        # Anything else (e.g. MitraV2's predict-time settings) comes from the first sub-model.
        if name.startswith("_") or not self.__dict__.get("_models"):
            raise AttributeError(name)
        return getattr(self._models[0], name)
