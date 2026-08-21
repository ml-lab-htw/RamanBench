"""RamanPFN — frozen-TabPFN dual-view (GCU + LVSE) model.

Reproduces Pan et al., "RamanPFN: learning from Raman spectral structure with
a tabular foundation model" (arXiv:2608.02157) as a genuinely separate
*model* (not a preprocessing recipe), because it needs both the GCU and LVSE
representations simultaneously as two internal TabPFN forward passes whose
predictions are then merged — unlike ``gcu``/``lvse`` in
``raman_preprocessing.py``, which are single representation-replacing
*preprocessing steps* usable by any downstream model.

Paper source: the fetched primary source (arxiv.org/abs/2608.02157 abstract
+ arxiv.org/html/2608.02157 HTML rendering) gives, verbatim:

- Signed triplet integration: ``p_triplet = -lambda*p_- + lambda*p_l + p_h``
  with ``lambda_j in {1, 0.75, 0.5}``, where ``p_h`` ("anchor") "supplies
  the main estimate", ``p_l`` ("local") and ``p_-`` ("negative") set a
  contrast term: "lambda*(p_l - p_-) adjusts it".
- Full aggregation: ``p_hat = p_ref + alpha * [A({p_triplet^(j)}_{j=1}^8) - p_ref]``,
  where "eight triplets are retained and combined" and ``A`` is "a mean,
  median or symmetrically trimmed mean". Alpha ("the correction magnitude")
  is "fixed or determined analytically from the training-target scale, the
  radius of the selected correction and agreement among the retained
  triplets" — the paper never gives alpha a concrete numeric value.
- Classification: "the same operation is applied coordinate-wise to
  class-aligned probability vectors, followed by projection onto the
  probability simplex."
- GCU rank selection: "intermediate factorization states yield a
  multiresolution family... protocol-matched OOF predictions rank the GCU
  and LVSE candidates" (already simplified to a single tunable rank by
  ``gcu_fit`` — see its docstring).
- LVSE: "the configured 32-way partition realizes 31 non-empty regions" (one
  example configuration, not stated as a fixed default across all tasks).
- TabPFN capacity: "TabPFN caps each forward pass at 500 features and, for
  wider inputs, increases the estimator count to a maximum of 32" — GCU/LVSE
  views are far narrower than 500 features by construction (``rho`` /
  ``n_regions * k_per_region``, both typically << 100), so this cap is a
  non-issue here; row/class caps are a separate matter (see below).

**What the paper itself never specifies** (confirmed by re-querying the
fetched HTML source directly): which of the two representations (GCU vs.
LVSE) is assigned to which of the three triplet roles (anchor/local/
negative); how the "eight triplets" are literally constructed from the two
representations (the multiresolution GCU-rank family and LVSE-region-count
family evidently supply the extra candidates, i.e. more than 2 raw views);
and any concrete numeric value for alpha.

**This implementation's explicit simplification** (documented here the same
way ``gcu_fit``'s docstring documents its own rank-selection simplification):
this class only computes *one* GCU view and *one* LVSE view (matching the
mixin's tunable-rank simplification already used for ``gcu``/``lvse`` as
preprocessing steps), not the paper's multiresolution rank/region family, so
there are 2 raw candidate predictions available, not the paper's 8. We
therefore implement a **single-triplet reduction** of the exact merge
formula above:

    p_h  = anchor  = GCU-view TabPFN prediction
    p_l  = local   = LVSE-view TabPFN prediction
    p_-  = negative = p_h   (no third independent candidate is available
                              with only 2 views, so the negative role is set
                              equal to the anchor -- this is a deliberate
                              choice, not an attempt to approximate the
                              paper's actual negative-role candidate)

Substituting p_- = p_h into the paper's own formula collapses it exactly to:

    p_triplet = -lambda*p_h + lambda*p_l + p_h = (1 - lambda) * p_h + lambda * p_l

i.e. a convex blend of the two views' predictions weighted by ``lambda``
(exposed as the ``lam`` hyperparameter, default 0.75 -- the paper's middle
triplet weight). We then take ``p_ref = p_h`` and ``alpha = 1`` (since the
paper's alpha is only defined in terms of the 8-triplet agreement structure
we do not reproduce), which makes the outer aggregation collapse to
``p_hat = p_triplet`` directly. Net effect: ``RamanPFNModel``'s final
prediction is ``(1 - lam) * pred_gcu + lam * pred_lvse`` for regression, and
the analogous convex blend of class-probability vectors (already exactly on
the probability simplex for ``lam`` in ``[0, 1]``, since it is a convex
combination of two valid probability distributions -- so the paper's
"projection onto the probability simplex" is applied defensively for
safety/generality only, in case ``lam`` is ever overridden outside
``[0, 1]``) for classification.

References
----------
Pan et al., "RamanPFN: learning from Raman spectral structure with a
tabular foundation model", arXiv:2608.02157.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase
from raman_bench.preprocessing.raman_preprocessing import (
    gcu_fit,
    gcu_transform,
    lvse_fit,
    lvse_transform,
)


def _project_to_simplex(proba):
    """Clip negatives to 0 and renormalize rows to sum to 1.

    A no-op for any row already on the simplex (true for a convex blend of
    two valid probability vectors with ``lam`` in ``[0, 1]``); kept as an
    explicit defensive step so this matches the paper's stated classification
    adaptation ("projection onto the probability simplex") even if ``lam``
    is overridden outside that range.
    """
    proba = np.clip(proba, 0.0, None)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return proba / row_sums


class RamanPFNModel(BaseEstimator):
    """RamanPFN — frozen TabPFN over GCU + LVSE views, merged via a signed blend.

    See the module docstring for the exact paper formula reproduced, and the
    explicit single-triplet simplification used here (2 views instead of the
    paper's 8-candidate multiresolution triplet ensemble).

    Parameters
    ----------
    rho : int
        GCU (NMF) rank -- see ``gcu_fit``. Default 16, matching the
        preprocessing mixin's ``prep_gcu_rho`` default.
    n_regions : int
        LVSE number of contiguous wavenumber-axis regions -- see
        ``lvse_fit``. Default 16, matching ``prep_lvse_n_regions``.
    k_per_region : int
        LVSE SVD components kept per region -- see ``lvse_fit``. Default 4,
        matching ``prep_lvse_k_per_region``.
    lam : float
        The triplet weight ``lambda`` from the paper's merge formula (one of
        ``{1, 0.75, 0.5}`` in the paper); this implementation exposes it as a
        continuous tunable hyperparameter defaulting to 0.75 (the paper's
        middle value). Controls the GCU/LVSE blend: final prediction is
        ``(1 - lam) * pred_gcu + lam * pred_lvse``.
    n_estimators : int
        Forwarded to ``TabPFNClassifier``/``TabPFNRegressor``.
    random_state : int | None
        Forwarded to ``gcu_fit`` and to both TabPFN backbones.
    device : str
        Forwarded to ``TabPFNClassifier``/``TabPFNRegressor`` (``"auto"``
        picks GPU when available).
    ignore_pretraining_limits : bool
        Forwarded to ``TabPFNClassifier``/``TabPFNRegressor``. TabPFN itself
        already raises a clear ``ValueError`` when the row count exceeds
        ``MAX_NUMBER_OF_SAMPLES`` (10_000) or the class count exceeds
        ``MAX_NUMBER_OF_CLASSES`` (10) unless this is set ``True`` -- see
        ``tabpfn.inference_config.InferenceConfig``. Left ``False`` (raise)
        by default rather than silently subsampling, matching the model-agent
        guidance to check what the raw ``tabpfn`` package does before adding
        a custom guard: it already does the right thing (a clear, catchable
        error) on its own. GCU/LVSE feature counts (``rho`` /
        ``n_regions * k_per_region``) are always far below TabPFN's 500
        feature cap, so only the row/class caps are ever relevant here.
    """

    def __init__(
        self,
        rho=16,
        n_regions=16,
        k_per_region=4,
        lam=0.75,
        n_estimators=8,
        random_state=None,
        device="auto",
        ignore_pretraining_limits=False,
    ):
        self.rho = rho
        self.n_regions = n_regions
        self.k_per_region = k_per_region
        self.lam = lam
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits

    def _make_backbone(self, is_classifier):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        kwargs = dict(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        if is_classifier:
            return TabPFNClassifier(**kwargs)
        return TabPFNRegressor(**kwargs)

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        X_np = np.asarray(X_np, dtype=np.float64)
        y_arr = np.asarray(y)

        if np.issubdtype(y_arr.dtype, np.floating):
            self.problem_type_ = "regression"
        else:
            unique = np.unique(y_arr)
            self.problem_type_ = "binary" if len(unique) == 2 else "multiclass"
        is_classifier = self.problem_type_ in ("binary", "multiclass")
        if is_classifier:
            self.classes_ = np.unique(y_arr)

        # Two internal, target-free, fold-safe representation views -- fit
        # directly via the raw gcu_fit/lvse_fit functions (NOT the
        # RamanPreprocessingMixin path: this model needs both views
        # simultaneously as internal features, computed from the *same*
        # input X, not as a single pipeline's single output).
        self._gcu_shift, self._gcu_nmf, W_gcu = gcu_fit(
            X_np, rho=self.rho, random_state=self.random_state
        )
        self._lvse_state, W_lvse = lvse_fit(
            X_np, n_regions=self.n_regions, k_per_region=self.k_per_region
        )

        self._backbone_gcu = self._make_backbone(is_classifier)
        self._backbone_lvse = self._make_backbone(is_classifier)
        self._backbone_gcu.fit(W_gcu, y_arr)
        self._backbone_lvse.fit(W_lvse, y_arr)

        return self

    def _views(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        X_np = np.asarray(X_np, dtype=np.float64)
        W_gcu = gcu_transform(X_np, self._gcu_shift, self._gcu_nmf)
        W_lvse = lvse_transform(X_np, self._lvse_state)
        return W_gcu, W_lvse

    def predict(self, X):
        if self.problem_type_ in ("binary", "multiclass"):
            proba = self.predict_proba(X)
            if proba.ndim == 1:
                # binary: proba is P(positive class)
                idx = (proba >= 0.5).astype(int)
                return self.classes_[idx]
            return self.classes_[np.argmax(proba, axis=1)]

        W_gcu, W_lvse = self._views(X)
        pred_gcu = np.asarray(self._backbone_gcu.predict(W_gcu), dtype=np.float64)
        pred_lvse = np.asarray(self._backbone_lvse.predict(W_lvse), dtype=np.float64)
        # p_triplet = (1 - lam) * p_h + lam * p_l, with p_h = GCU (anchor),
        # p_l = LVSE (local) -- see module docstring.
        return (1.0 - self.lam) * pred_gcu + self.lam * pred_lvse

    def predict_proba(self, X):
        W_gcu, W_lvse = self._views(X)
        proba_gcu = np.asarray(self._backbone_gcu.predict_proba(W_gcu), dtype=np.float64)
        proba_lvse = np.asarray(self._backbone_lvse.predict_proba(W_lvse), dtype=np.float64)

        # Both backbones are fit on the same y, but align columns to
        # self.classes_ defensively in case either backbone's own
        # classes_ ordering differs (e.g. a class missing an internal split).
        proba_gcu = self._align_proba(proba_gcu, self._backbone_gcu.classes_)
        proba_lvse = self._align_proba(proba_lvse, self._backbone_lvse.classes_)

        blended = (1.0 - self.lam) * proba_gcu + self.lam * proba_lvse
        blended = _project_to_simplex(blended)

        if self.problem_type_ == "binary":
            pos_idx = list(self.classes_).index(self.classes_[-1])
            return blended[:, pos_idx]
        return blended

    def _align_proba(self, proba, backbone_classes):
        if list(backbone_classes) == list(self.classes_):
            return proba
        aligned = np.zeros((proba.shape[0], len(self.classes_)))
        backbone_classes = list(backbone_classes)
        for j, cls in enumerate(self.classes_):
            if cls in backbone_classes:
                aligned[:, j] = proba[:, backbone_classes.index(cls)]
        return aligned


class _RamanPFNBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanPFNModel
    ag_key = "RAMANPFN"
    ag_name = "RamanPFN"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "rho": space.Categorical(8, 16, 32, 64),
            "n_regions": space.Categorical(8, 16, 24, 32),
            "k_per_region": space.Categorical(2, 4, 6, 8),
            "lam": space.Categorical(0.5, 0.75, 1.0),
        }


class Prep_RAMANPFN(_NoAugBase, _RamanPFNBridge):  # noqa: N801
    """RamanPFN wrapped with ``RamanPreprocessingMixin``.

    GCU and LVSE feature computation happens *inside* ``RamanPFNModel.fit()``/
    ``predict()`` itself (see that class), not via the mixin's ``gcu``/``lvse``
    preprocessing steps. ``prep_gcu_enabled``/``prep_lvse_enabled`` therefore
    MUST stay at their mixin default (``False``) here -- turning them on would
    silently double-apply GCU/LVSE (once via the mixin's own preprocessing
    pipeline before ``RamanPFNModel`` ever sees the data, and again inside
    ``RamanPFNModel.fit()`` itself on whatever representation-replaced output
    the mixin produced, which is not spectra anymore and would make a second
    GCU/LVSE pass meaningless). This class does not force that off explicitly
    because it is already the mixin's default for every model and
    ``_optimize_preprocessing`` is left at its own default (``False``, not set
    here), so the preprocessing search space -- and therefore
    ``prep_gcu_enabled``/``prep_lvse_enabled`` -- is never exposed to HPO for
    this model in the first place.

    Other, shape-preserving preprocessing steps (SNV, baseline correction,
    cosmic-ray removal, derivatives, ...) that would run *before* GCU/LVSE
    even start are still meaningfully composable ahead of this model, exactly
    as the paper's own pipeline includes an optional SNV pre-step ahead of
    GCU/LVSE -- so those stay available exactly like for any other model, and
    are left off by default (via ``_NoAugBase`` -> ``RamanPreprocessingMixin``
    defaults) but can be enabled through ordinary ``prep_*_enabled``
    hyperparameters if a user wants to explore them for this model.
    """
