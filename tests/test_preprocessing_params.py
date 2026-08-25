"""Tests for the ``preprocessing_params`` config-level hyperparameter override
mechanism.

``preprocessing_params`` is a flat ``param_name -> value`` dict, orthogonal
to ``preprocessing``/``preprocessing_config`` (which only controls each
step's ``*_enabled`` flag). It lets a config pin a step's own numeric/
categorical hyperparameter (e.g. ``prep_deriv_order``) without going through
HPO or a model class's ``_set_default_params`` default.
"""

import pytest
from raman_data import TASK_TYPE

pytest.importorskip("autogluon")

from raman_bench.config import (  # noqa: E402
    _ALL_PREPROCESSING_STEPS,
    _normalize_preprocessing_params,
)
from raman_bench.model import AutoGluonModel  # noqa: E402
from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS  # noqa: E402


def _prep_config(**enabled) -> dict:
    return {step: bool(enabled.get(step, False)) for step in _ALL_PREPROCESSING_STEPS}


def _params_for(
    model: str, preprocessing_config=None, preprocessing_params=None, optimize=False
) -> dict:
    m = AutoGluonModel(
        models=[model],
        ensemble=False,
        optimize=optimize,
        task_type=TASK_TYPE.Regression,
        preprocessing_config=preprocessing_config,
        preprocessing_params=preprocessing_params,
    )
    hp = m._build_model_hyperparameters()
    cls = PREPROCESSED_MODELS[model.upper()]
    assert cls in hp, f"{model} not in built hyperparameters"
    return hp[cls]


# ---------------------------------------------------------------------------
# config.py normalization/validation
# ---------------------------------------------------------------------------


def test_normalize_preprocessing_params_none():
    config = {}
    out = _normalize_preprocessing_params(config)
    assert out["preprocessing_params"] is None


def test_normalize_preprocessing_params_valid():
    config = {"preprocessing_params": {"prep_deriv_order": 2, "prep_deriv_wl": 15}}
    out = _normalize_preprocessing_params(config)
    assert out["preprocessing_params"] == {"prep_deriv_order": 2, "prep_deriv_wl": 15}


def test_normalize_preprocessing_params_rejects_typo():
    config = {"preprocessing_params": {"prep_derivv_order": 2}}
    with pytest.raises(ValueError, match="Unknown preprocessing_params"):
        _normalize_preprocessing_params(config)


def test_normalize_preprocessing_params_rejects_non_dict():
    config = {"preprocessing_params": [1, 2, 3]}
    with pytest.raises(TypeError):
        _normalize_preprocessing_params(config)


# ---------------------------------------------------------------------------
# _build_model_hyperparameters level (optimize=False, class-default path)
# ---------------------------------------------------------------------------


def test_override_reaches_merged_params_optimize_false():
    # Prep_PLS's class default already enables the derivative-adjacent step
    # only implicitly; force-enable derivative via the restriction and
    # confirm the numeric override (not the class/step default) reaches the
    # final params dict.
    params = _params_for(
        "PLS",
        preprocessing_config=_prep_config(derivative=True),
        preprocessing_params={"prep_deriv_order": 2, "prep_deriv_wl": 15},
    )
    assert params.get("prep_deriv_enabled") is True
    assert params.get("prep_deriv_order") == 2
    assert params.get("prep_deriv_wl") == 15


def test_override_applies_even_without_restriction():
    params = _params_for(
        "RF",
        preprocessing_config=None,
        preprocessing_params={"prep_emsc_poly_order": 6},
    )
    assert params.get("prep_emsc_poly_order") == 6


def test_override_does_not_flip_restriction_owned_enabled_flag():
    # The restriction explicitly disables derivative; an override attempting
    # to flip prep_deriv_enabled back on must be ignored (restriction wins
    # for the on/off decision).
    params = _params_for(
        "PLS",
        preprocessing_config=_prep_config(),  # everything False, incl. derivative
        preprocessing_params={"prep_deriv_enabled": True, "prep_deriv_order": 2},
    )
    assert params.get("prep_deriv_enabled") is False
    # The non-"_enabled" override on the same (disabled) step is still applied.
    assert params.get("prep_deriv_order") == 2


# ---------------------------------------------------------------------------
# _build_model_hyperparameters level (optimize=True, HPO search-space path)
# ---------------------------------------------------------------------------


def test_override_wins_over_hpo_search_space():
    from autogluon.common.space import Space

    params = _params_for(
        "PLS",
        preprocessing_config=_prep_config(derivative=True),
        preprocessing_params={"prep_deriv_order": 2},
        optimize=True,
    )
    assert params.get("prep_deriv_order") == 2
    assert not isinstance(params.get("prep_deriv_order"), Space)
