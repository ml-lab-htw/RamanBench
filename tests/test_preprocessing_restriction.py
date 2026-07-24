"""Tests that a config's ``preprocessing`` dict is applied as a fixed
specification — steps set ``True`` are force-*enabled* and steps set ``False``
are force-*disabled* — overriding each model's class defaults.

Regression guard for the fix in ``AutoGluonModel._build_model_hyperparameters``:
previously only force-*disable* (and augmentation force-enable) were honoured, so
enabling a step that a model's class default left off (e.g. MSC on Prep_RF, or
SNV on a tree model) was silently a no-op.
"""

from raman_data import TASK_TYPE

from raman_bench.config import _ALL_PREPROCESSING_STEPS
from raman_bench.model import AutoGluonModel
from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS


def _prep_config(**enabled) -> dict:
    """A normalized preprocessing_config dict: named steps True, rest False."""
    return {step: bool(enabled.get(step, False)) for step in _ALL_PREPROCESSING_STEPS}


def _params_for(model: str, preprocessing_config: dict) -> dict:
    """Build hyperparameters for a single model and return its params dict."""
    m = AutoGluonModel(
        models=[model],
        ensemble=False,
        optimize=False,
        task_type=TASK_TYPE.Regression,
        preprocessing_config=preprocessing_config,
    )
    hp = m._build_model_hyperparameters()
    cls = PREPROCESSED_MODELS[model.upper()]
    assert cls in hp, f"{model} not in built hyperparameters"
    return hp[cls]


def test_snv_only_forces_pls_bl_off_and_snv_on():
    # Prep_PLS class default = baseline_correction + denoising + SNV. An
    # SNV-only config must switch OFF baseline correction and denoising while
    # keeping SNV on.
    params = _params_for("PLS", _prep_config(snv=True))
    assert params.get("prep_snv_enabled") is True
    assert params.get("prep_bl_enabled") is False
    assert params.get("prep_denoise_enabled") is False


def test_snv_only_force_enables_on_tree_model():
    # Prep_RF class default = no preprocessing. Force-enabling SNV must take
    # effect (this is the case the old code got wrong).
    params = _params_for("RF", _prep_config(snv=True))
    assert params.get("prep_snv_enabled") is True
    assert params.get("prep_bl_enabled") is False


def test_msc_force_enabled_on_tree_model():
    # MSC is in no model's class default, so it exercises force-enable purely.
    params = _params_for("RF", _prep_config(msc=True))
    assert params.get("prep_msc_enabled") is True


def test_none_disables_all_pls_defaults():
    params = _params_for("PLS", _prep_config())
    assert params.get("prep_bl_enabled") is False
    assert params.get("prep_denoise_enabled") is False
    assert params.get("prep_snv_enabled") is False


def test_augmentation_not_forced_on_unsupported_model():
    # PLS has _supports_augmentation=False, so augmentation must stay off even
    # when the config enables it.
    params = _params_for("PLS", _prep_config(augmentation=True))
    assert params.get("prep_aug_enabled") is not True


def test_none_config_carries_no_prep_overrides():
    # preprocessing_config=None → no restriction applied. The class defaults
    # (Prep_PLS: bl + denoise + snv) are set later by AutoGluon at model
    # instantiation via _set_default_params, so the hyperparameters dict must
    # carry NO prep_*_enabled overrides — leaving those defaults intact.
    params = _params_for("PLS", None)
    assert not any(k.endswith("_enabled") for k in params), params


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
