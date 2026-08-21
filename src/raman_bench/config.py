"""Configuration loading and validation for the benchmark pipeline."""

import json
import os

_ALL_PREPROCESSING_STEPS = {
    "crop": True,
    "baseline_correction": True,
    "airpls": True,
    "arpls": True,
    "rubberband": True,
    "cosmic_ray_removal": True,
    "msc": True,
    "emsc": True,
    "denoising": True,
    "wavelet_denoise": True,
    "derivative": True,
    "snv": True,
    "vecnorm": True,
    "augmentation": True,
    "standard_scaling": True,
}


def _normalize_preprocessing_config(config):
    """Normalize the preprocessing field into a preprocessing_config dict.

    Handles three forms:
    - ``True``  → all steps enabled
    - ``False`` → no preprocessing (``None``)
    - ``dict``  → use as-is (missing keys default to ``False``)
    """
    raw = config.get("preprocessing", False)

    if isinstance(raw, bool):
        if raw:
            config["preprocessing_config"] = dict(_ALL_PREPROCESSING_STEPS)
        else:
            config["preprocessing_config"] = None
    elif isinstance(raw, dict):
        config["preprocessing_config"] = {k: raw.get(k, False) for k in _ALL_PREPROCESSING_STEPS}
    else:
        config["preprocessing_config"] = None

    return config


def _collect_known_prep_param_names():
    """Collect every known ``prep_*`` hyperparameter name across all steps.

    Pools the keys of each step's ``defaults`` dict in
    ``_PREP_STEP_DEFINITIONS`` (mixin.py) into one lookup set. Imported
    lazily (only when ``preprocessing_params`` overrides are actually being
    validated) since the mixin module requires autogluon, and config.py
    otherwise has no autogluon dependency.
    """
    from raman_bench.preprocessing.mixin import _PREP_STEP_DEFINITIONS

    names = set()
    for step_def in _PREP_STEP_DEFINITIONS.values():
        names.update(step_def["defaults"].keys())
    return names


def _normalize_preprocessing_params(config):
    """Normalize/validate the optional ``preprocessing_params`` override dict.

    ``preprocessing_params`` is a flat ``param_name -> value`` dict (e.g.
    ``{"prep_deriv_order": 2, "prep_deriv_wl": 15}``) that overrides a
    step's own hyperparameter values, independently of the ``preprocessing``
    / ``preprocessing_config`` enable/disable restriction. It is threaded
    through to ``AutoGluonModel`` and merged into each model's
    hyperparameters in ``model.py::_build_model_hyperparameters`` *after*
    the restriction enable/disable logic, so it can, e.g., force
    ``prep_deriv_order=2`` for a model whose class default (or the
    restriction) would otherwise leave it at 1 — but it cannot use an
    ``*_enabled`` key to flip a step on/off behind the restriction's back
    (see ``model.py`` for the exact precedence).

    Only names present in some step's ``defaults`` dict (collected via
    :func:`_collect_known_prep_param_names`) are accepted; anything else
    (most likely a typo) raises a clear ``ValueError`` at config-load time.
    """
    raw = config.get("preprocessing_params")
    if raw is None:
        config["preprocessing_params"] = None
        return config
    if not isinstance(raw, dict):
        raise TypeError(f"preprocessing_params must be a dict, got {type(raw)}")

    known = _collect_known_prep_param_names()
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"Unknown preprocessing_params key(s): {unknown}. "
            f"Must be one of the known prep_* hyperparameter names, e.g. "
            f"{sorted(known)[:8]}..."
        )

    config["preprocessing_params"] = dict(raw)
    return config


def _resolve_list_config(value, config_dir):
    """Resolve a list specification to a Python list or ``None``.

    Supports three modes:

    - ``"all"``  → returns ``None`` (sentinel meaning "load all at runtime")
    - ``list``   → returns the list as-is
    - ``str``    → file path relative to *config_dir*; loads a JSON array

    Parameters
    ----------
    value : str | list
        Raw value from the config file.
    config_dir : str
        Directory of the config file, used to resolve relative paths.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected str or list for list config, got {type(value)}")
    if value == "all":
        return None
    path = os.path.join(config_dir, value) if not os.path.isabs(value) else value
    with open(path) as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"List config file {path} must contain a JSON array")
    return items


def load_config(config_path=None):
    """Load and normalise a benchmark JSON config file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Loaded and normalised configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _normalize_preprocessing_config(config)
    config = _normalize_preprocessing_params(config)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key, resolved_key in [
        ("datasets_classification", "dataset_names_classification"),
        ("datasets_regression", "dataset_names_regression"),
    ]:
        if key in config:
            config[resolved_key] = _resolve_list_config(config[key], config_dir)
        else:
            config[resolved_key] = None

    if "models" in config:
        resolved = _resolve_list_config(config["models"], config_dir)
        if resolved is not None:
            config["models"] = resolved

    config.setdefault("log_level", "INFO")

    return config
