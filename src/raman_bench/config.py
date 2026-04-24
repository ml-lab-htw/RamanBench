"""Configuration loading and validation for the benchmark pipeline."""
import json
import os


_ALL_PREPROCESSING_STEPS = {
    "baseline_correction": True,
    "cosmic_ray_removal": True,
    "msc": True,
    "denoising": True,
    "snv": True,
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
        config["preprocessing_config"] = {
            k: raw.get(k, False) for k in _ALL_PREPROCESSING_STEPS
        }
    else:
        config["preprocessing_config"] = None

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
    with open(path, "r") as f:
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
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _normalize_preprocessing_config(config)

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
