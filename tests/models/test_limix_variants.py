"""LimiX-2-1016 (``Prep_LIMIX2_1016``) and LimiX-2M (``Prep_LIMIX_2M``): new LimiX checkpoints with their own keys, so their
results never share a folder with LIMIX2 / LIMIX. See ``wrapped_models.py`` for the reasoning.

Fitting needs a GPU (LimiX-2 imports ``triton``); these tests check the wiring that a fit relies on.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

pytest.importorskip("tabarena")

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_experiment():
    spec = importlib.util.spec_from_file_location("_run_experiment_limix", REPO_ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _classes():
    from raman_bench.preprocessing import wrapped_models as wm

    if wm.Prep_LIMIX2_1016 is None or wm.Prep_LIMIX_2M is None:
        pytest.skip("LimiX models unavailable in this tabarena build")
    return wm.Prep_LIMIX2_1016, wm.Prep_LIMIX_2M


def test_own_keys_and_result_folders():
    from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS, Prep_LIMIX, Prep_LIMIX2

    v1016, v2m = _classes()
    assert PREPROCESSED_MODELS["LIMIX2-1016"] is v1016 and PREPROCESSED_MODELS["LIMIX-2M"] is v2m
    names = {c.ag_name for c in (v1016, v2m, Prep_LIMIX2, Prep_LIMIX)}
    keys = {c.ag_key for c in (v1016, v2m, Prep_LIMIX2, Prep_LIMIX)}
    assert len(names) == 4 and len(keys) == 4  # a shared folder would let run_cached return the other checkpoint's folds


def test_generators_resolve(run_experiment):
    from raman_bench.experiment_utils import bag_experiment_kwargs
    from raman_bench.models.registry import infer_model_cls

    for key in ("LIMIX2-1016", "LIMIX-2M"):
        model_cls = infer_model_cls(key)
        gen = run_experiment._import_generator(key)
        assert gen.model_cls is model_cls
        experiments = gen.generate_all_bag_experiments(
            **bag_experiment_kwargs(num_random_configs=0, time_limit=60, num_bag_folds=2, scratch_dir=None)
        )
        assert experiments[0].method_kwargs["model_cls"] is model_cls


def test_1016_downloads_its_pinned_checkpoint(monkeypatch):
    import huggingface_hub

    v1016, _ = _classes()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/LimiX-2-1016.ckpt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert v1016.prefetch_weights() == "/cache/LimiX-2-1016.ckpt"
    assert calls[0]["repo_id"] == "stable-ai/LimiX-2-1016" and calls[0]["filename"] == "LimiX-2-1016.ckpt"
    assert len(calls[0]["revision"]) == 40  # pinned, so the file never changes under a run


def test_2m_defaults_to_its_checkpoint_and_configs(monkeypatch):
    _, v2m = _classes()
    monkeypatch.setattr(v2m, "prefetch_weights", classmethod(lambda cls: "/cache/LimiX-1_2M.ckpt"))
    def params(problem_type, **hyperparameters):
        model = v2m(problem_type=problem_type, hyperparameters=hyperparameters or None)
        model._init_params()  # as before a fit: the user's hyperparameters become the model's parameters
        return model._get_model_params()

    cls_params, reg_params = params("multiclass"), params("regression")
    assert cls_params["model_path"] == reg_params["model_path"] == "/cache/LimiX-1_2M.ckpt"
    assert cls_params["inference_config"] != reg_params["inference_config"]  # the classification and the regression config
    custom = params("regression", inference_config=[{"x": 1}])
    assert custom["inference_config"] == [{"x": 1}]  # an explicit setting wins


def test_class_caps_lifted_and_picklable():
    for cls in _classes():
        aux = cls(problem_type="multiclass")._get_default_auxiliary_params()
        assert aux["max_classes"] is None and aux["max_rows"] is None and aux["max_features"] is None
        assert pickle.loads(pickle.dumps(cls)) is cls


def test_leaderboard_and_scope_entries():
    import json

    from raman_bench.plotting import models as model_info

    scope = json.loads((REPO_ROOT / "configs" / "v1" / "scope_default.json").read_text())
    gpu = json.loads((REPO_ROOT / "cluster" / "gpu_models.json").read_text())
    for key in ("LIMIX2-1016", "LIMIX-2M", "FTFM"):
        assert key in gpu, key
        assert model_info.category(key) == "Tabular Foundation"
        assert model_info.provider(key) != key  # an organisation, not the model's own key
    assert model_info.provider("LIMIX2-1016") == model_info.provider("LIMIX2") == model_info.provider("LIMIX-2M")
    assert "LimiX-2" in model_info.display_name("LIMIX2-1016") and model_info.display_name("LIMIX-2M") == "LimiX-2M"
    assert "LIMIX2" not in scope["models"] and "LIMIX2-1016" in scope["models"]  # the new checkpoint replaces the old
