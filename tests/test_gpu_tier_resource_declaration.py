"""Regression test for a real, previously-undiscovered production bug: none of
RamanBench's custom PyTorch architectures ever declared a GPU resource
requirement to AutoGluon's own resource manager (`AbstractModel.minimum_num_gpus`/
`default_num_gpus`/`gpu_required` all default to 0/0/False), so AutoGluon's
`SequentialLocalFoldFittingStrategy` allocated `gpus=0` to every one of them
regardless of the pod's actual GPU or `cluster/gpu_models.json`'s own GPU-tier
tagging -- confirmed empirically (2026-09-25, a peer session's real 18-task
cluster test) to actually run CPU-only, not just a bookkeeping mismatch.

Fixed via `raman_bench.preprocessing.bridge_bases._GPURequiredBridge`, a mixin
listed ahead of `SklearnAutoGluonBridge` in each GPU-tier custom architecture's
own `_XBridge(...)` class. This test locks in that every GPU-tier model (per
`cluster/gpu_models.json`) declares it, and that CPU-tier custom models sharing
the same `SklearnAutoGluonBridge` base (PLS, ROCKET, HYDRA, ...) do NOT.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("autogluon.core")

from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 9 custom architectures confirmed GPU-tier in cluster/gpu_models.json that
# share SklearnAutoGluonBridge (i.e. excludes TabArena-native GPU wrappers like
# TABDPT/MITRA, which get their GPU declaration from upstream tabarena/autogluon
# code, not from RamanBench's own bridge classes).
GPU_TIER_CUSTOM_MODELS = [
    "COATNET",
    "DEEPCNN",
    "FCRESNEXT",
    "RAMANFORMER",
    "RAMANNET",
    "RAMANPFN",
    "RAMANTRANSFORMER",
    "REZERONET",
    "SANET",
]

# A few CPU-tier custom models sharing the same SklearnAutoGluonBridge base --
# must NOT pick up the GPU requirement.
CPU_TIER_CUSTOM_MODELS = ["PLS", "ROCKET", "HYDRA"]


def _load_gpu_models_json() -> set[str]:
    with open(REPO_ROOT / "cluster" / "gpu_models.json") as f:
        data = json.load(f)
    return set(data if isinstance(data, list) else data.get("gpu_models", data))


def test_confirmed_gpu_tier_models_are_actually_in_gpu_models_json():
    gpu_models = _load_gpu_models_json()
    missing = set(GPU_TIER_CUSTOM_MODELS) - gpu_models
    assert not missing, f"Test fixture drifted from cluster/gpu_models.json: {missing}"


@pytest.mark.parametrize("model_key", GPU_TIER_CUSTOM_MODELS)
def test_gpu_tier_custom_model_declares_gpu_requirement(model_key):
    cls = PREPROCESSED_MODELS.get(model_key)
    if cls is None:
        pytest.skip(f"{model_key} unavailable in this tabarena/autogluon build")
    inst = cls.__new__(cls)
    assert getattr(inst, "minimum_num_gpus", 0) >= 1, (
        f"{model_key} does not declare minimum_num_gpus -- AutoGluon's own "
        "SequentialLocalFoldFittingStrategy will allocate gpus=0 to it regardless "
        "of the pod's actual GPU (confirmed to actually run CPU-only, not just a "
        "bookkeeping mismatch)."
    )
    assert getattr(inst, "default_num_gpus", 0) >= 1
    resources = inst.get_minimum_resources(is_gpu_available=True)
    assert resources.get("num_gpus", 0) >= 1


@pytest.mark.parametrize("model_key", CPU_TIER_CUSTOM_MODELS)
def test_cpu_tier_custom_model_unaffected(model_key):
    cls = PREPROCESSED_MODELS.get(model_key)
    if cls is None:
        pytest.skip(f"{model_key} unavailable in this tabarena/autogluon build")
    inst = cls.__new__(cls)
    assert getattr(inst, "minimum_num_gpus", 0) == 0
    resources = inst.get_minimum_resources(is_gpu_available=True)
    assert "num_gpus" not in resources
