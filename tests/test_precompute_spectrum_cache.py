"""precompute_spectrum_cache.resolved_params gives the parameters a model applies, so the
precomputed step keys match the ones computed at run time."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import precompute_spectrum_cache as pc  # noqa: E402

from raman_bench.preprocessing.spectrum_cache import step_key  # noqa: E402


def test_recipe_defaults_and_shared_key(tmp_path):
    recipe = tmp_path / "sg_rubberband_snv.json"
    recipe.write_text(json.dumps({"preprocessing": {"denoising": True, "rubberband": True, "snv": True}}))
    _, p_pls = pc.resolved_params("PLS", str(recipe))
    _, p_ridge = pc.resolved_params("RIDGE", str(recipe))
    assert p_pls["prep_denoise_enabled"] and p_pls["prep_rubberband_enabled"]
    assert p_pls["prep_denoise_wl"] == 9 and p_pls["prep_denoise_po"] == 3  # mixin defaults
    assert not p_pls.get("prep_airpls_enabled", False)
    wn = np.linspace(100, 3000, 50)
    assert step_key(p_pls, wn) == step_key(p_ridge, wn) is not None
