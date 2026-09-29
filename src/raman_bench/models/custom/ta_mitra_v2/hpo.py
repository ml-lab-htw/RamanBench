from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ta_mitra_v2.model import Prep_MITRA_V2

# Mirrors tabarena.models.mitra_v2.hpo.gen_mitra_v2: Mitra-v2's fine-tuning recipe is
# frozen (see tabarena.models.mitra_v2._internal.recipe) -- the default configuration
# is the method, so the search space is empty and the only config is the default.
#
# max_features_budget=128 (lowered from tabarena's own default of 256, see
# tabarena.models.mitra_v2._internal.recipe.MAX_FEATURES_BUDGET): confirmed real CUDA
# memory-fragmentation OOM warnings on a k8s A100 pod (2026-09-29) even with the
# default 256-feature "wide table reducer" already active (log evidence: "Mitra-v2:
# reducing 724 features to 256 (select)" immediately preceding the OOM). Mitra-v2's own
# activation-memory estimate scales linearly with feature count
# (tabarena.models.mitra_v2.model: activation_mem ~ n_rows * n_features), so halving
# the post-reduction feature count directly cuts memory pressure. Paired with
# configs/v1/scope_default.json's new TA-MITRA-V2 row cap (3000, matching MITRA v1) --
# the two axes are independent (this is the model's own internal column reducer, not
# RamanBench's row-subsampling), and only Mitra-v2 has this knob: plain MITRA (v1) has
# no equivalent feature-reduction mechanism, only a raised (not reduced) max_features
# pass/fail cap.
gen_ta_mitra_v2 = ConfigGenerator(
    model_cls=Prep_MITRA_V2,
    search_space={},
    manual_configs=[{"max_features_budget": 128}],
)
