from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ta_mitra_v2.model import Prep_MITRA_V2

# Mirrors tabarena.models.mitra_v2.hpo.gen_mitra_v2: Mitra-v2's fine-tuning recipe is
# frozen (see tabarena.models.mitra_v2._internal.recipe) -- the default configuration
# is the method, so the search space is empty and the only config is the default.
#
# max_features_budget=1024 (raised from tabarena's own default of 256, see
# tabarena.models.mitra_v2._internal.recipe.MAX_FEATURES_BUDGET): confirmed real CUDA
# memory-fragmentation OOM warnings on a k8s A100 pod (2026-09-29) even with the
# default 256-feature "wide table reducer" already active (log evidence: "Mitra-v2:
# reducing 724 features to 256 (select)" immediately preceding the OOM). Deliberately
# NOT tightened further -- the primary fix is configs/v1/scope_default.json's new
# TA-MITRA-V2 row cap (3000, matching MITRA v1, previously absent entirely), which
# Mitra-v2's own activation-memory estimate (activation_mem ~ n_rows * n_features)
# says should dominate the OOM risk over feature count alone. Raising the budget to
# 1024 instead preserves full/near-full spectra for every routine-sweep Raman dataset
# under ~1024 features (i.e. most of them -- the reducer only activates above this
# threshold), trading a larger safety margin on rows for less information loss on
# features; only genuinely very wide datasets (already routed to the large/H100-H200
# pod, see large_datasets) still get reduced. Revisit if OOMs recur even with the row
# cap in place -- lowering this budget is the next lever, not the first one.
gen_ta_mitra_v2 = ConfigGenerator(
    model_cls=Prep_MITRA_V2,
    search_space={},
    manual_configs=[{"max_features_budget": 1024}],
)
