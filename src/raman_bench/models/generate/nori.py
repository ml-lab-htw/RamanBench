"""NORI: rebind TabArena's own (empty -- Nori has no tunable HPO surface, see
upstream's ``can_hpo=False``) search space onto ``Prep_NORI``.

``NoriModel`` lives only in TabArena's own package (``tabarena.models.nori.model``)
-- there is no AutoGluon-core counterpart. See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import and its ``ag_key``
override (``"TA-NORI"`` -> ``"NORI"``).

Nori is a **regression-only** in-context-learning foundation model
(``NoriModel.supported_problem_types() == ["regression"]``; ``_fit`` raises
``AssertionError`` for anything else) -- ``NORI`` is listed in
``wrapped_models.REGRESSION_ONLY_MODELS`` (the mirror of
``CLASSIFICATION_ONLY_MODELS``, which ROCKET/ARSENAL/TABPFN-WIDE/ORIONMSP use the
other way around). A full-benchmark run for this model should only target
``configs/datasets/regression_all.json`` -- not build a target list from
``classification_all.json`` at all, since every classification target would fail
fast with that same AssertionError.

Caps ``max_rows`` at 100_000 (its in-context-window limit) -- checked against
RamanBench's own precomputed dataset stats (``data/precomputed/dataset_stats.json``):
the largest regression dataset (``sugar_mixtures_low_snr``) has 7,840 rows,
nowhere near the cap, so no override was applied (see ``wrapped_models.py``'s
batch-3 comment block).

Only ``NoriModel`` (the base ~variant) is wired here, not upstream's separate
``Nori30MModel``/``gen_nori30m`` -- a distinct, smaller checkpoint TabArena tracks
as its own leaderboard entry. Can be onboarded the same way later if wanted.
"""

from __future__ import annotations

from tabarena.models.nori.hpo import gen_nori as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_NORI

gen_nori = rebind_tabarena_generator(_upstream, require_available(Prep_NORI, "NORI"))
