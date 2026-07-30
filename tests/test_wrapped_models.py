"""Every ``Prep_*`` class must carry ``ag_key``/``ag_name``.

Regression guard for the TabArena-based refactor: ``ConfigGenerator.__init__``
(tabarena.utils.config_utils) asserts ``model_cls.ag_key``/``ag_name`` are set,
since they're used to build the config-pool experiment names (e.g.
``PLS_c1_BAG_L1``). Built-in AutoGluon-backed models inherit these from their
AutoGluon base class; the custom Raman/bridge-backed models needed them added
explicitly.
"""

from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS


def test_all_models_have_ag_key_and_ag_name():
    missing = []
    for key, cls in PREPROCESSED_MODELS.items():
        ag_key = getattr(cls, "ag_key", None)
        ag_name = getattr(cls, "ag_name", None)
        if not ag_key or not ag_name:
            missing.append((key, cls.__name__, ag_key, ag_name))
    assert not missing, f"Models missing ag_key/ag_name: {missing}"
