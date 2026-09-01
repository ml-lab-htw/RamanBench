"""Per-model :class:`~tabarena.utils.config_utils.ConfigGenerator` modules.

Two kinds of model live here:

- **TabArena-native "built-in" AutoGluon models** (``rf.py``, ``cat.py``,
  ``xgb.py``, ``xt.py``, ``knn.py``, ``lr.py``, ``nn_torch.py``, ``fastai.py``,
  ``dummy.py``, ``realmlp.py``, ``mitra.py``, ``tabm.py``, ``tabdpt.py``,
  ``tabicl.py``, ``realtabpfn_v2.py``, ``realtabpfn_v25.py``): these already have
  a ``Prep_*`` class in ``preprocessing/wrapped_models.py`` (no per-model
  ``model.py`` of their own -- they're thin Raman-preprocessing subclasses of an
  existing AutoGluon model, not a from-scratch architecture). Most of these just
  rebind TabArena's own maintained search space
  (``tabarena.models.<key>.hpo.gen_<key>``) onto the ``Prep_*`` class via
  ``_tabarena_adapter.rebind_tabarena_generator`` -- see that module's docstring
  for why a straight rebind isn't always safe (``knn.py``) and which models have
  no upstream generator to reuse at all (``dummy.py``, ``realtabpfn_v2.py``).
- Any *other* future model added the old flat-file way, before being migrated to
  the newer ``raman_bench/models/custom/<key>/{model.py,hpo.py,info.py}``
  per-directory convention (see ``models/custom/ridge/`` for the reference
  implementation and ``RamanBench/.claude/agents/model-agent.md`` for the
  onboarding workflow) -- new custom *architectures* should use that convention
  instead, not this one.

Every RamanBench custom architecture that used to live here (``pls``, ``deepcnn``,
``ramannet``, ``sanet``, ``ramanformer``, ``ramantransformer``, ``rezeronet``,
``fcresnext``, ``coatnet``, ``rocket``, ``tabpfn_wide``, ``gbm``,
``ta_tabpfn_3``) has already moved to that per-directory convention.
"""
