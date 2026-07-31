"""Per-model :class:`~tabarena.utils.config_utils.ConfigGenerator` definitions.

Each ``{model_key}.py`` mirrors TabArena's own ``models/{key}/generate.py``
convention: a module-level ``search_space`` (or ``search_space_func``) plus a
``gen_{model_key}`` :class:`~tabarena.utils.config_utils.ConfigGenerator` /
:class:`~tabarena.utils.config_utils.CustomAGConfigGenerator`. Calling
``gen_{model_key}.generate_all_bag_experiments(num_random_configs=N)`` produces
the default config (``_c1``) plus ``N`` random-search configs (``_r1..rN``) —
the pool that TabArena's ``EndToEnd``/``PaperRunTabArena`` machinery recycles
into default / tuned / tuned+ensemble results with zero extra fitting (see
``raman_bench.cluster`` for how the pool is fanned out across cluster jobs).

Present so far (the 11 Raman-specific architectures with no TabArena
equivalent, plus ``gbm`` as a worked example of reusing a TabArena built-in
model's own search space): ``pls``, ``deepcnn``, ``ramannet``, ``sanet``,
``ramanformer``, ``ramantransformer``, ``rezeronet``, ``fcresnext``,
``coatnet``, ``rocket``, ``arsenal``, ``tabpfn_wide``, ``gbm``. The same
pattern extends to the remaining AutoGluon-built-in-backed models in
:data:`~raman_bench.preprocessing.wrapped_models.PREPROCESSED_MODELS`
(XGB, CAT, RF, XT, KNN, LR, NN_TORCH, FASTAI, REALMLP, MITRA, TABM, TABDPT,
TABFM, TABICL, REALTABPFN-V2/V2.5/V2.6, TABPFN-V3, TABPFN-V3-THINKING) by
following ``gbm.py``'s example.

**New models should not be added here.** ``ridge`` moved to
``raman_bench/models/custom/ridge/hpo.py`` as the reference implementation of
the new per-model-directory onboarding convention (mirroring upstream
TabArena's own ``models/<key>/{model.py,hpo.py,info.py}`` layout since its
post-fork restructure) -- see ``RamanBench/.claude/agents/model-agent.md``.
Migrating the remaining entries in this directory to that convention is
tracked future work, not required for them to keep functioning.
"""
