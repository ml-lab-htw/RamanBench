"""Legacy home for per-model :class:`~tabarena.utils.config_utils.ConfigGenerator` modules.

Empty as of the full migration to the per-model-directory convention --
every model that used to live here (``pls``, ``deepcnn``, ``ramannet``,
``sanet``, ``ramanformer``, ``ramantransformer``, ``rezeronet``, ``fcresnext``,
``coatnet``, ``rocket``, ``arsenal``, ``tabpfn_wide``, ``gbm``, ``ta_tabpfn_3``)
now has its ``ConfigGenerator`` in
``raman_bench/models/custom/<key>/hpo.py`` instead, alongside its
``model.py``/``info.py`` -- see ``models/custom/ridge/`` for the reference
implementation and ``RamanBench/.claude/agents/model-agent.md`` for the
onboarding workflow.

Kept as an empty package (rather than deleted outright) only because
``scripts/run_experiment.py::_import_generator`` still falls back to
``raman_bench.models.generate.<key>`` for any *future* model added the old
way before being migrated -- new models should not use this path.
"""
