"""Model registry for RamanBench.

Built-in AutoGluon models and 9 Raman-specific neural architectures are all
wrapped with :class:`~raman_bench.preprocessing.mixin.RamanPreprocessingMixin`
and available via :data:`~raman_bench.preprocessing.wrapped_models.PREPROCESSED_MODELS`.

Raman-specific architectures
-----------------------------
- :class:`~raman_bench.models.custom.deepcnn.DeepCNNModel`
  — Deep CNN (Liu et al., 2017)
- :class:`~raman_bench.models.custom.ramannet.RamanNetModel`
  — RamanNet (Ibtehaz et al., 2023)
- :class:`~raman_bench.models.custom.sanet.SANetModel`
  — SANet (Deng et al., 2021)
- :class:`~raman_bench.models.custom.ramanformer.RamanFormerModel`
  — RamanFormer (Koyun et al., 2024)
- :class:`~raman_bench.models.custom.ramantransformer.RamanTransformerModel`
  — RamanTransformer / ViT-style (Liu et al., 2023)
- :class:`~raman_bench.models.custom.rezeronet.ReZeroNetModel`
  — ReZeroNet (Lange et al., 2025)
- :class:`~raman_bench.models.custom.fcresnext.FCResNeXtModel`
  — FC-ResNeXt (Lange et al., 2025)
- :class:`~raman_bench.models.custom.coatnet.CoAtNetModel`
  — CoAtNet adapted for 1-D spectra (Lange et al., 2025)
- :class:`~raman_bench.models.custom.pls.PLSModel`
  — Partial Least Squares regression
"""
