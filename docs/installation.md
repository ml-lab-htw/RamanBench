# Installation

Three install levels, from lightest to full reproducibility. Start with Option 1.

## Option 1: datasets and leaderboard (recommended starting point)

```bash
pip install raman-bench
```

This is enough to:

- load every dataset (via `raman-data`)
- read the per-fold v1 results of all 69 leaderboard models (and the two AutoGluon reference systems), which ship with the
  package (`raman_bench.compare.load_reference()`, works offline)
- use the [preprocessing transformers and recipes](preprocessing.md)

With `pip install "raman-bench[plots]"` you can also score the results and rank your own
results against them (`raman_bench.compare`, see [Compare your model](compare-your-model.md)).
Running a model on the benchmark protocol needs TabArena and AutoGluon (Option 3).

**Versions.** The package's major version is the benchmark version: every `raman-bench 1.x`
release runs and scores RamanBench v1. 2.0.0 and 2.0.1 predate that rule and are benchmark v1
too (2.0.1 is yanked on PyPI). Pin the benchmark with `raman-bench>=1.2,<2`; the rules are in
[RELEASING.md](../RELEASING.md).

## Option 2: with all built-in models

Adds all Raman-specific architectures and standalone tabular foundation models,
all with a standard `fit(X, y)` / `predict(X)` interface:

```bash
pip install "raman-bench[models]"
```

This adds `torch`, `tabpfn`, `pytabkit`, `tabdpt`, `sktime`, `aeon`, and `ramanspy` to
the core package. AutoGluon is not needed for this path.

`tabarena` itself, and three wrapped models built on it (`Prep_TABFM`,
`Prep_SAP_RPT_OSS`, `Prep_ORIONMSP`), have git-only upstreams and cannot ship
in a PyPI package. Install them separately if you need them:

```bash
pip install -r requirements-git.txt
```

Until then, every `Prep_*` class that wraps a `tabarena.models.*` model
(which is most of them) raises a clear "not available" error; the
Raman-specific standalone architectures work without this step.

Some models need their own install because their dependencies conflict with the
main one (for example Lattice, which pins `torch==2.7.1`, and LimiX-2). They are in
`requirements-git-nodeps.txt`, installed with `pip install --no-deps -r`; its header says how.

## Option 3: full benchmark reproducibility

The paper's benchmark runs all models through AutoGluon's automated
preprocessing and HPO pipeline, on plain upstream AutoGluon (>=1.6.1, no fork):

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
uv pip install --prerelease=allow -e ".[models]"   # uv resolves bencheval automatically;
                                                    # plain pip needs it installed first,
                                                    # see the note in pyproject.toml
pip install -r requirements-git.txt                # tabarena is git-pinned, not on PyPI
```

### What replaced the AutoGluon fork

RamanBench previously depended on a patched AutoGluon fork to work around two limitations
of AutoGluon 1.5:

1. **Feature cap.** AutoGluon caps tabular foundation models (TabPFN v2, TabICL, TabDPT,
   Mitra) at 500 to 2000 features (varies by model); Raman spectra typically have 500 to
   4000 wavenumber points. RamanBench now lifts this cap itself, per model, using AutoGluon's
   own supported `_default_auxiliary_params_extra` subclass extension point; see `Prep_MITRA`,
   `Prep_TABDPT`, `Prep_TABICL`, `Prep_REALTABPFN_V2` and `Prep_REALTABPFN_V25` in
   `preprocessing/wrapped_models.py`. No fork needed. Datasets with more than 10 classes go
   through `tabpfn-extensions`' output-coding `ManyClassClassifier`, as the fork did:
   RamanBench applies it itself for the TabPFN v2/v2.5/v2.6 wrappers, and TabArena does for
   LightPFN and TabFM+.
2. **TabICL v2 regression.** AutoGluon 1.5 shipped TabICL v1 (classification only).
   AutoGluon 1.6 upgraded to TabICL v2 natively, adding regression support, so nothing needs
   overriding any more.
