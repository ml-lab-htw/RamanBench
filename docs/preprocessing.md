# Preprocessing interface

RamanBench's Raman preprocessing is available in two forms that run the same code: as
scikit-learn transformers you put in a `Pipeline`, and as `prep_*` hyperparameters of the
benchmark models. A named recipe gives identical output in both. This page covers the
transformer interface (since 1.6.0). The steps and recipes come from the RamanPreprocessing
study.

It needs only the core install (`pip install raman-bench`); AutoGluon and TabArena are imported
only on the benchmark-model path.

## Quick start

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from raman_bench.preprocessing import RamanPreprocessor

model = make_pipeline(
    RamanPreprocessor({"baseline_correction": True, "snv": True, "avgpool": True}),
    PLSRegression(n_components=10),
)
model.fit(X_train, y_train)   # spectra are rows; X is an array or a DataFrame
```

`RamanPreprocessor(preprocessing, preprocessing_params, wavenumbers)` takes a recipe in the
recipe-config schema: a dict from step key to `True`/`False`, plus optional `prep_*` parameter
overrides. It runs the enabled steps in the benchmark's fixed order. Stateful steps (the MSC/EMSC
reference spectrum, standard scaling, GCU, LVSE) are fitted on the training spectra only.

If `X` is a DataFrame whose column labels are Raman shifts (cm^-1), `PhysicalCrop` reads its axis
from them; otherwise pass `wavenumbers=`.

## Named recipes

26 recipes of the preprocessing study are available by name:

```python
from raman_bench.preprocessing import recipe, recipe_names, recipe_description, recipe_hyperparameters

recipe_names()                        # the 26 names
recipe_description("bc_snv_avgpool")  # 'AsLS baseline -> SNV -> average pooling'
model = make_pipeline(recipe("bc_snv_avgpool"), PLSRegression(n_components=10))
```

| Recipe | Steps in run order |
|---|---|
| `none` | no preprocessing |
| `snv`, `msc`, `emsc`, `stdscale`, `sg`, `bc`, `gcu`, `lvse`, `avgpool`, `decimate` | the one step |
| `bc_snv` | AsLS baseline, SNV |
| `bc_sg` | Savitzky-Golay smoothing, AsLS baseline |
| `bc_sg_snv` | Savitzky-Golay smoothing, AsLS baseline, SNV |
| `bc_sg_msc` | Savitzky-Golay smoothing, AsLS baseline, MSC |
| `bc_snv_avgpool` | AsLS baseline, SNV, average pooling |
| `airpls_snv`, `arpls_snv` | airPLS or arPLS baseline, SNV |
| `crr_snv` | cosmic-ray removal, SNV |
| `sg_d1_snv`, `sg_d2_snv` | Savitzky-Golay first or second derivative, SNV |
| `sg_rubberband_snv` | Savitzky-Golay smoothing, rubberband baseline, SNV |
| `wavelet_bc_snv` | wavelet denoising, AsLS baseline, SNV |
| `all` | cosmic-ray removal, smoothing, AsLS baseline, MSC, SNV, standard scaling |
| `gcu_lvse` | GCU and LVSE, concatenated |
| `prep_ensemble` | SNV, EMSC, airPLS baseline then SNV, and no preprocessing, concatenated |

Other helpers: `recipe_definition(name)` (the recipe-config dict, `{"preprocessing": {...},
"preprocessing_params": {...}}`), `recipe_steps(name)` (step keys in run order), and `STEP_ORDER` / `STEP_NAMES` in `raman_bench.preprocessing.recipes`.

The steps of a recipe always run in `STEP_ORDER` (crop, cosmic rays, smoothing, baseline, scatter
correction, derivative, normalisation, pooling, scaling, GCU/LVSE), whatever order the name lists
them in. `tests/test_preprocessing_recipes.py` checks this order against the code and is the
executable specification of the recipes.

## One transformer per step

`CosmicRayRemoval`, `AsLSBaseline`, `AirPLSBaseline`, `ArPLSBaseline`, `RubberbandBaseline`,
`SavitzkyGolaySmoothing`, `WaveletDenoising`, `SavitzkyGolayDerivative`, `MSC`, `EMSC`, `SNV`,
`VectorNormalization`, `AveragePooling`, `Decimation`, `Crop`, `PhysicalCrop`, `GCU` and `LVSE`
chain in any order and expose their parameters to `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV
from raman_bench.preprocessing import SNV, SavitzkyGolayDerivative

pipe = make_pipeline(SavitzkyGolayDerivative(order=1), SNV(), PLSRegression(n_components=10))
grid = GridSearchCV(pipe, {"savitzkygolayderivative__window_length": [9, 11, 15]}).fit(X_train, y_train)
```

Unlike `RamanPreprocessor`, a hand-built pipeline runs the steps in the order you give. Standard
scaling has no one-step transformer; use scikit-learn's `StandardScaler` in a hand-built pipeline. GCU takes
`random_state` (default 0); it is seeded everywhere, so a recipe with GCU is reproducible.

## Use a recipe with a benchmark model

The same recipe is the `prep_*` hyperparameters of a benchmark model, and both give the same
output:

```python
from raman_bench.evaluate import evaluate_estimator

evaluate_estimator("MY-PLS-BSA", regressor=PLSRegression(n_components=10),
                   hyperparameters=recipe_hyperparameters("bc_snv_avgpool"))
```

This is why the transformers are not a re-implementation. `RamanPreprocessor` and the `Prep_*`
model wrappers both run `RamanPreprocessingMixin`
(`src/raman_bench/preprocessing/mixin.py`), and a test fits real `Prep_PLS` and `Prep_RF` models
for 16 recipe forms and checks that their parameters and test-set output equal the transformer's.

## What is public, what is internal

Public (stable, re-exported from `raman_bench.preprocessing`):

- `raman_bench.preprocessing.transformers`: `RamanPreprocessor`, the one-step transformers,
  `recipe_params`.
- `raman_bench.preprocessing.recipes`: `recipe`, `recipe_names`, `recipe_definition`,
  `recipe_description`, `recipe_steps`, `recipe_hyperparameters`, `STEP_ORDER`, `STEP_NAMES`.

Internal (may change without notice):

- `mixin.py`: `RamanPreprocessingMixin`, the code both routes run.
- `raman_preprocessing.py`: the numerical implementations.
- `parallel.py`, `spectrum_cache.py`, `bridge_bases.py`, `wrapped_models.py`, `many_class_mitra.py`.

## Add a step

1. Implement the fit/transform functions in `raman_preprocessing.py`.
2. Add a `_PREP_STEP_DEFINITIONS` entry in `mixin.py` and register it in `_ALL_PREPROCESSING_STEPS`.
3. Add its HPO search space to `build_prep_model_hyperparameters()` (`src/raman_bench/model.py`).
4. Add a transformer in `transformers.py`, put the step in `STEP_ORDER`, and add tests, in particular
   for steps that change the number of columns.
5. Update [CHANGELOG.md](../CHANGELOG.md).

## Installation, dependencies and where this is going

The interface lives inside `raman-bench`. It needs numpy, pandas, scipy, scikit-learn, PyWavelets
(a core dependency since 1.6.0) and `torch` (also core; `raman_preprocessing.py` imports it at
module level, which is the heaviest dependency for a standalone package).
Whether it should become an independent package is open: the preprocessing code still imports
`raman_bench.config` and `raman_bench.seeding`, and the benchmark's `Prep_*` model wrappers run the
same mixin, which is what keeps a recipe's output identical in both routes. Until that is decided,
use it from `raman-bench`.
