# Models

The v1 leaderboard has 69 models plus two AutoGluon reference systems: classical
chemometrics, gradient boosting, tabular foundation models and deep networks built for
spectra. The live [leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
lists them all by category; [CHANGELOG.md](../CHANGELOG.md) says in which release each was added.

## Paper baselines (28 models)

All results in the paper were produced through the AutoGluon pipeline (Option 3 install in
[Installation](installation.md)).

| Category | Models |
|---|---|
| Classical spectroscopy | PLS, KNN, LR |
| Tree ensembles | GBM (LightGBM), XGB, CatBoost, RF, XT |
| Tabular deep learning | NN_TORCH, FastAI, RealMLP |
| Tabular foundation models | TabPFN v2, TabPFN v2.5, TabM, TabDPT, TabICL, MITRA |
| Time-series classifiers | ROCKET, Arsenal |
| Raman-specific DL | DeepCNN, RamanNet, SANet, RamanFormer, RamanTransformer, ReZeroNet, FC-ResNeXt, CoAtNet |
| AutoGluon ensemble | AUTOGLUON |

Models added since come mostly from TabArena's own registry and are wrapped with the Raman
preprocessing, for example TabPFN v2.6 and v3, TabPFN-Wide, TabSTAR, TabH2O, Seldon, TabAbax,
LightPFN, PrismBoost and RamanPFN. A system such as Chakra-Tab or TabFM+ does its own
validation and ensembling and runs through `scripts/run_system.py`
(see [Running the benchmark](running-the-benchmark.md)).

## Standalone scikit-learn wrappers (`raman-bench[models]`)

`raman-bench[models]` provides scikit-learn-compatible (`fit` / `predict`) wrappers for many of
the same algorithm families, usable directly without AutoGluon. These are **not** the exact
pipeline configurations from the paper (no AutoGluon preprocessing or HPO), but they use the
same underlying algorithms, and are a convenient starting point for building a new model.

| Class | Algorithm | Requires |
|---|---|---|
| `PLSModel` | Partial Least Squares | none |
| `DeepCNNModel` | Raman-specific CNN | `torch` |
| `RamanNetModel` | Raman-specific CNN | `torch` |
| `SANetModel` | Spectral attention net | `torch` |
| `RamanFormerModel` | Raman transformer | `torch` |
| `RamanTransformerModel` | Raman transformer | `torch` |
| `ReZeroNetModel` | ReZero CNN | `torch` |
| `FCResNeXtModel` | FC-ResNeXt | `torch` |
| `CoAtNetModel` | Conv + attention | `torch` |
| `RocketModel` | ROCKET regression/classification | `sktime` |
| `HydraModel` | Hydra + closed-form GPU ridge, regression/classification | `torch` |
| `AeonClassifierModel` | HIVE-COTE 2 or one of its components (STC, DrCIF, Arsenal, TDE), classification only | `aeon` |
| `TabPFNModel` | TabPFN v2 | `tabpfn` |
| `RealMLPModel` | RealMLP-TD | `pytabkit` |
| `TabMModel` | TabM-D | `pytabkit` |
| `TabDPTModel` | TabDPT | `tabdpt` |

All classes except `AeonClassifierModel` support classification and regression and
auto-detect the task from `y`. All package dependencies are included in `raman-bench[models]`.

```python
import numpy as np
from raman_bench.models.custom import DeepCNNModel, TabPFNModel

X = np.random.randn(200, 512).astype("float32")  # 200 spectra, 512 wavenumbers
y = np.random.randn(200)                          # regression targets

model = DeepCNNModel(n_epochs=50)                 # Raman-specific deep learning model
model.fit(X, y)
predictions = model.predict(X)

tfm = TabPFNModel()                               # tabular foundation model, no feature-count limit
tfm.fit(X, y)
predictions = tfm.predict(X)
```

## Known contamination and closed APIs

RamanPFN (Pan et al., 2026, arXiv:2608.02157) was developed and evaluated on the RamanBench
v0.1 datasets and task splits, with no separate development data reported. Its TabPFN weights
were not trained on RamanBench, but its fixed design choices may be tuned to these datasets, so
its scores may be optimistic. It is ranked like every other model and marked with † in figures
and tables (`raman_bench.plotting.models.KNOWN_CONTAMINATION`). Models behind a closed-source
hosted API (Chakra-Tab) are marked with ‡: the model and its code are not public, so their
results can't be reproduced independently (`raman_bench.plotting.models.CLOSED_API`).

Some weights carry a non-commercial licence (for example TabH2O and Lattice); check the model's
own licence before using it commercially.

## How the standalone and benchmark paths share model code

Custom models are written once as plain scikit-learn `BaseEstimator` subclasses, and the same
classes serve both usage modes:

```
  Custom model (e.g. DeepCNNModel)
  BaseEstimator, no AutoGluon dependency
  fit(X, y) / predict(X)
        |
        +--- Standalone path (pip install "raman-bench[models]")
        |      CUSTOM_MODELS["DEEPCNN"] -> DeepCNNModel().fit(X, y)
        |
        +--- AutoGluon pipeline path (pip install -e ".[models]" + TabArena)
               SklearnAutoGluonBridge._fit() -> DeepCNNModel(**params).fit(X_np, y_np)
               Prep_DEEPCNN(_RamanDLBase, _DeepCNNBridge)
```

`SklearnAutoGluonBridge` (in `preprocessing/wrapped_models.py`) is the only file that imports
AutoGluon; the model source files never do. To add a model, see
[CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-model).
