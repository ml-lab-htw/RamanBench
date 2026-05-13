# Contributing to RamanBench

Thank you for your interest in contributing!  This guide covers two main contribution paths:

1. **[Adding a new model](#adding-a-new-model)** — wrap an existing AutoGluon model or implement a new one
2. **[Adding a new dataset](#adding-a-new-dataset)** — contribute a new Raman spectroscopy dataset

For bug reports and feature requests, please open an issue on
[GitHub](https://github.com/ml-lab-htw/RamanBench/issues).

---

## Ecosystem Links

Before you start, here are the key resources:

| Resource                        | Link                                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------------------|
| **raman-data** (dataset loader) | [GitHub](https://github.com/ml-lab-htw/raman_data) · [PyPI](https://pypi.org/project/raman-data/)  |
| **raman-bench** (this repo)     | [GitHub](https://github.com/ml-lab-htw/RamanBench) · [PyPI](https://pypi.org/project/raman-bench/) |
| **Live Leaderboard**            | [huggingface.co/spaces/ml-lab-htw/RamanBench](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) |
| **Paper**                       | [arXiv:2605.02003](https://arxiv.org/abs/2605.02003)                                                             |

---

## Setting Up the Development Environment

```bash
git clone https://github.com/ml-lab-htw/RamanBench.git
cd RamanBench
pip install -e ".[full,dev]"
pre-commit install
```

---

## Adding a New Model

### Option A: Scikit-learn compatible model (simplest)

If your model has `.fit(X, y)` / `.predict(X)` methods, you can evaluate it
immediately using the leaderboard without touching the package source:

```python
from raman_bench import Leaderboard
from mypackage import MyModel

lb = Leaderboard.from_precomputed()
lb.evaluate_and_add("My Model", MyModel())
print(lb.rank())
```

### Option B: Wrapping an existing AutoGluon model

Use this path when the model already exists as an AutoGluon class (e.g. a new version
of TabPFN, a model from the AutoGluon registry, or any third-party model that ships
its own AutoGluon integration).

#### 1. Create or locate the AutoGluon model class

If the model is already in AutoGluon's registry (check
`autogluon.tabular.registry.ag_model_registry`), skip this step.

Otherwise subclass the closest existing AutoGluon base class. For example, a new
TabPFN version only needs to declare its checkpoint filenames:

```python
# autogluon/tabular/src/autogluon/tabular/models/tabpfnv2/tabpfnv2_5_model.py

class MyModelAG(TabPFNModel):
    ag_key = "MYMODEL"
    ag_name = "MyModel"
    default_classification_model = "mymodel-classifier-default.ckpt"
    default_regression_model     = "mymodel-regressor-default.ckpt"

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type):
        return []
```

Register it in `autogluon/tabular/src/autogluon/tabular/models/__init__.py` and
`autogluon/tabular/src/autogluon/tabular/registry/_ag_model_registry.py`.

#### 2. Create a `Prep_*` wrapper in RamanBench

Every model that runs through the benchmark pipeline must have a `Prep_*` class in
`src/raman_bench/preprocessing/wrapped_models.py`.

**Why the wrapper exists:** `_NoAugBase` and `RamanPreprocessingMixin` inject
Raman-specific preprocessing steps (baseline correction, denoising, SNV normalisation,
spectral augmentation) as AutoGluon hyperparameters directly into the model class.
This means the benchmark can later analyse which preprocessing combination worked best
for each model family without running a separate preprocessing sweep — the choices are
stored alongside the model's predictions.

Without the `Prep_*` wrapper, the pipeline puts the model's string key into the
hyperparameters dict instead of a class, causing a
`TypeError: issubclass() arg 1 must be a class` at runtime.

```python
# src/raman_bench/preprocessing/wrapped_models.py

# 1. Import the AutoGluon class at the top of the file
from autogluon.tabular.models import MyModelAG

# 2. Create the Prep_ wrapper (inherits _NoAugBase to disable spectral augmentation)
class Prep_MYMODEL(_NoAugBase, MyModelAG):  # noqa: N801
    pass

# 3. Register it in the PREPROCESSED_MODELS dict
PREPROCESSED_MODELS = {
    ...
    "MYMODEL": Prep_MYMODEL,
}
```

#### 3. Add the model key to config files

```bash
# Required
configs/models/all.json           # full benchmark model list
configs/models/raman.json         # or another appropriate group

# If the model uses a GPU
configs/models/gpu_models.json
```

If the model needs dataset-level subsampling on very large datasets (e.g. foundation
models that OOM on MLROD), add an entry to `configs/hpo_off.json`:

```json
"subsample": {
    "combinations": {
        "MYMODEL": ["mlrod_0"]
    }
}
```

#### 4. Update cluster scripts

In `cluster/submit_per_model.sh`, add a memory tier:

```bash
MITRA|REALTABPFN-V2|...|MYMODEL)
    MEM="128G" ;;
```

In `cluster/run_benchmark_single_model.sbatch`, add to `LARGE_GPU_MODELS` if the model
needs its seeds run sequentially (i.e. it can exhaust VRAM when two seeds run at once):

```bash
LARGE_GPU_MODELS="... MYMODEL"
```

#### 5. Add tests

Create `tests/models/test_my_model.py` (see `tests/models/test_deep_cnn.py` for an example).

#### 6. Open a Pull Request

Include benchmark results on at least the debug config:

```bash
raman-bench run --config configs/debug.json --model MYMODEL
```

---

### Option C: Custom Raman-specific model (full implementation)

Use this path when implementing a new neural network or algorithm from scratch.

#### 1. Implement the model

Create `src/raman_bench/models/custom/my_model.py` by subclassing `BaseCustomModel`:

```python
import torch.nn as nn
from raman_bench.models.custom.base import BaseCustomModel


class MyModel(BaseCustomModel):
    """My custom Raman spectroscopy model."""

    def _set_default_params(self):
        for k, v in {"hidden_dim": 256, "n_layers": 4, "learning_rate": 1e-3,
                     "n_epochs": 200, "patience": 20, "batch_size": 32}.items():
            self._set_default_param_value(k, v)

    def _fit(self, X, y, **kwargs):
        params = self._get_model_params()
        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        self.model = nn.Sequential(
            nn.Linear(X_np.shape[1], params["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(params["hidden_dim"], n_outputs),
        ).to(self._device)
        X_tr, X_v, y_tr, y_v = self._train_val_split(X_np, y_np, val_fraction=0.1)
        self._run_training_loop(X_tr, y_tr, X_v, y_v,
            n_epochs=params["n_epochs"], patience=params["patience"],
            time_limit=kwargs.get("time_limit"), criterion=criterion,
            batch_size=params["batch_size"], lr=params["learning_rate"])
```

Then follow steps 2–6 from Option B above (the `Prep_*` wrapper, config registration,
and cluster scripts are identical regardless of whether the model is custom or wrapped).

---

## Adding a New Dataset

New datasets are managed through the companion
[raman-data](https://github.com/ml-lab-htw/raman_data) package.
RamanBench automatically picks up any dataset added there.

### Step 1: Prepare and publish your data

1. **Host** your dataset on [HuggingFace Datasets](https://huggingface.co/datasets)
   or [Zenodo](https://zenodo.org).
2. **License** it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
   (or a more permissive licence).
3. **Format**: plain CSV or Parquet.  Wavenumber columns as floats, target
   column(s) as the last column(s).  Include a `raman_shifts` row or file if available.

### Step 2: Create a Croissant metadata file

Generate a [Croissant](https://mlcommons.org/working-groups/data/croissant/) metadata file
using the helper script in raman-data:

```bash
python scripts/generate_croissant.py --dataset my_dataset --source huggingface
```

### Step 3: Add a loader to raman-data

Open a Pull Request on [ml-lab-htw/raman_data](https://github.com/ml-lab-htw/raman_data)
with:

1. A new entry in the appropriate loader dict
   (`HuggingFaceLoader.DATASETS` or `ZenodoLoader.DATASETS`).
2. A `DatasetInfo` entry in `datasets.py`.
3. A Croissant metadata file in `croissant_files/`.
4. Tests in `tests/`.

See the existing entries in raman-data for examples.

### Step 4: Open an issue here

Once the raman-data PR is merged and a new raman-data release is published, open an issue
in this repository requesting the dataset be added to `configs/datasets/`.

### Dataset inclusion criteria

| Criterion | Details |
|---|---|
| **Freely accessible** | Publicly available (HuggingFace, Zenodo, Kaggle, etc.) under an open license (CC BY 4.0 or more permissive) |
| **Experimentally acquired** | Real instrument measurements — no simulated or synthetic spectra |
| **Supervised labels** | At least one regression target or classification label per spectrum |
| **Minimum size** | ≥ 10 labeled spectra total; for classification ≥ 9 spectra per class (rare classes removed; excluded if < 2 classes remain) |
| **Learnability** | Regression: R² > 0.05 with at least one model; Classification: ΔF1 > 0.05 above majority-class baseline (checked automatically during integration) |
| **Citation** | Published paper or preprint with DOI |
| **Format** | Spectra as rows, wavenumbers as columns; targets as named columns (continuous or categorical) |

### What to document

For each new dataset, write a section in `NEW_DATASETS.md` following the existing
format.  See [NEW_DATASETS.md](NEW_DATASETS.md) for examples.

---

## Code Style

```bash
ruff check src/          # linting
black src/ tests/        # formatting
mypy src/raman_bench/    # type checking
pytest                   # tests
```

All of these are run automatically in CI.

---

## Licence

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
