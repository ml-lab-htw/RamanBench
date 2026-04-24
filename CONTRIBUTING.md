# Contributing to RamanBench

Thank you for your interest in contributing!  This guide covers two main contribution paths:

1. **[Adding a new model](#adding-a-new-model)** — implement and evaluate a new ML model
2. **[Adding a new dataset](#adding-a-new-dataset)** — contribute a new Raman spectroscopy dataset

For bug reports and feature requests, please open an issue on
[GitHub](https://github.com/ml-lab-htw/RamanBench/issues).

---

## Ecosystem Links

Before you start, here are the key resources:

| Resource | Link |
|---|---|
| **raman-data** (dataset loader) | [GitHub](https://github.com/ml-lab-htw/raman_data) · [PyPI](https://pypi.org/project/raman-data/) |
| **raman-bench** (this repo) | [GitHub](https://github.com/ml-lab-htw/RamanBench) · [PyPI](https://pypi.org/project/raman-bench/) |
| **Live Leaderboard** | [huggingface.co/spaces/ml-lab-htw/RamanBench](https://huggingface.co/spaces/ml-lab-htw/RamanBench) |
| **Paper** (NeurIPS 2026) | [arXiv TBD](https://arxiv.org/abs/TBD) |

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

### Option B: AutoGluon-integrated model (full benchmark registration)

To include your model in the full benchmark pipeline (with Raman preprocessing as
tunable hyperparameters), follow these steps:

#### 1. Implement the model

Create `src/raman_bench/models/custom/my_model.py`:

```python
import numpy as np
import torch
import torch.nn as nn
from autogluon.core.models import AbstractModel
from raman_bench.models.custom.base import BaseCustomModel


class MyModel(BaseCustomModel):
    """My custom Raman spectroscopy model."""

    def _set_default_params(self):
        default_params = {
            "hidden_dim": 256,
            "n_layers": 4,
            "learning_rate": 1e-3,
            "n_epochs": 200,
            "patience": 20,
            "batch_size": 32,
        }
        for k, v in default_params.items():
            self._set_default_param_value(k, v)

    def _get_default_searchspace(self):
        from autogluon.common import space
        return {
            "hidden_dim": space.Categorical(128, 256, 512),
            "n_layers": space.Int(2, 8),
            "learning_rate": space.Real(1e-4, 1e-2, log=True),
        }

    def _fit(self, X, y, **kwargs):
        params = self._get_model_params()
        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        # Build your network
        self.model = nn.Sequential(
            nn.Linear(X_np.shape[1], params["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(params["hidden_dim"], n_outputs),
        ).to(self._device)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(X_np, y_np, val_fraction=0.1)
        self._run_training_loop(
            X_train_t, y_train_t, X_val_t, y_val_t,
            n_epochs=params["n_epochs"], patience=params["patience"],
            time_limit=kwargs.get("time_limit"),
            criterion=criterion,
            per_epoch_augmentation=kwargs.get("per_epoch_augmentation", False),
            batch_size=params["batch_size"],
            aug_noise_sigma=0.01, aug_mixup_alpha=0.0,
            lr=params["learning_rate"], weight_decay=1e-4,
            warmup_epochs=5, grad_clip_norm=1.0,
        )
```

#### 2. Add a preprocessing wrapper

In `src/raman_bench/preprocessing/wrapped_models.py`, add:

```python
from raman_bench.models.custom.my_model import MyModel

class Prep_MYMODEL(_RamanDLBase, MyModel):
    pass  # Inherits augmentation defaults from _RamanDLBase
```

And register it:

```python
PREPROCESSED_MODELS["MYMODEL"] = Prep_MYMODEL
```

#### 3. Add to the model list

Add `"MYMODEL"` to `configs/models/raman.json` (and optionally `all.json`).

#### 4. Add tests

Create `tests/models/test_my_model.py` (see `tests/models/test_deep_cnn.py` for an example).

#### 5. Open a Pull Request

Include benchmark results on at least the debug config:

```bash
raman-bench run --config configs/debug.json --model MYMODEL
```

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

### Dataset requirements

| Requirement | Details |
|---|---|
| **Min samples** | ≥ 20 total (≥ 9 per class for classification) |
| **Format** | Spectra as rows, wavenumbers as columns (one sample = one row) |
| **Targets** | Named columns, continuous or categorical |
| **License** | CC BY 4.0 or more permissive |
| **Citation** | Published paper or preprint with DOI |
| **Instrument** | Document excitation wavelength, instrument model, and spectral range |

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
