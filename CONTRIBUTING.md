# Contributing to RamanBench

This guide covers the two contribution paths:

1. **[Adding a new model](#adding-a-new-model)** — wrap an existing AutoGluon model, or implement a new one
2. **[Adding a new dataset](#adding-a-new-dataset)** — contribute a Raman spectroscopy dataset

For bug reports and feature requests, please open an issue on
[GitHub](https://github.com/ml-lab-htw/RamanBench/issues).

---

## Ecosystem Links

Before you start, here are the key resources:

| Resource                        | Link                                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------------------|
| **raman-data** (dataset loader) | [GitHub](https://github.com/ml-lab-htw/raman_data) · [PyPI](https://pypi.org/project/raman-data/)  |
| **raman-bench** (this repo)     | [GitHub](https://github.com/ml-lab-htw/RamanBench) · [PyPI](https://pypi.org/project/raman-bench/) |
| **Live Leaderboard**            | [huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) |
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

### Option B: Wrapping a TabArena-native model

Use this path when the model already exists in TabArena (check the pinned
commit's `tabarena.models.<key>` — see `pyproject.toml`'s `tabarena`
dependency comment for which commit is currently pinned and why) rather than
subclassing a raw AutoGluon model directly.

#### 1. Confirm TabArena has it, don't reinvent

Check `tabarena.models.<key>.{model,hpo,info}.py` in the pinned commit's
checkout. If it's there, you almost always want a thin rebind, not a
from-scratch wrapper — see `src/raman_bench/models/custom/ta_exaone_tabular/`
or `src/raman_bench/models/custom/causilo/` for the reference pattern: a
`Prep_*` class subclassing `_NoAugBase` + TabArena's own model class, in its
own `models/custom/<key>/{model,hpo,info}.py` directory (auto-discovered, no
manual registry edit needed — see `src/raman_bench/models/discover.py`).

If TabArena does NOT have it yet, that's real upstream contribution work
(open a PR to `autogluon/tabarena`), not something to fork around here.

#### 2. GPU tier

If the model needs a GPU, add its key to `cluster/gpu_models.json` — this is
what keeps the CPU/GPU cluster-tier split enforced (see
`cluster/opportunistic_scheduler.py`'s `compute_backlog`).

#### 3. Add tests

Create `tests/models/test_my_model.py` (see `tests/models/test_causilo.py`
or `tests/models/test_ta_mitra_v2.py` for a real, current example).

#### 4. Open a Pull Request

Run a real smoke test locally before opening the PR:

```bash
python scripts/run_experiment.py --dataset alzheimer --target-idx 0 \
    --model MYMODEL --repeat 0 --fold 0 --config-index 0 \
    --results-dir /tmp/smoke_results --scratch-dir /tmp/smoke_scratch
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

> **Just want to suggest a dataset without implementing it?** Add an entry
> (or open an issue/PR) to [`PROPOSED_DATASETS.md`](PROPOSED_DATASETS.md).
> We maintain it as a running queue of community-suggested datasets to
> consider for the next benchmark release. The full implementation steps
> below are only required when you want to integrate a loader yourself.

### Step 1: Prepare and publish your data

1. **Host** your dataset on [HuggingFace Datasets](https://huggingface.co/datasets)
   or [Zenodo](https://zenodo.org). HuggingFace is preferred — see the
   step-by-step [Uploading Raman Datasets to HuggingFace](https://github.com/ml-lab-htw/raman_data/blob/main/dataset_to_huggingface.md)
   guide in the `raman_data` repo for the recommended layout (spectra,
   wavenumbers, metadata files, README).
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
