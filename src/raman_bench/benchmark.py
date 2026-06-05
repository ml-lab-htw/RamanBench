"""Dataset loading, caching, and iteration for the RamanBench pipeline.

The :class:`RamanBenchmark` class loads datasets from the
`raman-data <https://github.com/ml-lab-htw/raman_data>`_ package, applies
train/test splits (with group-based splitting for regression to prevent data
leakage), and caches prepared splits to disk for faster subsequent runs.

Datasets and targets
--------------------
Each dataset may have multiple regression targets.  The benchmark expands every
(dataset, target) pair into a separate "key" of the form
``"{dataset_name}_{target_idx}"``.  Iterating over a ``RamanBenchmark`` instance
yields these keys, making it easy to run a model on every target independently.

Example
-------
::

    from raman_bench import RamanBenchmark

    bench = RamanBenchmark(
        dataset_names_classification=["wheat_lines"],
        dataset_names_regression=["amino_acids_glycine"],
        cache_dir=".cache",
    )
    bench.init_datasets()

    for train_df, test_df, key, task_type in bench:
        print(key, len(train_df), len(test_df))
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from raman_data import TASK_TYPE, raman_data
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from huggingface_hub downloads
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def _is_float_col(s: str) -> bool:
    """Check if string is a parseable float (for wavenumber column detection)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def configure_benchmark(config, init_benchmark: bool = True) -> "RamanBenchmark":
    """Instantiate a :class:`RamanBenchmark` from a loaded config dict.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration (see :func:`raman_bench.config.load_config`).
    init_benchmark : bool
        If ``True`` (default), call :meth:`RamanBenchmark.init_datasets` before
        returning so the benchmark is ready for iteration.

    Returns
    -------
    RamanBenchmark
    """
    benchmark = RamanBenchmark(
        dataset_names_classification=config.get("dataset_names_classification"),
        dataset_names_regression=config.get("dataset_names_regression"),
        test_size=config["test_size"],
        random_state=config["random_state"],
        cache_dir=config.get("cache_dir") or None,
        min_samples_per_class=config.get("min_samples_per_class", 9),
        group_regression_splits=config.get("group_regression_splits", True),
        use_mirror=config.get("use_mirror", True),
        mirror_repo=config.get("mirror_repo", "HTW-KI-Werkstatt/RamanBench"),
    )

    if init_benchmark:
        benchmark.init_datasets()
    return benchmark


class RamanBenchmark:
    """Manage dataset loading, caching, and iteration for the benchmark.

    Provides an iterator-like container of prepared train/test splits sourced
    from the `raman-data <https://pypi.org/project/raman-data/>`_ package.

    Parameters
    ----------
    dataset_names_classification : list[str] | None
        Classification dataset names.  ``None`` loads all available datasets.
    dataset_names_regression : list[str] | None
        Regression dataset names.  ``None`` loads all available datasets.
    test_size : float
        Fraction reserved for testing (default 0.2).
    random_state : int
        Seed for the train/test split (default 42).
    cache_dir : str | None
        Directory for cached dataset splits.  Defaults to ``".cache"``.
    min_samples_per_class : int
        Classes with fewer samples than this are removed before splitting
        (classification only).  Default 9.
    group_regression_splits : bool
        If ``True`` (default), use :class:`~sklearn.model_selection.GroupShuffleSplit`
        to keep samples from the same physical measurement in the same split.
        This prevents data leakage in multi-measurement datasets.
    use_mirror : bool
        If ``True`` (default), load datasets from the HuggingFace mirror repo.
        If ``False``, download from original sources via raman-data.
    mirror_repo : str
        HuggingFace dataset repo ID for the mirror (default
        ``"HTW-KI-Werkstatt/RamanBench"``).

    Examples
    --------
    ::

        bench = RamanBenchmark(
            dataset_names_classification=["wheat_lines"],
            dataset_names_regression=[],
            cache_dir=".cache",
        )
        bench.init_datasets()
        train_df, test_df, key, task_type = bench[0]
    """

    def __init__(
        self,
        dataset_names_classification: list[str] | None = None,
        dataset_names_regression: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cache_dir: str | None = None,
        min_samples_per_class: int = 9,
        group_regression_splits: bool = True,
        use_mirror: bool = True,
        mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
    ):
        if not cache_dir:
            cache_dir = ".cache"

        # Use separate cache directories for mirror vs. original sources
        raw_subdir = "datasets_mirror" if use_mirror else "datasets_raw"
        self.cache_dir_raw = os.path.join(cache_dir, raw_subdir)
        os.makedirs(self.cache_dir_raw, exist_ok=True)

        self.cache_dir_processed = os.path.join(
            cache_dir, "datasets_processed", f"seed_{random_state}"
        )
        os.makedirs(self.cache_dir_processed, exist_ok=True)

        self.test_size = test_size
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        self.group_regression_splits = group_regression_splits
        self.use_mirror = use_mirror
        self.mirror_repo = mirror_repo

        if dataset_names_classification is None:
            self.dataset_names_classification = raman_data(
                task_type=TASK_TYPE.Classification, cache_dir=self.cache_dir_raw
            )
        else:
            self.dataset_names_classification = list(dataset_names_classification)

        if dataset_names_regression is None:
            self.dataset_names_regression = raman_data(
                task_type=TASK_TYPE.Regression, cache_dir=self.cache_dir_raw
            )
        else:
            self.dataset_names_regression = list(dataset_names_regression)

        logger.info(
            "Datasets: %d classification, %d regression",
            len(self.dataset_names_classification),
            len(self.dataset_names_regression),
        )

        self._key_list: list[str] = []
        self._task_type_list: list[TASK_TYPE] = []
        self._index_file = os.path.join(self.cache_dir_processed, "index.json")
        self._index: dict[str, int] = self._load_index()
        self.is_initialized = False

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._key_list)

    def __iter__(self):
        if not self.is_initialized:
            self.init_datasets()
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> tuple[DataFrame, DataFrame, str, TASK_TYPE]:
        """Return ``(train_df, test_df, key, task_type)`` for index *idx*."""
        if not self.is_initialized:
            self.init_datasets()

        key = self._key_list[idx]
        task_type = self._task_type_list[idx]

        if self._has_dataset_in_cache(key):
            data_train, data_test = self._load_dataset_from_cache(key)
        else:
            data_train, data_test = self._load_dataset_from_key(key)
            if data_train is not None:
                self._save_dataset(key, data_train, data_test)

        return data_train, data_test, key, task_type

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_datasets(self):
        """Populate the key list and ensure all splits are cached."""
        self._load_datasets(self.dataset_names_regression)
        self._load_datasets(self.dataset_names_classification)

        for dataset_name in self.dataset_names_classification + self.dataset_names_regression:
            for target_idx in range(self._index.get(dataset_name, 0)):
                self._key_list.append(self.get_key(dataset_name, target_idx))
                task = (
                    TASK_TYPE.Classification
                    if dataset_name in self.dataset_names_classification
                    else TASK_TYPE.Regression
                )
                self._task_type_list.append(task)

        self._save_index()
        self.is_initialized = True

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_key(dataset_name: str, target_idx: int) -> str:
        """Return the cache key ``"{dataset_name}_{target_idx}"``."""
        dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
        return f"{dataset_name}_{target_idx}"

    @staticmethod
    def split_key(key: str) -> tuple[str, int]:
        """Reverse :meth:`get_key` → ``(dataset_name, target_idx)``."""
        parts = key.split("_")
        return "_".join(parts[:-1]), int(parts[-1])

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_paths(self, key: str) -> tuple[str, str]:
        train = f"{self.cache_dir_processed}/{key}_train.pkl"
        test = f"{self.cache_dir_processed}/{key}_test.pkl"
        return train, test

    def _has_dataset_in_cache(self, key: str) -> bool:
        train, test = self._get_cache_paths(key)
        return os.path.exists(train) and os.path.exists(test)

    def _save_dataset(self, key: str, train: DataFrame, test: DataFrame):
        train_path, test_path = self._get_cache_paths(key)
        train.to_pickle(train_path)
        test.to_pickle(test_path)

    def _load_dataset_from_cache(self, key: str) -> tuple[DataFrame | None, DataFrame | None]:
        train_path, test_path = self._get_cache_paths(key)
        try:
            data_train = pd.read_pickle(train_path)
            data_test = pd.read_pickle(test_path)
        except Exception as e:
            logger.warning("Cache for %s unreadable (%s); regenerating.", key, e)
            for f in (train_path, test_path):
                if os.path.exists(f):
                    os.remove(f)
            return self._load_dataset_from_key(key)

        combined = pd.concat([data_train, data_test], ignore_index=True)
        _, rare = self._filter_rare_classes(combined, key)
        if rare is None:
            return None, None
        data_train = self._drop_classes(data_train, key, rare)
        if data_train is None:
            return None, None
        data_test = self._drop_classes(data_test, key, rare)
        return data_train, data_test

    def _load_index(self) -> dict[str, int]:
        if not os.path.exists(self._index_file):
            return {}
        with open(self._index_file) as f:
            return json.load(f)

    def _save_index(self):
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    # ------------------------------------------------------------------
    # Mirror loading
    # ------------------------------------------------------------------

    def _load_raman_dataset(self, dataset_name: str):
        """Load dataset from mirror or original source.

        If use_mirror is True, loads from HuggingFace mirror repo with fallback
        to original sources. Otherwise, downloads from original sources via raman_data().
        """
        if self.use_mirror:
            dataset = self._load_from_mirror(dataset_name)
            if dataset is not None:
                return dataset
            # Fallback to original source if mirror fails
            logger.debug("Mirror load failed for %s; falling back to original source", dataset_name)
        return raman_data(dataset_name, cache_dir=self.cache_dir_raw)

    def _load_from_mirror(self, dataset_name: str):
        """Load dataset from HuggingFace mirror repo.

        Reconstructs RamanDataset from wide-format Parquet stored in mirror.
        Metadata is loaded from the separate metadata.json file if available.
        """
        from huggingface_hub import hf_hub_download
        from raman_data import RamanDataset as RamanDatasetType
        from raman_data.types import DatasetInfo as RamanDatasetInfo

        try:
            # Download Parquet file directly from HF Hub
            parquet_path = hf_hub_download(
                repo_id=self.mirror_repo,
                filename=f"{dataset_name}/data/train-00000-of-00001.parquet",
                repo_type="dataset",
                cache_dir=self.cache_dir_raw,
            )
        except Exception as e:
            logger.warning("Failed to load %s from mirror: %s", dataset_name, e)
            return None

        # Load Parquet to pandas
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning("Failed to read parquet for %s: %s", dataset_name, e)
            return None

        # Separate wavenumber columns (float-parseable) from target columns
        all_cols = df.columns.tolist()
        shift_cols = sorted(
            [c for c in all_cols if _is_float_col(c)],
            key=float
        )
        target_cols = [c for c in all_cols if not _is_float_col(c)]

        # Extract arrays
        spectra = df[shift_cols].values.astype(np.float32)
        raman_shifts = np.array([float(c) for c in shift_cols])

        # Try to load metadata from separate metadata.json file
        target_names = None
        task_type = None
        try:
            metadata_path = hf_hub_download(
                repo_id=self.mirror_repo,
                filename=f"{dataset_name}/metadata.json",
                repo_type="dataset",
                cache_dir=self.cache_dir_raw,
            )
            with open(metadata_path) as f:
                meta = json.load(f)
                target_names = meta.get("target_names")
                task_type_val = meta.get("task_type")
                # task_type might be a string or int
                if isinstance(task_type_val, int):
                    task_type = TASK_TYPE(task_type_val)
                elif task_type_val:
                    task_type_str = str(task_type_val).lower()
                    if task_type_str in ["classification", "0"]:
                        task_type = TASK_TYPE.Classification
                    elif task_type_str in ["regression", "1"]:
                        task_type = TASK_TYPE.Regression
        except Exception as e:
            logger.debug("Could not load metadata.json for %s: %s", dataset_name, e)

        # Infer target_names and task_type if not in metadata
        if target_names is None:
            target_names = target_cols
        if task_type is None:
            # Infer from target column type
            if len(target_cols) > 0:
                first_target = df[target_cols[0]]
                # Classification if non-numeric strings, regression otherwise
                if first_target.dtype == object or first_target.dtype.name.startswith('str'):
                    task_type = TASK_TYPE.Classification
                else:
                    task_type = TASK_TYPE.Regression
            else:
                task_type = TASK_TYPE.Regression

        # Reconstruct targets (single vs multi-target)
        if len(target_cols) == 1:
            targets = df[target_cols[0]].values
        else:
            targets = df[target_cols].values

        info = RamanDatasetInfo(
            id=dataset_name,
            name=dataset_name,
            loader=lambda: None,
            metadata={},
            task_type=task_type,
        )
        return RamanDatasetType(
            spectra=spectra,
            targets=targets,
            raman_shifts=raman_shifts,
            target_names=target_names,
            info=info,
        )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_datasets(self, dataset_names: list[str]):
        for dataset_name in tqdm(dataset_names, desc="Loading datasets"):
            if dataset_name in self._index and self._index[dataset_name] > 0:
                num_targets = self._index[dataset_name]
            else:
                dataset = self._load_raman_dataset(dataset_name)
                if dataset is None or dataset.targets is None:
                    num_targets = 0
                elif dataset.targets.ndim == 1:
                    num_targets = 1
                else:
                    num_targets = dataset.targets.shape[1]
                self._index[dataset_name] = num_targets
                self._save_index()

            for target_idx in range(num_targets):
                key = self.get_key(dataset_name, target_idx)
                if not self._has_dataset_in_cache(key):
                    data_train, data_test = self._load_dataset_from_key(key)
                    if data_train is not None:
                        self._save_dataset(key, data_train, data_test)

    def _load_dataset_from_key(self, key: str) -> tuple[DataFrame | None, DataFrame | None]:
        dataset_name, target_idx = self.split_key(key)
        dataset = self._load_raman_dataset(dataset_name)

        if dataset is None or dataset.targets is None:
            num_targets = 0
        elif dataset.targets.ndim == 1:
            num_targets = 1
        else:
            num_targets = dataset.targets.shape[1]

        if target_idx >= num_targets:
            raise ValueError(f"Target index {target_idx} out of range for dataset {dataset_name}")

        data_df = dataset.to_dataframe(target_idx)
        data_df = data_df.dropna()

        if len(data_df) == 0:
            logger.warning("Dataset %s has 0 samples after dropping NaNs.", key)
            return None, None

        data_df, _ = self._filter_rare_classes(data_df, key)
        if data_df is None:
            return None, None

        is_regression = dataset_name in self.dataset_names_regression
        if is_regression and self.group_regression_splits:
            if num_targets > 1:
                all_targets_df = pd.DataFrame(dataset.targets, columns=dataset.target_names).loc[
                    data_df.index
                ]
                return self._grouped_train_test_split(data_df, group_by_df=all_targets_df)
            return self._grouped_train_test_split(data_df)

        label_col = data_df.columns[-1]
        stratify = data_df[label_col] if not is_regression else None
        return train_test_split(
            data_df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    # ------------------------------------------------------------------
    # Rare-class filtering
    # ------------------------------------------------------------------

    def _filter_rare_classes(
        self, data_df: DataFrame, key: str
    ) -> tuple[DataFrame | None, list | None]:
        dataset_name, _ = self.split_key(key)
        if dataset_name not in self.dataset_names_classification:
            return data_df, []
        if self.min_samples_per_class <= 0:
            return data_df, []

        label_col = data_df.columns[-1]
        counts = data_df[label_col].value_counts()
        rare = counts[counts < self.min_samples_per_class].index.tolist()
        if rare:
            logger.warning(
                "Dataset %s: removing %d class(es) with < %d samples",
                key,
                len(rare),
                self.min_samples_per_class,
            )
            data_df = data_df[~data_df[label_col].isin(rare)]

        if data_df[label_col].nunique() < 2:
            logger.warning("Dataset %s: < 2 classes remain. Skipping.", key)
            return None, None

        return data_df, rare

    def _drop_classes(self, data_df: DataFrame, key: str, classes: list) -> DataFrame | None:
        if not classes:
            return data_df
        dataset_name, _ = self.split_key(key)
        if dataset_name not in self.dataset_names_classification:
            return data_df
        label_col = data_df.columns[-1]
        data_df = data_df[~data_df[label_col].isin(classes)]
        if data_df[label_col].nunique() < 2:
            return None
        return data_df

    # ------------------------------------------------------------------
    # Group-aware train/test split (regression leakage prevention)
    # ------------------------------------------------------------------

    def _grouped_train_test_split(
        self, data_df: DataFrame, group_by_df: DataFrame | None = None
    ) -> tuple[DataFrame, DataFrame]:
        """Split while keeping co-measured samples in the same partition.

        Two rows are considered the same physical measurement when their
        non-zero target values are identical.  Rows that share a group key
        always land in the same split, preventing the kind of train/test
        leakage that arises from replicate measurements of the same sample.

        Falls back to a plain :func:`~sklearn.model_selection.train_test_split`
        when every row is in its own unique group.
        """
        target_col = data_df.columns[-1]
        target_values = group_by_df if group_by_df is not None else data_df[[target_col]]

        def _row_key(row):
            nonzero = {col: val for col, val in row.items() if val != 0}
            return frozenset(nonzero.items()) if nonzero else None

        keys = target_values.apply(_row_key, axis=1)
        unique_counter = 0
        group_labels = []
        seen: dict = {}
        for k in keys:
            if k is None:
                group_labels.append(f"__unique_{unique_counter}")
                unique_counter += 1
            else:
                if k not in seen:
                    seen[k] = str(k)
                group_labels.append(seen[k])

        groups = np.array(group_labels)
        if len(np.unique(groups)) == len(data_df):
            return train_test_split(
                data_df, test_size=self.test_size, random_state=self.random_state
            )

        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(gss.split(data_df, groups=groups))
        return data_df.iloc[train_idx], data_df.iloc[test_idx]
