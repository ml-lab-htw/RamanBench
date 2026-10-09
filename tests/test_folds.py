"""raman_bench.folds: the exported folds are the runner's folds, keyed by spectrum id."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabarena")

from raman_bench import folds as folds_mod
from raman_bench.experiment_utils import build_task, prepare_task_dataframe
from raman_bench.splitting import GROUP_COL


def _frame(problem_type, *, grouped=False, n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"{400 + i}" for i in range(12)])
    if grouped:
        df[GROUP_COL] = np.repeat(np.arange(n // 3), 3)
    if problem_type == "classification":
        df["target"] = np.tile(["a", "b", "c"], n // 3)
        df.loc[[5, 6], "target"] = "rare"  # a class below min_samples_per_class
    else:
        df["target"] = rng.normal(size=n)
    df.iloc[[3, 40], 0] = np.nan  # rows with a NaN in the spectrum
    df.loc[[7], "target"] = np.nan  # an unlabelled row
    df.index.name = "spectrum_id"
    return df


@pytest.fixture
def fake_task(monkeypatch):
    """Point _task_frame at a synthetic task, prepared the way the runner prepares it."""

    def install(problem_type, **kw):
        raw = _frame(problem_type, **kw)
        df = prepare_task_dataframe(
            dataset_name="toy",
            target_idx=0,
            df=raw.copy(),
            raw_targets=raw["target"].to_numpy(),
            problem_type=problem_type,
        )
        spec = {"task": "toy__0", "dataset": "toy", "target_idx": 0, "n_repeats": 1, "n_folds": 3}
        monkeypatch.setattr(folds_mod, "_task_frame", lambda task, **_: (spec, df, problem_type))
        return raw, df

    return install


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
@pytest.mark.parametrize("grouped", [False, True])
def test_folds_equal_the_runners_task_splits(fake_task, problem_type, grouped):
    raw, df = fake_task(problem_type, grouped=grouped)
    folds = folds_mod.task_folds("toy__0")

    _, wrapper = build_task(
        dataset_name="toy",
        target_idx=0,
        df=raw.copy(),
        raw_targets=raw["target"].to_numpy(),
        problem_type=problem_type,
        n_repeats=1,
        n_splits=3,
    )
    for k in range(3):
        _, test_pos = wrapper.get_split_indices(fold=k, repeat=0)
        expected = np.sort(df.index.to_numpy()[test_pos])
        _, test_ids = folds_mod.train_test_ids(folds, "toy__0", k)
        np.testing.assert_array_equal(np.sort(test_ids), expected)


def test_dropped_rows_are_absent_and_ids_are_original(fake_task):
    _, df = fake_task("classification")
    folds = folds_mod.task_folds("toy__0")
    assert set(folds["spectrum_id"]) == set(df.index)
    assert not {3, 5, 6, 7, 40} & set(folds["spectrum_id"])  # NaN spectrum, rare class, no label
    assert folds["spectrum_id"].max() == 89  # ids are dataset rows, not positions after dropping


def test_every_row_is_tested_once_and_groups_stay_together(fake_task):
    _, df = fake_task("regression", grouped=True)
    folds = folds_mod.task_folds("toy__0")
    assert folds["spectrum_id"].is_unique
    group_folds = folds.assign(group=df.loc[folds["spectrum_id"], GROUP_COL].to_numpy()).groupby(
        "group"
    )["fold"]
    assert (group_folds.nunique() == 1).all()

    train, test = folds_mod.train_test_ids(folds, "toy__0", 0)
    assert not set(train) & set(test)
    assert len(train) + len(test) == len(df)


def test_export_writes_csv_and_parquet(fake_task, monkeypatch, tmp_path):
    fake_task("classification")
    monkeypatch.setattr(folds_mod, "protocol_tasks", lambda **_: [{"task": "toy__0"}])
    for name in ("folds.csv", "folds.parquet"):
        out = folds_mod.export_folds(tmp_path / name)
        back = (
            pd.read_csv(tmp_path / name)
            if name.endswith(".csv")
            else pd.read_parquet(tmp_path / name)
        )
        assert list(back.columns) == folds_mod.FOLD_COLUMNS
        assert len(back) == len(out)
