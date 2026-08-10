"""split_info must be written unconditionally, not gated behind a skip check.

See #6: compute_predictions returns at the "already_complete" skip *before*
writing ground truth, so metadata placed after that point would never appear
for already-computed keys. Mirrors test_metrics_no_delete.py's source-guard
style for the analogous concern.
"""

import json

import pytest

pytest.importorskip("autogluon")

from raman_bench.predictions import _atomic_write_json  # noqa: E402


def test_atomic_write_json_round_trips(tmp_path):
    path = tmp_path / "some_key_split_info.json"
    data = {"split_type": "grouped", "n_groups": 4, "largest_group_size": 3, "n_train": 8, "n_test": 4}
    _atomic_write_json(str(path), data)
    with open(path) as f:
        assert json.load(f) == data


def test_atomic_write_json_leaves_no_temp_file_on_success(tmp_path):
    path = tmp_path / "some_key_split_info.json"
    _atomic_write_json(str(path), {"split_type": "iid"})
    assert list(tmp_path.iterdir()) == [path]


def test_split_info_write_precedes_the_already_complete_skip():
    """Source guard: the split_info write must appear before every per-model
    'continue' that could skip an already-computed key, so a rerun that only
    adds a new model still records split_info for keys whose predictions
    already exist from a previous model."""
    import raman_bench.predictions as predictions_mod

    with open(predictions_mod.__file__) as f:
        lines = f.readlines()

    split_info_line = next(
        i for i, line in enumerate(lines) if "split_info_path = os.path.join(predictions_dir" in line
    )
    already_complete_line = next(
        i for i, line in enumerate(lines) if "already_complete = os.path.exists(pred_path)" in line
    )
    assert split_info_line < already_complete_line, (
        "split_info must be written before the already_complete skip check, "
        "or a rerun with a new model will never record it for existing keys"
    )
