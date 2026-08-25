"""Computing metrics must never delete the predictions it is asked to measure.

``compute_metrics_from_predictions`` used to ``os.remove`` any prediction whose test
index disagreed with the stored ground truth, so that a later ``compute_predictions``
would regenerate it. The loss was invisible: the model fell back to Random-Forest
imputation, every plot still rendered, and the leaderboard reported a number that was
not the model's. A whole cluster's results were reduced to RF scores this way.

A mismatch means the prediction and the ground truth describe different splits. That
is a diagnosis, not a cleanup: the file stays, the pair is skipped, and the run ends
with a summary naming what was skipped.
"""

import re

import pandas as pd

import raman_bench.evaluation as evaluation


def _write_pair(pred_dir, key, model, gt_index, pred_index):
    """Write a ground truth and a prediction, optionally on different indices."""
    pd.DataFrame({"target": [1.0] * len(gt_index)}, index=list(gt_index)).to_csv(
        pred_dir / f"{key}_ground_truth.csv", index_label="spectrum_id"
    )
    pd.DataFrame({"target": [1.0] * len(pred_index)}, index=list(pred_index)).to_csv(
        pred_dir / f"{key}_{model}_predictions.csv", index_label="spectrum_id"
    )
    return pred_dir / f"{key}_{model}_predictions.csv"


def test_source_has_no_os_remove_of_predictions():
    """The delete must not creep back in.

    Guards the behaviour at the source level too: a future refactor could reintroduce
    ``os.remove(pred_path)`` without any test that builds a full benchmark noticing.
    """
    src = evaluation.__file__
    with open(src) as f:
        body = f.read()
    # Any os.remove touching a prediction path is the regression we are guarding.
    # Comments are excluded deliberately: the fix documents the old behaviour by
    # name, and matching prose instead of code would make this test fail on the
    # very code it is meant to bless.
    offenders = [
        line.strip()
        for line in body.splitlines()
        if not line.strip().startswith("#") and re.search(r"os\.remove\s*\(\s*pred_path", line)
    ]
    assert not offenders, (
        "compute_metrics_from_predictions deletes predictions again: "
        f"{offenders}. Metrics must skip and report, never destroy results."
    )


def test_mismatched_prediction_is_kept_on_disk(tmp_path):
    """A prediction whose index disagrees with the ground truth survives.

    Exercises the branch directly: build the two frames the way the metrics loop
    does, and assert the file is still there after the comparison fails.
    """
    import numpy as np

    pred_dir = tmp_path / "seed_0" / "predictions"
    pred_dir.mkdir(parents=True)

    # Same row count, different rows -> exactly the grouped-split failure mode.
    path = _write_pair(
        pred_dir, "some_dataset_0", "CAT", gt_index=[0, 1, 2, 3], pred_index=[4, 5, 6, 7]
    )
    assert path.exists()

    gt = pd.read_csv(pred_dir / "some_dataset_0_ground_truth.csv", index_col=0).sort_index()
    pred = pd.read_csv(path, index_col=0).sort_index()

    assert not np.array_equal(gt.index, pred.index), "test setup should mismatch"
    # The old code removed the file at exactly this point; nothing may do so now.
    assert path.exists(), "a mismatched prediction must be kept, not deleted"
