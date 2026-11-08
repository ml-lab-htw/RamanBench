"""bag_experiment_kwargs: inner-fold validation switch (issue #162).

v1 keeps plain inner folds, so results stay reproducible; v2 switches the group-aware
inner folds on.
"""

import pytest

pytest.importorskip("tabarena")

from raman_bench.experiment_utils import bag_experiment_kwargs  # noqa: E402


def _kwargs(**extra):
    return bag_experiment_kwargs(num_random_configs=0, time_limit=60.0, num_bag_folds=3, **extra)


def test_task_specific_validation_is_off_by_default():
    protocol = _kwargs()["validation_protocol"]
    assert protocol.num_bag_folds == 3
    assert protocol.task_specific_validation is False


def test_task_specific_validation_can_be_switched_on():
    protocol = _kwargs(task_specific_validation=True)["validation_protocol"]
    assert protocol.task_specific_validation is True
