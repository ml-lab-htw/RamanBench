"""Torch models give the same predictions for the same AutoGluon seed (issue #171)."""

import numpy as np
import pandas as pd
import pytest

from raman_bench.seeding import model_seed, seed_global_rngs


class _Model:
    def __init__(self, random_seed):
        self.random_seed = random_seed


def test_model_seed_reads_the_autogluon_seed():
    assert model_seed(_Model(7)) == 7
    assert model_seed(_Model("NOTSET")) == 0
    assert model_seed(_Model(None)) == 0


def test_seed_global_rngs_makes_numpy_and_torch_repeatable():
    torch = pytest.importorskip("torch")
    seed_global_rngs(5)
    first = (np.random.rand(3), torch.rand(3))
    seed_global_rngs(5)
    second = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_same_seed_gives_identical_predictions_through_autogluon(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("autogluon")
    from raman_bench.models.custom.ramannet.model import Prep_RAMANNET

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(60, 256).astype(np.float32), columns=[f"f{i}" for i in range(256)])
    y = pd.Series(rng.choice([0, 1], size=60))
    hyperparameters = {
        "n_epochs": 2,
        "window_size": 50,
        "fc_dim": 64,
        "patience": 100,
        "warmup_epochs": 1,
    }

    def fit_predict(seed):
        model = Prep_RAMANNET(
            path=str(tmp_path / f"seed{seed}"),
            name="RAMANNET",
            problem_type="binary",
            eval_metric="log_loss",
            hyperparameters=hyperparameters,
        )
        model.fit(X=X, y=y, random_seed=seed)
        return model.predict_proba(X)

    assert np.array_equal(fit_predict(3), fit_predict(3))
    assert not np.allclose(fit_predict(3), fit_predict(4))
