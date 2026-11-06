"""The hosted TabPFN wrapper sends the right options to tabpfn-client (no network, no token).

tabpfn-client is replaced by a recording fake, so these tests run in CI without the package.
"""

import sys
import types

import numpy as np
import pytest

from raman_bench.models.api.tabpfn_api import THINKING_MAX_COLUMNS, TabPFNAPIClassifier, TabPFNAPIRegressor


class _FakeEstimator:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_X = None
        type(self).instances.append(self)

    def fit(self, X, y):
        self.fit_X = X
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def fake_client(monkeypatch):
    _FakeEstimator.instances = []
    module = types.ModuleType("tabpfn_client")
    module.TabPFNClassifier = type("TabPFNClassifier", (_FakeEstimator,), {"instances": []})
    module.TabPFNRegressor = type("TabPFNRegressor", (_FakeEstimator,), {"instances": []})
    monkeypatch.setitem(sys.modules, "tabpfn_client", module)
    return module


def _X(n=12, p=50, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


def _y(n=12):
    return np.arange(n) % 2


def test_thinking_options_reach_the_client(fake_client):
    clf = TabPFNAPIClassifier(thinking=True, thinking_effort="mini", use_groups=False)
    clf.fit(_X(), _y())
    kwargs = fake_client.TabPFNClassifier.instances[-1].kwargs
    assert kwargs["thinking_mode"] is True
    assert kwargs["thinking_effort"] == "mini"
    assert "thinking_timeout_s" not in kwargs


def test_timeout_is_passed_only_when_set(fake_client):
    TabPFNAPIClassifier(thinking=True, thinking_timeout_s=600.0, use_groups=False).fit(_X(), _y())
    assert fake_client.TabPFNClassifier.instances[-1].kwargs["thinking_timeout_s"] == 600.0


def test_default_call_sends_no_thinking_options(fake_client):
    TabPFNAPIClassifier(use_groups=False).fit(_X(), _y())
    kwargs = fake_client.TabPFNClassifier.instances[-1].kwargs
    assert "thinking_mode" not in kwargs and "thinking_effort" not in kwargs


def test_groups_go_to_the_api_only_with_use_groups(fake_client):
    groups = np.arange(12) // 3
    TabPFNAPIClassifier(use_groups=False).fit(_X(), _y(), groups=groups)
    assert "group_col" not in fake_client.TabPFNClassifier.instances[-1].kwargs
    TabPFNAPIClassifier(use_groups=True).fit(_X(), _y(), groups=groups)
    assert fake_client.TabPFNClassifier.instances[-1].kwargs["group_col"] == "_group_id"


def test_thinking_bins_spectra_wider_than_the_limit(fake_client):
    wide = _X(p=THINKING_MAX_COLUMNS + 500)
    TabPFNAPIClassifier(thinking=True, use_groups=False).fit(wide, _y())
    assert fake_client.TabPFNClassifier.instances[-1].fit_X.shape[1] == THINKING_MAX_COLUMNS


def test_non_thinking_keeps_every_column(fake_client):
    wide = _X(p=THINKING_MAX_COLUMNS + 500)
    TabPFNAPIClassifier(use_groups=False).fit(wide, _y())
    assert fake_client.TabPFNClassifier.instances[-1].fit_X.shape[1] == THINKING_MAX_COLUMNS + 500


def test_thinking_regressor_gets_the_same_options(fake_client):
    TabPFNAPIRegressor(thinking=True, thinking_effort="medium", use_groups=False).fit(_X(), np.arange(12.0))
    kwargs = fake_client.TabPFNRegressor.instances[-1].kwargs
    assert kwargs["thinking_mode"] is True and kwargs["thinking_effort"] == "medium"
