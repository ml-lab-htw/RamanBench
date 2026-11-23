"""The LLM-tree runner waits out gateway errors instead of failing every fold that starts during an outage."""

import importlib.util
import pathlib

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

_spec = importlib.util.spec_from_file_location(
    "run_llm_trees", pathlib.Path(__file__).parent.parent / "scripts" / "run_llm_trees.py"
)
run_llm_trees = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_llm_trees)


def _server_error():
    request = httpx.Request("POST", "http://gateway/v1/chat/completions")
    return openai.InternalServerError("no vllm", response=httpx.Response(500, request=request), body=None)


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(run_llm_trees.time, "sleep", waited.append)
    return waited


def test_gateway_error_is_retried_after_a_wait(monkeypatch, sleeps):
    calls = []

    def run_fold(*args):
        calls.append(args)
        if len(calls) < 3:
            raise _server_error()
        return {"status": "ok"}

    monkeypatch.setattr(run_llm_trees, "run_fold", run_fold)
    assert run_llm_trees.run_fold_waiting("task", 0) == {"status": "ok"}
    assert len(calls) == 3 and sleeps == [30, 60]


def test_other_errors_are_not_retried(monkeypatch, sleeps):
    def run_fold(*args):
        raise ValueError("Tree generation failed")

    monkeypatch.setattr(run_llm_trees, "run_fold", run_fold)
    with pytest.raises(ValueError):
        run_llm_trees.run_fold_waiting("task", 0)
    assert sleeps == []


def test_a_long_outage_still_raises(monkeypatch, sleeps):
    def run_fold(*args):
        raise _server_error()

    monkeypatch.setattr(run_llm_trees, "run_fold", run_fold)
    with pytest.raises(openai.InternalServerError):
        run_llm_trees.run_fold_waiting("task", 0)
    assert sleeps == list(run_llm_trees.GATEWAY_WAIT_S)
