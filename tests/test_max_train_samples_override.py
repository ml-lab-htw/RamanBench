"""Regression tests for the global, dataset-keyed max_train_samples override.

Added 2026-09-25 after REZERONET's real TimeLimitExceeded on mlrod (130,061
rows) under the reduced 600s/3-bag-fold compute-scaling settings -- 90/93
large-pod tasks succeeded with confirmed real GPU utilization (the GPU-tier
resource-declaration fix), only mlrod's 3 tasks failed on time budget.
Subsampling the row count directly addresses the cost driver, applied
globally (every model, both SLURM and k8s backends) via
cluster/scope_default.json's max_train_samples_overrides, mirroring
time_limit_overrides' dataset-keyed shape but resolved for every model (no
per-model variant needed yet, unlike model_time_limit_overrides).

The 8th jobspec field is written empty (not "0" or "None") for a dataset
without an override, so run_experiment.py's --max-train-samples flag is
simply omitted -- matching its own no-subsampling default -- rather than
passed a sentinel value.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster"


def _load_submit_job():
    if str(CLUSTER_DIR) not in sys.path:
        sys.path.insert(0, str(CLUSTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_submit_job_under_test_max_train_samples", CLUSTER_DIR / "submit_job.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jobspec_lines(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


class TestResolveMaxTrainSamples:
    def test_no_overrides_returns_none(self):
        submit_job = _load_submit_job()
        assert submit_job.resolve_max_train_samples("mlrod") is None
        assert submit_job.resolve_max_train_samples("mlrod", {}) is None

    def test_override_applies(self):
        submit_job = _load_submit_job()
        assert submit_job.resolve_max_train_samples("mlrod", {"mlrod": 10000}) == 10000

    def test_override_does_not_leak_to_other_datasets(self):
        submit_job = _load_submit_job()
        overrides = {"mlrod": 10000}
        assert submit_job.resolve_max_train_samples("mlrod", overrides) == 10000
        assert submit_job.resolve_max_train_samples("alzheimer", overrides) is None


class TestWriteJobspecMaxTrainSamples:
    def test_eighth_field_present_only_for_overridden_dataset(self, tmp_path, monkeypatch):
        submit_job = _load_submit_job()
        monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
        jobs = [
            ("mlrod", 0, 0, 0, 0, 10),
            ("alzheimer", 0, 0, 0, 0, 10),
        ]
        path = submit_job.write_jobspec(
            jobs,
            f"test_{uuid.uuid4().hex}",
            default_time_limit=600,
            max_train_samples_overrides={"mlrod": 10000},
        )
        lines = _read_jobspec_lines(path)
        by_dataset = {parts[0]: parts for parts in lines}

        mlrod_parts = by_dataset["mlrod"]
        assert len(mlrod_parts) == 8, f"expected 8 fields (incl. max_train_samples), got {mlrod_parts!r}"
        assert mlrod_parts[7] == "10000"

        alzheimer_parts = by_dataset["alzheimer"]
        assert len(alzheimer_parts) == 7, (
            f"expected 7 fields (no max_train_samples override -> field omitted "
            f"entirely by .split(), not a sentinel), got {alzheimer_parts!r}"
        )

    def test_no_overrides_omits_field_entirely(self, tmp_path, monkeypatch):
        submit_job = _load_submit_job()
        monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
        jobs = [("wheat_lines", 0, 0, 0, 0, 10)]
        path = submit_job.write_jobspec(jobs, f"test_{uuid.uuid4().hex}", default_time_limit=600)
        lines = _read_jobspec_lines(path)
        assert len(lines[0]) == 7


class TestSubmitJobsMaxTrainSamples:
    def test_max_train_samples_overrides_reach_the_jobspec(self, tmp_path, monkeypatch):
        submit_job = _load_submit_job()
        monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
        jobs = [
            ("mlrod", 0, 0, 0, 0, 10),
            ("alzheimer", 0, 0, 0, 0, 10),
        ]
        profile = {"name": "test_profile", "slurm": True}
        slug = f"test_{uuid.uuid4().hex}"

        job_ids = submit_job.submit_jobs(
            model="REZERONET",
            jobs=jobs,
            slug=slug,
            n_splits=3,
            num_random_configs=0,
            num_bag_folds=3,
            time_limit=600,
            results_dir="results/v1/data",
            cache_dir=".cache_v1",
            mirror_repo="HTW-KI-Werkstatt/RamanBench",
            profile=profile,
            throttle=8,
            dry_run=True,
            max_train_samples_overrides={"mlrod": 10000},
        )
        assert job_ids == []

        lines = _read_jobspec_lines(tmp_path / f"{slug}.txt")
        by_dataset = {parts[0]: parts for parts in lines}
        assert by_dataset["mlrod"][7] == "10000"
        assert len(by_dataset["alzheimer"]) == 7

    def test_single_dataset_submit_passes_max_train_samples(self, tmp_path, monkeypatch):
        """The single-(dataset,target) CLI path (submit()) also supports
        --max-train-samples, wrapping it into a one-entry override dict."""
        submit_job = _load_submit_job()
        monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
        profile = {"name": "test_profile", "slurm": True}

        submit_job.submit(
            dataset="mlrod",
            target_idx=0,
            model="REZERONET",
            n_repeats=1,
            n_splits=3,
            config_indices=[0],
            num_random_configs=0,
            num_bag_folds=3,
            time_limit=600,
            results_dir="results/v1/data",
            cache_dir=".cache_v1",
            mirror_repo="HTW-KI-Werkstatt/RamanBench",
            profile=profile,
            throttle=8,
            dry_run=True,
            max_train_samples=10000,
        )
        slug = "mlrod_0_REZERONET"
        lines = _read_jobspec_lines(tmp_path / f"{slug}.txt")
        assert all(parts[7] == "10000" for parts in lines)
