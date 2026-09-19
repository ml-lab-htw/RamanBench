"""Regression tests for cluster/opportunistic_scheduler.py's check_capacity().

Covers a real production bug found on 2026-09-19: the courtesy-ceiling and
max-pending checks counted this SLURM account's ENTIRE resident task count,
not just RamanBench's own (job name prefix ``RB_``) -- confirmed live, an
unrelated concurrent project's own 619-task array left RamanBench (with zero
jobs of its own resident) permanently reporting "at ceiling", even though
check (4) (distinct array count) already filtered on the same ``RB_`` prefix.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

_CLUSTER_DIR = Path(__file__).resolve().parent.parent / "cluster"
if str(_CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_DIR))

_MODULE_PATH = _CLUSTER_DIR / "opportunistic_scheduler.py"
_spec = importlib.util.spec_from_file_location("opportunistic_scheduler", _MODULE_PATH)
opportunistic_scheduler = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("opportunistic_scheduler", opportunistic_scheduler)
_spec.loader.exec_module(opportunistic_scheduler)

check_capacity = opportunistic_scheduler.check_capacity


def _completed(stdout: str):
    class _Result:
        def __init__(self, out):
            self.stdout = out

    return _Result(stdout)


def _run_side_effect(sinfo_out: str, squeue_out: str):
    def _side_effect(cmd, **kwargs):
        if cmd[0] == "sinfo":
            return _completed(sinfo_out)
        if cmd[0] == "squeue":
            return _completed(squeue_out)
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    return _side_effect


class TestCheckCapacityJobNameFiltering:
    def test_unrelated_project_jobs_do_not_trip_courtesy_ceiling(self):
        # 619 resident tasks, ALL belonging to a different, unrelated
        # project (no "RB_" prefix) -- RamanBench itself has zero jobs
        # resident. This must NOT count against courtesy_ceiling.
        squeue_lines = "\n".join(["RUNNING|RPPPA_T5_kfold_param_ablation_mlrod_subsample_retry"] * 611
                                  + ["RUNNING|TabICLMix"] * 7
                                  + ["RUNNING|RPPPA_T3_kfold_param_ablation_himem_retry"])
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("128/64/0/192\n", squeue_lines),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200,
            )
        assert has_room, reason
        assert "0 resident task(s)" in reason

    def test_own_jobs_over_ceiling_still_blocks(self):
        squeue_lines = "\n".join(f"PENDING|RB_EBM_full" for _ in range(250))
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("128/64/0/192\n", squeue_lines),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200,
            )
        assert not has_room
        assert "250 resident task(s)" in reason

    def test_mixed_own_and_unrelated_jobs_counts_only_own(self):
        squeue_lines = "\n".join(
            ["RUNNING|RPPPA_T5_kfold_param_ablation_mlrod_subsample_retry"] * 619
            + ["PENDING|RB_ROCKET_full"] * 10
        )
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("128/64/0/192\n", squeue_lines),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200, max_pending=20, max_concurrent_arrays=5,
            )
        assert has_room, reason
        assert "10 resident task(s)" in reason

    def test_max_pending_also_filtered_to_own_jobs(self):
        # An unrelated project's own array sitting 100% PENDING must not
        # trip RamanBench's max_pending fairshare-stall detection either.
        squeue_lines = "\n".join(["PENDING|RPPPA_T9_some_other_array"] * 20)
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("128/64/0/192\n", squeue_lines),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200, max_pending=5,
            )
        assert has_room, reason

    def test_own_pending_over_max_pending_blocks(self):
        squeue_lines = "\n".join(["PENDING|RB_EBM_full"] * 6)
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("128/64/0/192\n", squeue_lines),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200, max_pending=5,
            )
        assert not has_room
        assert "6 of my own PENDING" in reason

    def test_idle_cpu_check_still_applies(self):
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("0/8/0/192\n", ""),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200,
            )
        assert not has_room
        assert "idle CPU" in reason

    def test_max_concurrent_arrays_still_filtered_to_own_jobs(self):
        squeue_lines = "\n".join(
            ["PENDING|RB_EBM_full"] * 5 + ["PENDING|RB_ROCKET_full"] * 5
            + ["RUNNING|RPPPA_unrelated"] * 5
        )
        with patch.object(
            opportunistic_scheduler.subprocess, "run",
            side_effect=_run_side_effect("128/64/0/192\n", squeue_lines),
        ):
            has_room, reason = check_capacity(
                profile={"partition": "Debug_node", "cron_user": "koddenb"},
                min_idle_cpus=32, courtesy_ceiling=200, max_pending=20, max_concurrent_arrays=1,
            )
        assert not has_room
        assert "2 of my own RamanBench array-job(s)" in reason
