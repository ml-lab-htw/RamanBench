#!/usr/bin/env python
"""Submit the AutoGluon-extreme-1h tasks still missing after the BHT k8s runs
(rb-autogluon-extreme-1h / -b200 / -anygpu), to the HTW SLURM cluster instead.

Deliberately targets ONLY the exact missing (dataset, target_idx, repeat, fold)
tuples -- not a full re-sweep. HTW's results/cache directory is a SEPARATE
filesystem from BHT's PVC (no shared storage, no cache hits possible), so
submitting the full 405-task AutoGluon-extreme-1h target list there would
recompute everything the BHT k8s run already has. The missing set was computed
by diffing configs/v1/target_list.json's expected (dataset, target_idx, repeat,
fold) tuples against the real `results.pkl` files found under
AutoGluon_extreme_1h/ on the BHT PVC (via a PVC-mounted exec pod) -- 19 tasks
across 7 datasets, confirmed to exactly match model_progress.csv's reported gap
(288/303 full + 98/102 large = 386/405 done, 19 missing).

Uses cluster/run_autogluon_baseline.sbatch (the SLURM counterpart to
scripts/run_autogluon_baseline.py, sibling of run_experiment.sbatch) -- NOT
run_experiment.py/run_experiment.sbatch, since AutoGluon-extreme is a
whole-predictor AGExperiment baseline with its own jobspec format
(write_jobspec_autogluon's 8-field AutoGluonTask), not a per-model
ConfigGenerator run.

Usage:
    python cluster/submit_autogluon_1h_missing_htw.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cluster"))

from submit_job import (  # noqa: E402
    _sbatch_with_retry,
    resolve_gpu_flags,
    resolve_mem_flags,
    resolve_profile,
    write_jobspec_autogluon,
)

PROFILE_PATH = "/Users/koddenbrock/Repository/raman_bench_paper/cluster/profiles/htw.yaml"
RESULTS_DIR = "results/v1/data"
CACHE_DIR = ".cache_v1"
MIRROR_REPO = "HTW-KI-Werkstatt/RamanBench"
TIME_LIMIT = 3600
BUDGET_LABEL = "1h"
N_SPLITS = 3
THROTTLE = 4

# (dataset, target_idx, repeat, fold, n_repeats) -- exact diff of
# target_list.json's non-excluded (dataset, target_idx, repeat, fold) tuples
# against real AutoGluon_extreme_1h/*/*/results.pkl files on the BHT PVC.
MISSING = [
    ("bacteria_identification", 0, 0, 0, 1),
    ("bacteria_identification", 0, 0, 1, 1),
    ("bacteria_identification", 0, 0, 2, 1),
    ("cspp_serum_metabolites", 0, 0, 0, 1),
    ("cspp_serum_metabolites", 0, 0, 1, 1),
    ("cspp_serum_metabolites", 0, 0, 2, 1),
    ("diabetes_skin_ear_lobe", 0, 0, 0, 1),
    ("diabetes_skin_ear_lobe", 0, 0, 1, 1),
    ("diabetes_skin_ear_lobe", 0, 0, 2, 1),
    ("diabetes_skin_vein", 0, 0, 0, 1),
    ("diabetes_skin_vein", 0, 0, 1, 1),
    ("diabetes_skin_vein", 0, 0, 2, 1),
    ("marine_pathogens", 0, 0, 0, 1),
    ("marine_pathogens", 0, 0, 1, 1),
    ("marine_pathogens", 0, 0, 2, 1),
    ("marine_pathogens_binary", 0, 0, 0, 1),
    ("marine_pathogens_binary", 0, 0, 1, 1),
    ("marine_pathogens_binary", 0, 0, 2, 1),
    ("mlrod", 0, 0, 2, 1),
]


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    jobs = [
        (dataset, target_idx, repeat, fold, TIME_LIMIT, BUDGET_LABEL, n_repeats, N_SPLITS)
        for dataset, target_idx, repeat, fold, n_repeats in MISSING
    ]
    print(f"{len(jobs)} missing AutoGluon-extreme-{BUDGET_LABEL} task(s) to submit on HTW")

    profile = resolve_profile(PROFILE_PATH, None)
    assert profile.get("slurm"), f"{PROFILE_PATH} is not a SLURM profile"

    slug = f"extreme-{BUDGET_LABEL}-missing-htw"
    jobspec_path = write_jobspec_autogluon(jobs, slug)
    print(f"  jobspec: {jobspec_path}")

    sbatch_args = [
        "sbatch",
        f"--array=0-{len(jobs) - 1}%{THROTTLE}",
        f"--job-name=RB_AUTOGLUON-EXTREME-{BUDGET_LABEL.upper()}_{slug}",
        "--cpus-per-task", str(profile.get("default_cpus_per_task", 16)),
        "--time", profile.get("default_time", "10-00:00:00"),
    ]
    if profile.get("workspace"):
        sbatch_args += ["--chdir", str(Path(profile["workspace"]).expanduser())]
    sbatch_args += resolve_mem_flags(profile, "AUTOGLUON")
    sbatch_args += resolve_gpu_flags(profile, True)
    if profile.get("account"):
        sbatch_args.append(f"--account={profile['account']}")
    if profile.get("partition"):
        sbatch_args.append(f"--partition={profile['partition']}")
    if profile.get("mail_user"):
        sbatch_args += [
            f"--mail-user={profile['mail_user']}", f"--mail-type={profile.get('mail_type', 'FAIL')}"
        ]
    sbatch_args += profile.get("extra_sbatch_args", [])

    export_vars = ",".join([
        f"RESULTS_DIR={RESULTS_DIR}",
        f"CACHE_DIR={CACHE_DIR}",
        f"MIRROR_REPO={MIRROR_REPO}",
        "USE_GPU=1",
        f"JOBSPEC={jobspec_path}",
        f"ACTIVATION={profile.get('activation') or 'conda'}",
        f"CONDA_ENV={profile.get('conda_env') or ''}",
        f"VENV_PATH={profile.get('venv_path') or ''}",
        f"WORKSPACE={profile.get('workspace') or ''}",
    ])
    sbatch_args += ["--export", export_vars, str(REPO / "cluster" / "run_autogluon_baseline.sbatch")]

    print(f"  {' '.join(sbatch_args)}")
    if dry_run:
        print("Dry run -- not submitting.")
        return

    result = _sbatch_with_retry(sbatch_args)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
