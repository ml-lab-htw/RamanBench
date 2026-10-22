#!/usr/bin/env python
"""Run the leaderboard roster on benchmark v2's own targets (k8s / BHT).

The v2-only targets are the entries of configs/v2/target_list.json with ``since: "v2"``;
the v1 targets reuse the v1 results (RELEASING.md, "Branches"). Three parts, each its own
set of k8s Jobs:

* ``models``: every model of configs/v2/scope_default.json that runs through
  run_experiment.py, one Job per (image, CPU/GPU) group as in submit_dataset_sweep.py, but
  split into pods of ``--tasks-per-pod`` tasks. Tasks are ordered target-major, so one pod
  runs many models on one target instead of one slow model on many targets.
* ``systems``: TABFM-PLUS through run_system.py, one pod per target.
* ``autogluon``: the AUTOGLUON-EXTREME 5m and 1h references, as submit_autogluon_baseline.py.

``--job-per-model`` instead gives every model (and each AutoGluon budget) its own Job with a
single pod that runs all of that model's tasks, created paused (parallelism 0) so a controller
can start them one model at a time. Such a pod runs for days, so an eviction or preemption
(pod condition DisruptionTarget) does not count as a failure and a crash is retried twice;
exit code 1 (some task failed, the rest ran) ends the index. Finished tasks are skipped on a
retry.

Every pod installs raman-data (``--raman-data``, which knows the v2 datasets) and runs the
code snapshot ``--code-dir`` on the PVC instead of the image's copy. Results go to the scope's
results_dir (results/v2/data under the profile workspace), with min_groups_per_class=3.

Not run: CHAKRA-TAB and CHAKRA-TAB-FULL (hosted API, run from a workstation), and TABSTAR and
AUTOGLUON-EXTREME-4H (not in the reference). ``--skip-dataset`` (default glioma_idh, whose
results were copied from its v1 sweep) leaves datasets out.

Usage:
    python cluster/submit_v2_sweep.py --profile PROFILE --code-dir /data/RamanBench_v1/code_v2_<sha> \\
        --image REGISTRY/ramanbench@sha256:... [--part models systems autogluon] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR))
REPO_ROOT = CLUSTER_DIR.parent

from submit_autogluon_baseline import EXTREME_TIME_BUDGETS  # noqa: E402
from submit_job import (  # noqa: E402
    GPU_MODELS,
    _chunk,
    _max_memory_tier,
    build_k8s_job_manifest_autogluon,
    build_k8s_job_manifest_multimodel,
    resolve_k8s_image,
    resolve_profile,
    write_jobspec_autogluon,
    write_jobspec_multimodel,
)

NOT_RUN = {"CHAKRA-TAB", "CHAKRA-TAB-FULL", "TABSTAR", "AUTOGLUON-EXTREME-4H"}
SYSTEMS = {"TABFM-PLUS"}
AUTOGLUON = {"AUTOGLUON-EXTREME-5M": "5m", "AUTOGLUON-EXTREME-1H": "1h"}


def boot(code_dir: str, raman_data: str, entrypoint: str) -> list[str]:
    """Container command: install raman-data, then run *entrypoint* from the code snapshot."""
    return ["/bin/bash", "-c", "; ".join([
        f"pip install -q --no-deps --no-cache-dir --break-system-packages 'raman-data=={raman_data}'",
        f"cd {code_dir}",
        f"export PYTHONPATH={code_dir}/src",
        entrypoint,
    ])]


def set_container(manifest: dict, command: list[str], min_groups: int) -> None:
    c = manifest["spec"]["template"]["spec"]["containers"][0]
    c["command"] = command
    env = [e for e in c["env"] if e["name"] != "MIN_GROUPS_PER_CLASS"]
    c["env"] = env + [{"name": "MIN_GROUPS_PER_CLASS", "value": str(min_groups)}]


def run_alone(manifest: dict) -> None:
    """Make *manifest* a paused one-pod Job that survives evictions (see --job-per-model)."""
    spec = manifest["spec"]
    spec.update(parallelism=0, backoffLimit=3, backoffLimitPerIndex=2, maxFailedIndexes=1)
    spec["podFailurePolicy"] = {"rules": [
        {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]},
        {"action": "FailIndex", "onExitCodes": {"operator": "In", "values": [1]}},
    ]}


def apply(profile: dict, configmap_name: str, files: dict[str, Path], manifest: dict) -> None:
    namespace = profile.get("namespace", "default")
    cm_cmd = ["kubectl", "create", "configmap", configmap_name, "-n", namespace, "--dry-run=client", "-o", "yaml"]
    for key, path in files.items():
        cm_cmd += ["--from-file", f"{key}={path}"]
    cm_yaml = subprocess.run(cm_cmd, check=True, capture_output=True, text=True).stdout
    # --validate=false: see submit_dataset_sweep.py (API server timeouts on schema download).
    subprocess.run(["kubectl", "apply", "--validate=false", "-f", "-"], input=cm_yaml, text=True, check=True)
    import yaml

    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"], input=yaml.safe_dump(manifest), text=True, check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--code-dir", required=True, help="Code snapshot on the PVC (absolute path in the pod)")
    parser.add_argument("--image", required=True, help="Pinned main image (registry/ramanbench@sha256:...)")
    parser.add_argument("--raman-data", default="1.7.0", help="raman-data version installed in every pod")
    parser.add_argument("--scope", default=str(REPO_ROOT / "configs" / "v2" / "scope_default.json"))
    parser.add_argument("--part", nargs="+", default=["models", "systems", "autogluon"],
                        choices=["models", "systems", "autogluon"])
    parser.add_argument("--models", nargs="*", default=None, help="Only these models (default: the scope's)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Only these datasets (default: all v2 ones)")
    parser.add_argument("--skip-dataset", nargs="*", default=["glioma_idh"])
    parser.add_argument("--tag", default="", help="Suffix for Job names, e.g. 'test' for a verification run")
    parser.add_argument("--tasks-per-pod", type=int, default=30)
    parser.add_argument("--throttle", type=int, default=None, help="Pods at once per Job (default: the scope's)")
    parser.add_argument("--job-per-model", action="store_true",
                        help="One paused one-pod Job per model / AutoGluon budget (see the docstring)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scope = json.loads(Path(args.scope).read_text())
    targets = json.loads((Path(args.scope).parent / scope["targets_file"]).read_text())
    targets = [
        t for t in targets
        if t.get("since") == "v2" and not t.get("excluded") and t["dataset"] not in set(args.skip_dataset)
        and (not args.datasets or t["dataset"] in args.datasets)
    ]
    prefix = f"v2{args.tag}"
    n_splits, min_groups = scope["n_splits"], int(scope.get("min_groups_per_class", 0))
    results_dir, cache_dir = scope["results_dir"], scope["cache_dir"]
    mirror_repo = scope.get("mirror_repo", "HTW-KI-Werkstatt/RamanBench")
    throttle = args.throttle or scope.get("throttle", 6)
    roster = [m for m in (args.models or scope["models"]) if m not in NOT_RUN]
    print(f"{len(targets)} v2 target(s), {len(roster)} model(s)/system(s); results -> {results_dir}")

    profile = resolve_profile(args.profile, "k8s")
    if profile.get("backend") != "k8s":
        sys.exit("submit_v2_sweep.py is k8s-only.")
    profile = deepcopy(profile)
    profile["image"] = args.image
    for key in ("wandb_project", "wandb_entity"):
        profile.pop(key, None)
    created = []

    if "models" in args.part:
        groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
        for model in roster:
            if model not in SYSTEMS and model not in AUTOGLUON:
                key = (resolve_k8s_image(profile, model), model in GPU_MODELS)
                groups[(*key, model) if args.job_per_model else key].append(model)
        for (image, use_gpu, *_), models in groups.items():
            # target-major: a pod's tasks span many models of one (target, fold)
            tasks = [
                (m, t["dataset"], t["target_idx"], 0, fold, 0, t.get("n_repeats", 1))
                for t in targets for fold in range(n_splits) for m in models
            ]
            tier = "gpu" if use_gpu else "cpu"
            image_slug = image.rsplit("/", 1)[-1].replace("ramanbench", "rb").replace("@sha256:", "-")[:16]
            group_slug = f"{prefix}-{image_slug}-{tier}"
            tasks_per_pod = args.tasks_per_pod
            if args.job_per_model:
                group_slug = f"{prefix}-{models[0].lower().replace('_', '-').replace('.', '')}"
                tasks_per_pod = len(tasks)
            mem = _max_memory_tier([profile.get("mem_tiers", {}).get(m, profile.get("default_memory", "64G"))
                                    for m in models])
            for part, chunk in enumerate(_chunk(tasks, profile.get("max_array_size", 5000))):
                part_slug = f"p{part}"
                jobspec = write_jobspec_multimodel(
                    chunk, f"{group_slug}_{part_slug}",
                    default_time_limit=scope["time_limit"],
                    dataset_time_limit_overrides=scope.get("time_limit_overrides", {}),
                    model_time_limit_overrides=scope.get("model_time_limit_overrides", {}),
                    max_train_samples_overrides=scope.get("max_train_samples_overrides", {}),
                    model_max_train_samples_overrides=scope.get("model_max_train_samples_overrides", {}),
                )
                job, cm, n_pods, manifest = build_k8s_job_manifest_multimodel(
                    group_slug=group_slug, part_slug=part_slug, n_tasks=len(chunk), n_splits=n_splits,
                    num_random_configs=scope["num_random_configs"], num_bag_folds=scope["num_bag_folds"],
                    time_limit=scope["time_limit"], results_dir=results_dir, cache_dir=cache_dir,
                    mirror_repo=mirror_repo, profile=profile, image=image, mem_tier=mem, use_gpu=use_gpu,
                    tasks_per_pod=tasks_per_pod, throttle=throttle, min_groups_per_class=min_groups,
                )
                set_container(manifest, boot(args.code_dir, args.raman_data,
                                             "bash cluster/k8s_entrypoint_by_dataset.sh"), min_groups)
                if args.job_per_model:
                    run_alone(manifest)
                print(f"models {tier} {image.rsplit('/', 1)[-1][:40]}: {len(models)} model(s), {len(chunk)} task(s), "
                      f"{n_pods} pod(s) of {tasks_per_pod}, mem {mem}, job {job}")
                if not args.dry_run:
                    apply(profile, cm, {"jobspec.txt": jobspec}, manifest)
                    created.append(job)

    if "systems" in args.part:
        for system in sorted(SYSTEMS & set(roster)):
            tasks_file = REPO_ROOT / ".slurm_jobspecs" / f"{prefix}-{system.lower()}.txt"
            tasks_file.parent.mkdir(exist_ok=True)
            tasks_file.write_text("".join(f"{t['dataset']}__{t['target_idx']}\n" for t in targets))
            slug = f"{prefix}-{system.lower()}"
            job, cm, n_pods, manifest = build_k8s_job_manifest_multimodel(
                group_slug=slug, part_slug="p0", n_tasks=len(targets), n_splits=n_splits,
                num_random_configs=0, num_bag_folds=0, time_limit=scope["time_limit"],
                results_dir=results_dir, cache_dir=cache_dir, mirror_repo=mirror_repo, profile=profile,
                image=args.image, mem_tier=profile.get("default_memory", "64G"), use_gpu=True,
                tasks_per_pod=1, throttle=throttle, min_groups_per_class=min_groups,
            )
            run = (
                'TASK=$(sed -n "$((JOB_COMPLETION_INDEX + 1))p" /jobspec/jobspec.txt); '
                f'python scripts/run_system.py --system {system} --protocol configs/v2/protocol_v2_only.json '
                '--tasks "$TASK" --results-dir "$RESULTS_DIR" --cache-dir "$CACHE_DIR" --mirror-repo "$MIRROR_REPO"'
            )
            set_container(manifest, boot(args.code_dir, args.raman_data, run), min_groups)
            print(f"systems {system}: {len(targets)} pod(s), one target each, job {job}")
            if not args.dry_run:
                apply(profile, cm, {"jobspec.txt": tasks_file}, manifest)
                created.append(job)

    if "autogluon" in args.part:
        budgets = [b for m, b in AUTOGLUON.items() if m in roster]
        for budget_group in ([[b] for b in budgets] if args.job_per_model else [budgets]):
            submit_autogluon(args, profile, targets, budget_group, prefix, n_splits, min_groups,
                             results_dir, cache_dir, mirror_repo, throttle, created)

    print(f"\n{'Dry run, nothing submitted' if args.dry_run else f'Created {len(created)} Job(s): {created}'}")


def submit_autogluon(args, profile, targets, budgets, prefix, n_splits, min_groups,
                     results_dir, cache_dir, mirror_repo, throttle, created) -> None:
    jobs = [
        (t["dataset"], t["target_idx"], 0, fold, EXTREME_TIME_BUDGETS[b], b, t.get("n_repeats", 1), n_splits)
        for b in budgets for t in targets for fold in range(n_splits)
    ]
    if jobs:
        slug = f"{prefix}-ag-extreme-" + "-".join(budgets)
        jobspec = write_jobspec_autogluon(jobs, slug)
        job, cm, n_pods, manifest = build_k8s_job_manifest_autogluon(
            slug=slug, n_tasks=len(jobs), results_dir=results_dir, cache_dir=cache_dir,
            mirror_repo=mirror_repo, profile=profile, use_gpu=True,
            tasks_per_pod=len(jobs) if args.job_per_model else 4, throttle=throttle,
        )
        set_container(manifest, boot(args.code_dir, args.raman_data,
                                     "bash cluster/k8s_entrypoint_autogluon.sh"), min_groups)
        if args.job_per_model:
            run_alone(manifest)
        print(f"autogluon {'/'.join(budgets)}: {len(jobs)} task(s), {n_pods} pod(s), job {job}")
        if not args.dry_run:
            apply(profile, cm, {"jobspec.txt": jobspec}, manifest)
            created.append(job)


if __name__ == "__main__":
    main()
