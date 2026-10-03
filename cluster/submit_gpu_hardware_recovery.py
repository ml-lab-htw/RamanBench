#!/usr/bin/env python
"""One-off CPU-only k8s Job that runs scripts/write_gpu_hardware_files.py against
the real results tree on the PVC -- writes a gpu.json sidecar next to every
completed task's results.pkl (skipping any that already have one), so later
speed/timing analysis can compare only results produced on the same hardware.

No GPU needed (it only calls the wandb read API and writes small JSON files),
so this deliberately has no gpu node-selector/resources -- it can schedule on
any CPU node instead of competing with real benchmark jobs for a100 slots.

Usage::

    python cluster/submit_gpu_hardware_recovery.py --profile <path/to/k8s.yaml>
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml
from submit_job import _k8s_name, load_profile  # noqa: E402 (same dir)

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "write_gpu_hardware_files.py"


def build_manifest(profile: dict) -> tuple[dict, str, str, str]:
    namespace = profile["namespace"]
    job_name = _k8s_name("rb-gpu-hardware-recovery")
    configmap_name = _k8s_name("rb-gpu-hardware-recovery-script")

    env = [
        {"name": "RESULTS_DIR", "value": f"{profile.get('pvc_mount_path', '/data')}/RamanBench_v1/results/v1/data"},
    ]
    if profile.get("wandb_secret"):
        env.append({
            "name": "WANDB_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": profile["wandb_secret"], "key": "WANDB_API_KEY"}},
        })

    pod_spec = {
        "restartPolicy": "Never",
        "containers": [{
            "name": "write-gpu-hardware-files",
            "image": profile["image"],
            "imagePullPolicy": "Always",
            # Mounted from a ConfigMap (see configmap_name below), not baked into the
            # image -- this script didn't exist when :v1 was last built, and a one-off
            # admin task doesn't warrant a full image rebuild/push cycle.
            "command": ["/bin/bash", "-c",
                        'python /scripts/write_gpu_hardware_files.py --results-dir "$RESULTS_DIR"'],
            "env": env,
            "resources": {
                "requests": {"cpu": "2", "memory": "4G"},
                "limits": {"cpu": "4", "memory": "8G"},
            },
            "volumeMounts": [
                {"name": "workspace", "mountPath": profile.get("pvc_mount_path", "/data")},
                {"name": "script", "mountPath": "/scripts"},
            ],
        }],
        "volumes": [
            {"name": "workspace", "persistentVolumeClaim": {"claimName": profile["pvc_claim_name"]}},
            {"name": "script", "configMap": {"name": configmap_name}},
        ],
    }
    if profile.get("image_pull_secret"):
        pod_spec["imagePullSecrets"] = [{"name": profile["image_pull_secret"]}]
    if profile.get("priority_class_name"):
        pod_spec["priorityClassName"] = profile["priority_class_name"]

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace, "labels": {"app": "raman-bench"}},
        "spec": {
            "completions": 1,
            "parallelism": 1,
            "backoffLimit": 2,
            "template": {"metadata": {"labels": {"app": "raman-bench"}}, "spec": pod_spec},
        },
    }
    return manifest, job_name, namespace, configmap_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    manifest, job_name, namespace, configmap_name = build_manifest(profile)

    print(f"job: {job_name}  configmap: {configmap_name}  namespace: {namespace}")
    if args.dry_run:
        print(yaml.safe_dump(manifest))
        return

    cm_yaml = subprocess.run(
        ["kubectl", "create", "configmap", configmap_name, "-n", namespace,
         "--from-file", f"write_gpu_hardware_files.py={SCRIPT_PATH}", "--dry-run=client", "-o", "yaml"],
        check=True, capture_output=True, text=True,
    ).stdout

    # Same --validate=false rationale as submit_job.py: this cluster's API server has
    # repeatedly timed out on kubectl's client-side OpenAPI schema download.
    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"], input=cm_yaml, text=True, check=True,
    )
    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"],
        input=yaml.safe_dump(manifest), text=True, check=True,
    )
    print(f"Done: k8s Job {job_name!r} created in namespace {namespace!r} (CPU-only, 1 pod).")


if __name__ == "__main__":
    main()
