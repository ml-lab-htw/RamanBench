#!/usr/bin/env python
"""One-off CPU-only k8s Job that runs scripts/backfill_cpu_legacy_gpu_json.py
against the real results tree on the PVC -- reclassifies the "unknown"
sentinel to confirmed "cpu" for PLS/CAT/DUMMY/KNN/LR/RF/XGB/XT's historical
HTW-cluster results. Same ConfigMap-mounted-script pattern as
cluster/submit_gpu_hardware_recovery.py (this script doesn't exist in the
already-built image either).

Usage::

    python cluster/submit_backfill_cpu_legacy.py --profile <path/to/k8s.yaml>
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml
from submit_job import _k8s_name, load_profile  # noqa: E402 (same dir)

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "backfill_cpu_legacy_gpu_json.py"


def build_manifest(profile: dict) -> tuple[dict, str, str, str]:
    namespace = profile["namespace"]
    job_name = _k8s_name("rb-backfill-cpu-legacy")
    configmap_name = _k8s_name("rb-backfill-cpu-legacy-script")
    pvc_mount = profile.get("pvc_mount_path", "/data")

    env = [
        {"name": "RESULTS_DIR", "value": f"{pvc_mount}/RamanBench_v1/results/v1/data"},
    ]

    pod_spec = {
        "restartPolicy": "Never",
        "containers": [{
            "name": "backfill-cpu-legacy",
            "image": profile["image"],
            "imagePullPolicy": "Always",
            "command": ["/bin/bash", "-c",
                        'python /scripts/backfill_cpu_legacy_gpu_json.py --results-dir "$RESULTS_DIR"'],
            "env": env,
            "resources": {
                "requests": {"cpu": "2", "memory": "4G"},
                "limits": {"cpu": "4", "memory": "8G"},
            },
            "volumeMounts": [
                {"name": "workspace", "mountPath": pvc_mount},
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
         "--from-file", f"backfill_cpu_legacy_gpu_json.py={SCRIPT_PATH}", "--dry-run=client", "-o", "yaml"],
        check=True, capture_output=True, text=True,
    ).stdout

    subprocess.run(["kubectl", "apply", "--validate=false", "-f", "-"], input=cm_yaml, text=True, check=True)
    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"],
        input=yaml.safe_dump(manifest), text=True, check=True,
    )
    print(f"Done: k8s Job {job_name!r} created in namespace {namespace!r} (CPU-only, 1 pod).")


if __name__ == "__main__":
    main()
