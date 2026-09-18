"""Regression tests for cluster/submit_job.py's k8s manifest construction.

Covers the real production bugs found and fixed during the BHT k8s rollout
(2026-09-18): stale mutable-tag images silently reused (imagePullPolicy),
silent data loss from a `cd $WORKSPACE` that only existed in a directory
without code (_abs_under_workspace), missing TabPFN license token
(tabpfn_secret), and shared-tenancy scheduling courtesy (priority_class_name).
These were previously untestable because the manifest was built inline inside
`submit_jobs_k8s`, reachable only via real kubectl calls.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_CLUSTER_DIR = Path(__file__).resolve().parent.parent / "cluster"
if str(_CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_DIR))

_MODULE_PATH = _CLUSTER_DIR / "submit_job.py"
_spec = importlib.util.spec_from_file_location("submit_job", _MODULE_PATH)
submit_job = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("submit_job", submit_job)
_spec.loader.exec_module(submit_job)

_abs_under_workspace = submit_job._abs_under_workspace
_build_k8s_job_manifest = submit_job._build_k8s_job_manifest


def _minimal_profile(**overrides) -> dict:
    profile = {
        "namespace": "test-namespace",
        "image": "registry.example.com/ns/ramanbench:v1",
        "pvc_claim_name": "raman-bench-pvc",
    }
    profile.update(overrides)
    return profile


def _build(profile, **overrides):
    kwargs = dict(
        model="RamanPFN",
        part_slug="full",
        n_tasks=10,
        n_splits=5,
        num_random_configs=1,
        num_bag_folds=8,
        time_limit=3600,
        results_dir="results/v1/data",
        cache_dir="cache/v1",
        mirror_repo="HTW-KI-Werkstatt/RamanBench",
        profile=profile,
        use_gpu=True,
        tasks_per_pod=5000,
        throttle=10,
    )
    kwargs.update(overrides)
    return _build_k8s_job_manifest(**kwargs)


class TestAbsUnderWorkspace:
    def test_relative_path_is_joined_under_workspace(self):
        assert _abs_under_workspace("results/v1/data", "/data") == "/data/results/v1/data"

    def test_workspace_trailing_slash_does_not_double_up(self):
        assert _abs_under_workspace("results/v1/data", "/data/") == "/data/results/v1/data"

    def test_already_absolute_path_passes_through_unchanged(self):
        assert _abs_under_workspace("/already/absolute/path", "/data") == "/already/absolute/path"

    def test_empty_workspace_leaves_relative_path_unchanged(self):
        assert _abs_under_workspace("results/v1/data", "") == "results/v1/data"


class TestBuildK8sJobManifest:
    def test_image_pull_policy_is_always_regardless_of_profile(self):
        # A node that already cached an earlier pull of a mutable tag must not
        # silently keep running a stale image after a rebuild+push under the
        # same tag -- confirmed happening in practice on this cluster.
        _, _, _, manifest = _build(_minimal_profile())
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["imagePullPolicy"] == "Always"

    def test_results_and_cache_dirs_resolved_absolute_under_workspace(self):
        profile = _minimal_profile(workspace="/data")
        _, _, _, manifest = _build(profile, results_dir="results/v1/data", cache_dir="cache/v1")
        env = {e["name"]: e["value"] for e in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
               if "value" in e}
        assert env["RESULTS_DIR"] == "/data/results/v1/data"
        assert env["CACHE_DIR"] == "/data/cache/v1"

    def test_results_and_cache_dirs_unchanged_without_workspace(self):
        profile = _minimal_profile()
        _, _, _, manifest = _build(profile, results_dir="results/v1/data", cache_dir="cache/v1")
        env = {e["name"]: e["value"] for e in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
               if "value" in e}
        assert env["RESULTS_DIR"] == "results/v1/data"
        assert env["CACHE_DIR"] == "cache/v1"

    def test_tabpfn_secret_env_wired_when_configured(self):
        profile = _minimal_profile(tabpfn_secret="tabpfn-secret")
        _, _, _, manifest = _build(profile)
        env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        tabpfn_entries = [e for e in env if e["name"] == "TABPFN_TOKEN"]
        assert len(tabpfn_entries) == 1
        assert tabpfn_entries[0]["valueFrom"]["secretKeyRef"] == {
            "name": "tabpfn-secret", "key": "TABPFN_TOKEN",
        }

    def test_tabpfn_secret_env_absent_when_not_configured(self):
        _, _, _, manifest = _build(_minimal_profile())
        env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        assert not any(e["name"] == "TABPFN_TOKEN" for e in env)

    def test_hf_secret_wires_both_hf_env_vars(self):
        profile = _minimal_profile(hf_secret="hf-secret")
        _, _, _, manifest = _build(profile)
        env = {e["name"] for e in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} <= env

    def test_wandb_secret_and_optional_project_entity(self):
        profile = _minimal_profile(wandb_secret="wandb-secret", wandb_project="raman-bench",
                                    wandb_entity="ml-lab-htw")
        _, _, _, manifest = _build(profile)
        env = {e["name"]: e for e in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env["WANDB_API_KEY"]["valueFrom"]["secretKeyRef"] == {
            "name": "wandb-secret", "key": "WANDB_API_KEY",
        }
        assert env["WANDB_PROJECT"]["value"] == "raman-bench"
        assert env["WANDB_ENTITY"]["value"] == "ml-lab-htw"

    def test_wandb_env_absent_when_not_configured(self):
        _, _, _, manifest = _build(_minimal_profile())
        env = {e["name"] for e in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert "WANDB_API_KEY" not in env

    def test_priority_class_name_set_only_when_configured(self):
        profile = _minimal_profile(priority_class_name="unimportant")
        _, _, _, manifest = _build(profile)
        pod_spec = manifest["spec"]["template"]["spec"]
        assert pod_spec["priorityClassName"] == "unimportant"

    def test_priority_class_name_absent_by_default(self):
        _, _, _, manifest = _build(_minimal_profile())
        pod_spec = manifest["spec"]["template"]["spec"]
        assert "priorityClassName" not in pod_spec

    def test_kaggle_secret_adds_volume_and_mount_only_when_configured(self):
        profile = _minimal_profile(kaggle_secret="kaggle-secret")
        _, _, _, manifest = _build(profile)
        pod_spec = manifest["spec"]["template"]["spec"]
        assert any(v["name"] == "kaggle" for v in pod_spec["volumes"])
        container = pod_spec["containers"][0]
        assert any(m["name"] == "kaggle" for m in container["volumeMounts"])

    def test_kaggle_volume_absent_without_secret(self):
        _, _, _, manifest = _build(_minimal_profile())
        pod_spec = manifest["spec"]["template"]["spec"]
        assert not any(v["name"] == "kaggle" for v in pod_spec["volumes"])

    def test_image_pull_secret_node_selector_and_affinity_conditionals(self):
        profile = _minimal_profile(
            image_pull_secret="regcred",
            node_selector={"gpu": "a100"},
            node_affinity_match_expressions=[{"key": "gpu-mem", "operator": "Exists"}],
        )
        _, _, _, manifest = _build(profile)
        pod_spec = manifest["spec"]["template"]["spec"]
        assert pod_spec["imagePullSecrets"] == [{"name": "regcred"}]
        assert pod_spec["nodeSelector"] == {"gpu": "a100"}
        assert pod_spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"] == [{"matchExpressions": [{"key": "gpu-mem", "operator": "Exists"}]}]

    def test_no_optional_pod_spec_fields_without_profile_entries(self):
        _, _, _, manifest = _build(_minimal_profile())
        pod_spec = manifest["spec"]["template"]["spec"]
        assert "imagePullSecrets" not in pod_spec
        assert "nodeSelector" not in pod_spec
        assert "affinity" not in pod_spec

    def test_n_pods_is_ceil_division_of_tasks_per_pod(self):
        _, _, n_pods, _ = _build(_minimal_profile(), n_tasks=10001, tasks_per_pod=5000)
        assert n_pods == 3

    def test_n_pods_exact_multiple_does_not_overallocate(self):
        _, _, n_pods, _ = _build(_minimal_profile(), n_tasks=10000, tasks_per_pod=5000)
        assert n_pods == 2

    def test_parallelism_capped_by_throttle_and_n_pods(self):
        _, _, n_pods, manifest = _build(_minimal_profile(), n_tasks=3, tasks_per_pod=1, throttle=10)
        assert n_pods == 3
        assert manifest["spec"]["parallelism"] == 3

    def test_job_and_configmap_names_are_dns_1123_safe(self):
        job_name, configmap_name, _, _ = _build(_minimal_profile(), model="TabPFN_v2", part_slug="full")
        for name in (job_name, configmap_name):
            assert name == name.lower()
            assert all(c.isalnum() or c == "-" for c in name)
            assert len(name) <= 63

    def test_namespace_defaults_when_not_in_profile(self):
        profile = _minimal_profile()
        del profile["namespace"]
        _, _, _, manifest = _build(profile)
        assert manifest["metadata"]["namespace"] == "default"

    def test_extra_pod_labels_merged_into_job_labels(self):
        profile = _minimal_profile(extra_pod_labels={"team": "ml-lab"})
        _, _, _, manifest = _build(profile)
        assert manifest["metadata"]["labels"]["team"] == "ml-lab"
        assert manifest["metadata"]["labels"]["app"] == "raman-bench"
