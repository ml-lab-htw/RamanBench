#!/usr/bin/env python
"""Sync the completed-work tree (``results/`` -- ``results.pkl`` presence is
what ``opportunistic_scheduler.py`` checks before queuing a task, see its
module docstring) between two cluster storage domains, so a newly-added
cluster's own scheduler doesn't re-run work another cluster already finished.

Why this is needed at all: completion tracking across this project is purely
filesystem-based (a ``results.pkl`` existing under a shared cache/results
directory). SLURM clusters (htw, tu) share a filesystem reachable over
ssh/rsync from wherever this script runs. A Kubernetes cluster's PVC is a
SEPARATE storage domain -- nothing on it is visible to the SLURM clusters'
schedulers (or vice versa) unless explicitly synced. This script is that
sync, run once for the initial bulk copy and then periodically (e.g. via
cron, see raman_bench_paper/cluster/CRON.md) to keep them converged.

This script is deliberately generic (no institution-specific hosts/paths) --
real source/destination values are supplied via CLI args or a profile
lookup, and belong in the private raman_bench_paper repo's own wrapper
script/cron entry, not here.

Transport: plain ``rsync`` over whatever ``--source``/``--dest`` paths you
give it. For a k8s PVC with no direct network path (the common case -- PVCs
usually aren't reachable over ssh from outside the cluster), route through a
short-lived "rsync pod" that mounts the PVC and exposes it via `kubectl cp`
or an rsync-over-stdin pipe (`kubectl exec`); see the --via-pod flag.

Usage
-----
    # Direct rsync (both sides reachable as regular rsync targets, e.g. two
    # SLURM clusters sharing a filesystem, or a k8s PVC exposed via NFS at a
    # mount path this machine can also see):
    python cluster/sync_results.py \\
        --source htw:~/workspace/RamanBench_v1/results/v1/data/ \\
        --dest /mnt/k8s-pvc/RamanBench_v1/results/v1/data/

    # Via a helper pod (no direct network path to the PVC) -- spins up a
    # minimal busybox pod with the PVC mounted, rsyncs into it over
    # `kubectl exec`, then tears the pod down:
    python cluster/sync_results.py \\
        --source htw:~/workspace/RamanBench_v1/results/v1/data/ \\
        --via-pod --pvc-claim-name raman-bench-pvc --namespace default \\
        --dest /data/RamanBench_v1/results/v1/data/

    # Dry run first (always recommended before the first bulk sync):
    python cluster/sync_results.py --source ... --dest ... --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

_HELPER_POD_NAME = "raman-bench-sync-helper"

_HELPER_POD_MANIFEST = """\
apiVersion: v1
kind: Pod
metadata:
  name: {name}
  namespace: {namespace}
spec:
  restartPolicy: Never
  containers:
    - name: rsync
      image: instrumentisto/rsync-ssh:latest
      command: ["sleep", "3600"]
      volumeMounts:
        - name: workspace
          mountPath: /data
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: {pvc_claim_name}
"""


def _run(cmd: list[str], *, dry_run: bool, **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, **kwargs)


def sync_direct(source: str, dest: str, *, delete: bool, dry_run: bool, extra_rsync_args: list[str]) -> None:
    """Plain rsync -- correct when both sides are reachable as ordinary rsync
    targets (ssh remote, or a local/NFS-mounted path)."""
    cmd = ["rsync", "-av", "--update"]  # --update: never overwrite a newer dest copy
    if delete:
        cmd.append("--delete")
    if dry_run:
        cmd.append("--dry-run")
    cmd += extra_rsync_args
    cmd += [source, dest]
    _run(cmd, dry_run=False)  # rsync's own --dry-run above handles the preview; always actually invoke rsync


def sync_via_pod(
    source: str, dest_in_pvc: str, *, pvc_claim_name: str, namespace: str,
    delete: bool, dry_run: bool, extra_rsync_args: list[str],
) -> None:
    """Spin up a short-lived helper pod with the target PVC mounted, then
    sync in TWO hops: (1) plain rsync ``source`` -> a local staging dir on
    this machine, (2) rsync that staging dir into the pod over `kubectl exec`.

    Two hops, not one, because rsync's remote-shell mode only supports ONE
    remote endpoint per invocation -- ``source`` (e.g. an ssh host like
    ``htw:...``) and the pod (reached via `kubectl exec`, not ssh) can't both
    be "remote" in a single rsync call. This mirrors how you'd do it by hand:
    pull to a local scratch dir, then push from there into the cluster.

    Use this whole function when the PVC has no direct network path (the
    common case for a k8s cluster's storage); use ``sync_direct`` instead if
    your PVC IS reachable as a plain path (e.g. exposed via NFS)."""
    import shutil
    import tempfile

    staging_dir = tempfile.mkdtemp(prefix="raman_bench_sync_")
    try:
        print(f"Hop 1/2: {source} -> {staging_dir}/ (local staging)")
        sync_direct(source, staging_dir + "/", delete=delete, dry_run=dry_run, extra_rsync_args=extra_rsync_args)

        print(f"Creating helper pod {_HELPER_POD_NAME!r} in namespace {namespace!r} (PVC {pvc_claim_name!r})...")
        manifest = _HELPER_POD_MANIFEST.format(name=_HELPER_POD_NAME, namespace=namespace, pvc_claim_name=pvc_claim_name)
        if dry_run:
            print(manifest)
        else:
            # --validate=false: see cluster/submit_job.py's own comment on the same
            # flag -- client-side `kubectl apply` schema validation has repeatedly
            # timed out against this cluster's API server.
            subprocess.run(
                ["kubectl", "apply", "--validate=false", "-f", "-"], input=manifest, text=True, check=True,
            )
            print("Waiting for helper pod to become Ready...")
            subprocess.run(
                ["kubectl", "wait", "--for=condition=Ready", f"pod/{_HELPER_POD_NAME}",
                 "-n", namespace, "--timeout=120s"],
                check=True,
            )

        try:
            print(f"Hop 2/2: {staging_dir}/ -> pod:{dest_in_pvc}")
            # rsync's remote-shell mode invokes: <rsh-argv...> <host> <remote-command...>
            # -- it always inserts a placeholder <host> token (taken from the
            # `host:path` destination argument) right before the remote command,
            # which plain `kubectl exec -- rsync --server ...` would then try to
            # exec as a command and fail. The `sh -c 'shift; exec ...' sh` wrapper
            # is the standard workaround: the extra "sh" gives the script a $0, the
            # inserted placeholder becomes $1, and `shift` drops it before exec'ing
            # the real rsync --server invocation inside the pod. NOT yet verified
            # against a real cluster -- test with --dry-run against a throwaway
            # pod/PVC first.
            remote_shell = (
                f"sh -c 'shift; exec kubectl exec -n {namespace} -i {_HELPER_POD_NAME} -- \"$@\"' sh"
            )
            cmd = ["rsync", "-av", "--update", "--rsh", remote_shell]
            if delete:
                cmd.append("--delete")
            if dry_run:
                cmd.append("--dry-run")
            cmd += extra_rsync_args
            cmd += [staging_dir + "/", f"{_HELPER_POD_NAME}:{dest_in_pvc}"]
            # Confirmed real bug (2026-09-18): the helper pod is only actually
            # created when dry_run is False (see above), so a rsync --dry-run
            # invocation here still needs a live remote to compare against --
            # it can't just preview, it crashes ("pods ... not found" ->
            # rsync "unexpected end of file"). Unlike sync_direct/hop 1 (a
            # local<->ssh-remote rsync, where --dry-run alone is meaningful),
            # this hop's dry-run can only be a fully skipped print, not a
            # real preview, since the pod side of the comparison doesn't
            # exist yet.
            if dry_run:
                print(f"  (skipped: helper pod not created under --dry-run) $ {' '.join(cmd)}")
            else:
                _run(cmd, dry_run=False)
        finally:
            print(f"Deleting helper pod {_HELPER_POD_NAME!r}...")
            _run(
                ["kubectl", "delete", "pod", _HELPER_POD_NAME, "-n", namespace, "--ignore-not-found"],
                dry_run=dry_run,
            )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="rsync source (local path or user@host:path)")
    parser.add_argument("--dest", required=True, help="rsync destination path")
    parser.add_argument("--via-pod", action="store_true", help="Route through a helper pod with the PVC mounted")
    parser.add_argument("--pvc-claim-name", help="Required with --via-pod")
    parser.add_argument("--namespace", default="default")
    parser.add_argument(
        "--delete", action="store_true",
        help="Pass rsync --delete (remove dest files absent from source). Off by default -- "
             "a k8s cluster's own in-progress results shouldn't be deleted just because the "
             "SLURM source hasn't produced them yet.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rsync-arg", action="append", default=[], dest="extra_rsync_args")
    args = parser.parse_args()

    if args.via_pod and not args.pvc_claim_name:
        parser.error("--via-pod requires --pvc-claim-name")

    start = time.time()
    if args.via_pod:
        sync_via_pod(
            args.source, args.dest, pvc_claim_name=args.pvc_claim_name, namespace=args.namespace,
            delete=args.delete, dry_run=args.dry_run, extra_rsync_args=args.extra_rsync_args,
        )
    else:
        sync_direct(args.source, args.dest, delete=args.delete, dry_run=args.dry_run, extra_rsync_args=args.extra_rsync_args)
    print(f"Done in {time.time() - start:.1f}s{' (dry run)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()
