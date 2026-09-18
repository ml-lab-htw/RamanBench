#!/usr/bin/env python
"""Create/update wandb Workspace views (per model) and one cross-model Report.

Built for the per-model project sharding in run_experiment.wandb_project_for_model
(one wandb project per model, e.g. "raman-bench-cat", to stay under W&B's
documented 10,000-runs-per-project recommendation -- see that function's
docstring). A single live workspace can't span multiple projects, so this
script creates two different kinds of view:

    1. One Workspace per model's own project -- bar charts of mean
       metric_error / time_train_s / time_infer_s grouped by dataset, so you
       can see how one model does across every dataset it was run on.
    2. One cross-model Report -- a single PanelGrid whose Runsets each point
       at a DIFFERENT per-model project (wr.Runset(project=...) supports
       this), combined into shared bar charts grouped by the "model" config
       field, so you can compare all models against each other in one place.
       (Live cross-project Workspaces aren't supported by the API; a Report's
       PanelGrid is the documented way to combine multiple projects' runs.)

Idempotent via a local manifest file (`.wandb_views_manifest.json`, not
git-tracked -- it's just a per-machine record of which view/report URL to
update next time, not shared state) -- re-running this script updates the
SAME views in place (via Workspace.from_url / Report.from_url) instead of
creating duplicates every time. Delete the manifest file to force fresh
creation instead of updates.

Requires: pip install wandb-workspaces (kept out of the `tracking` extra --
this is a one-off provisioning script, not something every task run needs).

Usage
-----
    python scripts/setup_wandb_views.py --dry-run
    python scripts/setup_wandb_views.py
    python scripts/setup_wandb_views.py --models CAT XGB RF   # subset, for testing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_experiment import wandb_project_for_model  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "scripts" / ".wandb_views_manifest.json"

# Per-dataset breakdown metrics -- the ones confirmed present on a real cached
# result (see run_experiment.build_wandb_metrics) that are meaningful to
# average across a model's (repeat, fold, config) runs for one dataset.
_DATASET_BREAKDOWN_METRICS = ["metric_error", "time_train_s", "time_infer_s"]


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _model_workspace_sections(model_key: str) -> list:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    panels = [
        wr.BarPlot(
            title=f"{model_key}: mean {metric} by dataset",
            metrics=[metric],
            groupby=wr.Config("dataset"),
            groupby_aggfunc="mean",
            orientation="h",
        )
        for metric in _DATASET_BREAKDOWN_METRICS
    ]
    return [ws.Section(name="Per-dataset breakdown", panels=panels, is_open=True)]


def _existing_projects(entity: str) -> set[str]:
    """W&B requires a project to already exist (i.e. have at least one logged
    run) before a Workspace/Runset can be created against it -- confirmed
    live: creating a workspace for a not-yet-logged-to model's project fails
    with `WandbApiFailedError: project <entity>/<project> not found during
    upsertView`. Since scripts/backfill_wandb.py logs models in
    results.pkl-glob (sorted-path) order, not every model's project exists
    yet while it's still running -- this lets callers skip those cleanly
    instead of crashing partway through the model list."""
    import wandb

    return {p.name for p in wandb.Api().projects(entity)}


def upsert_model_workspace(*, entity: str, project: str, model_key: str, manifest: dict, dry_run: bool) -> None:
    manifest_key = f"workspace::{project}"
    existing_url = manifest.get(manifest_key)

    if dry_run:
        print(f"[dry-run] would {'update' if existing_url else 'create'} workspace for project={project}")
        return

    import wandb_workspaces.workspaces as ws

    if existing_url:
        workspace = ws.Workspace.from_url(existing_url)
        workspace.sections = _model_workspace_sections(model_key)
    else:
        workspace = ws.Workspace(
            name=f"{model_key} overview", entity=entity, project=project,
            sections=_model_workspace_sections(model_key),
        )
    workspace.save()
    manifest[manifest_key] = workspace.url
    _save_manifest(manifest)  # incremental -- a later crash (e.g. a not-yet-existing
    # model project, see _existing_projects) must not lose this successful upsert,
    # or a retry creates a DUPLICATE view instead of updating this one (confirmed
    # live: exactly this happened once before this fix was added).
    print(f"  {project}: {workspace.url}")


def _cross_model_blocks(entity: str, base_project: str, model_keys: list[str]) -> list:
    import wandb_workspaces.reports.v2 as wr

    runsets = [
        wr.Runset(entity=entity, project=wandb_project_for_model(m, base_project=base_project), name=m)
        for m in model_keys
    ]
    panels = [
        wr.BarPlot(
            title=f"Mean {metric} by model (all datasets combined)",
            metrics=[metric],
            groupby=wr.Config("model"),
            groupby_aggfunc="mean",
            orientation="h",
            max_bars_to_show=len(model_keys),
        )
        for metric in _DATASET_BREAKDOWN_METRICS
    ]
    return [wr.PanelGrid(runsets=runsets, panels=panels)]


def upsert_cross_model_report(*, entity: str, base_project: str, model_keys: list[str],
                               manifest: dict, dry_run: bool) -> None:
    manifest_key = "report::cross_model"
    existing_url = manifest.get(manifest_key)

    if dry_run:
        print(f"[dry-run] would {'update' if existing_url else 'create'} cross-model report "
              f"covering {len(model_keys)} model project(s)")
        return

    import wandb_workspaces.reports.v2 as wr

    if existing_url:
        report = wr.Report.from_url(existing_url)
        report.blocks = _cross_model_blocks(entity, base_project, model_keys)
    else:
        report = wr.Report(
            entity=entity, project=base_project, title="RamanBench v1: Cross-Model Comparison",
            description="Mean metric_error/time_train_s/time_infer_s per model, combined across "
                         "every per-model wandb project (see run_experiment.wandb_project_for_model).",
            blocks=_cross_model_blocks(entity, base_project, model_keys),
        )
    report.save()
    manifest[manifest_key] = report.url
    _save_manifest(manifest)
    print(f"  cross-model report: {report.url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--base-project", default=os.environ.get("WANDB_PROJECT", "raman-bench"))
    parser.add_argument("--scope", default=str(REPO_ROOT / "configs/v1/scope_default.json"))
    parser.add_argument("--models", nargs="+", default=None,
                         help="Subset of model keys (default: every model in --scope)")
    parser.add_argument("--skip-cross-model-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set -- export it first, or pass --dry-run to preview.")

    model_keys = args.models or json.loads(Path(args.scope).read_text())["models"]

    entity = args.entity
    if entity is None and not args.dry_run:
        import wandb

        entity = wandb.Api().default_entity
        print(f"No --entity given -- using your wandb default entity: {entity!r}")
    elif entity is None:
        entity = "<default-entity>"

    manifest = _load_manifest()

    if args.dry_run:
        available_models = model_keys
    else:
        existing_projects = _existing_projects(entity)
        available_models = [m for m in model_keys
                             if wandb_project_for_model(m, base_project=args.base_project) in existing_projects]
        skipped = [m for m in model_keys if m not in available_models]
        if skipped:
            print(f"Skipping {len(skipped)} model(s) with no wandb project yet "
                  f"(no runs logged for them so far -- e.g. the backfill hasn't reached them): {skipped}")

    print(f"Creating/updating {len(available_models)} per-model workspace(s)...")
    for model_key in available_models:
        project = wandb_project_for_model(model_key, base_project=args.base_project)
        upsert_model_workspace(entity=entity, project=project, model_key=model_key, manifest=manifest, dry_run=args.dry_run)

    if not args.skip_cross_model_report:
        print("Creating/updating the cross-model comparison report...")
        model_keys = available_models  # only include projects that actually exist
        upsert_cross_model_report(
            entity=entity, base_project=args.base_project, model_keys=model_keys,
            manifest=manifest, dry_run=args.dry_run,
        )

    if not args.dry_run:
        print(f"Manifest at {MANIFEST_PATH} is up to date (saved incrementally after each upsert).")


if __name__ == "__main__":
    main()
