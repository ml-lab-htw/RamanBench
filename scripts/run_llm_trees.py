#!/usr/bin/env python
"""Zero-shot LLM decision trees (``llm-trees``) on the v1 protocol folds.

The LLM sees no spectra. For each outer fold it gets a text description of the task (dataset name and
description, and the class names, from raman_data) and of 24 features, the mean min-max normalised
intensity in 24 equal-width wavenumber bands, and writes one decision tree. The training rows only
count the class fractions in the leaves (``predict_proba``). ``llm-trees`` talks to any OpenAI-compatible
server; point it at the HTW LiteLLM/vLLM service:

    export OPENROUTER_BASE_URL=https://f2ki-h100-1.f2.htw-berlin.de:11435/v1
    export OPENROUTER_API_KEY=$(cat ~/.litellm_key)      # never put the key in a file in the repo
    export OPENROUTER_MODEL=qwen3.8-27b
    python scripts/run_llm_trees.py --tasks diabetes_skin_ear_lobe__0 --folds 0

Per fold it writes ``<out>/<task>/fold<k>/`` with ``tree.py`` (the induced tree, the generated Python
source), ``prompt.txt`` (what the LLM was asked), ``result.json`` (metrics, features used, timing) and
``predictions.npz``. A finished fold is skipped, so an interrupted run resumes.

Needs the ``llm-trees`` package with ``OPENROUTER_BASE_URL`` support (ml-lab-htw/llm-trees#2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback

import numpy as np

N_BANDS = 24
N_SPLITS = 3
CLF_MAX_DEPTH_RANGE = (2, 5)
REG_MAX_DEPTH = 4


def band_features(X, n_bands: int = N_BANDS):
    """Per-spectrum min-max scaling, then the mean in *n_bands* equal-width bands. Returns (frame, edges)."""
    import pandas as pd

    shifts = X.columns.to_numpy(dtype=float)
    A = X.to_numpy(dtype=float)
    lo, hi = A.min(axis=1, keepdims=True), A.max(axis=1, keepdims=True)
    A = (A - lo) / np.where(hi > lo, hi - lo, 1.0)
    edges = np.linspace(shifts.min(), shifts.max(), n_bands + 1)
    idx = np.clip(np.searchsorted(edges, shifts, side="right") - 1, 0, n_bands - 1)
    cols = {}
    for b in range(n_bands):
        m = idx == b
        cols[f"band_{round(edges[b])}_{round(edges[b + 1])}"] = A[:, m].mean(axis=1) if m.any() else np.zeros(len(A))
    return pd.DataFrame(cols, index=X.index), edges


def describe(dataset_name, dataset, info, target_name, problem_type, labels, edges, names, y_range=None):
    """target and feature description dicts for llm-trees, from the raman_data metadata."""
    meta = info.metadata
    context = f"{meta.get('full_name', dataset_name)}. {meta.get('description', '').strip().rstrip('.')}"
    app = getattr(info, "application_type", None)
    if app:
        context += f" Application area: {getattr(app, 'name', app)}."
    if problem_type == "classification":
        class_names = list(dataset.class_names) if getattr(dataset, "class_names", None) is not None else None
        enc = ", ".join(
            f"{int(c)} = {class_names[int(c)] if class_names and int(c) < len(class_names) else int(c)}" for c in labels
        )
        target = {target_name: f"class of the measured sample ({enc}). Context: {context}"}
    else:
        lo, hi = y_range
        target = {
            target_name: f"{target_name} of the measured sample (numeric value in the unit of the source data; the values "
            f"in the training data range from {lo:.4g} to {hi:.4g}, so predict values inside this range). Context: {context}"
        }
    features = {
        n: (
            f"mean Raman intensity between {round(edges[i])} and {round(edges[i + 1])} cm-1 after scaling each spectrum to "
            "its own minimum 0 and maximum 1 (unitless, 0 to 1; higher means a stronger Raman signal in this band)"
        )
        for i, n in enumerate(names)
    }
    return target, features


def run_fold(task, fold, out_dir, protocol, cache_dir, mirror_repo, model_kwargs):
    from llm_trees import LLMTreeClassifier, LLMTreeRegressor
    from raman_data.datasets import get_dataset_info

    from raman_bench.compare import row_cap
    from raman_bench.experiment_utils import build_task, load_dataframe

    (spec,) = [t for t in protocol["tasks"] if t["task"] == task]
    name, target_idx = spec["dataset"], spec["target_idx"]
    dataset, df, sample_idx, problem_type = load_dataframe(
        name, target_idx, max_train_samples=row_cap(protocol, name),
        cache_dir=cache_dir, mirror_repo=mirror_repo,
    )
    task_name, wrapper = build_task(
        dataset_name=name, target_idx=target_idx, df=df, raw_targets=dataset.targets, problem_type=problem_type,
        n_repeats=spec["n_repeats"], n_splits=spec["n_folds"], sample_idx=sample_idx,
        min_samples_per_class=protocol["min_samples_per_class"], inner_group_folds=False,
    )
    if wrapper is None:
        return {"task": task, "fold": fold, "status": "skipped"}
    X_tr, y_tr, X_te, y_te = wrapper.get_train_test_split(fold=fold, repeat=0)
    F_tr, edges = band_features(X_tr)
    F_te, _ = band_features(X_te)
    names = list(F_tr.columns)
    target_name = str(spec["target_name"])
    if problem_type == "classification":
        labels = np.unique(np.concatenate([y_tr.to_numpy(), y_te.to_numpy()]))
        depth = int(min(CLF_MAX_DEPTH_RANGE[1], max(CLF_MAX_DEPTH_RANGE[0], math.ceil(math.log2(len(labels))) + 1)))
        cls = LLMTreeClassifier
    else:
        labels = None
        depth = REG_MAX_DEPTH
        cls = LLMTreeRegressor
    y_range = None if problem_type == "classification" else (float(y_tr.astype(float).min()), float(y_tr.astype(float).max()))
    target, features = describe(name, dataset, get_dataset_info(name), target_name, problem_type, labels, edges, names, y_range)

    fold_dir = os.path.join(out_dir, task, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    model = cls(
        target=target, features=features, max_tree_depth=depth,
        prompt_file=os.path.join(fold_dir, "prompt.txt"), **model_kwargs,
    )
    start = time.monotonic()
    model.fit(F_tr, y_tr.rename(target_name))
    fit_s = time.monotonic() - start
    with open(os.path.join(fold_dir, "tree.py"), "w", encoding="utf-8") as f:
        f.write(model.tree_code_ + "\n")
    pred = model.predict(F_te)
    res = {
        "task": task, "fold": fold, "status": "ok", "problem_type": problem_type, "max_tree_depth": depth,
        "n_train": len(F_tr), "n_test": len(F_te), "fit_s": fit_s, "features_used": list(model.support_),
        "model": os.environ.get("OPENROUTER_MODEL"), **model_kwargs,
    }
    arrays = {"y_test": y_te.to_numpy(), "pred": np.asarray(pred), "test_index": np.asarray(X_te.index)}
    yt = y_te.to_numpy()
    if problem_type == "classification":
        from sklearn.metrics import balanced_accuracy_score

        res["accuracy"] = float((pred == yt).mean())
        res["balanced_accuracy"] = float(balanced_accuracy_score(yt, pred))
        res["majority_accuracy"] = float((yt == y_tr.value_counts().idxmax()).mean())
        res["n_classes"] = int(len(labels))
        if model.tree_ is not None:
            arrays["proba"] = model.predict_proba(F_te)
            arrays["classes"] = model.classes_
    else:
        yt = yt.astype(float)
        res["rmse"] = float(np.sqrt(np.mean((pred - yt) ** 2)))
        res["rmse_mean_baseline"] = float(np.sqrt(np.mean((y_tr.to_numpy().astype(float).mean() - yt) ** 2)))
    np.savez(os.path.join(fold_dir, "predictions.npz"), **arrays)
    with open(os.path.join(fold_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res


GATEWAY_WAIT_S = (30, 60, 120, 300, 600, 600, 600, 600)  # waits between attempts: about 50 minutes in all


def run_fold_waiting(*args):
    """run_fold, waiting out gateway errors (HTTP 5xx, no connection to the LLM server) with growing pauses.

    The server behind the gateway can be down for a few minutes; without the wait every fold that starts
    meanwhile fails at once. Any other error, and a gateway outage longer than the waits, still raises.
    """
    import openai

    for attempt, wait in enumerate((*GATEWAY_WAIT_S, None)):
        try:
            return run_fold(*args)
        except (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            if wait is None:
                raise
            print(f"  gateway error ({type(e).__name__}); attempt {attempt + 1}, waiting {wait} s", flush=True)
            time.sleep(wait)


def main():
    from raman_bench.compare import load_protocol

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", help="Protocol tasks (<dataset>__<target_idx>); default: all")
    p.add_argument("--task-type", choices=["classification", "regression"])
    p.add_argument("--folds", type=int, nargs="*", default=list(range(N_SPLITS)))
    p.add_argument("--out-dir", default="results/llm_trees")
    p.add_argument("--cache-dir", default=".cache_v1")
    p.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th task, starting at i, to run several processes")
    p.add_argument("--reverse", action="store_true", help="work through the jobs from the last to the first (a second run from the other end)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-retry-llm", type=int, default=5)
    args = p.parse_args()

    protocol = load_protocol()
    tasks = [t for t in protocol["tasks"] if (not args.tasks or t["task"] in args.tasks)
             and (not args.task_type or t["problem_type"] == args.task_type)]
    model_kwargs = {"temperature": args.temperature, "seed": args.seed, "num_retry_llm": args.num_retry_llm}
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    # shard by task: the folds of one task share a local task cache that two processes must not write at once
    jobs = [(t["task"], f) for t in tasks[shard::n_shards] for f in args.folds]
    if args.reverse:
        jobs.reverse()
    summary = {"ok": 0, "skipped": 0, "failed": 0, "cached": 0}
    start = time.monotonic()
    for i, (task, fold) in enumerate(jobs, 1):
        done = os.path.join(args.out_dir, task, f"fold{fold}", "result.json")
        if os.path.exists(done):
            summary["cached"] += 1
            continue
        t0 = time.monotonic()
        try:
            res = run_fold_waiting(task, fold, args.out_dir, protocol, args.cache_dir, args.mirror_repo, model_kwargs)
            summary["ok" if res["status"] == "ok" else "skipped"] += 1
            shown = {k: round(v, 3) for k, v in res.items() if k in ("accuracy", "balanced_accuracy", "rmse", "rmse_mean_baseline")}
            print(f"[{i}/{len(jobs)}] {task} fold {fold}: {res['status']} {shown} ({time.monotonic() - t0:.0f} s)", flush=True)
        except Exception:  # one failed fold must not stop the others; the trace goes to the log
            summary["failed"] += 1
            print(f"[{i}/{len(jobs)}] {task} fold {fold}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"SUMMARY {summary} in {time.monotonic() - start:.0f} s, results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
