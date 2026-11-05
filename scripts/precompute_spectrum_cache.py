#!/usr/bin/env python
"""Precompute the per-spectrum preprocessing of a jobspec into a spectrum-cache directory.

For every dataset and recipe in a ``run_experiment_batch.py`` jobspec, apply the recipe's
per-spectrum steps (crop, CRR, smoothing, baseline correction) to all spectra of the dataset once
and write them to ``<out>/<step key>/<dataset>.npz`` (see
:mod:`raman_bench.preprocessing.spectrum_cache`). Batch runs started with
``--spectrum-cache <out>`` then read these rows instead of recomputing them in every pod, model,
fold and bag child. Run it on CPU nodes before the GPU jobs; it never needs a GPU.

The step key is resolved per (model, recipe) exactly as the model resolves its parameters (the
mixin's defaults overlaid with the recipe's hyperparameters), so models that share a recipe
share the rows. Rows a run does not find here are computed there as usual; the batch runner's
summary reports how many rows were served and computed.

    python scripts/precompute_spectrum_cache.py --jobspec jobs.jsonl --out /data/spectrum_cache \\
        --cache-dir /data/cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rex  # noqa: E402
import run_experiment_batch as rb  # noqa: E402

logger = logging.getLogger("precompute_spectrum_cache")


def resolved_params(model_key: str, recipe_config: str | None) -> tuple[object, dict]:
    """The model instance and its resolved prep_* parameters for a recipe, as in run_one."""
    from raman_bench.model import build_prep_model_hyperparameters
    from raman_bench.models.registry import infer_model_cls

    cls = infer_model_cls(model_key)
    base_cfg: dict = {}
    preprocessing_params = None
    if recipe_config is not None:
        preprocessing_config, preprocessing_params = rex._load_recipe_config(recipe_config)
        if preprocessing_config is not None:
            base_cfg["_prep_restriction"] = preprocessing_config
    hp = build_prep_model_hyperparameters(cls, base_cfg, preprocessing_params=preprocessing_params,
                                          optimize=False, model_extra_params=None) or {}
    model = cls(hyperparameters=hp, problem_type="regression", path=tempfile.mkdtemp())
    model.params = {}
    model._set_default_params()  # the mixin's prep_* defaults, as AutoGluon's initialize() sets them
    return model, {**model.params, **hp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobspec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-dir", default=".cache_v1")
    ap.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    ap.add_argument("--use-mirror", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Default row cap, as in the batch run (a line's own value wins)")
    args = ap.parse_args()
    rex.load_env()
    from raman_bench.experiment_utils import load_dataframe
    from raman_bench.preprocessing.spectrum_cache import row_digests, save_table, step_key

    jobs = rb.read_jobspec(args.jobspec)
    # Spectra do not depend on the target, but a row cap subsamples per dataset, so the unit of
    # work is (dataset, row cap); the key decides which recipes need computing.
    work: dict[tuple, set] = {}
    for j in jobs:
        cap = j.get("max_train_samples", args.max_train_samples)
        work.setdefault((j["dataset_name"], cap), set()).add((j["model_key"], j.get("recipe_config")))
    n_items = len(work)
    logger.info("Precompute: %d (dataset, row cap) units from %d jobs", n_items, len(jobs))
    t_start = time.perf_counter()
    n_written = n_skipped = 0
    for i, ((dataset, cap), pairs) in enumerate(sorted(work.items(), key=str), 1):
        _, df, _, _ = load_dataframe(dataset, 0, max_train_samples=cap, cache_dir=args.cache_dir,
                                     use_mirror=args.use_mirror, mirror_repo=args.mirror_repo)
        features = df.iloc[:, :-1]  # the last column is the label (see build_task)
        X = features.to_numpy(dtype=np.float64)
        try:
            wavenumbers = features.columns.to_numpy(dtype=float)
        except (TypeError, ValueError):
            wavenumbers = None
        digests = None
        done_keys: set[str] = set()
        name = f"{dataset}__cap{cap}" if cap else dataset
        for model_key, recipe in sorted(pairs, key=str):
            model, params = resolved_params(model_key, recipe)
            key = step_key(params, wavenumbers)
            if key is None or key in done_keys:
                continue
            done_keys.add(key)
            if (args.out / key / f"{name}.npz").exists():
                n_skipped += 1
                continue
            model._wavenumbers = wavenumbers
            t0 = time.perf_counter()
            out = model._apply_spectrumwise_steps(X, params, "Precompute")
            digests = digests or row_digests(X)
            save_table(args.out, key, dict(zip(digests, out)), name)
            n_written += 1
            print(f"[{i}/{n_items}] {dataset} cap={cap} {Path(recipe).stem if recipe else 'default'} "
                  f"-> {key}: {len(X)} spectra in {time.perf_counter() - t0:.1f}s", flush=True)
        elapsed = time.perf_counter() - t_start
        print(f"[{i}/{n_items}] {dataset} done, {len(done_keys)} step keys "
              f"(ETA {elapsed / i * (n_items - i) / 60:.0f} min)", flush=True)
    print(f"Precompute finished in {(time.perf_counter() - t_start) / 60:.1f} min: "
          f"{n_written} tables written, {n_skipped} already present", flush=True)


if __name__ == "__main__":
    main()
