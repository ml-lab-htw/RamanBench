# Plan: LimiX-2-1016, LimiX-2M and FTFM (branch `feat/limix2-1016-limix2m-ftfm`)

Agreed with the maintainer on 2026-11-27, from `ramanbench-new-models-papers.csv`.

## What

1. **LimiX-2-1016** (`LIMIX2-1016`): the 10 Nov 2026 checkpoint of LimiX-2 (`stable-ai/LimiX-2-1016`, 400M, classification and
   regression). **It replaces the old LimiX-2 run** on the leaderboard: `LIMIX2` leaves the v1 scope and the reference results, its
   result folders stay on disk as history (same convention as `TABICL` and `TA-TABPFN-3`).
2. **LimiX-2M** (`LIMIX-2M`): the 2M-parameter LimiX (`stable-ai/LimiX-1_2M`), classification and regression, retrieval config.
   Added next to LimiX (16M) and LimiX-2.
3. **FTFM** (`FTFM`): already wired (`Prep_FTFM`, classification only), not yet in the v1 scope. Put it in the scope and on the
   leaderboard (Grabocka group, arXiv 2610.25558).

## How

- `LIMIX2-1016`: subclass of TabArena's `LimiX2Model` that downloads the 1016 checkpoint at a pinned revision; own `ag_key` /
  `ag_name` so results never share a cache folder with the old LimiX-2. Same image, same Python 3.12 routing as `LIMIX2`.
  **Check first** that the pinned inference package (`LimiX@774aa3e`) loads the 1016 checkpoint; the upstream repo has since
  been repackaged as `ldm-limix`.
- `LIMIX-2M`: TabArena's vendored `LimiXModel` already ships `cls/reg_default_2M_retrieval.json`; subclass it with the 2M
  checkpoint and those configs, own `ag_key` / `ag_name`.
- `FTFM`: scope entry, display name, provider, release date, tests. No new code.
- Plot registry (`plotting/models.py`): display names, category, release dates, provider.
- Scope (`configs/v1/scope_default.json`), `cluster/gpu_models.json`, docs, CHANGELOG, tests.

## Not in this change

- Running the sweeps. The BHT cap is 3 GPU pods and the v2 rerun holds them; the runs wait for free slots and an explicit go.
- Aggregation, reference rebuild, Space refresh and release 1.7.0 (after the runs).
- Seldon+ (hosted API only) and the classification-only time-series models from the CSV.
