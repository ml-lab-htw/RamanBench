# Benchmark v2

Benchmark v2 lives on the long-lived `v2` branch; `main` stays benchmark v1 (see RELEASING.md,
"Branches"). v2 is every v1 target plus the targets of the datasets raman_data gained after
1.6.6 and `glioma_idh`. Its first release will be 2.1.0.

## Files

- `target_list.json`: all v2 targets. `since` says where a target comes from: `"v1"` (the 135
  targets of `configs/v1/target_list.json`, unchanged) or `"v2"` (67, of which 41 run).
- `datasets/v2_only.json`: the 41 datasets of the `since: "v2"` targets.
- `exclusions.json`: v2-only targets that are not run, with a reason, in the format of
  `configs/v1/quality_exclusions.json`.
- `scope_default.json`: the v1 scope plus `min_groups_per_class: 3`, `results/v2/data` and
  the large-dataset overrides of the new datasets.
- `protocol_v2_only.json`: the 41 v2-only tasks with their metric, for
  `scripts/run_system.py --protocol` (systems such as TABFM-PLUS).

Rebuild the v2-only part of the target list after a dataset change:

```bash
python scripts/build_target_list.py \
    --dataset-list configs/v2/datasets/v2_only.json \
    --quality-exclusions configs/v2/exclusions.json \
    --min-groups-per-class 3 --force-n-repeats 1 --output v2_only.json
```

and merge it with the v1 entries (`since: "v1"`).

## Protocol

As v1, with one change: **at least 3 groups per class** (`min_groups_per_class: 3`). Grouped
3-fold CV needs 3 groups in every class; with fewer, a test fold can hold a class that no
training fold has. Classes below that are dropped, like classes below the 9-spectra minimum.
On the v1 datasets the rule changes nothing (checked 2026-10-21: only `cspp_serum_metabolites`
and `locust_phase_hemolymph` are grouped classification datasets, and both have at least 3
groups per class), so v1 results count for v2 as they are.

## Results

- v1 targets: the v1 results (`results/v1/data`).
- v2-only targets: `results/v2/data` (BHT PVC `/data/RamanBench_v1/results/v2/data`).
  `glioma_idh` was swept with the v1 parameters before it moved to v2; its results are copied
  from `results/v1/data`.
- `cluster/submit_v2_sweep.py` runs the roster on the v2-only targets (all models except
  Chakra-Tab, which runs through its hosted API, and TABSTAR and AUTOGLUON-EXTREME-4H, which
  are not in the reference).
- A model added later on `main` is run on the v2-only targets once `main` is merged into `v2`:
  `cluster/submit_v2_sweep.py --models <MODEL>`.

## Decisions (2026-10-21)

From a DUMMY/PLS/RF fold-0 run on every new key:

- **Excluded, too few spectra** (`too_few_classes`): `glucose_liquid_{10,14}level`
  (5 per class, and no wavenumber axis), `ocean_polymers` (at most 4 per class), and
  `uraman_mofs`, `uraman_mof74_1000_2000`, `uraman_mof74_70_3000` (1 per class).
- **Excluded, duplicates** (`duplicate`): `microplastics_training_{1,2,4,8}cm` hold the
  same 186 measurements as `microplastics_training_clean`, as 20 augmented rows each at
  four grid resolutions. Only the clean key is in v2.
- **Glaucoma** (`out_of_scope`): only the glaucoma-severity targets of each visit,
  microperimetry, RNFL and GCC (baseline 0, 1, 2; 18 months 7, 8, 10).
- **Kept although small or weak on fold 0:** `rod_minerals` and `nist_ecsers_drugs`
  (only 8 and 5 classes reach 9 spectra), `aged_oils_raman_2` (no sample ids),
  `cwa_dmmp` (198 saturated spectra), `sers_pesticide_produce` (n = 33).
- **`glioma_idh`** moved from the planned v1 addition to v2 (2026-10-21).
