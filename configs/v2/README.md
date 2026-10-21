# Benchmark v2: dataset candidates

Benchmark v1 (`configs/v1/`) is frozen: its task set and protocol stay as published.
This folder collects the datasets that raman_data gained after 1.6.6 as candidates for
benchmark v2 (package 2.1.0). Nothing here is run by the v1 sweep.

## Files

- `datasets/candidates.json`: the 40 raman_data keys added after 1.6.6 (raman_data 1.7.0).
- `candidate_exclusions.json`: candidate targets that are not run, with a reason, in the
  format of `configs/v1/quality_exclusions.json`.

Build the target list:

```bash
python scripts/build_target_list.py \
    --dataset-list configs/v2/datasets/candidates.json \
    --quality-exclusions configs/v2/candidate_exclusions.json \
    --min-groups-per-class 3 \
    --output configs/v2/target_list.json
```

## Protocol changes against v1

- **At least 3 groups per class** (`--min-groups-per-class 3` for `run_experiment.py`
  and `build_target_list.py`). Grouped 3-fold CV needs 3 groups in every class;
  with fewer, a test fold can hold a class that no training fold has. Classes below that
  are dropped, like classes below the 9-spectra minimum. This affects
  `microplastics_training_clean` (polyester, ABS, POM) and `raman_mpi` (polyurethane,
  silicone). v1 keeps the rule off.

## Decisions (2026-10-21)

From a DUMMY/PLS/RF fold-0 run on every new key:

- **Excluded, too few spectra** (`too_few_classes`): `glucose_liquid_{10,14}level`
  (5 per class, and no wavenumber axis), `ocean_polymers` (at most 4 per class), and
  `uraman_mofs`, `uraman_mof74_1000_2000`, `uraman_mof74_70_3000` (1 per class).
  `build_target_list.py` also flags such targets on its own now.
- **Excluded, duplicates** (`duplicate`): `microplastics_training_{1,2,4,8}cm` hold the
  same 186 measurements as `microplastics_training_clean`, as 20 augmented rows each at
  four grid resolutions. Only the clean key is a candidate.
- **Glaucoma** (`out_of_scope`): only the glaucoma-severity targets of each visit,
  microperimetry, RNFL and GCC (baseline 0, 1, 2; 18 months 7, 8, 10). With all 22
  targets of about 100 spectra from 21-23 patients, glaucoma would be a third of the
  v2 tasks.
- **Kept although small or weak on fold 0:** `rod_minerals` and `nist_ecsers_drugs`
  (only 8 and 5 classes reach 9 spectra), `aged_oils_raman_2` (no sample ids, so
  replicates of a bottle can sit on both sides of a split), `cwa_dmmp` (198 saturated
  spectra), `sers_pesticide_produce` (n = 33).
