# Plan: four new TabArena models (2026-11-20)

Agreed with the user: add all four, then submit the v1 runs on the BHT GPU cluster.

1. Bump the tabarena pin from bfa8c50c to 39c3b4a8 (2026-11-20) and add the extras inpriora, tabpack, lattice.
2. Wire INPRIORA, TABPACK, LATTICE, LATTICE-TINY like TABH2O: optional import and Prep class in wrapped_models.py,
   generate/*.py, plotting/models.py, cluster/gpu_models.json, configs/models/*.json, configs/v1/scope_default.json,
   the generator tests, CHANGELOG.
3. Run the model tests in a clone of the tabarena env with the new pin.
4. Check the image still resolves with the new extras (Python 3.11); build and test with a real fit.
5. PR to main. After the merge: submit the v1 runs (kq / cluster/submit_job.py), then rebuild the reference results.

Lattice: the weights are CC-BY-NC-4.0 (non-commercial); say so in the README and CHANGELOG.
