# Proposed datasets for the next RamanBench release

A running list of datasets suggested by external contributors or spotted in
the literature that we want to evaluate adding to the benchmark.

When a dataset is added to [`raman_data`](https://github.com/ml-lab-htw/raman_data),
flip the **In `raman_data`?** column to ✓ and note the loader entry name.

| Dataset | Link | Task | License | Suggested by | Comment | In `raman_data`? |
|---|---|---|---|---|---|---|
| Chlorinated sample identification | https://github.com/AaronFlanagan20/Analysis-of-Data-Synthesis-for-Raman-Spectroscopy / [paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c00761) | Binary classification (chloroform present / absent) | **Unspecified upstream** — no LICENSE file in repo; data provided by [Analyze IQ Limited](https://www.analyzeiq.com/). Author contacted to confirm redistribution terms. | Aaron Flanagan (Univ. of Galway) — email to MK, 2026-05-19 | 230 spectra × 2473 wavenumbers; predefined 3-fold splits in repo | in progress (GitHubLoader, key `chlorinated_samples`) |

## How to add a new entry

1. Append a row above with the source link, suggester, and a short comment.
2. Open an issue/PR in [`raman_data`](https://github.com/ml-lab-htw/raman_data)
   to add a loader. Reference this file in the PR description.
3. Once merged, update the **In `raman_data`?** column with the loader/key name.
4. Run the benchmark with the new dataset enabled in `configs/`.