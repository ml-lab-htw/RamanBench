# New Datasets Released with RamanBench

This document describes the 17 datasets released for the first time as part of RamanBench, spanning six biological and two chemical dataset groups. All datasets are accessible via the `raman-data` Python library and published under CC BY 4.0.

## Loading Datasets

Install the library:

```bash
pip install raman-data
```
 
Load any dataset by its `raman_data-key`:

```python
from raman_data import raman_data

# Load a dataset by raman_data-key
raman_data_key = "kaiser_ecoli_fermentation"
dataset = raman_data(raman_data_key)

# Access data
X = dataset.spectra          # np.ndarray (n_samples × n_wavenumbers)
w = dataset.raman_shifts     # np.ndarray of wavenumber values in cm⁻¹
y = dataset.targets          # np.ndarray of labels or values
target_names = dataset.target_names  # list of target column names

# Convert to pandas DataFrame
df = dataset.to_dataframe()
```

## Overview

| Group                                                                                | Domain | Task | Datasets | Targets | Samples | Features | Range (cm⁻¹) |
|--------------------------------------------------------------------------------------|---|---|---|---|---|---|---|
| [E. coli Fermentation (Kaiser)](#e-coli-fermentation-kaiser)                         | Biological | Regression | 2 | 8 | 28 | 1,699 | 301–1999 |
| [E. coli Fermentation (Time-Gated)](#e-coli-fermentation-time-gated)                 | Biological | Regression | 2 | 8 | 25 | 114 | 604–1508 |
| [S. thermophilus Fermentation (Kaiser)](#s-thermophilus-fermentation-kaiser)         | Biological | Regression | 1 | 4 | 14 | 1,501 | 300–1800 |
| [S. thermophilus Fermentation (Time-Gated)](#s-thermophilus-fermentation-time-gated) | Biological | Regression | 1 | 4 | 14 | 117 | 487–1423 |
| [E. coli Metabolites](#e-coli-metabolites)                                           | Biological | Regression | 2 | 5 | 2,304 | 594 | 402–1599 |
| [Bio-Catalysis Monitoring of AXP](#bio-catalysis-monitoring-of-axp)                  | Biological | Regression | 1 | 4 | 344 | 2,048 | −32–3385 |
| [Yeast Fermentation](#yeast-fermentation)                                            | Biological | Regression | 1 | 4 | 58 | 1,900 | 401–2300 |
| [R. eutropha Copolymer Fermentations](#r-eutropha-copolymer-fermentations)           | Biological | Regression | 1 | 6 | 82 | 2,776 | 405–3180 |
| [Gasoline Properties (Benchtop)](#gasoline-properties-benchtop)                      | Chemical | Regression | 1 | 12 | 179 | 961 | 98–3801 |
| [Gasoline Properties (Handheld)](#gasoline-properties-handheld)                      | Chemical | Regression | 1 | 12 | 179 | 1,901 | 400–2300 |
| [Adenine SERS (Colloidal)](#adenine-sers-colloidal)                                  | Chemical | Regression | 2 | 2 | 855 | 534 | 400–1999 |
| [Adenine SERS (Solid)](#adenine-sers-solid)                                          | Chemical | Regression | 2 | 2 | 2,661 | 534 | 400–1999 |

---

## E. coli Fermentation (Kaiser)

**raman_data-keys:** `kaiser_ecoli_fermentation`, `kaiser_ecoli_fermentation_supernatant`

**Paper:** Kogler et al. (2018), [doi:10.1002/btpr.2665](https://doi.org/10.1002/btpr.2665)

Two datasets of *E. coli* fed-batch fermentation spectra recorded with a Kaiser NIR-Raman spectrometer, capturing both the fermentation broth and the cell-free supernatant. Regression targets are OD600, glucose, acetate, and fermentation time. Samples were collected at defined time points, centrifuged for supernatant preparation, and stored at −80 °C. Reference concentrations were determined by enzymatic assay. Measurements were performed in aluminium microwell plates (20 µL per well).

**Instrument:** Kaiser RXN1, 785 nm excitation, 135 mW, non-immersion probe (NA = 0.29), 20 s × 5 accumulations, 4 cm⁻¹ resolution, CCD cooled to −40 °C.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `kaiser_ecoli_fermentation` | 14 | 1,699 | 301–1999 | [HuggingFace](https://huggingface.co/datasets/chlange/kaiser_raman_ecoli_fermentation) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/kaiser_ecoli_fermentation.json) |
| `kaiser_ecoli_fermentation_supernatant` | 14 | 1,699 | 301–1999 | [HuggingFace](https://huggingface.co/datasets/chlange/kaiser_raman_ecoli_fermentation_supernatant) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/kaiser_ecoli_fermentation_supernatant.json) |

---

## E. coli Fermentation (Time-Gated)

**raman_data-keys:** `tg_ecoli_fermentation`, `tg_ecoli_fermentation_supernatant`

**Paper:** Kogler et al. (2018), [doi:10.1002/btpr.2665](https://doi.org/10.1002/btpr.2665)

Two datasets of *E. coli* fed-batch fermentation spectra recorded with a time-gated Raman spectrometer, capturing both the fermentation broth and the cell-free supernatant. These are the time-gated counterparts to the Kaiser E. coli datasets above; the four targets (OD600, glucose, acetate, fermentation time) and sample preparation protocol are identical.

**Instrument:** TimeGate TGM1, pulsed 532 nm Nd:YVO₄ laser (~30 mW, 100 ps pulses), fiber-optic probe (NA = 0.22). A temporal gate of 1.2–2.1 ns after the laser pulse suppresses fluorescence from the culture medium. Resolution 10 cm⁻¹, non-cooled SPAD array detector; ~15 min per spectrum.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `tg_ecoli_fermentation` | 12 | 114 | 604–1508 | [HuggingFace](https://huggingface.co/datasets/chlange/tg_raman_ecoli_fermentation) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/tg_ecoli_fermentation.json) |
| `tg_ecoli_fermentation_supernatant` | 13 | 114 | 604–1508 | [HuggingFace](https://huggingface.co/datasets/chlange/tg_raman_ecoli_fermentation_supernatant) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/tg_ecoli_fermentation_supernatant.json) |

---

## S. thermophilus Fermentation (Kaiser)

**raman_data-key:** `streptococcus_thermophilus_fermentation_kaiser`

Offline Raman spectra from batch cultivations of *Streptococcus thermophilus* in shake flasks, recorded with a Kaiser RXN1 spectrometer. The dataset covers two independent 24-hour fermentation runs. It is in the tiny-data regime (N = 14) and targets lactose, galactose, lactate, and OD600.

**Instrument:** Kaiser RXN1, 785 nm excitation (same setup as the *E. coli* Kaiser datasets above).

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `streptococcus_thermophilus_fermentation_kaiser` | 14 | 1,501 | 300–1800 | [HuggingFace](https://huggingface.co/datasets/chlange/streptococcus_thermophilus_fermentation_kaiser) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/streptococcus_thermophilus_fermentation_kaiser.json) |

---

## S. thermophilus Fermentation (Time-Gated)

**raman_data-key:** `streptococcus_thermophilus_fermentation_timegate`

Offline Raman spectra from the same *Streptococcus thermophilus* batch cultivations as above, recorded with a time-gated Raman spectrometer. Temporal gating suppresses the fluorescence background characteristic of complex fermentation media. Two independent 24-hour fermentation runs are included. Targets are identical to the Kaiser variant (lactose, galactose, lactate, OD600); both datasets are in the tiny-data regime (N < 50).

**Instrument:** TimeGate spectrometer, pulsed 532 nm laser.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `streptococcus_thermophilus_fermentation_timegate` | 14 | 117 | 487–1423 | [HuggingFace](https://huggingface.co/datasets/chlange/streptococcus_thermophilus_fermentation_timegate) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/streptococcus_thermophilus_fermentation_timegate.json) |

---

## E. coli Metabolites

**raman_data-keys:** `ecoli_metabolites`, `ecoli_metabolites_dig4bio`

**Paper:** Lange et al. (2025), [doi:10.1002/bit.70006](https://doi.org/10.1002/bit.70006)

Both datasets contain Raman spectra of aqueous mixtures of key *E. coli* fermentation metabolites, acquired with an automated high-throughput Raman system integrated into a liquid handling station. `ecoli_metabolites` covers binary glucose–acetate mixtures; `ecoli_metabolites_dig4bio` extends the composition to include magnesium sulfate.

**Instrument:** Metrohm Raman Plus 785 (785 nm, 455 mW), fiber-optic BAC102 probe, 10 s acquisition, 18 µL flow-through quartz cuvette. Samples were robotically pipetted by a Tecan EVO 200 liquid handling station via a multiplexer valve. Concentrations pipetted by the robot served as ground truth, cross-validated against enzymatic assays.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `ecoli_metabolites` | 1,920 | 594 | 402–1599 | [HuggingFace](https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEcoliMetabolites) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/ecoli_metabolites.json) |
| `ecoli_metabolites_dig4bio` | 384 | 1,869 | 402–1599 | [HuggingFace](https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEcoliMetabolitesDig4Bio) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/ecoli_metabolites_dig4bio.json) |

---

## Bio-Catalysis Monitoring of AXP

**raman_data-key:** `ht_raman_bio_catalysis_axp`

High-throughput Raman spectra for real-time monitoring of biocatalytic reactions involving adenosine phosphates (AMP, ADP, ATP; collectively AXP). The reaction medium uses Deep Eutectic Solvents (DES) as an alternative solvent system and a Tris(hydroxymethyl)aminomethane buffer to fix pH between 7 and 9. When trained as a regression model, the dataset can serve as an analytical method to evaluate the suitability of different enzymes. The four regression targets cover the key phosphorylated forms of adenosine (adenosine, AMP, ADP, ATP).

**Instrument:** Metrohm Raman Plus 785 (785 nm, 455 mW), fiber-optic BAC102 probe, 25 s acquisition per spectrum, 18 µL flow-through quartz cuvette (same automated measurement platform as the E. coli Metabolites datasets).

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `ht_raman_bio_catalysis_axp` | 344 | 2,048 | −32–3385 | [HuggingFace](https://huggingface.co/datasets/chlange/HTRamanBioCatalysisAXP) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/ht_raman_bio_catalysis_axp.json) |

---

## Yeast Fermentation

**raman_data-key:** `yeast_fermentation`

**Paper:** Legner et al. (2019), [doi:10.1002/bit.27112](https://doi.org/10.1002/bit.27112)

Online Raman spectra of the continuous ethanolic fermentation of sucrose by *Saccharomyces cerevisiae* immobilized in calcium alginate beads. Immobilization enables stable continuous processing and unobstructed optical access to the liquid phase. The dataset was originally published without providing access to the raw data. Four regression targets (glucose, fructose, glycerol, ethanol) capture the dynamic evolution of key metabolites during continuous operation.

**Instrument:** Handheld IDRaman mini 2.0 (Ocean Optics), 785 nm excitation, 400–2300 cm⁻¹, 13 cm⁻¹ resolution, QS 0.5 mm quartz flow cell. Data acquisition automated via MATLAB with cloud upload. Reference concentrations determined by HPLC.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `yeast_fermentation` | 58 | 1,900 | 401–2300 | [HuggingFace](https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/yeast_fermentation.json) |

---

## R. eutropha Copolymer Fermentations

**raman_data-key:** `ralstonia_fermentations`

**Paper:** Lange et al. (2024), [doi:10.1016/B978-0-443-28824-1.50510-X](https://doi.org/10.1016/B978-0-443-28824-1.50510-X)

In-line Raman spectra from four independent batch cultivations of *Ralstonia eutropha* for biosynthesis of the biodegradable copolymer P(HB-co-HHx). Two cultivations used canola oil as carbon substrate, two used fructose; starting concentrations of RCDW, fructose, and urea were varied to maximise diversity in biomass and metabolite profiles. The dataset combines experimental and high-fidelity synthetic spectra to address multicollinearity between correlated process variables. Six regression targets cover cell dry weight, substrate consumption, and copolymer fractions.

**Instrument:** Multi-Spec Raman spectrometer (Tec5), 785 nm excitation, up to 500 mW, 365–3180 cm⁻¹, in-line sapphire optical window probe.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `ralstonia_fermentations` | 82 | 2,776 | 405–3180 | [HuggingFace](https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/ralstonia_fermentations.json) |

---

## Gasoline Properties (Benchtop)

**raman_data-key:** `fuel_benchtop`

**Paper:** Voigt et al. (2019), [doi:10.1016/j.fuel.2018.09.006](https://doi.org/10.1016/j.fuel.2018.09.006)

FT-Raman spectra of 179 commercial gasoline samples recorded with a benchtop 1064 nm spectrometer. The sample set spans RON 95–102.2 (Super, Super Plus, and premium grades) from refineries and petrol stations. Twelve regression targets include Research Octane Number (RON), Motor Octane Number (MON), ethanol content, oxygenate additives (MTBE, ETBE), and benzene content. Ground-truth RON and MON were measured using a CFR motor per ASTM D2699/D2885. The 1064 nm excitation suppresses fluorescence from aromatic gasoline components that interferes at shorter wavelengths.

**Instrument:** NXR FT-Raman module (Thermo Fisher Scientific) coupled to a Nicolet 6700 FT-IR spectrometer, 1064 nm excitation, 900 mW, 64 averaged spectra, 8 cm⁻¹ resolution, 2 mL glass vials.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `fuel_benchtop` | 179 | 961 | 98–3801 | [HuggingFace](https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/fuel_benchtop.json) |

---

## Gasoline Properties (Handheld)

**raman_data-key:** `fuel_handheld`

**Paper:** Legner et al. (2019), [doi:10.1021/acs.energyfuels.9b02944](https://doi.org/10.1021/acs.energyfuels.9b02944)

Raman spectra of the same 179 commercial gasoline samples as the Benchtop dataset, recorded with a portable handheld spectrometer. The twelve regression targets are identical, enabling direct evaluation of cross-instrument transferability between laboratory and field form factors.

**Instrument:** IDRaman mini 2.0 (Ocean Optics), 785 nm excitation, 100 mW, 400–2300 cm⁻¹, 13 cm⁻¹ resolution, 2 mL glass vials.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `fuel_handheld` | 179 | 1,901 | 400–2300 | [HuggingFace](https://huggingface.co/datasets/HTW-KI-Werkstatt/FuelRamanSpectraHandheld) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/fuel_handheld.json) |

---

## Adenine SERS (Colloidal)

**raman_data-keys:** `adenine_colloidal_gold`, `adenine_colloidal_silver`

**Paper:** Fornasaro et al. (2020), [doi:10.1021/acs.analchem.9b05658](https://doi.org/10.1021/acs.analchem.9b05658)

Two datasets from a large-scale European interlaboratory SERS study (COST Action Raman4Clinics), using colloidal nanoparticle substrates. Up to 18 European laboratories participated, each using its own 785 nm Raman spectrometer. Citrate-reduced gold and silver nanoparticle suspensions were provided centrally. Aqueous adenine solutions were prepared in PBS (pH 7.4) at multiple concentration levels; the regression target is adenine concentration. The two sub-datasets differ only in substrate metal (Au vs. Ag), enabling comparison of inter-laboratory reproducibility across substrate types.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `adenine_colloidal_gold` | 225 | 534 | 400–1999 | [Zenodo 3572359](https://doi.org/10.5281/zenodo.3572359) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/adenine_colloidal_gold.json) |
| `adenine_colloidal_silver` | 630 | 534 | 400–1999 | [Zenodo 3572359](https://doi.org/10.5281/zenodo.3572359) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/adenine_colloidal_silver.json) |

---

## Adenine SERS (Solid)

**raman_data-keys:** `adenine_solid_gold`, `adenine_solid_silver`

**Paper:** Fornasaro et al. (2020), [doi:10.1021/acs.analchem.9b05658](https://doi.org/10.1021/acs.analchem.9b05658)

Two datasets from the same European interlaboratory SERS study as above, using sputtered solid substrates instead of colloidal suspensions. Metal-coated nanostructured surfaces (sputtered gold and silver) were distributed centrally to all sites. All other experimental conditions (adenine solutions, PBS preparation, 785 nm excitation) are identical to the colloidal variants. With 2,661 spectra total, this is the larger of the two Adenine sub-collections.

| raman_data-key | Samples | Features | Range (cm⁻¹) | Source | Croissant |
|---|---|---|---|---|---|
| `adenine_solid_gold` | 810 | 534 | 400–1999 | [Zenodo 3572359](https://doi.org/10.5281/zenodo.3572359) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/adenine_solid_gold.json) |
| `adenine_solid_silver` | 1,851 | 534 | 400–1999 | [Zenodo 3572359](https://doi.org/10.5281/zenodo.3572359) | [croissant](https://github.com/ml-lab-htw/raman_data/tree/main/croissant_files/adenine_solid_silver.json) |
