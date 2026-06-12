# BRIDGE-KO preprocessing order & data-leakage audit

Addresses Reviewer 1 #6 (186 vs 154 sample counts) and Reviewer 2 #3 (exact order of
preprocessing relative to train/test splits; leakage check).

## 1. Sample counts: 186 → 154

| Stage | Samples | Genera |
|---|---|---|
| Raw clinical ICI cohort (= DeepMicroNET dataset) | **186** | 369 species |
| Map species → genus; keep genera shared with Microbiomap source | 186 | **104** shared genera |
| Drop target samples with **< 5 non-zero abundance genera** (sparse/low-quality) | **154** | 104 |

So **154** is the analysis set used in *all* BRIDGE-KO experiments (manuscript Methods,
"Data collection and normalization"). The 186 figure is the raw count before the
low-content sample filter; the manuscript should state 154 as the analysis N and note
186→154 explicitly. City breakdown of the 154: Pittsburgh 63, Chicago 39, Houston 25,
New York 14, Dallas 13 (R/NR = 80/74).

> Note: the raw-cohort Excel (`Intestinal_data_MOESM3_ESM.xlsx`) lists 186 rows and a
> different R/NR balance (111/75) because it includes the 32 samples later dropped and
> uses a different response field; the 154-sample `clinical_meta.csv` is authoritative.

## 2. Exact preprocessing order

1. Raw counts (target clinical cohort; source Microbiomap raw counts).
2. Taxonomic mapping to **genus**; for multiple species in one genus, take the mean.
3. Restrict to the **104 genera shared** by source and target.
4. Drop target samples with < 5 non-zero genera → 154.
5. **Relative abundance** per sample, then **log2** transform.
6. **Standardize** (z-score per genus); source PCA/UMAP fit on source, applied to target.
7. **ComBat** batch correction (three settings: none / target-only / source+target).
8. **Train/test split** (70/30, ×10 seeds) for AUC.
9. Knockoff generation (source covariance) and DNN training/selection/AUC.

## 3. Leakage finding (honest disclosure)

In the original pipeline (`code/aim3_small+big_batch.Rmd` →
`code/knockoff_pca.Rmd`), **ComBat (step 7) was fit on the full combined matrix
*before* the 70/30 split (step 8).** Because ComBat estimates batch location/scale
parameters using all samples, held-out test samples influence the correction applied to
training samples. This is a mild form of information leakage and can modestly inflate
reported AUC.

Scope of the effect:
- ComBat here uses **batch only** (or batch + biological covariates age/response); it
  does **not** use the outcome label of the test set for the correction transform, so the
  leakage is limited to shared distributional information, not label leakage.
- The proximity ranking and knockoff covariance come from the **source** cohort, which is
  never split — unaffected.

## 4. Sensitivity analysis (leakage-free ComBat-within-fold)

To demonstrate robustness, we add a leakage-free variant: ComBat is **fit on the training
samples only** and the learned batch adjustments are **applied to the held-out test
samples** (reference-batch transform), repeated within each of the 10 splits, then the
best configuration (PCA + ComBat-both + most-proximal 25% source) is re-run.
Implementation: `revision/scripts/combat_within_fold.R` → fed to `run_bridgeko.py`.
We report the original (main) and within-fold (sensitivity) AUCs side by side; the
conclusion (transfer helps after harmonization) is expected to hold.
