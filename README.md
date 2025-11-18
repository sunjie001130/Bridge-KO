***BRIDGE-KO: Knockoff-Assisted Transfer Learning for Microbiome Immunotherapy Prediction***

**Overview**

BRIDGE-KO (Background-Regularized Inference for Differential Knockoff Signatures) is a transfer-learning framework designed for microbiome-based immunotherapy prediction when the target clinical cohort is small.
It addresses two major challenges:

Distribution shift between large reference microbiome datasets and small clinical cohorts (batch effects, demographic/lifestyle differences, sequencing depth).

Overfitting and unstable feature selection due to limited sample size.

**BRIDGE-KO integrates:**

Representation learning (PCA/UMAP embeddings) on a large external dataset (e.g., AGP / MicrobioMap).

Statistical knockoff generation for feature selection stability.

Transfer learning to align reference embeddings with the clinical cohort before final modeling.

AUC-evaluation using LLM-based model (LLM_AUC.ipynb).

This framework yields more robust, interpretable predictors of immunotherapy response from gut microbiome abundance profiles.

**Pipeline summary:**

        Large Reference Cohort             Clinical Immunotherapy Cohort
         (MicroBioMap / AGP)                       (Pitt / non-Pitt)
                      │                                      │
                      ├──────────┐                ┌──────────┤
                      ▼          ▼                ▼          ▼
              PCA / UMAP Embedding        Preprocessed Bacterial Counts
                      │          └─────►      PCA/UMAP Projection
                      │                        (Combat-adjusted)
                      ▼
          Knockoff Construction (PCA / UMAP space)
                      │
                      ▼
          Feature Importance + Selection
                      │
                      ▼
              Transfer-Learning Classifier
       (LLM-AUC scoring + cross-validation)
                      │
                      ▼
            Final RA/NR Biomarkers & AUC

**Repository Structure**

Below is a description of each file provided, grouped by functional stage of BRIDGE-KO.

# 1. Data Preprocessing

*aim3_bac_final_count.R*

Loads raw bacterial count tables for Pitt/non-Pitt cohorts.

Performs filtering, normalization checks, keeps shared bacterial taxa.

Outputs the final count matrix used in downstream steps.

aim3_relative_small+big_batch.Rmd

Converts counts → relative abundance.

Merges small batch (clinical cohort) with big batch (MicroBioMap).

Performs batch correction (Combat).

Produces a combined abundance matrix for embedding.

# 2. Representation Learning (Embeddings)

*PCA_distance_aim3-Combat2.ipynb*

Runs PCA on the combined dataset.

Calculates pairwise PCA distances between small-cohort samples and the large cohort.

Used for quantifying distribution alignment.

*UMAP_distance_aim3-Combat2.ipynb*

Performs UMAP embedding of combined microbiome data.

Computes UMAP distance metrics between cohorts.

Helps compare PCA- vs UMAP-based representations.

*microbiomap_PCA.ipynb*

Runs PCA only on the large reference dataset (e.g., MicroBioMap).

Saves PCA loadings for transfer to the clinical cohort.

Forms the basis for initial transfer learning.

# 3. Knockoff Feature Construction

*knockoff_pca.Rmd*

Constructs model-X knockoffs in PCA space.

Ensures:

Pairwise correlations 𝑋𝑗,𝑋𝑘 preserved

Cross-correlations 𝑋𝑗,𝑋~𝑘 match the knockoff constraints

Computes feature importance statistics (W-scores).

Produces a selected set of PCA components or bacteria.

*knockoff_umap.Rmd*

Same workflow as above but using UMAP embeddings.

Captures nonlinear structure of microbiome communities.

Saves knockoff-selected features.

# 4. Downstream Prediction & Evaluation
   
*LLM_AUC.ipynb*

Loads selected feature sets from the knockoff stage.

Trains a predictor (LLM-based or classical model).

Evaluates performance through:

AUC

Cross-validation

Robustness across seeds

Outputs final immunotherapy prediction results.
