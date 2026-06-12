#!/usr/bin/env Rscript
# build_covariates.R  [Phase G: R2#6 / R2#7]
# Join the uploaded Intestinal_data_MOESM3_ESM.xlsx (sheet "Metadata", 186 rows)
# to the 154 analysis samples via clean_id, and emit a tidy covariate table aligned
# to the row order of small_scaled_combat2.csv (the target matrix).
#
# Usable covariates (per data inspection): Age, Sex, BMI, Immunotherapy_Drug (treatment),
# Abx_Use, H2RA_PPI_Use, Pre_Treatment_Disease_Stage. Many public-cohort samples carry
# "N_A" for treatment/Abx/stage (only Pittsburgh has them) -> kept as explicit "unknown".
# ECOG performance status is NOT in the file (acknowledge as limitation).
#
# Output: revision/results/covariates_154.csv  (rownames = target sample IDs)
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
  library(readxl); library(dplyr)
})

WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# target sample order + response + city
small <- read.csv(file.path(WD, "small_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
target_samples <- rownames(small)
meta <- read.csv(file.path(WD, "clinical_meta.csv"))          # Sample, Study_Clin_Response, Study
smc  <- read.csv(file.path(WD, "smallData_meta_country.csv")) # Sample -> clean_id

# map analysis Sample -> clean_id (the Excel key)
clean_map <- setNames(smc$clean_id, smc$Sample)
clean_ids <- clean_map[target_samples]
stopifnot(!any(is.na(clean_ids)))

xl <- read_excel(file.path(WD, "Intestinal_data_MOESM3_ESM.xlsx"), sheet = "Metadata")
keep <- c("Sample", "Age", "Sex", "BMI", "Immunotherapy_Drug",
          "Abx_Use", "H2RA_PPI_Use", "Pre_Treatment_Disease_Stage")
xl <- xl[, keep]

idx <- match(clean_ids, xl$Sample)
stopifnot(!any(is.na(idx)))
cov <- xl[idx, ]

out <- data.frame(
  sample = target_samples,
  clean_id = clean_ids,
  city = meta$Study[match(target_samples, meta$Sample)],
  response = meta$Study_Clin_Response[match(target_samples, meta$Sample)],
  response_binary = ifelse(meta$Study_Clin_Response[match(target_samples, meta$Sample)] == "Responder", 1, 0),
  age = suppressWarnings(as.numeric(cov$Age)),
  sex = cov$Sex,
  bmi = suppressWarnings(as.numeric(cov$BMI)),
  treatment = cov$Immunotherapy_Drug,
  abx = cov$Abx_Use,
  ppi = cov$H2RA_PPI_Use,
  stage = cov$Pre_Treatment_Disease_Stage,
  stringsAsFactors = FALSE
)

# --- sequencing method (verified from McCulloch et al. 2022 Nat Med Suppl. Table 3) ---
# All five cohorts contributed *shotgun metagenomic* data (McCulloch re-analyzed the four
# external cohorts through a unified pipeline; Suppl. Fig. 7), so the residual technical
# axis is the original cohort taxonomic-profiling method, which is 1:1 with city/cohort:
#   Pittsburgh (this study) ....... shotgun, JAMS / Last-Known-Taxon (in-house, NCI)
#   Houston   (Gopalakrishnan 2018) shotgun, MetaOMineR
#   Chicago   (Matson 2018) ........ shotgun, MetaPhlAn2
#   New_York  (Peters 2019) ........ shotgun, MetaPhlAn2 + HUMAnN2
#   Dallas    (Frankel 2017) ....... shotgun, MetaPhlAn + HUMAnN + FMAP
# We encode the 3 distinct profiler FAMILIES so the covariate pools cities (not a trivial
# relabel of `city`). Chicago/New_York/Dallas all use MetaPhlAn-family profilers.
seq_family <- c(Pittsburgh = "JAMS", Houston = "MetaOMineR",
                Chicago = "MetaPhlAn", New_York = "MetaPhlAn", Dallas = "MetaPhlAn")
seq_detail <- c(Pittsburgh = "shotgun_JAMS", Houston = "shotgun_MetaOMineR",
                Chicago = "shotgun_MetaPhlAn2", New_York = "shotgun_MetaPhlAn2_HUMAnN2",
                Dallas = "shotgun_MetaPhlAn_HUMAnN_FMAP")
out$seq_method <- unname(seq_family[out$city])
out$seq_detail <- unname(seq_detail[out$city])
stopifnot(!any(is.na(out$seq_method)))

# normalize "N_A"/NA to explicit "unknown" for categoricals; median-impute age
norm_cat <- function(x) { x[is.na(x) | x %in% c("N_A", "NA", "")] <- "unknown"; x }
for (cc in c("sex", "treatment", "abx", "ppi", "stage")) out[[cc]] <- norm_cat(out[[cc]])
out$age[is.na(out$age)] <- median(out$age, na.rm = TRUE)
out$bmi[is.na(out$bmi)] <- median(out$bmi, na.rm = TRUE)

write.csv(out, file.path(OUT, "covariates_154.csv"), row.names = FALSE)

cat("=== coverage (non-unknown) ===\n")
for (cc in c("treatment", "abx", "ppi", "stage")) {
  n_known <- sum(out[[cc]] != "unknown")
  cat(sprintf("  %-10s %d/%d known\n", cc, n_known, nrow(out)))
}
cat("\n=== treatment ===\n"); print(table(out$treatment))
cat("=== abx ===\n"); print(table(out$abx))
cat("=== stage ===\n"); print(table(out$stage))
cat("=== seq_method (profiler family) ===\n"); print(table(out$seq_method))
cat("=== seq_method x city ===\n"); print(table(out$seq_method, out$city))
cat("=== city x response ===\n"); print(table(out$city, out$response))
cat("DONE build_covariates ->", file.path(OUT, "covariates_154.csv"), "\n")
