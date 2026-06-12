#!/usr/bin/env Rscript
# compile_summary.R — gather every arm's AUC into one tidy table for the response letter.
.libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
RES <- "/home/sunj107/scratch/Bridge_revision/revision/results"
rows <- list()
add <- function(arm, group, v) if (length(v)) rows[[length(rows)+1]] <<-
  data.frame(arm = arm, group = group, mean_auc = mean(v), sd_auc = sd(v), n = length(v))

# cached PCA both (main BRIDGE-KO)
cap <- read.csv(file.path(RES, "cached_auc_from_pdf.csv"), comment.char = "#")
for (g in c(25,50,75,100)) add("BRIDGE-KO (PCA, ComBat-both)", g,
    subset(cap, setting=="pca_both" & group==g)$auc)

# helper to read a run_bridgeko group dir
grp_auc <- function(dir, arm) for (g in c(25,50,75,100)) {
  f <- file.path(RES, dir, paste0("group_", g), paste0("auc_g", g, ".csv"))
  if (file.exists(f)) add(arm, g, read.csv(f)$auc)
}
grp_auc("random_subset_auc",  "Random source subset")
grp_auc("within_fold_auc",    "Leakage-free (within-fold ComBat)")
grp_auc("source_only_auc",    "ComBat source-only")
grp_auc("covaware_m2_auc",    "Covariate-aware ComBat (~response)")

# knockoff methods (best config, group 25 only)
for (m in c("equi","sdp","asdp")) {
  f <- file.path(RES, paste0("km_", m, "_auc"), paste0("auc_km_", m, ".csv"))
  if (file.exists(f)) add(paste0("Knockoff method: ", m), 25, read.csv(f)$auc)
}
# baselines (group 25 equivalent)
bf <- file.path(RES, "baselines", "baseline_aucs.csv")
if (file.exists(bf)) { b <- read.csv(bf); for (mm in unique(b$method))
  add(paste0("Baseline: ", mm), 25, b$auc[b$method==mm]) }
# covariate prediction
cf <- file.path(RES, "covariate_pred", "covariate_prediction.csv")
if (file.exists(cf)) { d <- read.csv(cf)
  add("Covariate: micro-only", 25, d$micro_only)
  add("Covariate: covariates-only", 25, d$cov_only)
  add("Covariate: micro+covariates", 25, d$micro_plus_cov) }

out <- do.call(rbind, rows)
out$mean_auc <- round(out$mean_auc, 3); out$sd_auc <- round(out$sd_auc, 3)
write.csv(out, file.path(RES, "ALL_AUC_SUMMARY.csv"), row.names = FALSE)
print(out, row.names = FALSE)
cat("\nDONE -> ALL_AUC_SUMMARY.csv\n")
