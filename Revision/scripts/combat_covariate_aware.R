#!/usr/bin/env Rscript
# combat_covariate_aware.R  [Phase G: R2#7]  -- REWRITTEN
# Covariate-aware ComBat applied to the TARGET across its 5 city batches (the small
# dataset's batch variable), with biological covariates preserved via mod=. This mirrors
# the manuscript's target ComBat (aim3_small_batch.Rmd used ~status+age) and directly
# tests whether preserving clinical covariates changes harmonization/AUC. Settings:
#   m1: batch(city) only                                   [reference]
#   m2: ~ response                                         (preserve response)
#   m3: ~ response + treatment + abx + stage               (preserve clinical covariates)
# (PDF p37 already found ~status+age worked but ~status+age+sex was confounded — so we
#  expect richer models may fail; we fall back and report which are estimable.)
#
# Output: corrected target matrices revision/results/combat_covaware/small_scaled_<m>.csv
# (samples x genes), to feed proximity+knockoff prep + run_bridgeko for AUC comparison.
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(sva)
})

WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results/combat_covaware"); dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

# pre-target-ComBat scaled target (small_scaled.csv = before combined ComBat)
small <- as.matrix(read.csv(file.path(WD, "small_scaled.csv"), row.names = 1, check.names = FALSE))
cov <- read.csv(file.path(WD, "revision/results/covariates_154.csv"))
cov <- cov[match(rownames(small), cov$sample), ]
stopifnot(!any(is.na(cov$sample)))

city <- factor(cov$city)
dat <- t(small)   # genes x samples for ComBat

# drop city batches with <2 samples (ComBat requirement) — none expected (min Dallas=13)
run_combat <- function(mod, tag) {
  adj <- tryCatch(
    ComBat(dat = dat, batch = city, mod = mod, par.prior = TRUE, prior.plots = FALSE),
    error = function(e) { cat("  ComBat failed (", tag, "):", conditionMessage(e), "\n"); NULL })
  if (is.null(adj)) return(invisible(FALSE))
  write.csv(t(adj), file.path(OUT, sprintf("small_scaled_%s.csv", tag)))
  cat("wrote", tag, "\n"); invisible(TRUE)
}

resp <- factor(ifelse(cov$response_binary == 1, "R", "NR"))
trt  <- factor(cov$treatment); abx <- factor(cov$abx); stg <- factor(cov$stage)

run_combat(NULL, "m1_batchonly")
run_combat(model.matrix(~ resp), "m2_response")
# richer model: include only covariates with >1 non-unknown level to reduce confounding
df3 <- data.frame(resp = resp, trt = trt, abx = abx, stg = stg)
keep <- names(Filter(function(x) nlevels(droplevels(x)) > 1, df3))
f3 <- as.formula(paste("~", paste(keep, collapse = " + ")))
cat("m3 formula:", deparse(f3), "\n")
run_combat(model.matrix(f3, data = df3), "m3_resp_trt_abx_stage")
cat("DONE combat_covariate_aware ->", OUT, "\n")
