#!/usr/bin/env Rscript
# knockoff_covariance.R  [Phase A: R1#1 / R2#1]
# Compare sdp / asdp / equi knockoffs on (1) runtime, (2) covariance preservation.
#
# Covariance preservation (reviewer's requested metric): for the target data X and
# its knockoff X~ generated from the source covariance, for each variable pair (j,k)
# compute |cov(X)_{jk} - cov(X~)_{jk}|, and the diagonal original-vs-knockoff
# correlation |cor(X_j, X~_j)| (lower = less "self-leakage", more valid control).
#
# Uses the SOURCE covariance (both-ComBat) -> the BRIDGE-KO transfer setting.
# Outputs:
#   revision/results/knockoff_runtime.csv
#   revision/results/knockoff_covpres_pairs.csv   (long: method, abs_diff)
#   revision/results/knockoff_covpres_summary.csv
#   revision/results/knockoff_selfcor.csv
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
  library(knockoff)
})

set.seed(2025)
WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

big <- read.csv(file.path(WD, "big_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
small <- read.csv(file.path(WD, "small_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
stopifnot(all(colnames(small) == colnames(big)))
X <- as.matrix(small)
Sigma <- cov(as.matrix(big)) + diag(1e-6, ncol(big))
covX <- cov(X)

methods <- c("equi", "sdp", "asdp")
runtime <- list(); covpairs <- list(); selfcor <- list()

ut <- upper.tri(covX)  # unique off-diagonal pairs

for (m in methods) {
  cat("method:", m, "\n")
  t0 <- proc.time()[["elapsed"]]
  Xk <- tryCatch(
    create.gaussian(X = X, method = m, Sigma = Sigma, mu = rep(0, ncol(X))),
    error = function(e) { cat("  FAILED:", conditionMessage(e), "\n"); NULL })
  dt <- proc.time()[["elapsed"]] - t0
  runtime[[m]] <- data.frame(method = m, seconds = dt,
                             failed = is.null(Xk))
  if (is.null(Xk)) next
  covXk <- cov(Xk)
  absdiff <- abs(covX[ut] - covXk[ut])
  covpairs[[m]] <- data.frame(method = m, abs_diff = absdiff)
  selfcor[[m]] <- data.frame(method = m,
                             self_cor = abs(diag(cor(X, Xk))))
}

runtime_df <- do.call(rbind, runtime)
write.csv(runtime_df, file.path(OUT, "knockoff_runtime.csv"), row.names = FALSE)

covpairs_df <- do.call(rbind, covpairs)
write.csv(covpairs_df, file.path(OUT, "knockoff_covpres_pairs.csv"), row.names = FALSE)

summ <- do.call(rbind, lapply(split(covpairs_df, covpairs_df$method), function(d)
  data.frame(method = d$method[1],
             mean_abs_diff = mean(d$abs_diff),
             median_abs_diff = median(d$abs_diff),
             max_abs_diff = max(d$abs_diff),
             q90_abs_diff = quantile(d$abs_diff, 0.90))))
write.csv(summ, file.path(OUT, "knockoff_covpres_summary.csv"), row.names = FALSE)

selfcor_df <- do.call(rbind, selfcor)
write.csv(selfcor_df, file.path(OUT, "knockoff_selfcor.csv"), row.names = FALSE)

cat("\n=== runtime ===\n"); print(runtime_df)
cat("\n=== covariance preservation summary ===\n"); print(summ)
cat("\n=== self-correlation (mean by method) ===\n")
print(aggregate(self_cor ~ method, selfcor_df, mean))
cat("DONE knockoff_covariance\n")
