#!/usr/bin/env Rscript
# prep_knockoffs_3methods.R  [Phase A4: UpSet]
# Generate equi/sdp/asdp knockoffs for the BEST CONFIG (PCA, ComBat-both, top-25% source),
# so the DNN can produce selected-genera lists per method for the UpSet overlap plot.
# Output dir per method: revision/results/km_<method>/ (train/test/meta in run_bridgeko layout)
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(knockoff)
})
set.seed(2025)
WD <- "/home/sunj107/scratch/Bridge_revision"
big <- as.matrix(read.csv(file.path(WD, "big_scaled_combat2.csv"), row.names = 1, check.names = FALSE))
small <- as.matrix(read.csv(file.path(WD, "small_scaled_combat2.csv"), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(WD, "clinical_meta.csv"))
resp <- ifelse(meta$Study_Clin_Response[match(rownames(small), meta$Sample)] == "Responder", 1, 0)

# top-25% proximity source subset (same rule as prep_pca_knockoffs.R)
pca <- prcomp(big, center = TRUE, scale. = FALSE)
pc_big <- pca$x[, 1:2]; center <- colMeans(predict(pca, small)[, 1:2])
d <- sqrt((pc_big[,1]-center[1])^2 + (pc_big[,2]-center[2])^2)
src <- big[d <= quantile(d, 0.25), , drop = FALSE]
zv <- which(apply(src, 2, var) == 0); if (length(zv)) src[, zv] <- src[, zv] + rnorm(nrow(src)*length(zv),0,1e-6)
Sigma <- cov(src) + diag(1e-6, ncol(src))

set.seed(0817)
n <- nrow(small); tr <- sample(seq_len(n), floor(0.7*n)); te <- setdiff(seq_len(n), tr)

for (m in c("equi", "sdp", "asdp")) {
  OUT <- file.path(WD, "revision/results", paste0("km_", m)); dir.create(OUT, recursive=TRUE, showWarnings=FALSE)
  set.seed(2025)
  kn <- create.gaussian(X = small, method = m, Sigma = Sigma, mu = rep(0, ncol(small)))
  comb <- cbind(as.data.frame(small), as.data.frame(kn))
  colnames(comb) <- c(colnames(small), paste0("K", seq_len(ncol(kn))))
  write.csv(comb[tr, ], file.path(OUT, "train_big_equi_knockoffs_70_25_pca.csv"), row.names = FALSE)
  write.csv(comb[te, ], file.path(OUT, "test_big_equi_knockoffs_30_25_pca.csv"), row.names = FALSE)
  write.csv(data.frame(response_binary = resp[tr]), file.path(OUT, "meta_train_70.csv"), row.names = FALSE)
  write.csv(data.frame(response_binary = resp[te]), file.path(OUT, "meta_test_30.csv"), row.names = FALSE)
  cat("method", m, "-> ", OUT, "\n")
}
cat("DONE prep_knockoffs_3methods\n")
