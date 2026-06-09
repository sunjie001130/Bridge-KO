#!/usr/bin/env Rscript
# prep_ablations.R  [Phase C: R1#5]
# Generate knockoff train/test inputs for two ablation arms, both on PCA proximity:
#   (C1) RANDOM source subset: randomly sample source to the SAME counts as the
#        25/50/75/100% proximity groups (contrast random vs proximity-ranked).
#   (C2) ComBat SOURCE-ONLY harmonization context is handled in combat_source_only.R
#        (produces a different big_scaled matrix); this script's --big arg can point to
#        either the both-ComBat (default) or source-only matrix to build that arm.
#
# Reuses the same PCA-distance grouping + source-cov equi knockoff machinery as
# prep_pca_knockoffs.R, but for the random arm the subset is drawn at random.
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(knockoff)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- which(args == flag); if (length(i)) args[i + 1] else default
}
WD <- "/home/sunj107/scratch/Bridge_revision"
BIG  <- get_arg("--big",  file.path(WD, "big_scaled_combat2.csv"))
SMALL <- get_arg("--small", file.path(WD, "small_scaled_combat2.csv"))
OUTDIR <- get_arg("--outdir", file.path(WD, "revision/results/random_subset"))
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

big <- read.csv(BIG, row.names = 1, check.names = FALSE)
small <- read.csv(SMALL, row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(WD, "clinical_meta.csv"))
stopifnot(all(colnames(small) == colnames(big)))

# group sizes from PCA proximity (to match counts exactly)
pca <- prcomp(big, center = TRUE, scale. = FALSE)
pc_big <- pca$x[, 1:2]; pc_small <- predict(pca, small)[, 1:2]
center <- colMeans(pc_small)
d <- sqrt((pc_big[, 1] - center[1])^2 + (pc_big[, 2] - center[2])^2)
q <- quantile(d, c(0.25, 0.5, 0.75))
grp <- ifelse(d <= q[1], "25", ifelse(d <= q[2], "50", ifelse(d <= q[3], "75", "100")))
target_counts <- table(grp)[c("25", "50", "75", "100")]
cat("proximity group sizes (to match):\n"); print(target_counts)

make_combined <- function(src_rows) {
  src <- as.matrix(big[src_rows, , drop = FALSE])
  zv <- which(apply(src, 2, var) == 0)
  if (length(zv)) src[, zv] <- src[, zv] + rnorm(nrow(src) * length(zv), 0, 1e-6)
  Sigma <- cov(src) + diag(1e-6, ncol(src))
  smat <- as.matrix(small)
  kn <- create.gaussian(X = smat, method = "equi", Sigma = Sigma, mu = rep(0, ncol(smat)))
  combined <- cbind(as.data.frame(small), as.data.frame(kn))
  colnames(combined) <- c(colnames(small), paste0("K", seq_len(ncol(kn))))
  combined
}

resp <- ifelse(meta$Study_Clin_Response == "Responder", 1, 0)
set.seed(0817)
n <- nrow(small); train_idx <- sample(seq_len(n), floor(0.7 * n))
test_idx <- setdiff(seq_len(n), train_idx)
write.csv(data.frame(response_binary = resp[train_idx]), file.path(OUTDIR, "meta_train_70.csv"), row.names = FALSE)
write.csv(data.frame(response_binary = resp[test_idx]),  file.path(OUTDIR, "meta_test_30.csv"),  row.names = FALSE)

set.seed(2025)
all_src <- rownames(big)
for (tag in names(target_counts)) {
  k <- as.integer(target_counts[[tag]])
  src_rows <- sample(all_src, k)               # RANDOM subset of matching size
  combined <- make_combined(src_rows)
  write.csv(combined[train_idx, ], file.path(OUTDIR, sprintf("train_big_equi_knockoffs_70_%s_pca.csv", tag)), row.names = FALSE)
  write.csv(combined[test_idx, ],  file.path(OUTDIR, sprintf("test_big_equi_knockoffs_30_%s_pca.csv", tag)),  row.names = FALSE)
  cat(sprintf("random subset tag %s: %d source samples\n", tag, k))
}
cat("DONE prep_ablations ->", OUTDIR, "\n")
