#!/usr/bin/env Rscript
# prep_pca_knockoffs.R
# Regenerate the PCA-distance proximity subsets + source-covariance equi knockoffs
# for the both-ComBat target/source matrices, then write train/test combined CSVs
# in the layout run_bridgeko.py expects.
#
# This reproduces code/knockoff_pca.Rmd but (a) self-contained, (b) regenerates the
# proximity subsets here (the /ix/hpark .../PCA_dist_combat2 CSVs are gone), using the
# exact rule from PCA_distance_aim3-Combat2.ipynb:
#   - PCA(2 comp) fit on source (big_scaled_combat2), applied to target
#   - target centroid in PC space; Euclidean distance of each source sample to it
#   - quartile groups: top_25/25_50/50_75/75_100
#
# Inputs (working dir): big_scaled_combat2.csv, small_scaled_combat2.csv, clinical_meta.csv
# Output dir: revision/results/pca_knockoffs/
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
  library(knockoff)
})

set.seed(2025)
WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results/pca_knockoffs")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# --- load scaled matrices (rows = samples, cols = 104 genera) ---
big <- read.csv(file.path(WD, "big_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
small <- read.csv(file.path(WD, "small_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(WD, "clinical_meta.csv"))
stopifnot(all(colnames(small) == colnames(big)))
cat(sprintf("big: %d x %d ; small: %d x %d\n", nrow(big), ncol(big), nrow(small), ncol(small)))

# --- PCA on source, project target; quartile groups by distance to target centroid ---
pca <- prcomp(big, center = TRUE, scale. = FALSE)   # big already StandardScaled upstream
pc_big <- pca$x[, 1:2]
pc_small <- predict(pca, small)[, 1:2]
center <- colMeans(pc_small)
d <- sqrt((pc_big[, 1] - center[1])^2 + (pc_big[, 2] - center[2])^2)
q <- quantile(d, c(0.25, 0.5, 0.75))
grp <- ifelse(d <= q[1], "top_25",
       ifelse(d <= q[2], "25_50",
       ifelse(d <= q[3], "50_75", "75_100")))
cat("group sizes:\n"); print(table(grp))

group_levels <- c("top_25", "25_50", "50_75", "75_100")
group_tag <- c(top_25 = "25", `25_50` = "50", `50_75` = "75", `75_100` = "100")

# --- helper: build source-cov equi knockoffs for the target, given a source subset ---
make_combined <- function(src_rows) {
  src <- as.matrix(big[src_rows, , drop = FALSE])
  # guard against zero-variance columns in the subset (jitter, as in knockoff_pca.Rmd)
  zv <- which(apply(src, 2, var) == 0)
  if (length(zv)) src[, zv] <- src[, zv] + rnorm(nrow(src) * length(zv), 0, 1e-6)
  Sigma <- cov(src) + diag(1e-6, ncol(src))
  smat <- as.matrix(small)
  kn <- create.gaussian(X = smat, method = "equi", Sigma = Sigma,
                        mu = rep(0, ncol(smat)))
  combined <- cbind(as.data.frame(small), as.data.frame(kn))
  colnames(combined) <- c(colnames(small), paste0("K", seq_len(ncol(kn))))
  combined
}

# --- response labels aligned to small's row order ---
# clinical_meta Sample order matches small rows (both from same pipeline)
resp <- ifelse(meta$Study_Clin_Response == "Responder", 1, 0)
stopifnot(length(resp) == nrow(small))

# --- 70/30 split (matches knockoff_pca.Rmd: set.seed(0817)) ---
set.seed(0817)
n <- nrow(small)
train_idx <- sample(seq_len(n), floor(0.7 * n))
test_idx <- setdiff(seq_len(n), train_idx)

write.csv(data.frame(response_binary = resp[train_idx]),
          file.path(OUT, "meta_train_70.csv"), row.names = FALSE)
write.csv(data.frame(response_binary = resp[test_idx]),
          file.path(OUT, "meta_test_30.csv"), row.names = FALSE)

# also export sample IDs + city + response for each split (used by stratified bootstrap)
city <- meta$Study[match(rownames(small), meta$Sample)]
write.csv(data.frame(sample = rownames(small)[train_idx],
                     city = city[train_idx], response_binary = resp[train_idx]),
          file.path(OUT, "split_train_70.csv"), row.names = FALSE)
write.csv(data.frame(sample = rownames(small)[test_idx],
                     city = city[test_idx], response_binary = resp[test_idx]),
          file.path(OUT, "split_test_30.csv"), row.names = FALSE)

set.seed(2025)
for (g in group_levels) {
  src_rows <- rownames(big)[grp == g]
  combined <- make_combined(src_rows)
  tag <- group_tag[[g]]
  write.csv(combined[train_idx, ],
            file.path(OUT, sprintf("train_big_equi_knockoffs_70_%s_pca.csv", tag)),
            row.names = FALSE)
  write.csv(combined[test_idx, ],
            file.path(OUT, sprintf("test_big_equi_knockoffs_30_%s_pca.csv", tag)),
            row.names = FALSE)
  cat(sprintf("wrote group %s (tag %s): %d source samples\n", g, tag, length(src_rows)))
}
cat("DONE prep_pca_knockoffs ->", OUT, "\n")
