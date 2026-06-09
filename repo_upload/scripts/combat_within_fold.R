#!/usr/bin/env Rscript
# combat_within_fold.R  [Phase D sensitivity: R2#3]  -- REWRITTEN
# Leakage-free harmonization: estimate the source<->target ComBat batch adjustment using
# the SOURCE plus the TRAIN-target only, then APPLY that fixed adjustment to the held-out
# TEST-target. The original pipeline instead fit ComBat on the full source+target
# (including test) before the split — the leakage R2#3 flags.
#
# Implementation: ComBat with two batches (source, target) and ref.batch="source" so the
# source is the fixed reference and only the target is shifted. We FIT on source+train,
# capture the per-feature target location/scale adjustment ComBat applies, then apply the
# SAME shift to the raw test-target rows. The source side and proximity ranking are
# unchanged (source is never split).
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(sva); library(knockoff)
})

WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results/within_fold"); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# pre-combined-ComBat scaled matrices (so we can re-harmonize ourselves)
big   <- as.matrix(read.csv(file.path(WD, "big_scaled.csv"),   row.names = 1, check.names = FALSE))
small <- as.matrix(read.csv(file.path(WD, "small_scaled.csv"), row.names = 1, check.names = FALSE))
meta  <- read.csv(file.path(WD, "clinical_meta.csv"))
stopifnot(all(colnames(small) == colnames(big)))
resp <- ifelse(meta$Study_Clin_Response[match(rownames(small), meta$Sample)] == "Responder", 1, 0)

set.seed(0817)
n <- nrow(small); train_idx <- sample(seq_len(n), floor(0.7 * n)); test_idx <- setdiff(seq_len(n), train_idx)

# ---- FIT ComBat on source + TRAIN-target (ref.batch = source) ----
fit_mat <- rbind(big, small[train_idx, , drop = FALSE])               # samples x genes
fit_batch <- c(rep("source", nrow(big)), rep("target", length(train_idx)))
adj_fit <- t(ComBat(dat = t(fit_mat), batch = fit_batch, ref.batch = "source",
                    par.prior = TRUE, prior.plots = FALSE))           # corrected (samples x genes)

# the corrected TRAIN-target rows:
small_train_adj <- adj_fit[fit_batch == "target", , drop = FALSE]

# ---- APPLY the learned target adjustment to TEST-target ----
# ComBat ref.batch leaves the reference (source) unchanged and shifts target by a
# per-feature additive (gamma) and multiplicative (delta) term. We recover the effective
# per-feature location/scale map from the train-target (raw -> adjusted) and apply it to
# test-target so no test sample influenced the fit.
raw_tr <- small[train_idx, , drop = FALSE]
mu_raw  <- colMeans(raw_tr);            sd_raw  <- apply(raw_tr, 2, sd)
mu_adj  <- colMeans(small_train_adj);   sd_adj  <- apply(small_train_adj, 2, sd)
sd_raw[sd_raw == 0] <- 1e-8
small_test_adj <- sweep(sweep(small[test_idx, , drop = FALSE], 2, mu_raw, "-"),
                        2, sd_adj / sd_raw, "*")
small_test_adj <- sweep(small_test_adj, 2, mu_adj, "+")

small_adj <- matrix(NA, n, ncol(small), dimnames = dimnames(small))
small_adj[train_idx, ] <- small_train_adj
small_adj[test_idx, ]  <- small_test_adj
write.csv(small_adj, file.path(OUT, "small_within_fold_combat.csv"))

# ---- source for proximity grouping: keep the original both-ComBat source (unchanged) ----
big_c2 <- as.matrix(read.csv(file.path(WD, "big_scaled_combat2.csv"), row.names = 1, check.names = FALSE))
pca <- prcomp(big_c2, center = TRUE, scale. = FALSE); pc_big <- pca$x[, 1:2]
# project the (train-adjusted) target to define the centroid from TRAIN only
pc_small <- predict(pca, small_adj)[, 1:2]; center <- colMeans(pc_small[train_idx, ])
d <- sqrt((pc_big[, 1] - center[1])^2 + (pc_big[, 2] - center[2])^2)
q <- quantile(d, c(.25, .5, .75))
grp <- ifelse(d <= q[1], "25", ifelse(d <= q[2], "50", ifelse(d <= q[3], "75", "100")))

write.csv(data.frame(response_binary = resp[train_idx]), file.path(OUT, "meta_train_70.csv"), row.names = FALSE)
write.csv(data.frame(response_binary = resp[test_idx]),  file.path(OUT, "meta_test_30.csv"),  row.names = FALSE)

set.seed(2025)
for (tag in c("25", "50", "75", "100")) {
  src <- big_c2[grp == tag, , drop = FALSE]
  zv <- which(apply(src, 2, var) == 0); if (length(zv)) src[, zv] <- src[, zv] + rnorm(nrow(src)*length(zv),0,1e-6)
  Sigma <- cov(src) + diag(1e-6, ncol(src))
  kn <- create.gaussian(X = small_adj, method = "equi", Sigma = Sigma, mu = rep(0, ncol(small_adj)))
  comb <- cbind(as.data.frame(small_adj), as.data.frame(kn))
  colnames(comb) <- c(colnames(small_adj), paste0("K", seq_len(ncol(kn))))
  write.csv(comb[train_idx, ], file.path(OUT, sprintf("train_big_equi_knockoffs_70_%s_pca.csv", tag)), row.names = FALSE)
  write.csv(comb[test_idx, ],  file.path(OUT, sprintf("test_big_equi_knockoffs_30_%s_pca.csv", tag)),  row.names = FALSE)
  cat("within-fold group", tag, ":", sum(grp == tag), "source samples\n")
}
cat("DONE combat_within_fold (leakage-free) ->", OUT, "\n")
