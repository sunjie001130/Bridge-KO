#!/usr/bin/env Rscript
# combat_source_only.R  [Phase C: R1#5]  -- REWRITTEN
# Ablation "harmonize source only": correct source toward the (unchanged) target, the
# mirror image of the manuscript's target-only setting. Implemented as 2-batch ComBat
# (source, target) with ref.batch="target" so ONLY the source is shifted; the target
# stays at its raw scaled values.
#
# Output: revision/results/combat_source_only/{big_scaled_srcCombat.csv, small_scaled_raw.csv}
# big_scaled_srcCombat = source corrected toward target ; small = unchanged target.
# Feed these to a prep that builds proximity groups + knockoffs (combat_source_only_prep).
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(sva)
})
WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results/combat_source_only"); dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

big   <- as.matrix(read.csv(file.path(WD, "big_scaled.csv"),   row.names = 1, check.names = FALSE))
small <- as.matrix(read.csv(file.path(WD, "small_scaled.csv"), row.names = 1, check.names = FALSE))
stopifnot(all(colnames(small) == colnames(big)))

combined <- t(rbind(big, small))                                  # genes x samples
batch <- c(rep("source", nrow(big)), rep("target", nrow(small)))
# The 168k-source vs 154-target size imbalance breaks ComBat's parametric prior
# (convergence NaN). Use mean.only=TRUE (location shift only), which is stable here and
# matches the "move source toward target" intent without scaling on a degenerate prior.
adj <- ComBat(dat = combined, batch = batch, ref.batch = "target",
              mean.only = TRUE, par.prior = TRUE, prior.plots = FALSE)
adj <- t(adj)                                                     # samples x genes
big_src <- adj[batch == "source", , drop = FALSE]                 # source moved toward target
# target unchanged (ref.batch): write raw scaled target for symmetry
write.csv(big_src,  file.path(OUT, "big_scaled_srcCombat.csv"))
write.csv(small,    file.path(OUT, "small_scaled_raw.csv"))
cat("DONE combat_source_only: source corrected toward target (ref.batch=target)\n")
cat("source:", nrow(big_src), "x", ncol(big_src), "; target unchanged:", nrow(small), "\n")
