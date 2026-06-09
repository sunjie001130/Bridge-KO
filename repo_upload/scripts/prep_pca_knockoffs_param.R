#!/usr/bin/env Rscript
# prep_pca_knockoffs_param.R — parametrized PCA-proximity + source-cov equi knockoff prep.
# Same logic as prep_pca_knockoffs.R but with configurable source/target matrices, so it
# serves the source-only-ComBat (Phase C) and covariate-aware-ComBat (Phase G) arms.
#   Rscript prep_pca_knockoffs_param.R --big <src.csv> --small <tgt.csv> --outdir <dir>
suppressMessages({ .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(knockoff) })
args <- commandArgs(trailingOnly = TRUE)
ga <- function(f, d) { i <- which(args == f); if (length(i)) args[i+1] else d }
WD <- "/home/sunj107/scratch/Bridge_revision"
BIG <- ga("--big", file.path(WD, "big_scaled_combat2.csv"))
SMALL <- ga("--small", file.path(WD, "small_scaled_combat2.csv"))
OUT <- ga("--outdir", file.path(WD, "revision/results/pca_knockoffs"))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
set.seed(2025)

big <- as.matrix(read.csv(BIG, row.names = 1, check.names = FALSE))
small <- as.matrix(read.csv(SMALL, row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(WD, "clinical_meta.csv"))
stopifnot(all(colnames(small) == colnames(big)))
resp <- ifelse(meta$Study_Clin_Response[match(rownames(small), meta$Sample)] == "Responder", 1, 0)

pca <- prcomp(big, center = TRUE, scale. = FALSE)
pc_big <- pca$x[, 1:2]; center <- colMeans(predict(pca, small)[, 1:2])
d <- sqrt((pc_big[,1]-center[1])^2 + (pc_big[,2]-center[2])^2)
q <- quantile(d, c(.25, .5, .75))
grp <- ifelse(d <= q[1], "25", ifelse(d <= q[2], "50", ifelse(d <= q[3], "75", "100")))

set.seed(0817)
n <- nrow(small); tr <- sample(seq_len(n), floor(0.7*n)); te <- setdiff(seq_len(n), tr)
write.csv(data.frame(response_binary = resp[tr]), file.path(OUT, "meta_train_70.csv"), row.names = FALSE)
write.csv(data.frame(response_binary = resp[te]), file.path(OUT, "meta_test_30.csv"), row.names = FALSE)

set.seed(2025)
for (tag in c("25","50","75","100")) {
  src <- big[grp == tag, , drop = FALSE]
  zv <- which(apply(src, 2, var) == 0); if (length(zv)) src[, zv] <- src[, zv] + rnorm(nrow(src)*length(zv),0,1e-6)
  Sigma <- cov(src) + diag(1e-6, ncol(src))
  kn <- create.gaussian(X = small, method = "equi", Sigma = Sigma, mu = rep(0, ncol(small)))
  comb <- cbind(as.data.frame(small), as.data.frame(kn))
  colnames(comb) <- c(colnames(small), paste0("K", seq_len(ncol(kn))))
  write.csv(comb[tr, ], file.path(OUT, sprintf("train_big_equi_knockoffs_70_%s_pca.csv", tag)), row.names = FALSE)
  write.csv(comb[te, ], file.path(OUT, sprintf("test_big_equi_knockoffs_30_%s_pca.csv", tag)), row.names = FALSE)
  cat("group", tag, ":", sum(grp==tag), "source samples\n")
}
cat("DONE prep_pca_knockoffs_param ->", OUT, "\n")
