#!/usr/bin/env Rscript
# prep_fulltarget_knockoffs.R  [for R2#7 covariate logistic test]
# Write the FULL 154-sample combined original+knockoff matrix for the best config
# (PCA distance, ComBat-both, most-proximal 25% source), plus response + sample order.
# Used by cross_fit_scores.py to produce out-of-fold BRIDGE-KO scores for every sample.
suppressMessages({ .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())); library(knockoff) })
set.seed(2025)
WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results/fulltarget"); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

big <- as.matrix(read.csv(file.path(WD, "big_scaled_combat2.csv"), row.names = 1, check.names = FALSE))
small <- as.matrix(read.csv(file.path(WD, "small_scaled_combat2.csv"), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(WD, "clinical_meta.csv"))
resp <- ifelse(meta$Study_Clin_Response[match(rownames(small), meta$Sample)] == "Responder", 1, 0)

# most-proximal 25% source by PCA distance to target centroid
pca <- prcomp(big, center = TRUE, scale. = FALSE)
pc_big <- pca$x[, 1:2]; center <- colMeans(predict(pca, small)[, 1:2])
d <- sqrt((pc_big[,1]-center[1])^2 + (pc_big[,2]-center[2])^2)
src <- big[d <= quantile(d, 0.25), , drop = FALSE]
zv <- which(apply(src, 2, var) == 0); if (length(zv)) src[, zv] <- src[, zv] + rnorm(nrow(src)*length(zv),0,1e-6)
Sigma <- cov(src) + diag(1e-6, ncol(src))

kn <- create.gaussian(X = small, method = "equi", Sigma = Sigma, mu = rep(0, ncol(small)))
comb <- cbind(as.data.frame(small), as.data.frame(kn))
colnames(comb) <- c(colnames(small), paste0("K", seq_len(ncol(kn))))
write.csv(comb, file.path(OUT, "full_big_equi_knockoffs_pca25.csv"), row.names = FALSE)
write.csv(data.frame(sample = rownames(small), response_binary = resp),
          file.path(OUT, "full_meta.csv"), row.names = FALSE)
cat("DONE prep_fulltarget_knockoffs: 154 x", ncol(comb), "->", OUT, "\n")
