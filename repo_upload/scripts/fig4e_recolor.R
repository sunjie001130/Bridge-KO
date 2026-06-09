#!/usr/bin/env Rscript
# fig4e_recolor.R  [Phase E3: R1#6 / R2#6]
# Recolor the Fig 4E PCA scatter (source+target, both-ComBat) two ways:
#   (1) target points colored by the 5 recruiting cities
#   (2) target points colored by R vs NR
# Source shown as light grey background. Saves PDFs and records filenames for HJ.
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
  library(ggplot2)
})

WD <- "/home/sunj107/scratch/Bridge_revision"
FIG <- file.path(WD, "revision/figures"); dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

big <- read.csv(file.path(WD, "big_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
small <- read.csv(file.path(WD, "small_scaled_combat2.csv"), row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(WD, "clinical_meta.csv"))

pca <- prcomp(big, center = TRUE, scale. = FALSE)
pc_big <- as.data.frame(pca$x[, 1:2]); names(pc_big) <- c("PC1", "PC2")
# downsample the source backdrop to keep the PDF light (168k pts -> 12k; density identical)
set.seed(1); if (nrow(pc_big) > 12000) pc_big <- pc_big[sample(nrow(pc_big), 12000), ]
pc_small <- as.data.frame(predict(pca, small)[, 1:2]); names(pc_small) <- c("PC1", "PC2")
pc_small$city <- meta$Study[match(rownames(small), meta$Sample)]
pc_small$response <- meta$Study_Clin_Response[match(rownames(small), meta$Sample)]
ve <- round(100 * pca$sdev[1:2]^2 / sum(pca$sdev^2), 1)

base <- function() list(
  geom_point(data = pc_big, aes(PC1, PC2), color = "grey80", alpha = 0.3, size = 0.6),
  labs(x = sprintf("PC1 (%.1f%%)", ve[1]), y = sprintf("PC2 (%.1f%%)", ve[2])),
  theme_classic())

p_city <- ggplot() + base() +
  geom_point(data = pc_small, aes(PC1, PC2, color = city), size = 1.8, alpha = 0.9) +
  scale_color_brewer(palette = "Set1") +
  ggtitle("Fig 4E recolored by recruiting city (ComBat both)")
f1 <- "E_fig4e_by_city.pdf"; ggsave(file.path(FIG, f1), p_city, width = 6.5, height = 5)

p_resp <- ggplot() + base() +
  geom_point(data = pc_small, aes(PC1, PC2, color = response), size = 1.8, alpha = 0.9) +
  scale_color_manual(values = c(Responder = "#E64B35", Non_Responder = "#4DBBD5")) +
  ggtitle("Fig 4E recolored by response (ComBat both)")
f2 <- "E_fig4e_by_response.pdf"; ggsave(file.path(FIG, f2), p_resp, width = 6.5, height = 5)

cat("wrote:\n  ", file.path(FIG, f1), "\n  ", file.path(FIG, f2), "\n")
cat("DONE fig4e_recolor\n")
