#!/usr/bin/env Rscript
# make_figures.R — build all revision figures from results CSVs.
# Mirrors the boxplot style of code/aim3_AUC_boxplots.Rmd (ggpubr, wilcox tests).
# Each figure is guarded so missing inputs are skipped with a message (lets us run
# incrementally as phases complete).
suppressMessages({
  .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
  library(ggplot2); library(dplyr)
  has_upset <- requireNamespace("UpSetR", quietly = TRUE)
})

# ggpubr's stat_compare_means is unavailable (quantreg/car fails to build on this
# toolchain). Replace with a base-R pairwise Wilcoxon annotation helper.
pairwise_wilcox_labels <- function(df, group_col = "group", val_col = "auc") {
  lv <- levels(df[[group_col]]); cmb <- combn(lv, 2, simplify = FALSE)
  do.call(rbind, lapply(cmb, function(pr) {
    a <- df[[val_col]][df[[group_col]] == pr[1]]
    b <- df[[val_col]][df[[group_col]] == pr[2]]
    if (length(a) < 2 || length(b) < 2) return(NULL)
    p <- suppressWarnings(wilcox.test(a, b)$p.value)
    data.frame(g1 = pr[1], g2 = pr[2], p = p)
  }))
}

WD <- "/home/sunj107/scratch/Bridge_revision"
RES <- file.path(WD, "revision/results")
FIG <- file.path(WD, "revision/figures"); dir.create(FIG, showWarnings = FALSE, recursive = TRUE)
have <- function(p) { ok <- file.exists(p); if (!ok) cat("skip (missing):", p, "\n"); ok }

grp_levels <- c("25", "50", "75", "100")
grp_labels <- c("top 25%", "25%-50%", "50%-75%", "75%-100%")

box_by_group <- function(df, title, file) {
  df$group <- factor(df$group, levels = grp_levels, labels = grp_labels)
  lab <- pairwise_wilcox_labels(df)
  sub <- if (!is.null(lab)) paste0("Wilcoxon p: ",
            paste(sprintf("%s vs %s=%.3f", lab$g1, lab$g2, lab$p), collapse = "; ")) else ""
  p <- ggplot(df, aes(group, auc, fill = group)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.15, alpha = 0.7, size = 1) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = title, subtitle = sub, x = "Source-proximity group", y = "AUC") +
    theme_classic() + theme(plot.subtitle = element_text(size = 6)) +
    coord_cartesian(ylim = c(0.3, 1))
  ggsave(file.path(FIG, file), p, width = 6, height = 4); cat("wrote", file, "\n")
}

# ---- Phase A: knockoff covariance preservation + runtime ----
if (have(file.path(RES, "knockoff_covpres_pairs.csv"))) {
  d <- read.csv(file.path(RES, "knockoff_covpres_pairs.csv"))
  p <- ggplot(d, aes(method, abs_diff, fill = method)) +
    geom_violin(alpha = 0.6) + geom_boxplot(width = 0.15, outlier.shape = NA) +
    scale_fill_brewer(palette = "Set1") +
    labs(title = "Covariance preservation by knockoff method",
         x = "method", y = expression(group("|", cov(X)[jk] - cov(tilde(X))[jk], "|"))) +
    theme_classic()
  ggsave(file.path(FIG, "A_covariance_preservation.pdf"), p, width = 5, height = 4)
  cat("wrote A_covariance_preservation.pdf\n")
}
if (have(file.path(RES, "knockoff_runtime.csv"))) {
  d <- read.csv(file.path(RES, "knockoff_runtime.csv"))
  p <- ggplot(d, aes(method, seconds, fill = method)) + geom_col() +
    scale_fill_brewer(palette = "Set1") +
    labs(title = "Knockoff generation runtime", y = "seconds") + theme_classic()
  ggsave(file.path(FIG, "A_runtime.pdf"), p, width = 4, height = 4)
  cat("wrote A_runtime.pdf\n")
}

# ---- Phase A: downstream AUC by knockoff method (best config) ----
km_files <- file.path(RES, paste0("km_", methods <- c("equi","sdp","asdp"), "_auc"),
                      paste0("auc_km_", c("equi","sdp","asdp"), ".csv"))
if (all(file.exists(km_files))) {
  km <- do.call(rbind, lapply(c("equi","sdp","asdp"), function(m)
    transform(read.csv(file.path(RES, paste0("km_", m, "_auc"), paste0("auc_km_", m, ".csv"))),
              method = m)))
  km$method <- factor(km$method, levels = c("equi","sdp","asdp"))
  p <- ggplot(km, aes(method, auc, fill = method)) +
    geom_boxplot(outlier.shape = NA) + geom_jitter(width = 0.15, alpha = 0.7, size = 1) +
    scale_fill_brewer(palette = "Set1") +
    labs(title = "Downstream AUC by knockoff method (best config)", x = NULL, y = "AUC") +
    theme_classic() + coord_cartesian(ylim = c(0.3, 1)) + guides(fill = "none")
  ggsave(file.path(FIG, "A_auc_by_method.pdf"), p, width = 4.5, height = 4)
  cat("wrote A_auc_by_method.pdf\n")
} else cat("skip (missing): km_*_auc\n")

# ---- Phase A: UpSet of selected genera by method ----
# Aggregate per-run selections from each method's DNN output (km_<method>_auc/run*/...),
# keeping genera selected in >=6/10 runs (the "frequent" stability set) per method.
methods <- c("equi", "sdp", "asdp")
agg_sel <- function(m) {
  d <- file.path(RES, paste0("km_", m, "_auc"))
  rf <- list.files(d, pattern = "selected_features_run.*\\.csv$", recursive = TRUE, full.names = TRUE)
  if (!length(rf)) return(NULL)
  tab <- table(unlist(lapply(rf, function(f) as.character(read.csv(f)$feature))))
  names(tab)[tab >= 6]
}
sets <- Filter(Negate(is.null), setNames(lapply(methods, agg_sel), methods))
if (length(sets) >= 2 && has_upset) {
  pdf(file.path(FIG, "A_upset_selected_genera.pdf"), width = 7, height = 4)
  print(UpSetR::upset(UpSetR::fromList(sets), order.by = "freq"))
  dev.off(); cat("wrote A_upset_selected_genera.pdf\n")
} else cat("skip (missing): per-method selected genera for UpSet\n")

# ---- Phase B: baselines vs BRIDGE-KO ----
if (have(file.path(RES, "baselines/baseline_aucs.csv"))) {
  b <- read.csv(file.path(RES, "baselines/baseline_aucs.csv"))
  # add BRIDGE-KO best-config (cached PCA both top-25%)
  cap <- read.csv(file.path(RES, "cached_auc_from_pdf.csv"), comment.char = "#")
  bk <- subset(cap, setting == "pca_both" & group == 25)
  bridge <- data.frame(method = "BRIDGE-KO", run = bk$run, auc = bk$auc)
  allm <- rbind(b[, c("method", "run", "auc")], bridge)
  p <- ggplot(allm, aes(reorder(method, auc, median), auc, fill = method)) +
    geom_boxplot(outlier.shape = NA) + geom_jitter(width = 0.15, alpha = 0.7, size = 1) +
    labs(title = "BRIDGE-KO vs baselines (PCA, ComBat-both, top 25%)",
         x = NULL, y = "AUC") + theme_classic() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
    coord_cartesian(ylim = c(0.3, 1)) + guides(fill = "none")
  ggsave(file.path(FIG, "B_baselines.pdf"), p, width = 6, height = 4)
  cat("wrote B_baselines.pdf\n")
}

# ---- Phase C: random vs proximity source subset ----
rs_dir <- file.path(RES, "random_subset_auc")
rs_files <- file.path(rs_dir, paste0("group_", grp_levels), paste0("auc_g", grp_levels, ".csv"))
if (all(file.exists(rs_files))) {
  rand <- do.call(rbind, lapply(grp_levels, function(g)
    transform(read.csv(file.path(rs_dir, paste0("group_", g), paste0("auc_g", g, ".csv"))),
              group = g)))
  rand$group <- factor(rand$group, levels = grp_levels)
  box_by_group(rand, "Random source subset (PCA, ComBat-both)", "C_random_subset.pdf")
  # proximity (cached) vs random overlay
  cap <- read.csv(file.path(RES, "cached_auc_from_pdf.csv"), comment.char = "#")
  prox <- subset(cap, setting == "pca_both")[, c("group", "auc")]; prox$type <- "proximity"
  rnd2 <- rand[, c("group", "auc")]; rnd2$type <- "random"
  both <- rbind(prox, rnd2)
  both$group <- factor(both$group, levels = grp_levels, labels = grp_labels)
  p <- ggplot(both, aes(group, auc, fill = type)) +
    geom_boxplot(outlier.shape = NA, position = position_dodge(0.8)) +
    scale_fill_manual(values = c(proximity = "#4DBBD5", random = "#E64B35")) +
    labs(title = "Proximity-ranked vs random source subset (PCA, ComBat-both)",
         x = "Source subset size", y = "AUC") +
    theme_classic() + coord_cartesian(ylim = c(0.3, 1))
  ggsave(file.path(FIG, "C_proximity_vs_random.pdf"), p, width = 6.5, height = 4)
  cat("wrote C_proximity_vs_random.pdf\n")
} else cat("skip (missing): random_subset_auc group files\n")

# ---- Phase E: bootstrap CI + permutation null ----
if (have(file.path(RES, "stability/stability_permutation.csv"))) {
  perm <- read.csv(file.path(RES, "stability/stability_permutation.csv"))
  summ <- jsonlite::fromJSON(file.path(RES, "stability/summary_permutation.json"))
  p <- ggplot(perm, aes(auc)) + geom_histogram(bins = 30, fill = "grey70") +
    geom_vline(xintercept = summ$obs_mean_auc, color = "red", linewidth = 1) +
    labs(title = sprintf("Outcome-label permutation null (p=%.3f)", summ$emp_p),
         x = "AUC", y = "count") + theme_classic()
  ggsave(file.path(FIG, "E_permutation_null.pdf"), p, width = 5, height = 4)
  cat("wrote E_permutation_null.pdf\n")
}

# ---- Phase F: selection-frequency + randomization null ----
if (have(file.path(RES, "genera_randomization_pvalues.csv"))) {
  d <- read.csv(file.path(RES, "genera_randomization_pvalues.csv"))
  d$sig <- ifelse(d$emp_p < 0.05, "p<0.05", "ns")
  d <- d[order(-d$selection_count), ]; d <- head(d, 30)
  d$genus <- factor(d$genus, levels = rev(d$genus))
  p <- ggplot(d, aes(genus, selection_count, fill = sig)) +
    geom_col() + coord_flip() +
    geom_point(aes(y = null_mean_freq), shape = 4, size = 1.5, color = "black") +
    scale_fill_manual(values = c("p<0.05" = "#E64B35", "ns" = "grey70")) +
    labs(title = "Selection frequency vs randomization null (x = null mean)",
         x = NULL, y = "selection count (/10)") + theme_classic()
  ggsave(file.path(FIG, "F_selection_frequency.pdf"), p, width = 6, height = 7)
  cat("wrote F_selection_frequency.pdf\n")
}

# ---- Phase G: covariate-augmented prediction ----
if (have(file.path(RES, "covariate_pred/covariate_prediction.csv"))) {
  d <- read.csv(file.path(RES, "covariate_pred/covariate_prediction.csv"))
  long <- reshape(d, varying = c("micro_only", "cov_only", "micro_plus_cov"),
                  v.names = "auc", timevar = "model",
                  times = c("micro_only", "cov_only", "micro_plus_cov"),
                  direction = "long")
  long$model <- factor(long$model, levels = c("cov_only", "micro_only", "micro_plus_cov"))
  p <- ggplot(long, aes(model, auc, fill = model)) +
    geom_boxplot(outlier.shape = NA) + geom_jitter(width = 0.15, alpha = 0.7, size = 1) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Covariate-augmented prediction (best config)", x = NULL, y = "AUC") +
    theme_classic() + coord_cartesian(ylim = c(0.3, 1)) + guides(fill = "none")
  ggsave(file.path(FIG, "G_covariate_prediction.pdf"), p, width = 5, height = 4)
  cat("wrote G_covariate_prediction.pdf\n")
}

# ---- Phase G: covariate logistic-coefficient test (OOF score vs response) ----
if (have(file.path(RES, "covariate_logistic/oof_scores.csv"))) {
  d <- read.csv(file.path(RES, "covariate_logistic/oof_scores.csv"))
  d$response <- factor(ifelse(d$response_binary == 1, "Responder", "Non-responder"))
  p <- ggplot(d, aes(response, bridgeko_score, fill = response)) +
    geom_boxplot(outlier.shape = NA) + geom_jitter(width = 0.15, alpha = 0.6, size = 1) +
    scale_fill_manual(values = c(Responder = "#E64B35", `Non-responder` = "#4DBBD5")) +
    labs(title = "Out-of-fold BRIDGE-KO score by response",
         x = NULL, y = "OOF BRIDGE-KO score") + theme_classic() + guides(fill = "none")
  ggsave(file.path(FIG, "G_oof_score_by_response.pdf"), p, width = 4.5, height = 4)
  cat("wrote G_oof_score_by_response.pdf\n")
}

# ---- Phase D: within-fold (leakage-free) vs original AUC ----
wf_dir <- file.path(RES, "within_fold_auc")
wf_files <- file.path(wf_dir, paste0("group_", grp_levels), paste0("auc_g", grp_levels, ".csv"))
if (all(file.exists(wf_files))) {
  wf <- do.call(rbind, lapply(grp_levels, function(g)
    transform(read.csv(file.path(wf_dir, paste0("group_", g), paste0("auc_g", g, ".csv"))), group = g)))
  wf$type <- "within-fold (leakage-free)"
  cap <- read.csv(file.path(RES, "cached_auc_from_pdf.csv"), comment.char = "#")
  orig <- subset(cap, setting == "pca_both")[, c("group", "auc")]; orig$type <- "original (main)"
  both <- rbind(orig, wf[, c("group", "auc", "type")])
  both$group <- factor(both$group, levels = grp_levels, labels = grp_labels)
  p <- ggplot(both, aes(group, auc, fill = type)) +
    geom_boxplot(outlier.shape = NA, position = position_dodge(0.8)) +
    scale_fill_manual(values = c("original (main)" = "#4DBBD5", "within-fold (leakage-free)" = "#E64B35")) +
    labs(title = "ComBat leakage sensitivity: original vs within-fold", x = "Source subset", y = "AUC") +
    theme_classic() + coord_cartesian(ylim = c(0.3, 1))
  ggsave(file.path(FIG, "D_within_fold_sensitivity.pdf"), p, width = 6.5, height = 4)
  cat("wrote D_within_fold_sensitivity.pdf\n")
} else cat("skip (missing): within_fold_auc group files\n")

cat("make_figures DONE\n")
