#!/usr/bin/env Rscript
# genera_stability_stats.R  [Phase F: R1#9]
# Three analyses on the equi best-config selected-genera (10 runs):
#  1. Selection-frequency randomization test (preserve per-run selection count)
#  2. Enrichment of literature-supported ICI genera among frequently-selected
#     (Fisher's exact + hypergeometric), threshold sensitivity at >=5/6/7
#  (Outcome-label permutation is a separate DNN job: permutation_genera.py)
#
# Inputs:
#   Bac_final_model/selected_features_run{1..10}.csv  (col "feature")
#   104 candidate genera = columns of small_scaled_combat2.csv
# Outputs:
#   revision/results/genera_selection_frequency.csv
#   revision/results/genera_randomization_pvalues.csv
#   revision/results/genera_enrichment.csv
suppressMessages({ .libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths())) })

set.seed(2025)
WD <- "/home/sunj107/scratch/Bridge_revision"
OUT <- file.path(WD, "revision/results"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
N_RAND <- 1000

# --- candidate genera (the 104 shared) ---
hdr <- readLines(file.path(WD, "small_scaled_combat2.csv"), n = 1)
candidates <- strsplit(gsub('"', '', hdr), ",")[[1]][-1]
n_cand <- length(candidates)
stopifnot(n_cand == 104)

# --- observed selections per run ---
run_files <- file.path(WD, "Bac_final_model",
                        sprintf("selected_features_run%d.csv", 1:10))
stopifnot(all(file.exists(run_files)))
runs <- lapply(run_files, function(f) as.character(read.csv(f)$feature))
per_run_counts <- sapply(runs, length)
cat("per-run selection counts:", paste(per_run_counts, collapse = ", "), "\n")

# observed selection frequency per genus (0..10)
obs_freq <- table(factor(unlist(runs), levels = candidates))
obs_freq <- as.integer(obs_freq); names(obs_freq) <- candidates

freq_df <- data.frame(genus = candidates, selection_count = obs_freq,
                      stability = ifelse(obs_freq >= 6, "frequent",
                                  ifelse(obs_freq == 5, "intermediate", "infrequent")))
freq_df <- freq_df[order(-freq_df$selection_count), ]
write.csv(freq_df, file.path(OUT, "genera_selection_frequency.csv"), row.names = FALSE)
n_freq_obs <- sum(obs_freq >= 6)
cat(sprintf("observed #frequently-selected (>=6/10): %d\n", n_freq_obs))

# --- (1) selection-frequency randomization test ---
# Null: each run independently selects per_run_counts[i] genera at random from 104.
null_freq_mat <- matrix(0L, nrow = N_RAND, ncol = n_cand,
                        dimnames = list(NULL, candidates))
null_nfreq <- integer(N_RAND)
for (b in seq_len(N_RAND)) {
  cnt <- integer(n_cand); names(cnt) <- candidates
  for (i in seq_along(per_run_counts)) {
    sel <- sample(candidates, per_run_counts[i])
    cnt[sel] <- cnt[sel] + 1L
  }
  null_freq_mat[b, ] <- cnt
  null_nfreq[b] <- sum(cnt >= 6)
}

# per-genus empirical p-value: P(null freq >= observed freq)
pgenus <- sapply(candidates, function(g)
  (1 + sum(null_freq_mat[, g] >= obs_freq[g])) / (N_RAND + 1))
# overall: P(null #frequent >= observed #frequent)
p_overall <- (1 + sum(null_nfreq >= n_freq_obs)) / (N_RAND + 1)

rand_df <- data.frame(genus = candidates, selection_count = obs_freq,
                      null_mean_freq = colMeans(null_freq_mat),
                      emp_p = pgenus)
rand_df <- rand_df[order(-rand_df$selection_count), ]
write.csv(rand_df, file.path(OUT, "genera_randomization_pvalues.csv"), row.names = FALSE)
cat(sprintf("null #frequent: mean=%.2f, 95%%=%d ; observed=%d ; overall emp p=%.4f\n",
            mean(null_nfreq), quantile(null_nfreq, 0.95), n_freq_obs, p_overall))

# --- (2) enrichment of literature ICI genera ---
# 8 genera with prior ICI evidence (manuscript Results paras 54-55).
lit_genera <- c("Akkermansia", "Collinsella", "Parabacteroides", "Roseburia",
                "Barnesiella", "Enterocloster", "Escherichia", "Bifidobacterium")
lit_in_cand <- intersect(lit_genera, candidates)
cat("literature genera in 104:", paste(lit_in_cand, collapse = ", "),
    sprintf("(%d/%d)\n", length(lit_in_cand), length(lit_genera)))

enrich_rows <- list()
for (thr in c(5, 6, 7)) {
  selected <- names(obs_freq)[obs_freq >= thr]
  a <- length(intersect(selected, lit_in_cand))          # lit & selected
  b <- length(setdiff(selected, lit_in_cand))            # nonlit & selected
  c <- length(setdiff(lit_in_cand, selected))            # lit & not selected
  d <- n_cand - a - b - c                                # nonlit & not selected
  ft <- fisher.test(matrix(c(a, b, c, d), 2, 2), alternative = "greater")
  # hypergeometric: P(X >= a) drawing |selected| from 104 with |lit| successes
  hyp <- phyper(a - 1, length(lit_in_cand), n_cand - length(lit_in_cand),
                length(selected), lower.tail = FALSE)
  enrich_rows[[as.character(thr)]] <- data.frame(
    threshold = sprintf(">=%d/10", thr), n_selected = length(selected),
    lit_selected = a, lit_total = length(lit_in_cand),
    odds_ratio = unname(ft$estimate), fisher_p = ft$p.value, hyper_p = hyp)
}
enrich_df <- do.call(rbind, enrich_rows)
write.csv(enrich_df, file.path(OUT, "genera_enrichment.csv"), row.names = FALSE)
cat("\n=== enrichment ===\n"); print(enrich_df, row.names = FALSE)
cat("DONE genera_stability_stats\n")
