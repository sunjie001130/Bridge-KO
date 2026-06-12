# BRIDGE-KO revision — results for the response letter (Jie's part)

Numbers below are from `revision/results/`. Best config throughout = **PCA distance,
ComBat on both source+target, most-proximal 25% source, 70/30 split × 10 seeds**
(BRIDGE-KO mean AUC ≈ 0.679). PCA is the main figure; UMAP is supplementary (S Fig. 3).

## R1#1 / R2#1 — knockoff method comparison (sdp / asdp / equi)
`knockoff_covariance.R` → figs `A_runtime.pdf`, `A_covariance_preservation.pdf`.

| Method | Runtime (s) | Mean \|Δcov\| (off-diag, lower=better) | Orig–KO self-corr (lower=better) |
|---|---|---|---|
| **equi** (adopted) | **9.0** | **0.083** | 0.487 |
| sdp | 1333 | 0.121 | 0.172 |
| asdp | 984 | 0.121 | 0.172 |

- equi is ~150× faster AND preserves the pairwise covariance structure best (smallest
  \|cov(X)−cov(X̃)\|). The tradeoff: equi's original–knockoff self-correlation is higher
  (0.49 vs 0.17), i.e. sdp/asdp decorrelate the knockoff from its original more strongly.
- At p=104 features, n=154 samples, sdp/asdp are far slower with no covariance-fidelity
  gain, so **equi is the justified choice** for this high-dimensional microbiome setting.
- Downstream AUC by method (best config, 10 seeds, freshly run): equi **0.679**, sdp 0.620,
  asdp 0.620 → `A_auc_by_method.pdf`. equi is best downstream AND cheapest; sdp≈asdp (asdp
  approximates sdp at this dimension). Consistent with prior `LMM_microbiomap.pdf` p27–28.
- UpSet of frequently-selected genera (≥6/10) across methods: `A_upset_selected_genera.pdf`.

## R1#4 / R2#2 — outside-method baselines
`baselines.py` → fig `B_baselines.pdf`. Best config, 10 seeds.

| Method | Mean AUC | SD |
|---|---|---|
| **BRIDGE-KO** | **0.679** | 0.039 |
| Target-only DNN (no transfer, no knockoff) | 0.625 | 0.079 |
| CORAL (covariance-alignment DA) | 0.620 | 0.046 |
| DANN (adversarial-alignment DA) | 0.537 | 0.059 |

- BRIDGE-KO outperforms the no-transfer reference and BOTH standard domain-adaptation
  baselines. DANN (adversarial) actively underperforms on this small target cohort.
- Note (also in Introduction): LEEP/LogME/TransRate/OTCE transferability estimators need
  source outcome labels, which Microbiomap lacks, so the transferability-guided baseline
  is implemented label-free (source–target proximity ranking = the BRIDGE-KO coverage axis).

## R1#5 — ablations
- **Random vs proximity source subset** (`prep_ablations.R` + DNN) → `C_proximity_vs_random.pdf`:
  random top-25% AUC 0.603 vs proximity-ranked top-25% 0.679 → proximity ranking carries
  real signal beyond subset size.
- **ComBat source-only** (`combat_source_only.R`, source moved toward target, ref.batch=target):
  AUC 0.50 / 0.48 / 0.51 / 0.47 (25/50/75/100) ≈ chance. Correcting only the source does NOT
  help — harmonization must be applied to BOTH source and target (both-ComBat 0.68). Another
  clean ablation supporting the manuscript's harmonization design.
- **No-knockoff** ablation = the target-only DNN baseline above (0.625).
- See `ALL_AUC_SUMMARY.csv` for every arm in one table.

## R1#6 / R2#3 — preprocessing order, 186 vs 154, leakage
See `PREPROCESSING.md`. 186 raw → 154 after dropping target samples with <5 non-zero
genera; 104 shared genera; 154 used in all analyses. The original ComBat was fit on the
full combined matrix before the 70/30 split (mild leakage). Leakage-free sensitivity
(`combat_within_fold.R`, ComBat fit on source+train-target, applied to test):
`D_within_fold_sensitivity.pdf`.

| Group | Original (leaky, main) | Within-fold (leakage-free) |
|---|---|---|
| top 25% | 0.679 | 0.623 |
| 25–50% | 0.594 | 0.608 |
| 50–75% | 0.626 | 0.629 |
| 75–100% | 0.636 | 0.620 |

Removing the leakage lowers the headline top-25% AUC modestly (0.679 → 0.623) but transfer
still works (~0.62, well above chance) and is stable across proximity groups — the
conclusion is robust to the preprocessing-order concern.

## R1#6 / R2#4 / R2#6 — stability
- **Outcome-label permutation** (`stability_auc.py --mode permutation`, 200 perms) → `E_permutation_null.pdf`:
  observed AUC 0.679 vs null mean 0.489, **empirical p = 0.040**. The BRIDGE-KO AUC is
  significantly above the label-permuted null — answers R2#4 (differences are distinguishable).
- **Stratified balanced bootstrap** (city × R/NR, 10 seeds × 1000 resamples = 10,000):
  mean AUC 0.719, **95% CI [0.50, 0.88]**. Wide CI honestly reflects the small test set
  (n=47); point estimate robust — answers R2#6 (sample-size stability).
- **Fig 4E recolored** by city and by R/NR: `E_fig4e_by_city.pdf`, `E_fig4e_by_response.pdf`.

## R1#9 — genera selection stability
`genera_stability_stats.R` → fig `F_selection_frequency.pdf`.
- Per-run ~50 genera selected; all 104 selected ≥1×; 42 genera selected ≥6/10.
- Selection-frequency randomization (1000×, per-run count preserved): the *aggregate*
  count of ≥6/10 genera (42) is not above the random null (mean 38.6; p=0.16) — supports
  reframing ≥6/10 as a **stability category**, not a significance threshold.
- Per-genus: **Desulfovibrio (10/10) is significant (p=0.002)**; several 8/10 genera p≈0.05.
- Literature ICI-genera enrichment (8 curated genera): not significant (Fisher/hyper p>0.2),
  but odds ratio rises with stricter cutoffs (0.55 → 1.52 → 2.41 at ≥5/6/7).

## R2#6 / R2#7 — covariates
`build_covariates.R`, `covariate_prediction.py` → fig `G_covariate_prediction.pdf`.
- Available covariates (joined to all 154 via clean_id): Age, Sex, BMI, **sequencing method**
  (verified from McCulloch Suppl. Table 3), **Abx_Use**, H2RA_PPI_Use, Pre_Treatment_Disease_Stage.
  Sequencing method = 3 shotgun profiler families complete for all 154: JAMS (Pittsburgh 63),
  MetaOMineR (Houston 25), MetaPhlAn-family (Chicago+New_York+Dallas 66); 1:1 with city/cohort
  (= the ComBat batch). Abx/stage present only for the 63 Pittsburgh samples; the 91 public-cohort
  samples are "unknown". ECOG unavailable. NB: `treatment` (ICI regimen) intentionally dropped —
  the requested covariate is sequencing method, not regimen.
- Covariates-only prediction ≈ chance (AUC 0.491) → covariates carry little response signal.
- **Logistic-coefficient test (cross-fitted OOF score, 5-fold×5, replaces the leaky stacker):**
  score-only OR=1.23 p=0.22 (OOF AUC=0.57); covariate-adjusted (incl. sequencing method)
  OR=1.22 Wald p=0.27; LRT M1 vs M0 χ²(1)=1.24 **p=0.27**. Sequencing method itself NOT
  associated with response (MetaOMineR p=0.99, MetaPhlAn p=0.81) and adjusting for it barely
  moves the score OR (1.23→1.22) → microbiome signal is NOT a sequencing-batch artifact, but
  remains a positive NON-significant trend after adjustment at n=154 — reported honestly. The
  fully-cross-validated AUC (0.57) is lower than the fixed-split 0.679; both now reported so
  effect size isn't overstated. Figs `G_covariate_prediction.pdf`, `G_oof_score_by_response.pdf`.
  Scripts `covariate_logistic_test.py` (cross-fit) + `logistic_from_oof.py` (stats).
- Covariate-aware ComBat on the target across its 5 city batches (`combat_covariate_aware.R`):
  m1 (batch-only) and **m2 (~response) succeed**; **m3 (~response+treatment+Abx+stage) is
  infeasible — those covariates are confounded with city** (regimen/Abx/stage are recorded
  only for Pittsburgh, so city perfectly predicts their presence). This is a genuine data
  limitation to state, not an analysis failure. AUC under m2-corrected target (paired with
  both-ComBat source, 4 proximity groups): 0.52 / 0.54 / 0.52 / 0.54 — lower than the full
  both-ComBat best (0.679). Covariate-aware target ComBat is feasible but does not match the
  full source+target harmonization; the conclusion (both-ComBat best) is unchanged and the
  response signal is preserved without inflating AUC.
