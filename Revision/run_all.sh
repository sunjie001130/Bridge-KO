#!/bin/bash
# run_all.sh — full BRIDGE-KO revision pipeline (documentation + driver).
# Heavy DNN/prep steps go to SLURM (batch_cpu, --qos=1d). R analyses can run on the
# login node with thread limits. Env setup is in README.md.
#
# This is a DEPENDENCY-ORDERED reference, not a single blocking script: SLURM jobs are
# async. Submit a stage, wait for it (squeue), then submit the next that depends on it.
set -eo pipefail
WD=/home/sunj107/scratch/Bridge_revision
cd "$WD"
RP="sbatch revision/sbatch/rprep.sbatch"
export R_LIBS_USER=/home/sunj107/scratch/Rlibs

echo "Phase 0: env already built (micromamba env 'bridgeko' + R_LIBS_USER). DL vendored in code/DL."

echo "Data prep (best config knockoffs):"
echo "  $RP revision/scripts/prep_pca_knockoffs.R"

echo "Phase A (knockoff comparison):"
echo "  $RP revision/scripts/knockoff_covariance.R           # runtime + covariance preservation"
echo "  $RP revision/scripts/prep_knockoffs_3methods.R       # equi/sdp/asdp knockoffs (best config)"
echo "  sbatch revision/sbatch/run_one.sbatch revision/results/km_<m> revision/results/km_<m>_auc km_<m>  # x3"

echo "Phase B (baselines):"
echo "  sbatch revision/sbatch/baselines.sbatch              # target-only, CORAL, DANN"

echo "Phase C (ablations):"
echo "  $RP revision/scripts/prep_ablations.R                # random source subset"
echo "  sbatch --array=0-3 revision/sbatch/run_bridgeko.sbatch revision/results/random_subset revision/results/random_subset_auc"
echo "  $RP revision/scripts/combat_source_only.R            # source-only ComBat (mean.only)"
echo "  $RP revision/scripts/prep_pca_knockoffs_param.R --big revision/results/combat_source_only/big_scaled_srcCombat.csv --small revision/results/combat_source_only/small_scaled_raw.csv --outdir revision/results/source_only_knockoffs"
echo "  sbatch --array=0-3 revision/sbatch/run_bridgeko.sbatch revision/results/source_only_knockoffs revision/results/source_only_auc"

echo "Phase D (leakage):"
echo "  $RP revision/scripts/combat_within_fold.R            # leakage-free ComBat (fit train, apply test)"
echo "  sbatch --array=0-3 revision/sbatch/run_bridgeko.sbatch revision/results/within_fold revision/results/within_fold_auc"

echo "Phase E (stability):"
echo "  sbatch revision/sbatch/stability.sbatch bootstrap"
echo "  sbatch revision/sbatch/stability.sbatch permutation"
echo "  Rscript revision/scripts/fig4e_recolor.R"

echo "Phase F (genera stats):"
echo "  Rscript revision/scripts/genera_stability_stats.R"

echo "Phase G (covariates):"
echo "  Rscript revision/scripts/build_covariates.R"
echo "  sbatch revision/sbatch/covariate_pred.sbatch"
echo "  $RP revision/scripts/combat_covariate_aware.R        # m1/m2 OK; m3 confounded w/ city"
echo "  $RP revision/scripts/prep_pca_knockoffs_param.R --big big_scaled_combat2.csv --small revision/results/combat_covaware/small_scaled_m2_response.csv --outdir revision/results/covaware_m2_knockoffs"
echo "  sbatch --array=0-3 revision/sbatch/run_bridgeko.sbatch revision/results/covaware_m2_knockoffs revision/results/covaware_m2_auc"

echo "Figures (after results land; incremental, skips missing):"
echo "  Rscript revision/scripts/make_figures.R"
echo "Summary: revision/RESPONSE_NOTES.md ; preprocessing: revision/PREPROCESSING.md"
