#!/bin/bash
#SBATCH --job-name=bk_covlogit
#SBATCH --partition=batch_cpu
#SBATCH --qos=1d
#SBATCH --output=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.out
#SBATCH --error=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -eo pipefail
WD=/home/sunj107/scratch/Bridge_revision
F=$WD/revision/results/fulltarget
module load Miniforge3/24.1.2-0
export MAMBA_ROOT_PREFIX=/home/sunj107/scratch/micromamba
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TF_NUM_INTEROP_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4} TF_CPP_MIN_LOG_LEVEL=3
micromamba run -n bridgeko python "$WD/revision/scripts/covariate_logistic_test.py" \
  --combined "$F/full_big_equi_knockoffs_pca25.csv" \
  --meta "$F/full_meta.csv" \
  --covariates "$WD/revision/results/covariates_154.csv" \
  --outdir "$WD/revision/results/covariate_logistic" --n-seeds 5 --folds 5
