#!/bin/bash
#SBATCH --job-name=bg_reg
#SBATCH --partition=batch_cpu
#SBATCH --qos=1d
#SBATCH --output=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.out
#SBATCH --error=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Tune the L_BG background-regularization penalty on the headline arm
# (PCA, ComBat-both, top-25% source) and report tuned vs lambda_BG=0.
#   sbatch revision/sbatch/bg_reg.sbatch
set -eo pipefail

module load Miniforge3/24.1.2-0
export MAMBA_ROOT_PREFIX=/home/sunj107/scratch/micromamba
# Login/compute nodes cap pthreads; bound TF threading to avoid pthread_create crashes.
export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2

ROOT=/home/sunj107/scratch/Bridge_revision
SCR=$ROOT/revision/scripts
PREP=$ROOT/revision/results/pca_knockoffs

micromamba run -n bridgeko python "$SCR/tune_bg_reg.py" \
  --xtrain "$PREP/train_big_equi_knockoffs_70_25_pca.csv" \
  --ytrain "$PREP/meta_train_70.csv" \
  --xtest  "$PREP/test_big_equi_knockoffs_30_25_pca.csv" \
  --ytest  "$PREP/meta_test_30.csv" \
  --outdir "$ROOT/revision/results/bg_reg" --n-seeds 10
