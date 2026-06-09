#!/bin/bash
#SBATCH --job-name=bk_stability
#SBATCH --partition=batch_cpu
#SBATCH --qos=1d
#SBATCH --output=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.out
#SBATCH --error=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#   sbatch stability.sbatch <bootstrap|permutation>
set -eo pipefail
MODE="${1:?mode}"
WD=/home/sunj107/scratch/Bridge_revision
P=$WD/revision/results/pca_knockoffs
module load Miniforge3/24.1.2-0
export MAMBA_ROOT_PREFIX=/home/sunj107/scratch/micromamba
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TF_NUM_INTEROP_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4} TF_CPP_MIN_LOG_LEVEL=3

if [ "$MODE" = "bootstrap" ]; then
  COV="$P/split_test_30.csv"
else
  COV="$P/split_test_30.csv"
fi
micromamba run -n bridgeko python "$WD/revision/scripts/stability_auc.py" \
  --mode "$MODE" \
  --xtrain "$P/train_big_equi_knockoffs_70_25_pca.csv" --ytrain "$P/meta_train_70.csv" \
  --xtest  "$P/test_big_equi_knockoffs_30_25_pca.csv"  --ytest  "$P/meta_test_30.csv" \
  --covariates "$COV" \
  --outdir "$WD/revision/results/stability" --n-seeds 10 --n-boot 1000 --n-perm 200
