#!/bin/bash
#SBATCH --job-name=bk_one
#SBATCH --partition=batch_cpu
#SBATCH --qos=1d
#SBATCH --output=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.out
#SBATCH --error=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#   sbatch run_one.sbatch <prep_dir> <out_dir> <tag>
set -eo pipefail
PREP="${1:?}"; OUT="${2:?}"; TAG="${3:?}"
module load Miniforge3/24.1.2-0
export MAMBA_ROOT_PREFIX=/home/sunj107/scratch/micromamba
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TF_NUM_INTEROP_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4} TF_CPP_MIN_LOG_LEVEL=3
micromamba run -n bridgeko python /home/sunj107/scratch/Bridge_revision/revision/scripts/run_bridgeko.py \
  --xtrain "$PREP/train_big_equi_knockoffs_70_25_pca.csv" --ytrain "$PREP/meta_train_70.csv" \
  --xtest  "$PREP/test_big_equi_knockoffs_30_25_pca.csv"  --ytest  "$PREP/meta_test_30.csv" \
  --outdir "$OUT" --tag "$TAG" --n-seeds 10
