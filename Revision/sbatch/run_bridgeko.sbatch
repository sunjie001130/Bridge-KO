#!/bin/bash
#SBATCH --job-name=bridgeko
#SBATCH --partition=batch_cpu
#SBATCH --qos=1d
#SBATCH --output=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%A_%a.out
#SBATCH --error=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Generic BRIDGE-KO DNN runner. Pass a config dir + group via env or args:
#   sbatch --array=0-3 run_bridgeko.sbatch <prep_dir> <out_dir>
# array index maps to proximity group {25,50,75,100}.
set -eo pipefail
PREP_DIR="${1:?prep dir}"; OUT_DIR="${2:?out dir}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "$IDX" in
  0) G=25 ;; 1) G=50 ;; 2) G=75 ;; 3) G=100 ;;
  *) echo "bad array idx $IDX"; exit 1 ;;
esac

module load Miniforge3/24.1.2-0
export MAMBA_ROOT_PREFIX=/home/sunj107/scratch/micromamba
SCR=/home/sunj107/scratch/Bridge_revision/revision/scripts

micromamba run -n bridgeko python "$SCR/run_bridgeko.py" \
  --xtrain "$PREP_DIR/train_big_equi_knockoffs_70_${G}_pca.csv" \
  --ytrain "$PREP_DIR/meta_train_70.csv" \
  --xtest  "$PREP_DIR/test_big_equi_knockoffs_30_${G}_pca.csv" \
  --ytest  "$PREP_DIR/meta_test_30.csv" \
  --outdir "$OUT_DIR/group_${G}" --tag "g${G}" --n-seeds 10
