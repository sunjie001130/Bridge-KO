#!/bin/bash
#SBATCH --job-name=bk_rprep
#SBATCH --partition=batch_cpu
#SBATCH --qos=1d
#SBATCH --output=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.out
#SBATCH --error=/home/sunj107/scratch/Bridge_revision/revision/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Run an R prep/analysis script on a compute node.
#   sbatch rprep.sbatch <script.R> [args...]
set -euo pipefail
SCRIPT="${1:?R script path}"; shift || true
module load R/4.5.0-foss-2024a
export R_LIBS_USER=/home/sunj107/scratch/Rlibs
Rscript "$SCRIPT" "$@"
