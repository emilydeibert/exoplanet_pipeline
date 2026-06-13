#!/bin/bash
#SBATCH --job-name=mascara1b-fe-grid
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=retrieval/slurm/logs/%x-%j.out
#SBATCH --error=retrieval/slurm/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p retrieval/slurm/logs

module load python/3.11

# Choose one environment activation style and edit the path/name.
# source "${HOME}/venvs/exopipe-prt/bin/activate"
# source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# conda activate astro

export PETITRADTRANS_INPUT_DATA="/path/to/petitRADTRANS/input_data"

python -m retrieval.check_retrieval_environment \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --require-local-opacities

python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --n-jobs "${SLURM_CPUS_PER_TASK}"
