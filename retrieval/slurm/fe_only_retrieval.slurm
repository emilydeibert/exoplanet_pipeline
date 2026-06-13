#!/bin/bash
#SBATCH --job-name=mascara1b-fe-retrieval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
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

# Only run this after the smoke test and Fe Kp-Vsys grid have been inspected.
python -m retrieval.run_fe_sampler \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --confirm-grid-validated
