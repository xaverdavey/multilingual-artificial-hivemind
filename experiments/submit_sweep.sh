#!/bin/bash
# Submit one SLURM array task per active (uncommented) model in models.yaml.
# Usage: experiments/submit_sweep.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N=$(python -c "import yaml; print(len(yaml.safe_load(open('experiments/models.yaml'))))")
echo "submitting array of $N models"
mkdir -p logs
sbatch --array=0-$((N - 1)) experiments/sbatch_run.sh
