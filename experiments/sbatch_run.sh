#!/bin/bash
# One model per job. Submit via experiments/submit_sweep.sh -- see README for setup.

#SBATCH -A NAISS2025-22-1727
#SBATCH -J hivemind
#SBATCH -o logs/sweep-%j.out

set -euo pipefail

: "${MODEL:?MODEL env var not set -- submit via experiments/submit_sweep.sh}"

cd "$SLURM_SUBMIT_DIR"

SCRATCH_BASE=/mimer/NOBACKUP/groups/naiss2025-22-675/xaver-hivemind
export HF_HOME=$SCRATCH_BASE/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
# Pick up the cephyr-cached token since HF_HOME no longer points at it.
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.0.0
source "$SCRATCH_BASE/envs/hivemind/bin/activate"

echo "[$(date -Is)] $MODEL on $(hostname) with ${SLURM_GPUS_PER_NODE:-?}"

python -m experiments.run_generation \
  --model "$MODEL" \
  --out-dir "$SCRATCH_BASE/results/raw"
