#!/bin/bash
# One model per job. Submit via experiments/submit_sweep.sh, which sets MODEL
# and the per-model --gpus-per-node / -t overrides.
#
# Run-once setup (not done by this script):
#   - create venv with vllm, pandas, huggingface_hub, pyyaml, torch (CUDA build)
#   - huggingface-cli login (needs gated-model access for meta-llama, mistralai)
#   - mkdir -p $MIMER_BASE/{hf-cache,results/raw,logs}

#SBATCH -A NAISS2025-22-1727
#SBATCH -J hivemind
#SBATCH -o logs/sweep-%j.out

set -euo pipefail

: "${MODEL:?MODEL env var not set -- submit via experiments/submit_sweep.sh}"

cd "$SLURM_SUBMIT_DIR"

MIMER_BASE=/mimer/NOBACKUP/groups/naiss2025-22-675/xaver-hivemind
export HF_HOME=$MIMER_BASE/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
# Pick up the cephyr-cached token since HF_HOME no longer points at it.
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.0.0
source "$MIMER_BASE/envs/hivemind/bin/activate"

echo "[$(date -Is)] $MODEL on $(hostname) with ${SLURM_GPUS_PER_NODE:-?}"

python -m experiments.run_generation \
  --model "$MODEL" \
  --out-dir "$MIMER_BASE/results/raw"
