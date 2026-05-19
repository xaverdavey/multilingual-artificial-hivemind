#!/bin/bash
# One model per array task. Submit via experiments/submit_sweep.sh, which
# auto-sizes the array to the number of active entries in models.yaml.
#
# Run-once setup (not done by this script):
#   - create venv with vllm, pandas, huggingface_hub, pyyaml, torch (CUDA build)
#   - huggingface-cli login (needs gated-model access for meta-llama, mistralai)
#   - mkdir -p $MIMER_BASE/{hf-cache,results/raw,logs}

#SBATCH -A NAISS2025-22-1727
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J hivemind
#SBATCH -o logs/sweep-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

MIMER_BASE=/mimer/NOBACKUP/groups/naiss2025-22-675/xaver-hivemind
export HF_HOME=$MIMER_BASE/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
# Pick up the cephyr-cached token since HF_HOME no longer points at it.
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")

module load Python/3.12.3-GCCcore-13.3.0
source "$MIMER_BASE/envs/hivemind/bin/activate"

MODEL=$(python -c "import yaml; print(yaml.safe_load(open('experiments/models.yaml'))[$SLURM_ARRAY_TASK_ID])")
echo "[$(date -Is)] task $SLURM_ARRAY_TASK_ID -> $MODEL on $(hostname)"

python -m experiments.run_generation \
  --model "$MODEL" \
  --out-dir "$MIMER_BASE/results/raw"
