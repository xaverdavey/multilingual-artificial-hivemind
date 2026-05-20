#!/bin/bash
# Submit one SLURM job per active (uncommented) model in models.yaml, applying
# the per-model gpus/time override declared in the YAML.
# Usage: run from the repo root, i.e. ./experiments/submit_sweep.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."

mkdir -p logs

# Emit one TSV line per active model: id<TAB>gpus<TAB>time
mapfile -t entries < <(python -c "
import yaml
for m in yaml.safe_load(open('experiments/models.yaml')):
    print(f\"{m['id']}\t{m['gpus']}\t{m['time']}\")
")

echo "submitting ${#entries[@]} models"
for line in "${entries[@]}"; do
  IFS=$'\t' read -r id gpus time <<<"$line"
  echo "  $id  ($gpus, $time)"
  sbatch \
    --gpus-per-node="$gpus" \
    -t "$time" \
    -J "hivemind:$id" \
    --export=ALL,MODEL="$id" \
    experiments/sbatch_run.sh
done
