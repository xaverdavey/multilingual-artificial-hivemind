#!/bin/bash
# Submit one SLURM job per active (uncommented) model in models.yaml, applying
# the per-model gpus/time override declared in the YAML.
# Usage: run from the repo root, i.e. ./experiments/submit_sweep.sh [generation|competence]
# The competence benchmarks are ~12K greedy short translations and ~12K one-token
# answers per model, far lighter than generation, so their walltime is capped.
set -euo pipefail
TASK=${1:-generation}
case "$TASK" in generation|competence) ;; *) echo "unknown task: $TASK" >&2; exit 1;; esac
cd "$(dirname "$(realpath "$0")")/.."

mkdir -p logs

# Emit one TSV line per active cluster-bound model: id<TAB>gpus<TAB>time.
# Entries without gpus/time are API models and are skipped.
mapfile -t entries < <(TASK="$TASK" python -c "
import os, yaml
cap = '02:00:00' if os.environ['TASK'] == 'competence' else None
for m in yaml.safe_load(open('experiments/models.yaml')):
    if not m.get('gpus') or not m.get('time'):
        continue
    t = min(m['time'], cap) if cap else m['time']   # zero-padded HH:MM:SS sorts as text
    print(f\"{m['id']}\t{m['gpus']}\t{t}\")
")

echo "submitting ${#entries[@]} models ($TASK)"
for line in "${entries[@]}"; do
  IFS=$'\t' read -r id gpus time <<<"$line"
  echo "  $id  ($gpus, $time)"
  sbatch \
    --gpus-per-node="$gpus" \
    -t "$time" \
    -J "hivemind-$TASK:$id" \
    --export=ALL,MODEL="$id",TASK="$TASK" \
    experiments/sbatch_run.sh
done
