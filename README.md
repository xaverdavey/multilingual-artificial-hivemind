# Multilingual Artificial Hivemind

Does the Artificial Hivemind effect (Jiang et al. 2025) generalize multilingually? See [`PLAN.md`](PLAN.md) for the full plan, hypotheses, and metric definitions.

## Layout

| Path | Purpose |
|---|---|
| `PLAN.md` | Research questions, hypotheses, dataset, models, protocol, metrics, analysis, limitations |
| `experiments/select_oasst2_prompts.py` | Build the locked OASST2 prompt set (seed=42, K=30/lang × 13 langs) → `experiments/oasst2_prompts.parquet` |
| `experiments/run_generation.py` | One model × full prompt set → `results/raw/{model}.parquet` + `.meta.json` (vLLM) |
| `experiments/sweep_models.py` | Sweep `run_generation` over a list of HF model ids with overlapping prefetch + per-model cache purge |
| `experiments/models.yaml` | Sweep manifest: `id` / `gpus` / `time` per model |
| `experiments/submit_sweep.sh` + `sbatch_run.sh` | Submit one SLURM job per active model with per-model GPU/time overrides |
| `analysis.ipynb` | Scratch notebook (PCA visualizations of generations; not load-bearing) |
| `artificial-hivemind/` | Cloned reference repo for the original Jiang et al. paper (read-only, gitignored) |

`results/` (gitignored) is where generations and downstream metrics land.

## Reproducing a run

```sh
# 1. select prompts (deterministic given seed=42)
python -m experiments.select_oasst2_prompts

# 2. generate for one model
python -m experiments.run_generation --model Qwen/Qwen2.5-7B-Instruct

# 3. or sweep over many
python -m experiments.sweep_models --models Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct
```

Requires vLLM, pandas, huggingface_hub, sentence-transformers, scikit-learn, matplotlib. (Dependency manifest TODO.)

## Running the sweep on a SLURM cluster

The submit scripts assume a multi-GPU SLURM cluster with typed GRES (e.g. `--gpus-per-node=A40:1`). Module names, account string, and storage paths in `experiments/sbatch_run.sh` reflect one specific cluster — edit them for yours.

One-time setup (login node):

```sh
# 1. clone
ssh <login-node>
git clone <this-repo>.git && cd multilingual-artificial-hivemind

# 2. scratch dirs on cluster-persistent storage
export SCRATCH_BASE=/path/to/project/scratch
mkdir -p "$SCRATCH_BASE"/{hf-cache,envs,results/raw} logs

# 3. venv with CUDA-built torch + vllm (adjust module names to your cluster)
module load Python/3.12 CUDA/13.0
python -m venv "$SCRATCH_BASE/envs/hivemind"
source "$SCRATCH_BASE/envs/hivemind/bin/activate"
pip install vllm pandas pyyaml huggingface_hub[hf_transfer]

# 4. HF auth -- needed for gated models (meta-llama, mistralai, google/gemma).
# Request access on each model's HF page first, then:
huggingface-cli login   # writes token to ~/.cache/huggingface/token
```

Cluster-specific bits to edit before submitting:
- `experiments/sbatch_run.sh`: `#SBATCH -A` account, `module load` lines, and the scratch path (currently `MIMER_BASE`).
- `experiments/models.yaml`: the `gpus` field — typed-GRES clusters use `A40:1` / `A100:2`; others may want a bare count (`1`, `2`) plus a partition selector.

To submit a sweep:

```sh
# Edit experiments/models.yaml -- uncomment each model you want to run.
# Each entry declares its own --gpus-per-node and walltime.
./experiments/submit_sweep.sh

squeue -u "$USER"                # job status
ls logs/sweep-*.out              # per-job stdout
ls "$SCRATCH_BASE/results/raw"   # generations as they finish
```

Add a new model by appending `{id, gpus, time}` to `models.yaml`. Sizing rule: total VRAM ≥ 1.5 × params × bytes_per_param (2 for fp16/bf16, 1 for FP8); stay within your cluster's single-node ceiling unless you wire up multi-node tensor parallelism.
