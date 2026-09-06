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
| `experiments/run_embeddings.py` | Embed generations with any OpenAI or sentence-transformers model → `results/embeddings/` |
| `experiments/run_metrics.py` | Intra/inter similarity, F-tests, plots for one embedding model |
| `experiments/run_lexical.py` | Embedder-free diversity: mean pairwise character n-gram Jaccard (`--truncate` for the length control) |
| `experiments/run_robustness.py` | Compare the homogenization gap across every metric on disk → `results/robustness/` |
| `experiments/run_fertility.py` | Tokenization fertility per (tokenizer, language) on FLORES-200 parallel text |
| `experiments/run_linguistic.py` | Regress the gap on fertility / competence / resource / morphology (crossed random effects) |
| `experiments/plot_respondent_matrix.py` | Respondent-respondent similarity matrices, drawn at final print size |
| `experiments/plot_linguistic.py` | Two-panel figure for the linguistic-property analysis |
| `experiments/make_paper_tables.py` | Emit the appendix LaTeX tables from the analysis outputs |
| `analysis.ipynb` | Scratch notebook (PCA visualizations of generations; not load-bearing) |
| `artificial-hivemind/` | Cloned reference repo for the original Jiang et al. paper (read-only, gitignored) |

`results/` (gitignored) is where generations and downstream metrics land.

## Pipeline

The core is still three stages. Each embedding-model ablation re-runs stages 2–3
with a different `--embed-model`/`--out-dir`; `run_lexical` is the embedder-free
stand-in for both (it emits `run_metrics`' output schema on purpose). Everything
else is an analysis layer that only reads stage-3 outputs from disk.

```
1 generate     select_oasst2_prompts → sweep_models            → results/raw/
2 embed        run_embeddings   (once per embedding model)     → results/embeddings*/
3 metrics      run_metrics      (once per embedding model)     → results/metrics*/
               run_lexical      (no embedder; stages 2+3)      → results/lexical/
  predictors   run_fluency (GlotLID) · run_fertility (FLORES)  → results/fluency|linguistic/
4 analysis     run_robustness   (every metric found on disk)   → results/robustness/
               run_linguistic   (gap ~ predictors, crossed RE) → results/linguistic*/
5 paper        plot_respondent_matrix · plot_linguistic · make_paper_tables
```

Stages 4–5 are deterministic given the parquets and regenerate the paper's
figures and tables byte-for-byte.

## Reproducing a run

```sh
# 1. select prompts (deterministic given seed=42)
python -m experiments.select_oasst2_prompts

# 2. generate for one model
python -m experiments.run_generation --model Qwen/Qwen2.5-7B-Instruct

# 3. or sweep over many
python -m experiments.sweep_models --models Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct
```

Dependencies pinned in [`requirements.txt`](requirements.txt): `pip install -r requirements.txt`.

## Running the sweep on a SLURM cluster

The submit scripts assume a multi-GPU SLURM cluster with typed GRES (e.g. `--gpus-per-node=A40:1`). Module names, account string, and storage paths in `experiments/sbatch_run.sh` reflect one specific cluster — edit them for yours.

One-time setup (login node):

```sh
# 1. get the repo onto the cluster
ssh <login-node>
git clone <this-repo>.git && cd multilingual-artificial-hivemind
# If the repo is private and the cluster has no GitHub SSH key, rsync from your laptop instead:
#   rsync -av --exclude={'.git/objects','__pycache__','results/','logs/','.DS_Store'} \
#       /path/to/repo/ <user>@<cluster>:/path/to/repo/

# 2. scratch dirs on cluster-persistent storage
export SCRATCH_BASE=/path/to/project/scratch
mkdir -p "$SCRATCH_BASE"/{hf-cache,envs,results/raw} logs

# 3. venv with CUDA-built torch + vllm (adjust module names to your cluster)
module load Python/3.12 CUDA/13.0
python -m venv "$SCRATCH_BASE/envs/hivemind"
source "$SCRATCH_BASE/envs/hivemind/bin/activate"
pip install -r requirements.txt

# 4. HF auth -- needed for gated models (meta-llama, mistralai, google/gemma).
# Request access on each model's HF page first, then:
huggingface-cli login   # writes token to ~/.cache/huggingface/token

# 5. build the prompt parquet (deterministic, seed=42; gitignored so not in the repo)
python -m experiments.select_oasst2_prompts
```

Cluster-specific bits to edit before submitting:
- `experiments/sbatch_run.sh`: `#SBATCH -A` account, `module load` lines, and `SCRATCH_BASE`.
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
