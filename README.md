# Multilingual Artificial Hivemind

Does the Artificial Hivemind effect (Jiang et al. 2025) generalize multilingually? See [`PLAN.md`](PLAN.md) for the full plan, hypotheses, and metric definitions.

## Layout

| Path | Purpose |
|---|---|
| `PLAN.md` | Research questions, hypotheses, dataset, models, protocol, metrics, analysis, limitations |
| `experiments/select_oasst2_prompts.py` | Build the locked OASST2 prompt set (seed=42, K=30/lang × 13 langs) → `experiments/oasst2_prompts.parquet` |
| `experiments/run_generation.py` | One model × full prompt set → `results/raw/{model}.parquet` + `.meta.json` (vLLM) |
| `experiments/sweep_models.py` | Sweep `run_generation` over a list of HF model ids with overlapping prefetch + per-model cache purge |
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
