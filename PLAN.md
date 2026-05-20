# Multilingual Hivemind — Plan

Does the Artificial Hivemind (Jiang et al. 2025) generalize multilingually? Sample N=50 completions from ~22 chat models on a shared OASST2 prompt set across 13 languages; compare intra-model and inter-model diversity to the human-response baseline.

## Hypotheses

| | Hypothesis | Falsifier |
|---|---|---|
| **H1** | Intra-model sim > human sim in every lang | fails in >2 langs |
| **H2** | Inter-model sim ≥ intra-model sim in every lang | fails in >2 langs |
| **H3** | Per (family, lang) Spearman ρ(model size, intra-sim) > 0 across each family's scale ladder | ρ ≤ 0 in >2 (family × lang) cells |

Cross-language and cross-family claims use BH-FDR at q=0.05 (across 13 langs for H1/H2; across the family × lang grid for H3). Only families with ≥3 sizes contribute to H3 cleanly; ladders with 2 sizes report direction but not Spearman.

## Dataset

OASST2 root prompts, filtered to `~deleted & review_result & ≥3 sibling assistant responses & open-ended` (LLM classifier vs. Jiang's 6-category taxonomy), then K=30 per language (seed=42) across 13 langs: en, es, ru, zh, de, fr, ca, pt-BR, it, ja, eu, pl, vi. Human baseline = the sibling responses. Selection: `experiments/select_oasst2_prompts.py`.

## Models (18)

| Family | Sizes | Provider |
|---|---|---|
| Anthropic Claude | Haiku · Sonnet · Opus | Anthropic API |
| DeepSeek-R1-Distill-Qwen | 1.5B · 7B · 14B | Hugging Face / vLLM |
| Mistral | 3B · 8B · 14B | Hugging Face / vLLM |
| OLMo 2 | 7B · 13B · 32B | Hugging Face / vLLM |
| OpenAI GPT-5 | nano · mini · (full) | OpenAI API |
| Qwen2.5 | 3B · 14B · 32B | Hugging Face / vLLM |

## Protocol

T=1.0, top_p=0.9, max_tokens=2048, N=50, seed=42 — matches Jiang. Backend auto-detected per model: vLLM for HuggingFace models; OpenAI API (n=50 per call); Anthropic API (50 concurrent calls per prompt). Generation via `experiments/run_generation.py`; HF model sweep via `experiments/sweep_models.py` (overlapping prefetch + per-model cache purge). 18 × 13 × 30 × 50 ≈ 351K generations.

## Metrics

| # | Metric | Role |
|---|---|---|
| 1 | Mean pairwise cosine on `text-embedding-3-small` (OpenAI) | Primary similarity metric |
| 2 | MMD² (RBF) + permutation p-value | Two-sample distributional test |
| 3 | HDBSCAN: cluster count, top-mode mass, JS-div(LLM, human) | Mode collapse mean-sim misses |
| 4 | Distinct-2 + self-BLEU | Embedder-free surface robustness |
| 5 | BGE-M3 cosine, 2-3 lang subset | Cross-encoder check vs. embedding model artifacts |

All gap metrics signed so positive = LLM less diverse than human. Demote any (model × lang × prompt) cell where <3/4 of {text-embedding-3-small, MMD-direction, distinct-2, self-BLEU} agree on positive gap.

Alternatives if reviewers prefer something less coarse than mean cosine: Vendi score (entropy of the similarity-matrix eigenvalues — same intent, sees full spectrum) or `log det` of the kernel matrix (DPP-style). On unit-norm embeddings, mean pairwise cosine and total embedding variance are monotone-equivalent (`intra_sim ≡ 1 − tr(Σ) + 1/N`), so a separate "variance" metric would be a rename.

`intra_sim` = `(2/(N(N−1))) Σ_{i<j} ⟨eᵢ, eⱼ⟩` over the N=50 L2-normalized text-embedding-3-small embeddings of the LLM samples; equivalently `(N·‖c‖² − 1)/(N−1)` where `c` is the centroid, monotone in `1 − total embedding variance`. `human_sim` is the same over the ≥3 human responses. `inter_sim` per (lang, prompt): for each `sample_idx ∈ 0..49` take one sample per model at that idx, compute pairwise mean across the M models; report the mean over the 50 slices.

## Analysis

Per (model, lang): bootstrap 95% CI over 30 prompts (10K resamples). Population-level gap: hierarchical bootstrap (langs, then prompts). H3: per (family, lang) Spearman ρ on intra-sim vs. parameter count; BH-FDR across the family × lang grid.

## Outputs

```
results/raw/{model}.parquet         # row per (prompt × sample) + sidecar .meta.json
results/metrics/per_cell.parquet    # row per (model × lang × prompt)
results/{tables,figures}/...
```

## Limitations

- OASST2 misses Hindi/Arabic/Swahili/Korean; lang set is European + East-Asian-leaning.
- Prompts non-parallel across langs — cross-lang *gap* claims are clean; raw-diversity across langs is not.
- M ≥ 3 human responses per prompt is small; bootstrap over 30 prompts handles it.
- 22-model set is opportunistic, not stratified.
