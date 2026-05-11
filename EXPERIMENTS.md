# RQ1 — Experimental Setup

Does the Artificial Hivemind (Jiang et al. 2025) generalize multilingually?

## Hypotheses

- **H1** Intra-model similarity > human similarity in every tested language.
- **H2** Inter-model similarity ≥ intra-model in every tested language.
- **H3** Intra-model similarity increases monotonically with Qwen scale in every language.
- **H4** Aya-Expanse-32B less repetitive than Qwen2.5-32B at matched scale (multilingual SFT effect).
- **H5** Designed chat-stubs reproduce preliminary river-mode dominance multilingually.

## Datasets

| Dataset | Role | Languages | Prompts/lang | Human responses |
|---|---|---|---|---|
| OASST2 | Headline gap | 13: en, es, ru, zh, de, fr, ca, pt-BR, it, ja, eu, pl, vi | 30 | ≥3 per prompt |
| Aya Evaluation Suite | Parallel-prompt cross-lang | 7: en, tr, ar, zh, pt, hi, te | 100 | none |
| Designed chat-stubs | Jiang + preliminary bridge | 13 | 5 | n/a |

OASST2 prompts filtered: `parent_id=null`, `deleted=False`, `review_result=True`, ≥3 sibling assistant responses, open-ended (LLM classifier vs. Jiang's 6-category taxonomy).

## Models

Run all in E1. API tier is *optional, only if budget + time allow.

| Family | Models |
|---|---|
| Qwen2.5 | 3B, 7B, 14B, 32B, 72B-Instruct-FP8 |
| Qwen3 | 8B, 32B (if multilingual-suitable) |
| Llama | Llama-3.2-3B, Llama-3.1-8B, Llama-3.1-70B-FP8 |
| Mistral | Mistral-7B-v0.3, Mistral-Nemo-12B, Mistral-Small-3-24B |
| Gemma-2 | 9B, 27B |
| Aya | Aya-Expanse-8B, Aya-Expanse-32B |
| Other | DeepSeek-R1-Distill-Qwen-32B, Phi-4-14B, OLMo-2-13B |
| API* | GPT-4o*, Claude-3.5-Sonnet*, Gemini-1.5-Pro* |

~22 open-weight + up to 3 API = 22-25 models.

## Generation protocol

| Setting | Value |
|---|---|
| Method | `llm.chat([{role:"user", content: prompt}])` |
| N samples per prompt | 50 |
| Max tokens | 2048 |
| Decoding | T=1.0, top_p=0.9 |
| Inference | vLLM on Berzelius/Alvis SLURM |

## Experiments

| Run | Models | Dataset | Cells | Generations |
|---|---|---|---|---|
| **E1** main | all (22–25) | OASST2, 30 prompts × 13 langs | 22×13 | ~430K |
| **E2** parallel cross-lang | 6 main | Aya Eval, 100 prompts × 7 langs | 6×7 | ~210K |
| **E3** Jiang/preliminary bridge | 6 main | 5 designed stubs × 13 langs | 6×13 | ~20K |

Total core: ~630K generations.

**6 main models** for E2/E3: Qwen2.5-72B-FP8, Llama-3.1-70B-FP8, Aya-Expanse-32B, Qwen2.5-32B (Aya pair), Gemma-2-27B, Mistral-Small-3.

## Optional extensions

- **E4 prose-continuation regime shift.** Qwen-72B + Aya-32B × 5 langs × 200 Wikipedia prefixes × 20 samples. Tests whether Hivemind generalizes beyond chat-prompt regime.
- **E5 Hindi/Arabic human collection.** 30 Aya prompts × 5 human responses × 2-3 langs via Prolific. Extends LLM-vs-human gap to non-Latin-script languages OASST2 misses.

## Metrics

Five metrics, each doing distinct work; no redundant duplicates.

| # | Metric | Role |
|---|---|---|
| 1 | Mean pairwise cosine on `multilingual-e5-small` | Anchor to Jiang |
| 2 | MMD (RBF kernel) on e5 embeddings | Proper two-sample test for distributional difference |
| 3 | Mode structure: HDBSCAN cluster count + top-mode mass + JS-div between LLM and human cluster proportions | Captures "river-mode dominance" mean-similarity misses |
| 4 | Distinct-2 + self-BLEU | Embedder-free surface robustness |
| 5 | BGE-M3 cosine on 2-3 lang subset | Cross-encoder check against e5 cross-lingual artifacts |

LLM-judge deliberately excluded (circular, expensive, low marginal info). BGE-M3 only run where it matters — a robustness subset, not the full grid.

Per (model, language, prompt):
- `intra_model_sim` — mean pairwise sim over N=50 LLM samples
- `human_sim` — mean pairwise sim over M ≥3 human OASST2 responses
- `hivemind_gap = intra_model_sim − human_sim`
- `mmd_llm_vs_human`, `top_mode_mass_llm`, `top_mode_mass_human`, `cluster_count_llm`, `cluster_count_human`, `mode_js_div`
- `distinct2_gap`, `selfbleu_gap`

Per (language, prompt):
- `inter_model_sim` — mean pairwise sim across one sample from each model

## Inference

- Per (model, language): bootstrap CI on each metric's mean over 30 prompts (10K resamples).
- Cross-language claim ("gap holds in every tested language"): per-language test + BH-FDR at q=0.05 across 13 languages.
- Population-level claim ("language-aggregated gap"): hierarchical bootstrap (resample languages, then prompts within language).

## Falsifiers

- H1 fails in >2 langs → not multilingually universal.
- H2 fails in >2 langs → models more individuated than Jiang found.
- H3 non-monotonic in >2 langs → refine theory.
- H4 reversed → multilingual SFT doesn't help.
- H5 river-mode share <50% in >4 langs → preliminary finding was English-leaning.
- MMD finds no distributional difference where mean-cosine does → mean-cosine result is metric-artifact.
- Mean-cosine, MMD, distinct-2, self-BLEU disagree on gap direction → demote.

## Limitations

- OASST2 missing Hindi, Arabic, Swahili, Korean — partially covered by E2 (no human baseline) or E5.
- OASST2 prompts non-parallel across languages → cross-language *gap* claims are clean; raw-diversity comparisons need care.
- ≥3 human responses per prompt is small; aggregate over 30 prompts with bootstrap CIs.

## Effort

| Phase | Days |
|---|---|
| Data prep + open-ended classification | 1–2 |
| Generation on cluster (E1–E3) | 2–3 wall-clock |
| Metrics + analysis + figures | 2–3 |
| Pre-registration | 1 |
| **Total** | **~8–10 working days** |
