# Expected Results

Output spec, paired with `EXPERIMENTS.md`. Numbers below are placeholders.

## File layout

```
results/
  raw/{run}_{model}_{lang}.parquet         # Layer 1, one row per sample
  metrics/per_cell.parquet                 # Layer 2, one row per (run × model × lang × prompt_id)
  metrics/bootstrap_dists.parquet          # bootstrap distributions for CI re-derivation
  tables/{name}.{csv,tex}
  figures/{name}.pdf
```

## Layer 1 — raw generations

| col | type | example |
|---|---|---|
| run | str | `E1` / `E2` / `E3` |
| model | str | `Qwen2.5-72B-Instruct-FP8` |
| dataset | str | `oasst2` / `aya_eval` / `stub` |
| language | str | `de` |
| prompt_id | str | `oasst2:002c4715-...` |
| prompt_text | str | `Dame los pasos...` |
| sample_idx | int | `0..49` |
| response_text | str | `Para ser un desarrollador...` |
| human_responses | list[str] | (OASST2 sibling responses; null elsewhere) |
| n_tokens | int | `512` |

## Layer 2 — per-cell metrics

One row per `(run × model × language × prompt_id)`. Columns:

**Anchor (Jiang-comparable)**
- `intra_sim_e5` — mean pairwise e5 cosine sim over N=50 LLM samples
- `human_sim_e5` — mean pairwise e5 cosine sim over M ≥3 human responses (OASST2 only)
- `gap_e5` = `intra_sim_e5 − human_sim_e5`
- `inter_sim_e5` — mean pairwise e5 sim across one sample per model

**Distributional rigor**
- `mmd_llm_human` — MMD² (RBF kernel) between LLM and human response embeddings
- `mmd_pvalue` — permutation-test p-value

**Mode structure** (HDBSCAN on combined LLM+human embeddings)
- `cluster_count_llm`, `cluster_count_human`
- `top_mode_mass_llm`, `top_mode_mass_human` — fraction of samples in dominant cluster
- `mode_js_div` — JS divergence between LLM and human cluster-proportion distributions

**Surface (embedder-free)**
- `distinct2_llm`, `distinct2_human`, `distinct2_gap`
- `selfbleu_llm`, `selfbleu_human`, `selfbleu_gap`

**Robustness subset** (2-3 langs only)
- `intra_sim_bge`, `human_sim_bge`, `gap_bge`

**Agreement**
- `metric_agreement_count` ∈ {0..4} — number of {e5, mmd, distinct2, selfbleu} agreeing with e5 on gap direction.

## Headline tables

### T1 — Multilingual Hivemind gap (E1, OASST2, primary model)

| Lang | Human sim | LLM sim | Gap | 95% CI | Agreement |
|---|---:|---:|---:|---|---:|
| en | 0.62 | 0.81 | 0.19 | [0.15, 0.23] | 4/4 |
| es | 0.58 | 0.78 | 0.20 | [0.16, 0.24] | 4/4 |
| zh | 0.54 | 0.72 | 0.18 | [0.13, 0.22] | 4/4 |
| ... | | | | | |

Headline result. Primary model = Qwen2.5-72B-FP8 (or aggregate across 6 main models).

### T2 — Cross-family gap (E1, OASST2)

| Model | en | es | ru | zh | de | fr | ca | pt-BR | it | ja | eu | pl | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-72B | 0.19 | 0.20 | ... | | | | | | | | | | 0.18 |
| Llama-70B | | | | | | | | | | | | | |
| Aya-32B | | | | | | | | | | | | | |
| Qwen-32B (pair) | | | | | | | | | | | | | |
| Gemma-27B | | | | | | | | | | | | | |
| Mistral-Small-3 | | | | | | | | | | | | | |

Shows the gap is not family-specific. Aya vs. Qwen-32B is the H4 row pair.

### T3 — Inter-model homogeneity (E1 + E2)

| Lang | OASST2 inter | OASST2 intra (mean over models) | Aya-eval inter |
|---|---:|---:|---:|
| en | 0.85 | 0.81 | 0.83 |
| es | 0.82 | 0.78 | 0.80 |
| ... | | | |

Tests H2: inter ≥ intra means models converge more on each other than on themselves.

### T4 — Bridge to preliminary work (E3, designed chat-stubs)

| Lang | Diversity | River | Weaver | Sculptor | Other |
|---|---:|---:|---:|---:|---:|
| en | 0.07 | 45 | 0 | 5 | 0 |
| de | 0.08 | 35 | 3 | 2 | 10 |
| es | 0.09 | 30 | 5 | 0 | 15 |
| ... | | | | | |

Mode counts out of N=50 per language for "metaphor about time" stub. Anchors to preliminary findings and to Jiang.

## Headline figures

### F1 — Qwen scale curve (E1, OASST2)

Line chart, x = model size {3B, 7B, 14B, 32B, 72B-FP8}, y = intra-model similarity. One line per language; dashed line per language for human similarity floor. H3 confirmed if monotone.

### F2 — Aya vs. Qwen-32B (E1, OASST2)

Bar chart, x = language, y = Hivemind gap, two bars per language. H4 confirmed if Aya bars consistently shorter.

## Appendix

- **A1 — Metric agreement.** T1 reproduced per metric in {mean-cosine, MMD, distinct-2, self-BLEU}; flags cells where metrics disagree on gap direction.
- **A2 — Mode structure.** Per (model, language): mean cluster count, top-mode mass, JS divergence to human cluster proportions. Backs the "what's actually collapsing" narrative.
- **A3 — BGE-M3 cross-encoder check.** T1 reproduced on 2-3 representative languages with BGE-M3 to rule out e5-specific cross-lingual artifacts.
- **A4 — Parallel-prompt cross-language (E2, Aya Eval Suite).** Inter-model similarity per language on identical translated prompts; controls for the OASST2 non-parallel-prompts caveat. Covers Hindi, Arabic, Telugu.
- **A5 — Per-prompt detail.** Distribution of per-prompt gaps within each language; bootstrap CI on cell means.
- **A6 — Open-ended classification.** Agreement between LLM-judge open-ended labels and a 100-prompt human-verified subset.
