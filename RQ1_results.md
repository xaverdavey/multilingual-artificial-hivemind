# RQ1 — Expected Results, Tables, and Figures

Output specification for RQ1, paired with `RQ1_experimental_setup.md`. Defines the data schema we save, the headline tables for the paper, and the appendix tables. All numbers shown below are **illustrative placeholders**.

## Output file layout

```
results/
  raw/
    {model_id}_{run}.parquet              # Layer 1, one row per sample
  metrics/
    per_cell.parquet                      # Layer 2, one row per (run × model × corpus × language × prefix_length)
    bootstrap_dists.parquet               # bootstrap distributions for CI re-derivation
  tables/
    table1_main_gap.{csv,tex}
    table2_corpus_robustness.{csv,tex}
    table3_bridge_modes.{csv,tex}
    table4_cross_family.{csv,tex}
    appendix_metric_robustness.{csv,tex}
  figures/
    fig1_scale_curve.pdf
    fig2_prefix_length_curve.pdf          # only if H2 sweep is run
    fig3_aya_vs_qwen.pdf
```

## Layer 1 — raw generations (one row per sample)

| col | type | example |
|---|---|---|
| run | str | `E1` |
| model | str | `Qwen2.5-72B-Instruct-FP8` |
| corpus | str | `wiki` / `ccnews` / `stub` |
| language | str | `de` |
| prefix_length | int | `256` |
| prefix_id | int | `0..199` |
| sample_idx | int | `0..19` |
| prefix_text | str | `Mount Vesuvius is a stratovolcano…` |
| continuation_text | str | `the catastrophic destruction of Pompeii…` |
| human_continuation | str | (real continuation; null for E4 stubs) |
| n_tokens_continuation | int | `127` |

## Layer 2 — per-cell aggregated metrics

One row per `(run × model × corpus × language × prefix_length)`. Columns:

- For each metric in {e5, bge_m3, distinct2, distinct3, self_bleu, llmjudge}:
  - `within_input_div_<metric>` — mean over K prefixes of pairwise diversity over N=20 samples.
  - `acrossinput_div_llm_<metric>` — bootstrap mean of pairwise diversity over K-sized populations (1 sample per prefix, 100 reps).
  - `acrossinput_div_llm_<metric>_ci_lo`, `_ci_hi` — bootstrap 95% CIs.
  - `acrossinput_div_human_<metric>` — pairwise diversity over the K human continuations.
  - `hivemind_gap_<metric>` — `acrossinput_div_human_<metric> − acrossinput_div_llm_<metric>`.
  - `hivemind_gap_<metric>_ci_lo`, `_ci_hi`.
- `metric_agreement_count` — int in {0,1,2,3,4} — number of robustness metrics that agree with e5 on gap sign.

## Headline tables for the paper

### Table 1 — Multilingual Hivemind gap (E1, Qwen2.5-72B-Instruct-FP8, Wikipedia, prefix_length=256)

| Language | Across-input human | Across-input LLM | Gap (h − l) | 95% CI | Agreement |
|---|---:|---:|---:|---|---:|
| English | 0.52 | 0.34 | 0.18 | [0.15, 0.21] | 4/4 |
| German  | 0.48 | 0.28 | 0.20 | [0.17, 0.23] | 4/4 |
| Spanish | 0.50 | 0.31 | 0.19 | [0.16, 0.22] | 4/4 |
| Mandarin | 0.41 | 0.27 | 0.14 | [0.10, 0.17] | 3/4 |
| Hindi | … | … | … | … | … |
| ⋮ |

The load-bearing main result. One row per language.

### Table 2 — Corpus robustness (E1 vs. E3, 5 representative languages)

| Language | Wikipedia gap | CC-News gap | Δ |
|---|---:|---:|---:|
| English | 0.18 | 0.21 | +0.03 |
| Spanish | 0.19 | 0.18 | −0.01 |
| Mandarin | 0.14 | 0.16 | +0.02 |
| Hindi | … | … | … |
| Arabic | … | … | … |

Tests whether the gap holds across registers. Small ∆ ⇒ corpus-robust.

### Table 3 — Bridge to preliminary work (E4, designed stubs in chat-imperative form)

| Language | Diversity | River | Weaver | Sculptor | Other |
|---|---:|---:|---:|---:|---:|
| English | 0.07 | 18 | 0 | 2 | 0 |
| German | 0.08 | 14 | 1 | 1 | 4 |
| Spanish | 0.09 | 12 | 2 | 0 | 6 |
| ⋮ |

Mode counts out of N=20 per language for the "metaphor about time" stub. Anchors our prose-continuation results to the preliminary river-mode finding and to Jiang's framing.

### Table 4 — Cross-family Hivemind gap (E1, all model families, Wikipedia, prefix_length=256)

| Model | EN | ES | ZH | HI | AR | … | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-72B-Instruct-FP8 | 0.18 | 0.19 | 0.14 | … | … | | 0.17 |
| Llama-3.1-70B-Instruct-FP8 | 0.16 | 0.17 | 0.12 | … | … | | 0.15 |
| Aya-Expanse-32B | 0.12 | 0.14 | 0.13 | … | … | | 0.13 |
| Qwen2.5-32B-Instruct (Aya-paired) | 0.17 | 0.18 | 0.13 | … | … | | 0.16 |
| Gemma-2-27B-it | … | … | … | … | … | | … |

Demonstrates the gap is not Qwen-specific. The Aya vs. Qwen-32B paired row is the single-line summary of the multilingually-trained vs. English-translated SFT comparison.

## Headline figures

### Figure 1 — Scale curve within Qwen family (E2, Wikipedia, prefix_length=256)

Line chart, x = model size on log axis {3B, 7B, 14B, 32B, 72B-FP8}, y = across-input LLM diversity. One line per language (5 representative). Optional dashed horizontal line per language for human across-input diversity.

If diversity decreases monotonically and converges toward an English-like floor, H2 confirmed.

### Figure 2 — Prefix-length curve (Qwen family, if prefix-length sweep is run)

Line chart, x = prefix length {32, 128, 512, 2048}, y = within-input diversity. One line per (model size, language) — or aggregated to one line per model if too cluttered. Tests whether Hivemind is context-poverty driven.

### Figure 3 — Aya vs. Qwen-32B at matched scale, per language

Bar chart, x = language, y = Hivemind gap, two bars per language (Aya-Expanse-32B vs. Qwen2.5-32B-Instruct). Tests whether multilingually-trained post-training reduces the gap.

## Appendix tables

### A1 — Metric robustness for Table 1 gaps

One column per metric in {e5, bge_m3, distinct2, distinct3, self_bleu, llmjudge}. Same row structure as Table 1; values are gaps. Demonstrates the demotion rule was not triggered (or shows where it was).

### A2 — Within-input diversity per language (E1, default prefix length)

One row per language, columns for each metric. Exploratory; informs the relationship between within-input and across-input Hivemind.

### A3 — Per-prefix-length within-input diversity (Qwen family only, if prefix-length sweep is run)

One row per (model, language, prefix_length). Backs Figure 2.
