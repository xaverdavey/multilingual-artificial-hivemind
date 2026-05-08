# RQ1 — Experimental Setup (draft)

Builds on `RESEARCH_PLAN.md`. Implementation-level protocol for RQ1, to be locked in (pre-registered) before the full sweep.

## Question

> Does the Artificial Hivemind hold up multilingually under prose-continuation conditions, and how does LLM output diversity depend on context length and model scale?

Posture-neutral: outcomes from "Hivemind universal and severe" to "Hivemind is largely a context-poverty artifact" are all publishable.

## Hypotheses

- **H1 (Hivemind gap).** Across-input diversity of LLM continuations is lower than across-input diversity of human continuations on matched prefixes, in every tested language.
- **H2 (Context dependence).** Within-input diversity rises monotonically with prefix length.
- **H3 (Scale dependence).** Across-input LLM diversity decreases monotonically with model size in every language.

## Method

Prose continuation. For each language, sample K prefixes from a multilingual prose corpus and the original K human continuations. For each prefix, generate N LLM continuations. This single dataset supports two diversity measures: *within-input* (variability across N LLM samples for the same prefix) and *across-input* (variability across one sample per prefix on K different prefixes). The latter compares directly against the K human continuations — a clean human-vs-LLM gap, no multi-response baseline required.

## Data

| Source | Role | Languages |
|---|---|---|
| Wikipedia | Primary corpus | all 13 |
| CC-News | Robustness corpus (different register) | all 13 |

K = 200 prefixes per (language, corpus). Prefixes sampled uniformly from articles ≥ 1024 tokens.

## Generation protocol

| Setting | Value |
|---|---|
| Prefix lengths | {32, 128, 512, 2048} tokens |
| Continuation length | 128 tokens (or natural stop) |
| N (LLM samples per prefix) | 20 |
| K (prefixes per language) | 200 |
| Decoding | T = 1.0, top_p = 1.0 |
| Inference | vLLM on Fenrir |

## Models

| Run | Model(s) | Languages | Prefix lengths |
|---|---|---|---|
| **E0** — pilot | Qwen2.5-72B-Instruct-FP8 | English + 1 other | all 4 |
| **E1** — full sweep | Qwen2.5-72B-Instruct-FP8 | all 13 | all 4 |
| **E2** — scale curve | Qwen2.5 {3B, 7B, 14B, 32B, 72B-FP8} | 5 representative | 512 only |

E0 gates E1 (don't burn compute on the full sweep until the metric triangulates).

## Metrics

**Primary:** `1 − mean pairwise cosine` in `multilingual-e5-small`.

**Robustness:** BGE-M3 cosine, distinct-n (n=2,3), self-BLEU, LLM-judge similarity on 10% subsample.

Per cell:
- **Within-input diversity** — mean pairwise diversity over N LLM completions, averaged across K prefixes.
- **Across-input diversity (LLM)** — bootstrap 1 of N samples per prefix → K-sized population → pairwise diversity; 100 bootstraps.
- **Across-input diversity (human)** — K human suffixes, pairwise diversity.
- **Hivemind gap** — `across-input human − across-input LLM`.

**Demotion rule.** If the primary metric disagrees with ≥3 of the 4 robustness metrics on the gap direction, the primary result is demoted in favor of the robustness consensus.

## Analysis (pre-registered)

- Within-input diversity vs. prefix length (E1) — one curve per language.
- Across-input Hivemind gap per language (E1, prefix length 512) — bar chart with 95% bootstrap CIs.
- Scale curve (E2) — across-input LLM diversity vs. model size, one line per language.

## Falsifiers

- Hivemind gap ≤ 0 in > 2 languages → H1 fails; reframe.
- Within-input diversity flat or non-monotonic in prefix length across > 2 languages → H2 fails; context-poverty story doesn't hold.
- Across-input LLM diversity non-monotonic in scale across > 2 languages → H3 fails.
- Primary metric disagrees with ≥3 robustness metrics on the gap direction → metric-driven artifact; demote.

## Effort

| Phase | Time |
|---|---|
| Corpus prep (Wikipedia + CC-News, 13 langs, prefix sampling) | 2 days |
| LLM generation (E0, E1, E2 on Fenrir) | 3–4 days compute |
| Metric computation + analysis + figures | 2–3 days |
| Pre-registration | 1 day |
| **Total** | **~8–10 working days** |

## Open decisions

1. Prefix-length set {32, 128, 512, 2048} — tune after E0 if endpoints are uninformative.
2. CC-News as parallel robustness corpus from the start, or only on demand if Wikipedia results are equivocal?
3. LLM-judge model: Claude vs. GPT-4. Claude is methodologically cleanest given Qwen as study target.
