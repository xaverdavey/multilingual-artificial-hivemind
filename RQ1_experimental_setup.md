# RQ1 — Experimental Setup

Implementation protocol for RQ1, with the methodology validated through pilot experiments on 2026-05-08 (`prefix_vs_chat.py`, `prefill_natural_prose.py`, `prefix_length_sweep.py` in `preliminary_diversity_investigation/`).

## Question

Does the Artificial Hivemind hold up multilingually? On matched prose-continuation tasks, do LLMs converge more than humans, and how does the gap depend on language and model scale?

## Method

For each language, sample K prefixes from a multilingual prose corpus along with the K original human continuations of those prefixes. For each prefix, generate N LLM continuations via raw `llm.generate(prefix)` — no chat template, no prefill. Pilot confirmed raw prose continuation works on instruct models for natural prose of any reasonable length; the chat-template scaffolding is only needed for designed quiz-style stubs (used separately for the bridge experiment to preliminary work / Jiang).

The same dataset yields two diversity measures:

- **Within-input:** pairwise diversity over the N LLM samples for the same prefix.
- **Across-input:** pairwise diversity over K samples (1 per prefix) across the K different prefixes. Compares directly against the K original human continuations — a clean LLM-vs-human gap with no multi-response baseline required.

## Data

- **Wikipedia** in 13 languages — primary corpus.
- **CC-News** in 5 representative languages — robustness corpus (different register).
- K = 200 prefixes per (language, corpus). Prefixes sampled uniformly from articles ≥ 1024 tokens.
- Prefix length: 256 tokens by default (well above the ~10-token off-task threshold observed in pilot).

## Hypotheses

- **H1 (Hivemind gap).** Across-input LLM diversity is lower than across-input human diversity in every tested language.
- **H2 (Scale dependence).** Across-input LLM diversity decreases monotonically with model size in every language.
- **H3 (Bridge replication).** Designed quiz-style stubs in chat-imperative form reproduce the preliminary work's river-mode dominance multilingually.

## Models and runs

| Run | Model(s) | Languages | Prefixes |
|---|---|---|---|
| **E1** (main sweep) | Qwen2.5-72B-Instruct-FP8 | all 13 | 200 Wikipedia |
| **E2** (scale curve) | Qwen2.5 {3B, 7B, 14B, 32B, 72B-FP8} | 5 representative | 200 Wikipedia |
| **E3** (corpus robustness) | Qwen2.5-72B-Instruct-FP8 | 5 representative | 200 CC-News |
| **E4** (Jiang bridge) | Qwen2.5-72B-Instruct-FP8 | all 13 | 5 designed stubs (chat-imperative) |

## Generation protocol

| Setting | Value |
|---|---|
| Method (E1–E3) | raw `llm.generate(prefix)`, no chat template |
| Method (E4) | `llm.chat([{role:user, content: stub}])`, replicating preliminary work |
| Samples per prefix (N) | 20 |
| Continuation length | 128 tokens |
| Decoding | T = 1.0, top_p = 0.95 |
| Inference | vLLM on Fenrir |

## Metrics

**Primary:** `1 − mean pairwise cosine` in `multilingual-e5-small`.

**Robustness:** BGE-M3 cosine, distinct-n (n=2,3), self-BLEU, LLM-judge similarity on 10% subsample.

**Demotion rule:** if the primary metric disagrees with ≥3 of the 4 robustness metrics on the gap direction, primary is demoted in favor of the robustness consensus.

## Analysis plan (pre-registered)

- **Main result:** per-language Hivemind gap (across-input human − LLM, E1) with 95% bootstrap CIs.
- **Scale curve:** across-input LLM diversity vs. model size (E2), one line per language.
- **Robustness:** corpus comparison (E1 vs. E3) on the 5 overlap languages.
- **Bridge:** mode-frequency table for E4 (river / weaver / sculptor / etc.) per language, alongside diversity numbers, to anchor results to preliminary work and Jiang.
- **Exploratory:** within-input diversity per language at the default prefix length.

## Falsifiers

- H1 fails (gap ≤ 0) in > 2 languages → Hivemind is not multilingually universal under this measurement.
- H2 non-monotonic in > 2 languages → refine theory.
- H3 river-mode share < 50% in > 4 languages on E4 → preliminary "river is universal" finding was English-leaning.
- Primary metric disagrees with ≥3 robustness metrics on the gap direction → metric-driven artifact; demote.

## Effort

| Phase | Time |
|---|---|
| Wikipedia + CC-News prefix extraction (13 langs) | 2 days |
| LLM generation (E1–E4 on Fenrir) | 3 days compute |
| Metric computation + analysis + figures | 2 days |
| Pre-registration document | 1 day |
| **Total** | **~8 working days** |

## Open decisions

1. **5 representative languages** for E2 / E3. Default: English, Spanish, Mandarin, Hindi, Arabic (covers Latin / Han / Devanagari / Arabic scripts and a range of resource levels).
2. **LLM-judge model.** Claude is methodologically cleanest given Qwen as study target.
