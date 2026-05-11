# Multilingual Hivemind — Research Plan

Builds on `preliminary_diversity_investigation/FINDINGS.md` (2026-05-06) and pilot experiments validating the methodology (2026-05-08).

## Question

Does the Artificial Hivemind (Jiang et al. 2025) generalize multilingually? Specifically: do the two phenomena Jiang identifies — intra-model repetition and inter-model homogeneity — hold across typologically diverse languages, and how does the gap to the human-response distribution depend on language, model scale, and post-training data composition?

## Sub-claims

1. **Generalization.** Hivemind appears in every tested language, not just English.
2. **Scale.** Intra-model repetition strengthens with model size in every language.
3. **Multilingual SFT.** Models post-trained on natively-multilingual data (Aya) are less Hivemind-prone than English-translated-SFT counterparts (Qwen-32B) at matched scale.
4. **Bridge.** The preliminary river-mode dominance reproduces under controlled multilingual measurement.

Concrete falsifiers, hypotheses, and run matrix in `EXPERIMENTS.md`. Output specification in `RESULTS.md`.

## Contribution

- First multilingual measurement of the Hivemind effect with proper human baseline (OASST2 multi-human responses), at ~25 models × 12 languages.
- Test whether multilingual SFT mitigates the effect (Aya vs. Qwen at matched scale).
- Bridge between Jiang's chat-stub regime and our preliminary prose-completion findings.

## Future work (out of scope for this paper)

- **Mechanism.** Is Hivemind primarily pretraining-driven or post-training-driven? Base vs. instruct comparison on matched prefixes. Requires base-model access and careful prompting controls.
- **Intervention ceiling.** Can prompt-level or decoding-level interventions reach mode structure? Mode-ban depth, prompt-steering decomposition, decoding sweeps.
- **Cultural-native modes.** Beyond measuring overlap, identify culturally-grounded mode candidates (Arabic time-as-sword, etc.) and test whether models retrieve them.

## Methodological decisions locked in

1. Diversity metric: `1 − mean pairwise cosine` on `multilingual-e5-small` (primary); BGE-M3, distinct-{2,3}, self-BLEU, LLM-judge as robustness; demotion rule if ≥3/4 disagree.
2. Generation: `llm.chat`, T=1.0, top_p=0.9, N=50 samples per prompt — matches Jiang.
3. Primary data: OASST2 root prompts with ≥3 sibling assistant responses, open-ended filtered.
4. Open-ended classifier: LLM judge vs. Jiang's 6-category taxonomy.
5. Pre-registration before E1 generation begins.

## Risks

- **OASST2 language coverage skews European/East-Asian.** Hindi, Arabic, Swahili, Korean absent from the primary measurement; partially mitigated by Aya Evaluation Suite auxiliary (no human baseline) and optional fresh collection.
- **Cross-language comparison of absolute diversity is muddied** by non-parallel prompt sets in OASST2; gap measurements are clean, raw-diversity comparisons need care.
- **Multi-human response count per prompt is small (≥3, rarely >5).** Aggregate over 30 prompts per language with bootstrap CIs.

## Title

Deferred until experiments land. Working placeholder: *The Multilingual Hivemind*.
