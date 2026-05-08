# Multilingual Hivemind — Research Plan (draft)

Builds on `preliminary_diversity_investigation/FINDINGS.md` (2026-05-06). This document is the formal research plan: working thesis, research questions with motivation, hypotheses, experiments, falsifiers, and a sequencing proposal. It is intended to be argued with and revised before any further experiments are run.

## Working thesis

**Hivemind in instruction-tuned LLMs is structural and largely pretraining-driven — not primarily an RLHF artifact — and manifests cross-lingually as English conceptual modes imposed on other languages.**

The multilingual sweep is our *investigative lens*: it lets us locate Hivemind's source in a way within-English studies cannot, by exploiting cross-language mode overlap as a signature of English-imposition, and by contrasting models with different post-training data composition (English-translated synthetic SFT vs. natively multilingual). If the thesis holds, the contribution is twofold:

1. **Mechanism relocation.** We move the proximate cause of Hivemind from RLHF (Jiang et al. 2025) to pretraining + the structure of (translated) instruction-tuning data, with implications for what interventions can ever work.
2. **Cross-cultural cost.** We document multilingual instruction-tuning as a vector for conceptual flattening across cultures — a sociotechnical concern that is invisible to standard multilingual evaluation (which tracks fluency and accuracy, not conceptual diversity).

## Research questions

Each RQ states *why it matters* before *what we'll measure*. The four are designed to compose into a single argument: RQ1 establishes the phenomenon, RQ2 characterizes its cross-cultural shape, RQ3 locates its origin, RQ4 bounds the intervention surface.

---

### RQ1 — Generalization. Does Hivemind generalize across languages and intensify with scale?

**Why it matters.** Jiang et al. document Hivemind in English. If it is English-specific, it is a narrower phenomenon — plausibly an artifact of English-dominant alignment data — and most of the worry is overblown. If it generalizes, it is a property of large-scale instruction-tuned LMs as such, with consequences in every language these models are used in. Confirming the scale dependence cross-lingually rules out "scale will fix it" as a hope. RQ1 is necessary scaffolding for RQ2–4: without a confirmed cross-language phenomenon on more than one prompt, the rest of the project rests on a single data point.

**Hypothesis.** Hivemind appears in all 13 languages tested. Diversity decreases monotonically with model size in every language. The asymptote of larger scale is *English's already-saturated diversity floor* — i.e., scale makes non-English languages behave like English.

**Experiments.**
- **E1 — Prompt-bank multilingual sweep.** 15 prompts (Jiang et al. categories: creative generation, opinion, slogan, brainstorm, descriptive, joke, moral dilemma) × 13 languages × Qwen2.5-72B-Instruct-FP8 × n=20. Replaces the preliminary single-prompt result. Reports mean diversity per (prompt, language) and language-aggregate.
- **E2 — Scale curve.** Same prompt bank × {Qwen2.5-3B, 7B, 14B, 32B, 72B-FP8} × English + 4 representative languages (Spanish, Mandarin, Hindi, Arabic). Smaller language set because this is the most expensive sweep.

**What would falsify it.** Diversity is non-monotonic in scale, or one or more languages show no Hivemind effect. Would force a more careful theory of when Hivemind kicks in (resource level? script? typological distance?).

---

### RQ2 — Origin shape. Are non-English mode structures culturally native, or imposed from English?

**Why it matters.** This is the project's most consequential question for the broader community. If the dominant conceptual modes in non-English languages are English modes in local clothing, multilingual instruction-tuning is silently flattening cross-cultural conceptual variation — a harm that is invisible to standard multilingual benchmarks (which test fluency and task accuracy, not what the model *says*). It also implicates training-data sourcing: translated synthetic SFT data is the obvious mechanism. Crucially, this is a question that *can only be asked multilingually*: cross-language mode overlap is a signature of English-imposition that doesn't exist in monolingual studies. Preliminary work already shows the Arabic outlier (a culturally-native mode) and the river/weaver/sculptor universals — RQ2 turns those observations into a measurement.

**Hypothesis.** Across languages, dominant modes overlap heavily with the English dominant mode (river → weaver → sculptor hierarchy). A model post-trained on natively-multilingual data (Aya-Expanse) exhibits lower English-mode-overlap and higher within-language mode entropy than a comparably-sized model post-trained on English-translated synthetic data (Qwen2.5).

**Experiments.**
- **E3 — Cross-lingual mode annotation.** For E1's outputs, label the dominant conceptual mode of each generation (LLM-judge with held-out human-verified subset for calibration). Compute *mode-overlap-with-English* and *per-language mode entropy*. This is the paper's main figure.
- **E4 — Aya vs. Qwen.** Aya-Expanse-32B vs. Qwen2.5-32B (size-matched) on the prompt bank, English + 4 languages. Predicts Aya has lower English-mode-overlap if the imposition mechanism is translated synthetic SFT.
- **E5 — Culturally-grounded prompts.** Add 5 prompts with documented culturally-native modes in different languages (e.g. Arabic time-as-sword, Japanese mono-no-aware imagery, Mandarin chengyu-structured idiom). Test whether models retrieve these or default to English-translated equivalents.

**What would falsify it.** Mode overlap with English is no higher than between any two non-English languages (suggests convergence on universal human modes, not English-specifically). Or: Aya shows the same English-mode-overlap as Qwen (suggests imposition mechanism is upstream of post-training).

---

### RQ3 — Mechanism. Does Hivemind originate in pretraining or in post-training?

**Why it matters.** This is the headline experiment and the most likely source of an actionable claim. Jiang et al. attribute Hivemind to RLHF. If they are right, the intervention surface is the post-training pipeline (preference-data diversity, DPO objectives, KL-regularization choices). If we are right that the homogenization is already in the pretraining distribution, then post-training fixes are fundamentally limited — you cannot recover diversity via RLHF that wasn't in the base distribution to begin with. This also explains preliminary finding #5 (corpus-invariant optimal μ; decoding cannot close the gap): if Hivemind lives in pretraining, decoding-layer fixes were never going to work.

**Hypothesis.** Base models already exhibit substantial Hivemind on matched-prefix completion tasks — less than their instruct-tuned counterparts, but far less of a gap than an RLHF-mechanism story would predict. The *modes* base models converge to substantially overlap with the modes their instruct-tuned counterparts converge to. RLHF amplifies and crystallizes Hivemind; it does not create it.

**Experiments.**
- **E6 — Same-family base vs. instruct.** Qwen2.5-72B base vs. Qwen2.5-72B-Instruct. Few-shot completion-style prompting on both (3 in-context exemplars + open prefix; identical prefix for both models so the comparison is apples-to-apples). English + 4 representative languages. n=20 samples. Report diversity gap and base–instruct mode overlap on matched prefixes.
- **E7 — Cross-family robustness.** Same protocol on OLMo-2 (32B base/instruct, fully open) and Llama-3.1-70B base/Instruct. Robustness against Qwen-specific artifacts.
- **E8 — Few-shot anchoring control.** Vary in-context exemplars (uniform mode / diverse modes / adversarial / paraphrased) to characterize how much base-model diversity depends on prompting choices. Without this control, any base-vs-instruct gap could be a prompting artifact.

**What would falsify it.** Base models show diversity an order of magnitude higher than instruct, with little mode overlap on matched prefixes. Would confirm Jiang et al.'s RLHF-mechanism story rather than relocate it. (Still publishable — extends Jiang multilingually — but a much weaker novelty story.)

---

### RQ4 — Intervention ceiling. Can surface-level (prompt or decoding) interventions reach mode structure?

**Why it matters.** The set of interventions a downstream user can apply without retraining is small: prompting, decoding parameters, sampling tricks. If none of these reach the mode level, the practical conclusion is that diversity-recovery requires training-time intervention — which connects directly back to RQ3. If *some* surface interventions do reach modes, those are immediately deployable and important. Either answer is useful; the value depends on rigor. Preliminary findings #4 and #5 already gesture at "no", but only on one prompt and without isolating mode-shift from surface-shift.

**Hypothesis.** Prompt-based novelty steering shifts surface lexical diversity but not mode distribution. Decoding tweaks (temperature, top-p, min-p, T-distribution shape) likewise affect within-mode spread but not mode probability. Sequential mode-banning walks down a finite mode stack — at least three layers deep — without unlocking real spread, and the marginal gain from each ban is small relative to the gap to a human-prose baseline.

**Experiments.**
- **E9 — Mode-ban depth probing.** For ~5 prompts with cleanly-identified modes from E3: progressive bans (mode-1, mode-1+2, mode-1+2+3, mode-1+...+4). Measure diversity gain per ban and whether new modes emerge or distribution flattens. Repeat across English + 3 languages.
- **E10 — Prompt-steering decomposition.** Compare {baseline, "be unique", "avoid clichés", "be culturally specific", "give an unusual answer"}. Annotate each generation for mode-change vs. surface-change to separate lexical from mode-level effects.
- **E11 — Decoding ceiling.** Fixed-T sweep, top-p sweep, min-p, T-distribution (re-run from preliminary work for new prompts), plus an oracle ablation: best-of-N at high T with diversity-aware reranking. Establishes the upper bound decoding alone can reach.

**What would falsify it.** A specific surface intervention shifts mode distribution as much as a training-time change would (reframes the project: diversity is recoverable post-hoc). Particularly interesting failure mode: cultural-specificity prompts work *more* in less English-dominated languages — would suggest the imposition is fragile in those settings.

## Sequencing and scope

The full plan is more than a single course project can fit. Recommend tiered execution:

**Core (must do — supports each RQ at minimum):**
- E1 — prompt-bank multilingual sweep on Qwen2.5-72B
- E3 — mode annotation, English-overlap measurement
- E6 + E8 — Qwen base vs. instruct, with anchoring control
- E9 — mode-ban depth probing on prompt bank

**High-value extensions (do if time permits):**
- E4 — Aya vs. Qwen (single biggest payoff for RQ2)
- E2 — full scale curve
- E10 — prompt-steering decomposition

**Optional / future work:**
- E5 (culturally-grounded prompts), E7 (cross-family base/instruct), E11 (full decoding ceiling)

## Methodological decisions to lock in *before* running anything

These decisions propagate through every experiment; settling them up front avoids mid-stream methodology shifts that would make results non-comparable.

1. **Prompt translation pipeline.** Options: fixed NMT (NLLB) / LLM-translated-and-human-checked / human-verified end-to-end. Recommended: NLLB + native-speaker spot-check on the 5 most diversity-discriminating prompts.
2. **Diversity metric robustness.** Currently `1 − mean pairwise cosine` on `multilingual-e5-small`. Embedder choice plausibly affects cross-language comparisons most. Recommend locking in BGE-M3 as a robustness check on RQ2 specifically, plus a non-embedding metric (n-gram or edit-distance) on a few key results.
3. **Mode-annotation protocol.** Manual is gold-standard but slow; LLM-judge is fast but partly circular (we're studying LLM modes with an LLM). Recommend LLM-judge with a human-verified ~10% subset for calibration, and inter-annotator agreement reported.
4. **Few-shot prefix design for base models.** Length, exemplar selection rule, stop-token handling. Settle the protocol once and use identical prefixes for base and instruct in E6/E7.
5. **Coherence gate.** When small-model output in low-resource languages is incoherent, exclude or keep? Recommend: report both, mark incoherent cells in figures.
6. **Human-prose baseline.** Out of scope for the core, but committing now to whether we collect a small English baseline (≈20 responses × 10 prompts) determines whether RQ4's "intervention ceiling" can be expressed in absolute terms or only relative.

## Title and framing — deferred

Working title: **"The Multilingual Hivemind"** (placeholder). The four RQs unify under a "how deep does the phenomenon go" frame — each rules out a shallower explanation (language-specific, post-training, decoder-level) — and that organizing principle can stay regardless of outcome. The *claim* a title makes (e.g. "pretraining-driven", "beyond alignment", "structural") presupposes RQ3's result and should be chosen after the core experiments land. Do not relitigate framing mid-project; commit final title once E6 + E8 are in.

## Risks to flag

- **RQ3 is the highest-variance bet.** If E6 produces a clean "base ≪ instruct" result, the headline becomes a multilingual extension of Jiang et al. rather than a relocation of the mechanism. Still a paper — just a less novel one.
- **Mode-annotation is the biggest single methodological exposure.** The paper's main figure (RQ2) depends on it. Worth doing the calibration subset *first* and only scaling up once agreement is acceptable.
- **Course-project scope.** Even the "core" tier is ambitious. If forced to cut further, RQ1 (E1) + RQ3 (E6+E8) is the minimum viable arc — RQ2 and RQ4 become discussion-section claims rather than experimental ones.
