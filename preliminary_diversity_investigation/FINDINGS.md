# Preliminary Diversity Investigation

Notes from a single-day exploration (2026-05-06) of the Multilingual Hivemind project, run on Fenrir (RTX PRO 6000 Blackwell, 96 GB VRAM). All scripts use vLLM for sampling and `intfloat/multilingual-e5-small` for embeddings. Diversity is reported as `1 − mean pairwise cosine similarity` over the n=20 sampled responses to a prompt (so higher = more diverse). The default model unless noted is `RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic`.

## Scripts

| Script | Purpose |
|---|---|
| `diversity.py` | Smallest-possible diversity test on one prompt with one model. Sanity check for the metric and the inference stack. |
| `temp_dist_compare.py` | Compares fitted temperature distribution families (beta, lognormal, power-law) against fixed-T baselines, all matched on (median=0.9, p95=1.7). Tests whether *shape* of the temperature distribution matters once quantiles are matched. |
| `fit_temperature_dist.py` | Fits an optimal token-level mixture distribution μ(T) by EM, maximizing corpus log-likelihood under p_μ(x_t) = Σ_T μ(T) · p_T(x_t). Run on `merve/poetry` and `wikimedia/wikipedia`. |
| `diversity_multilingual.py` | Sweep of the same "metaphor about time" prompt translated into 13 languages. Used with both Qwen 3B and 72B-FP8. |
| `diversity_multilingual_ban_river.py` | Same multilingual sweep, but the prompt explicitly bans "river/water" metaphors. Tests whether the strongest mode is uniquely tempting or part of a hierarchy. |
| `diversity_multilingual_ban_river_weaver.py` | Double ban: river/water *and* weaver/loom/tapestry/thread. |
| `diversity_multilingual_creative.py` | Same multilingual sweep, but prompt explicitly asks for "unique and unusual" metaphor and to avoid clichés. Tests whether prompt-based novelty steering reaches mode structure. |

## Headline findings

### 1. The Hivemind effect strengthens with model scale across all languages

Mean diversity dropped from 0.105 (Qwen2.5-3B) to 0.057 (Qwen2.5-72B-FP8) across 13 languages. Every language tightened toward English's already-saturated level. English itself was effectively unchanged (already at the floor in 3B). This is the cross-lingual analog of Jiang et al. 2025's within-English finding that bigger / better-aligned models show *more* mode collapse.

| Language | 3B | 72B-FP8 | Δ |
|---|---:|---:|---:|
| Arabic | 0.134 | 0.021 | −0.113 |
| Hindi | 0.177 | 0.067 | −0.110 |
| Swedish | 0.152 | 0.081 | −0.071 |
| Japanese | 0.117 | 0.055 | −0.062 |
| Italian | 0.122 | 0.059 | −0.063 |
| German | 0.106 | 0.045 | −0.061 |
| Russian | 0.111 | 0.051 | −0.060 |
| Indonesian | 0.117 | 0.061 | −0.056 |
| French | 0.104 | 0.047 | −0.057 |
| Spanish | 0.096 | 0.047 | −0.049 |
| Portuguese | 0.103 | 0.060 | −0.043 |
| Mandarin | 0.078 | 0.064 | −0.014 |
| English | 0.052 | 0.051 | −0.001 |

Important caveat: the 3B numbers in low-resource languages were partly driven by *incoherence* (broken Swedish, Russian, Hindi; Italian code-switching into Portuguese). The 72B-FP8 run produced clean, coherent outputs in every language, so those numbers are interpretable at face value.

### 2. The "Time is a river" mode is universal across 12 of 13 languages

In the unconstrained 72B-FP8 run, the same conceptual mode appears across nearly every language:

> English: "Time is a relentless river…"
> Spanish: "El tiempo es un río implacable…"
> French: "Le temps est un fleuve incessant…"
> German: "Die Zeit ist wie ein stetiger Fluss…"
> Portuguese: "O tempo é um rio implacável…"
> Italian: "Il tempo è un fiume inesorabile…"
> Mandarin: "时间如同 flowing river…" / "时间如同流水…"
> Japanese: "時間は流れゆく川…"
> Russian: "Время — это река…"
> Hindi: "समय एक नदी जैसा है…" (time is like a river)
> Indonesian: "Waktu adalah sungai…" (time is a river)
> Swedish: "Tiden är en konstant ström…" (constant stream)

Strongly suggests that multilingual instruction-tuning data is dominated by English-translated synthetic content, imposing English conceptual structure across the world's languages.

Arabic was the lone exception: it converged on a culturally Arab proverb (الوقت كالسيف إن لم تقطعه قطعك, "Time is like a sword: if you don't cut it, it will cut you"), with diversity 0.021 — *more* collapsed than the river languages. So some languages have their own culturally-specific dominant modes, even more collapsed than the universal river mode.

### 3. The mode hierarchy goes at least three deep — and plateaus by depth 2–3

Banning "river/water" lifts mean diversity from 0.057 to 0.078 (+37%) and reveals a second universal mode: **weaver / tapestry / loom / threads**. 8 of 13 languages converge on this image:

> English: "Time is a weaver, threading moments… intricate tapestry…"
> Spanish: "tejedor silencioso, urdiendo el tapiz…"
> French: "tisserand silencieux, brodant chaque instant…"
> German: "gewebtes Tuch, unentwegt von unsichtbaren Fingern geflochten…"
> Mandarin: "时间如同织布机上的线…" (threads on a loom)

Banning *both* river and weaver yields **no further gain on average** (mean 0.078 → 0.075) and *decreases* diversity in 6 of 13 languages — those languages converge on a third universal mode (silent gardener for Romance languages, silent sculptor chiseling marble for English):

> English: "Time is a relentless sculptor, chiseling away at the marble…"
> French/Italian/Portuguese: "Le temps / Il tempo / O tempo é un jardinier / giardiniere / jardineiro silencieux / silenzioso…"

So the universal hierarchy is: **river/water → weaver/tapestry → sculptor/gardener**, at least three layers deep, with similar diversity at each level. Each ban walks down the stack rather than unlocking spread.

| Condition | Mean diversity |
|---|---:|
| Baseline | 0.057 |
| Ban river/water | 0.078 |
| Ban river+weaver | 0.075 |
| "Be unique" prompt | 0.087 |

### 4. Prompt-based novelty steering ("be unique and unusual") barely helps and doesn't reach modes

Mean diversity for the explicit "unique and unusual; avoid clichés" prompt is 0.087 — only modestly above ban-river (0.078). And the river mode *still reappears* in Italian, Mandarin, Japanese, and Russian despite the prompt explicitly demanding novelty:

> Italian creative: "Il tempo è un **fiume** di sabbia…"
> Mandarin creative: "时间是一条蜿蜒的**河流**…"
> Japanese creative: "**静かに流れゆく川の水**のように…"
> Russian creative: "непрекращающийся **поток невидимой воды**…"

The model interprets "unique" as license to add adjectives ("silent," "spectral," "ghostly") within the same mode, not as instruction to change modes. Damning finding for prompt-based diversification: meta-instructions don't penetrate to the mode level.

### 5. Per-token temperature distribution shape is not where the action is

Two independent results converge on this:

**a. Distributional family doesn't matter once quantiles are matched.** Beta, log-normal, and power-law sampling distributions, all matched on (median=0.9, p95=1.7), give statistically identical diversity at n=20 prompts:
- beta: 0.0984 ± 0.0058
- lognormal: 0.1008 ± 0.0060
- power_law: 0.0977 ± 0.0052
- fixed T=0.9: 0.0868 ± 0.0064 (loses to all distribution-based by ~10–15%)

So the shape of the temperature distribution doesn't carry information; the Pareto frontier is parameterized by 2–3 quantile numbers.

**b. Optimal token-level μ(T) doesn't depend on the corpus.** Fitting μ by EM to maximize per-token corpus likelihood gives nearly identical distributions for poetry and Wikipedia (both peak at T=1.10 with weight ≈ 0.27, p95=1.30, mean ≈ 1.06). The corpus difference shows up entirely in the achieved NLL (3.15 vs 1.87), not in the optimal μ.

Together these say: **the homogenization is structural, not in the sampling layer**. Decoding tweaks (temperature distributions, sampling strategies) cannot close the LM-vs-human diversity gap. The convergence is in the *modes*, not in per-token sharpness.

## Suggested narrative for the paper

Five-result arc with a clear thesis — *The Multilingual Hivemind is structural, layered, and resistant to surface interventions:*

1. The Hivemind effect generalizes across languages.
2. It *strengthens* with model scale.
3. It has a layered universal mode hierarchy (≥3 deep).
4. Prompt-based novelty steering doesn't reach mode level.
5. Per-token decoding tweaks can't fix it either.

## Caveats and known limitations

- All single-prompt findings (river-mode universality, mode hierarchy, creative-prompt failure) come from one prompt: "Write a one-sentence metaphor about time." The story might collapse if mode-stack universality is specific to time-metaphors. **The next experiment should replicate across a small bank of prompts (e.g. 10–20 from different open-ended categories: opinions, slogans, jokes, descriptions, moral dilemmas).** Until that's done, treat the qualitative findings as suggestive rather than confirmed.
- All numbers are from one model family (Qwen2.5). Need to verify with at least one non-Qwen model (Llama-3.3-70B, Mistral-Large, or Aya-Expanse-32B) to rule out a Qwen-specific artifact. Aya is particularly worth testing because its instruction-tuning is explicitly multilingual rather than English-translated.
- The diversity metric is `1 − mean pairwise cosine similarity` in `multilingual-e5-small` space. The embedder choice could shape cross-language comparisons. Worth a robustness check with a stronger embedder (e.g. BGE-M3) and possibly a non-embedding metric (n-gram, edit distance) for a few key results.
- No human-prose baseline. We measure the model-output distribution but don't compare it to a real human distribution per language. This is the gap the original Jiang et al. paper has, and our project inherits it. Worth at minimum a small human-response collection for English (e.g. via WritingPrompts subreddit data) to anchor the metric in absolute terms.
- Per-token vs per-generation temperature distribution distinction (in `fit_temperature_dist.py`): the fitted μ is per-token; per-generation μ would tell a different story. We didn't run that variant.
- The poetry corpus (`merve/poetry`) is mostly pre-1923 public-domain works, not strictly post-WWII. Easy to swap to a modern corpus once one is identified.
- All scripts assume the Fenrir CPATH workaround for Triton (see project memory `reference_fenrir.md`).

## Suggested next experiments (in priority order)

1. **Prompt-bank replication.** Run the baseline + ban-river + creative-prompt comparison on 10–20 prompts spanning the Jiang et al. taxonomy (creative generation, opinion, slogan, brainstorm, descriptive, jokes, moral dilemmas). Confirms whether the layered-mode-stack finding generalizes beyond "metaphor about time."
2. **Cross-model replication.** Same multilingual sweep on Llama-3.3-70B (after gating) and Aya-Expanse-32B. Establishes that the universal-mode finding isn't Qwen-specific. Aya is the most diagnostic since it's explicitly multilingual-trained — if Aya *also* shows English-translation-imposed modes, the case for "synthetic translated training data drives the Hivemind" weakens.
3. **Coherence gate.** Add an LM-as-judge or perplexity filter to the diversity metric so quality-confounded numbers (the 3B Hindi/Swedish/Russian results) can be partially recovered, and so future experiments at the competence cliff are interpretable.
4. **English human-prose baseline.** Collect ~20 human responses to ~10 of the Infinity-Chat prompts (e.g. via Prolific or even informally) and compute the same diversity metric. Gives an absolute reference point for what "real diversity" looks like on these prompts.
5. **Mode-bank annotation.** For the unconstrained 72B-FP8 multilingual run, manually label which conceptual mode each generated metaphor falls into (river/weaver/sculptor/gardener/painter/thief/journey/etc.). Lets you compute *mode entropy* directly rather than relying on embedding similarity, and produces a much sharper figure for the paper.
