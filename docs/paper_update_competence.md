# Paper update: independent competence measure (reviewer response)

Handoff for whoever edits the paper. Branch: `competence-benchmarks`. Every number below is
from `results/linguistic*/regression.txt` and `results/tables/tab_regression.tex` as committed.

## Why

A reviewer called §5's competence analysis circular: "model competence" was the GlotLID
target-language match rate of the *same* generations the similarity scores come from, and
formulaic text can score high on both. They asked for an independent capability benchmark and
for the result to be stated as an association, not as "improving competence reduces diversity."

## What was done

- **Competence is now FLORES+ translation quality**: each model translates the 1,012 FLORES+
  devtest sentences from English into the target language (greedy, zero-shot, instruction
  "Translate … Reply with the translation only"), scored with corpus chrF++ (sacrebleu). It shares
  no prompts or outputs with the generations, and it measures *writing* in the language, which
  speaks directly to the reviewer's "formulaic text" point. chrF++ ranges 1.9–70.6 (median 50.3);
  the weakest models are Gemma-3 270M, Qwen3 0.6B and OLMo-2 1B, the strongest Gemma-3 27B,
  Mistral Small 3.2 24B and Gemma-3 12B.
- **English is excluded** from §5 (translation is from English): 12 languages, 252 cells;
  spec A 210 cells / 10 languages (morphology undefined for zh/ja); spec C 230 cells.
- **21 models**, not 23: Qwen3-235B (too large for one GPU) and Qwen3-30B-A3B (MoE kernels failed
  to build) have no scores and are dropped from §5. Every other section still uses 23 models.
- Qwen3 thinking was disabled for the benchmark (`enable_thinking=False`).
- GlotLID is no longer a predictor. It remains the spec C filter (drops 22 cells with match rate
  < 0.80) and a fifth predictor in the new **spec F**.
- **Do not report Belebele.** It was also run, but scored at the first generated token, where many
  models write "(" or "**" rather than the letter; the scores include below-chance accuracies and
  are invalid. The scorer is fixed in code for any future rerun; `run_linguistic` ignores the old scores.

Regenerate everything (no GPU needed) with:
```sh
python -m experiments.run_linguistic        # default --competence-measure flores_chrf
python -m experiments.run_linguistic --embed-model BAAI/bge-m3 --out-dir results/linguistic_bge
python -m experiments.run_linguistic --embed-model Qwen/Qwen3-Embedding-0.6B --out-dir results/linguistic_qwen
python -m experiments.plot_linguistic       # Fig. 7 panels -> results/plots/linguistic_gap_{competence,coefs}.png
python -m experiments.make_paper_tables     # Table 5 -> results/tables/tab_regression.tex
```

## Results (OpenAI / BGE-M3 / Qwen3-Emb)

| | OpenAI | BGE-M3 | Qwen3-Emb |
|---|---|---|---|
| Competence, spec A | +0.036 | +0.038 | +0.039 |
| Competence, spec B | +0.038 | +0.040 | +0.041 |
| Competence, spec C | +0.044 | +0.044 | +0.043 |
| (all p < 0.001) | | | |
| Spec F: competence | +0.023 (p = 0.003) | +0.029 (p < 0.001) | +0.029 (p < 0.001) |
| Spec F: GlotLID | +0.024 (p < 0.001) | +0.018 (p < 0.001) | +0.018 (p < 0.001) |
| Spec H (within-language competence), B | +0.034 | +0.032 | +0.034 (all p < 0.001) |
| Spec H, F: competence / GlotLID | +0.023 / +0.023 | +0.025 / +0.017 | +0.026 / +0.017 (all p < 0.001) |
| Resource level, A–C | p ≥ 0.13 | p ≥ 0.24 | p ≥ 0.30 |
| Resource level, spec H (B) | +0.031, p = 0.008 | +0.025, p = 0.013 | +0.024, p = 0.025 |
| Fertility, A–C | p ≥ 0.10 | p ≥ 0.42 | p ≥ 0.29 |
| Morphology (TTR), A | p = 0.32 | p = 0.14 | p = 0.12 |
| Cell Spearman, gap vs chrF (Fig. 7a) | ρ = 0.27, p < 0.001, n = 252 | 0.35 | 0.31 |

Other numbers:
- chrF vs GlotLID across cells: Spearman ρ = 0.15 (p = 0.016) — nearly unrelated measures.
- Competence vs resource level across cells: Pearson r = 0.29.
- Within-model share of competence variance: 0.69 (plenty of spread across languages).
- Language level (n = 12): competence vs gap ρ = 0.06 / 0.25 / 0.15, all n.s. — the effect lives
  across models within a language, as the old Fig. 7 caption already says.
- Placebo (spec E, human intra-similarity on the predictors, n = 12 languages): competence
  β = +0.037, p = 0.012 under OpenAI; p = 0.25 and 0.062 elsewhere. Positive means languages with
  higher chrF have *more similar* human answers, which would shrink the gap, so it cannot produce
  the positive effect; spec H, which removes all between-language variation in competence, is
  unaffected.
- Resource decomposition (unchanged): log Wikipedia vs LLM intra-sim ρ = 0.52 (p = 0.07) vs human
  ρ = 0.02 (OpenAI); LLM side ρ = 0.41 and 0.37 in the other two spaces, human 0.15 and 0.00.
- Fertility decomposition: vs LLM intra-sim ρ = −0.75 / −0.80 / −0.65; vs human −0.30 / −0.58 /
  −0.40 (p = 0.32 / 0.04 / 0.18); vs gap n.s.

## Edits to the paper

Line numbers are from the 7 Sep submission PDF; the current draft's §5 Results paragraph already
has Belebele-era numbers in it, which must all be replaced. Search for "competence",
"target-language fidelity" and "match rate".

1. **Abstract, L13–16.** "The size of the homogenization gap is associated with a model's
   competence in a language, measured by held-out translation quality, robustly across embedding
   spaces, but not with the language's tokenization fertility or morphological complexity." Replace
   "Output diversity is thus lowest where models are strongest" with "models write least diversely
   in the languages they translate into best."
2. **Intro contribution, L32–33.** "…tracks how well a model handles a language, measured
   independently of the generations, rather than…".
3. **§5 Predictors, L204–205.** *Model competence is the chrF++ of the model's zero-shot
   translation of the 1,012 FLORES+ devtest sentences from English into the language, a held-out
   task that shares no prompts or outputs with the generations; it is undefined for English, which
   §5 therefore omits.* Add: *The GlotLID target-language match rate of the generations is used only
   as an outcome-quality filter and as a comparison predictor (specification F).*
4. **§5 Method, L208–211.** Now 21 models and 12 languages (252 cells). Footnote: *Scores are
   unavailable for Qwen3-235B and Qwen3-30B-A3B for hardware reasons.*
5. **§5 Results.** Replace with (adjust \ref names to the draft):

   > \paragraph{Results.} Fitting the model in all three embedding spaces separates one robust
   > predictor from three that are not (Figure~\ref{fig:linguistic_gap}b, Table~\ref{tab:regression}).
   > Competence is associated with the gap in every space ($\beta = +0.036$/$+0.038$/$+0.039$ in
   > specification A, $p < 0.001$ in every specification A--C). Resource level ($p \geq 0.13$),
   > tokenization fertility ($p \geq 0.10$) and morphological complexity ($p \geq 0.12$) are not
   > significant in any space. Resource level becomes significant only when competence is measured
   > relative to the other models in each language (specification H), which leaves the
   > between-language part of competence to the resource term, so we do not treat it as robust either
   > way; the two are only moderately related ($r = 0.29$ across cells).
   >
   > In our data, then, the gap is more consistently associated with a model's command of a language
   > than with the typological proxies we measured. Basque, the highest-fertility ($1.77\times$
   > English) and lowest-resource language, has the smallest gap under every metric, though as it is
   > also among the languages models translate into worst, it cannot separate these explanations.
   >
   > Four checks support this reading. Dropping the 22 cells with a GlotLID match rate below $0.80$
   > (specification C) leaves the coefficient intact ($+0.044$/$+0.044$/$+0.043$), so degenerate
   > generation is not what carries it. Standardising competence within each language
   > (specification H; $+0.034$/$+0.032$/$+0.034$, $p < 0.001$) removes any between-language
   > difference in chrF, such as its dependence on script. Resource level tracks LLM intra-similarity
   > ($\rho = 0.52$, $p = 0.07$, $n = 13$) but not human intra-similarity ($\rho = 0.02$), which
   > suggests the association runs through the model side, though it weakens in the other two spaces
   > ($\rho = 0.41$, $0.37$). Fertility tracks raw LLM intra-similarity ($\rho = -0.75$) but not the
   > gap, partly offset by a weaker shift on the human side. Finally, adding the match rate to B as a
   > fifth predictor (F) leaves competence significant ($\beta = +0.023$/$+0.029$/$+0.029$,
   > $p \leq 0.003$) while the match rate is itself significant ($+0.024$/$+0.018$/$+0.018$,
   > $p < 0.001$): on-language output is associated with homogenization over and above competence,
   > although because the match rate is measured on the same generations, part of that association
   > may be mechanical.
   >
   > This is consistent with the performance--diversity pattern suggested in
   > Section~\ref{sec:results}: in our data, the better a model translates into a language, the
   > \emph{less} diverse what it writes in it. Because the design is observational, we do not claim
   > that improving competence causes the loss of diversity (Section~\ref{sec:limitations}).

6. **Figure 7.** Regenerate both panels. (a) x-axis is now "FLORES+ en→xx chrF++"; subcaption
   "Spearman ρ = 0.27, p < 0.001, n = 252". (b) top row "Competence (FLORES+ chrF++)"; it plots
   spec A, n = 210 cells, 21 models, 10 languages. Caption: "Competence is associated with the gap in
   all three spaces; resource level, tokenization fertility and morphological complexity in none."
7. **Limitations, Scope (L255–256).** Drop GlotLID from "coarse proxies". Add: *Competence is
   measured as translation quality from English with a surface metric (chrF++), so English itself is
   not covered, chrF++ levels are not comparable across scripts (hence the language random effect and
   specification H), and two models lack scores.*
8. **Conclusion, L261–263.** "Its magnitude is predicted by how well a model handles a language" →
   "is associated with"; "should be expected to grow" → "may grow".
9. **Appendix D / Table 5.** `\input` the regenerated `results/tables/tab_regression.tex`; its
   caption already states the result. Mention specification H and the placebo in one sentence each.
10. **Bibliography.** Add FLORES+/NLLB (NLLB Team, 2022, "No Language Left Behind: Scaling
    Human-Centered Machine Translation") and chrF++ (Popović, 2017, "chrF++: words helping character
    n-grams", WMT).

## Reviewer response (draft)

> We agree that the GlotLID match rate is computed from the same generations as the similarity
> scores. We re-ran the analysis with an independent measure: each model's translation quality
> (chrF++) on the FLORES+ benchmark, translating from English into each target language. It shares
> no prompts or outputs with the diversity measurement, and because it scores what the model
> *writes*, generic or formulaic output cannot score well on it. Competence remains a positive,
> significant predictor of the homogenization gap in every specification and all three embedding
> spaces (standardised β = 0.036–0.044, p < 0.001), including when competence is measured relative to
> the other models in the same language. With the match rate added alongside it, both remain
> significant, suggesting that target-language fidelity carries an association with output
> similarity beyond competence, consistent with the reviewer's concern. Following the reviewer, we
> now describe these results as associations rather than evidence that improving competence would
> reduce diversity. English (the translation source) and two models without scores are excluded from
> this analysis, leaving 21 models and 12 languages.
