# Paper update: independent competence measure (reviewer response)

Handoff for whoever edits the paper. Branch: `competence-benchmarks`.

## Why

A reviewer called §5's competence analysis circular: "model competence" was the GlotLID
target-language match rate of the *same* generations the similarity scores come from, and
formulaic text can score high on both. They asked for an independent capability benchmark
and for the result to be stated as an association, not as "improving competence reduces
diversity."

## What was done

- **Belebele** (Bandarkar et al., 2024): reading comprehension, 900 four-way MC questions per
  language, **parallel across all 13 languages**, zero-shot, scored by argmax over A–D at the
  first answer token (no sampling or parsing). Chance = 0.25. This is the new competence predictor.
- **FLORES+** (NLLB Team, 2022) en→xx translation, greedy, chrF++ (and optionally COMET).
  Undefined for en. Used as a robustness check (Spec G in `regression.txt`).
- Qwen3 thinking disabled for both benchmarks (`enable_thinking=False`).
- **21 models**, not 23: Qwen3-235B (too large for one GPU) and Qwen3-30B-A3B (MoE kernels
  failed to build) have no benchmark scores. They are dropped from the whole §5 analysis
  (273 cells; spec A 231, spec C 251). All other sections still use 23 models.
- GlotLID is no longer a predictor. It remains (i) the spec C filter for degenerate output and
  (ii) a fifth predictor in the new **Spec F**, which tests whether it carries signal beyond competence.

Code: `experiments/run_competence.py`, `score_competence.py`, `run_linguistic.py`
(`--competence-measure belebele_acc|flores_chrf|flores_comet`). Regenerated table:
`results/tables/tab_regression.tex` (now has rows A, B, C, F and a GlotLID column).

## Numbers available now (from `results/tables/tab_regression.tex`)

Standardised coefficients; `*` p<0.05, `***` p<0.001.

| Space | Competence A / B / C | Competence F | GlotLID F | Resource A / B / C / F | Fertility A / B / C / F |
|---|---|---|---|---|---|
| OpenAI | +0.018\* / +0.017\* / +0.016\* | +0.013\* | +0.025\*\*\* | +0.030\* / +0.025\* / +0.021\* / +0.021 | −0.007 / −0.010\* / −0.009\* / −0.005 |
| BGE-M3 | +0.021\*\*\* / +0.020\*\*\* / +0.018\*\*\* | +0.017\*\*\* | +0.019\*\*\* | +0.016 / +0.014 / +0.011 / +0.010 | −0.006 / −0.007\* / −0.006\* / −0.003 |
| Qwen3-Emb | +0.020\*\*\* / +0.019\*\*\* / +0.017\*\*\* | +0.016\*\*\* | +0.019\*\*\* | +0.013 / +0.011 / +0.008 / +0.008 | −0.007\* / −0.008\* / −0.007\* / −0.005 |

Morphology (TTR, spec A only): +0.005 / +0.009 / +0.010, never significant.

Reading: competence is positive and significant in all 12 fits. GlotLID stays significant
next to it, so the match rate carries an association beyond competence (plausibly the
reviewer's "formulaic target-language text"). Resource level is significant only under
text-embedding-3-small and not in F. Fertility is now small and **significant in 7 of 9 non-F
fits**, but in no F fit.

## Numbers still to pull (run on the branch once `results/linguistic*/` is pushed)

```sh
git checkout competence-benchmarks && git pull
R=results/linguistic/regression.txt
sed -n '/Competence diagnostics/,/=== Spearman/p' $R      # ceiling/chance counts, within-model spread, Belebele-vs-GlotLID rho
sed -n '/=== Spearman correlations/,/=== Predictor/p' $R  # cell-level rho of each predictor with the gap
for s in "Spec A" "Spec B" "Spec F" "Spec G"; do grep -A 25 "=== $s" $R | grep -E "^z_|^n = "; done   # exact p-values
sed -n '/Gap decomposition/,/=== Language-level/p' $R     # replaces the rho=0.49 / 0.02 numbers at L225-227
python -m experiments.plot_linguistic                      # prints the Fig. 7a subcaption (Spearman rho, n)
python -c "import pandas as pd; t=pd.read_parquet('results/linguistic/linguistic_table.parquet'); print(t[['competence','log_wiki']].corr().iloc[0,1].round(2)); print(t.groupby('model').competence.agg(['min','max']).round(2))"
```

The last line gives the competence–resource correlation (replaces "r = 0.25" at L218) and
each model's Belebele range. Repeat the `sed`/`grep` lines with `results/linguistic_bge/` and
`results/linguistic_qwen/` for the other two spaces.

## Edits to the paper

Line numbers are from the 7 Sep submission PDF. Later drafts may call the predictor
"target-language fidelity" (see commit 993b6e8), so search for both that and "model competence".

1. **Abstract, L13–16.** Keep "predicted by a model's competence in a language", but name the
   measure ("held-out reading-comprehension accuracy"). Replace "Output diversity is thus lowest
   where models are strongest" with an association phrasing: "models produce the least diverse
   text in the languages they handle best." Drop "but not by the language's tokenization
   fertility" or soften it to "and only weakly by tokenization fertility" (see item 5).
2. **Intro contribution, L32–33.** "tracks how well a model handles a language, as measured by an
   independent benchmark, rather than…"; same caveat about fertility.
3. **§5 Predictors, L204–205.** Replace the GlotLID definition with: *Model competence is the
   model's zero-shot accuracy on Belebele, 900 reading-comprehension questions that are identical
   across all 13 languages and share no prompts or outputs with the generations (chance 0.25).*
   Add one sentence: *GlotLID target-language match rate on the generations themselves is
   used only as an outcome-quality filter and as a comparison predictor (Spec F).*
4. **§5 Method, L208–211.** "21 models" / "273" are now literally correct. Add a footnote:
   *Benchmark scores are unavailable for Qwen3-235B and Qwen3-30B-A3B for hardware reasons;
   §5 uses the remaining 21 models.*
5. **§5 Results, L212–220.** Replace β = +0.024/+0.019/+0.019 with the new spec A values
   (+0.018/+0.021/+0.020), giving exact p-values from `regression.txt`; OpenAI is p<0.05 only, not
   p<0.001. Resource level: same story as before (significant only under text-embedding-3-small,
   β = +0.030 in spec A). **L219–220 is now false**: fertility *is* significant (β ≈ −0.006 to −0.010)
   in most specifications, though not once GlotLID enters. Rewrite as a small negative association
   that is not robust to Spec F. Morphology is still null in every space. Update "r = 0.25" (L218).
6. **New paragraph after L220 (Spec F).** *Adding the GlotLID match rate as a fifth predictor
   leaves competence significant in every space (β = +0.013/+0.017/+0.016) while the match rate is
   itself significant (β = +0.025/+0.019/+0.019, p<0.001): fidelity to the target language is
   associated with homogenization over and above competence, consistent with formulaic but
   on-language output.*
7. **Checks, L224–227.** Spec C now drops 22 of 273 cells (251 kept); competence stays significant
   in all three spaces (+0.016/+0.018/+0.017). Re-take the decomposition numbers (ρ = 0.49, 0.02)
   from `regression.txt`: they move with the 21-model set.
8. **L228–229.** Replace the causal sentence ("improving a model's command of a language should be
   expected to *reduce* the diversity…") with: *Across models and languages, higher competence is
   associated with less diverse output; whether improving competence causes this is not something
   this observational design can establish.*
9. **Figure 7.** Regenerate with `python -m experiments.plot_linguistic` (writes
   `results/plots/linguistic_gap_competence.png` and `linguistic_gap_coefs.png`). Panel (a) x-axis is
   now Belebele accuracy; the subcaption takes ρ, p, and n = 273 from the script's printout. Panel (b)'s
   top row is now "Competence (Belebele accuracy)". Caption: n = 231 cells, 21 models, 11 languages
   for (b) is unchanged; replace "Model competence predicts the gap in all three spaces; resource level
   in only one; tokenization fertility and morphological complexity in none" to reflect item 5.
10. **Limitations, Scope (L255–256).** "GlotLID match rate … coarse proxies" → Wikipedia count only.
    Add: *Belebele measures comprehension, not generation; it was run without reasoning
    (`enable_thinking=False`) for the Qwen3 models, which reason during the main generation sweep,
    and two models lack scores.*
11. **Conclusion, L261–263.** "Its magnitude is predicted by how well a model handles a language"
    → "is associated with". Soften "should be expected to grow" to "may grow".
12. **Appendix D / Table 5.** `\input` the regenerated `results/tables/tab_regression.tex`. Its
    caption is now descriptive only and needs a results sentence; suggested text:
    *Competence is positive and significant in every specification and embedding space, including
    F, where the GlotLID match rate is also significant. Resource level reaches p<0.05 only under
    text-embedding-3-small, and tokenization fertility only outside F; neither is robust.* Better to
    add it to `regression_table()` in `experiments/make_paper_tables.py` so it regenerates with the
    table. Add one appendix paragraph on the protocol: Belebele prompt (the authors' zero-shot
    instruction; English instruction, target-language passage, question and options),
    `max_tokens=1`, argmax over A–D among the top-20 logprobs. Report any cells where the model put
    little probability on A–D (`belebele_letter_mass` < 0.5, counted in the diagnostics).
13. **Bibliography.** Add Belebele (Bandarkar et al., ACL 2024, "The Belebele Benchmark: a Parallel
    Reading Comprehension Dataset in 122 Language Variants") and FLORES+/NLLB (NLLB Team, 2022,
    "No Language Left Behind") if FLORES+ results are reported.

## Reviewer response (draft)

> We agree that the GlotLID match rate is computed from the same generations as the similarity
> scores. We re-ran the analysis with an independent measure: Belebele reading-comprehension
> accuracy (900 parallel questions in all 13 languages), which shares no prompts or outputs with
> the diversity measurement. Competence remains a positive, significant predictor of the
> homogenization gap in every specification and all three embedding spaces (standardised
> β = 0.016–0.021). When the match rate is added alongside it, both remain significant,
> suggesting that target-language fidelity carries an association with output similarity beyond
> competence, consistent with the reviewer's concern about formulaic target-language text. We now
> describe these results as associations. Two models lack benchmark scores for hardware reasons,
> leaving 21 for this analysis.
