"""Explain the per-language homogenization gap with linguistic / model properties.

The gap for one (model, language) cell is the LLM-minus-human intra-similarity
difference already computed by run_metrics, read from the `model_language` level
of its F-test table. Positive = that model's samples cluster more tightly than
the human responses to the same prompts.

Four candidate predictors, matching the four the reviewer suggested:

  fertility_vs_en   per (model, language). Subword tokens this model's tokenizer
                    spends on FLORES-200 parallel text relative to English.
                    From run_fertility.
  competence        per (model, language). Held-out benchmark score, from
                    run_competence + score_competence; --competence-measure picks
                    belebele_acc (default; reading comprehension, parallel across
                    all 13 languages), flores_chrf or flores_comet (en -> xx
                    translation; undefined for en, so those fits drop English).
  log_wiki          per language. log10 Wikipedia articles, a standard
                    resource-level proxy (fetched live, cached).
  ttr               per language. Type-token ratio on the FLORES parallel text,
                    a morphological-richness proxy. Undefined for zh and ja,
                    which have no whitespace word boundaries -- those two cells
                    are NaN rather than silently mis-segmented.

Competence used to be the GlotLID target-language match rate of the generations
themselves. That is circular -- it comes from the same samples as the similarity
scores, and generic formulaic text scores well on both -- so it is no longer a
predictor. It survives in two clearly separate roles: as an outcome-quality
filter (spec C drops cells whose generations are not in the target language),
and in spec F, which puts it next to the independent measure to ask whether
it carries any signal beyond competence.

Analysis is deliberately two-level, because the levels carry different power:
  - cell level (n=299): mixed model with a random intercept per model, so
    repeated languages within a model do not count as independent evidence.
  - language level (n=13): Spearman correlations on the model-averaged gap.
    Reported as exploratory -- 13 points is thin and we say so.

Outputs (to --out-dir):
  linguistic_table.parquet   one row per (model, language) with gap + predictors
  language_summary.parquet   one row per language, model-averaged
  regression.txt             mixed-model fit + correlation table
"""

import argparse
import json
import re
import string
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Project language code -> Wikipedia subdomain. pt-BR maps to the (unified)
# Portuguese Wikipedia; there is no separate pt-BR edition.
WIKI_CODE = {"en": "en", "es": "es", "ru": "ru", "zh": "zh", "de": "de", "fr": "fr",
             "ca": "ca", "pt-BR": "pt", "it": "it", "ja": "ja", "eu": "eu",
             "pl": "pl", "vi": "vi"}

# Languages without whitespace word boundaries: word-level morphology measures
# are not defined for them without a segmenter, so we leave them missing.
NO_WORD_BOUNDARY = {"zh", "ja"}

FLORES_CODE = {
    "en": "eng_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "de": "deu_Latn", "fr": "fra_Latn", "ca": "cat_Latn", "pt-BR": "por_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "eu": "eus_Latn", "pl": "pol_Latn",
    "vi": "vie_Latn",
}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def wikipedia_articles(cache: Path) -> dict[str, int]:
    """Article count per language Wikipedia. Cached; throttled to avoid 429s."""
    if cache.exists():
        return json.loads(cache.read_text())
    counts = {}
    for lang, code in WIKI_CODE.items():
        url = (f"https://{code}.wikipedia.org/w/api.php?action=query&meta=siteinfo"
               f"&siprop=statistics&format=json")
        req = urllib.request.Request(url, headers={"User-Agent": "hivemind-research/1.0"})
        counts[lang] = json.load(urllib.request.urlopen(req, timeout=30))["query"]["statistics"]["articles"]
        print(f"  {lang:<6s} {counts[lang]:>10,} articles")
        time.sleep(2.0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(counts, indent=2))
    return counts


_PUNCT = str.maketrans("", "", string.punctuation + "«»„“”‘’—–…¡¿。、，；：？！（）")


def morphology_from_flores(flores_dir: Path) -> pd.DataFrame:
    """Type-token ratio and mean word length on the FLORES parallel corpus.

    The corpus is parallel, so differences across languages reflect the language
    rather than the subject matter.
    """
    rows = []
    for lang, code in FLORES_CODE.items():
        text = (flores_dir / f"{code}.devtest").read_text(encoding="utf-8")
        if lang in NO_WORD_BOUNDARY:
            rows.append({"language": lang, "ttr": np.nan, "mean_word_len": np.nan})
            continue
        words = [w for w in re.split(r"\s+", text.lower().translate(_PUNCT)) if w]
        rows.append({
            "language": lang,
            "ttr": len(set(words)) / len(words),
            "mean_word_len": float(np.mean([len(w) for w in words])),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def load_gap(f_tests: Path) -> pd.DataFrame:
    """LLM-minus-human intra-similarity per (model, language)."""
    f = pd.read_parquet(f_tests)
    ml = f[f["level"] == "model_language"].copy()
    return pd.DataFrame({
        "model": ml["group"], "language": ml["language"],
        "llm_intra": ml["mean_a"], "human_intra": ml["mean_b"],
        "gap": ml["mean_a"] - ml["mean_b"],
    }).reset_index(drop=True)


COMPETENCE_MEASURES = ["belebele_acc", "flores_chrf", "flores_comet"]


def build_table(f_tests: Path, fertility: Path, fluency: Path, competence: Path,
                measure: str, flores_dir: Path, wiki_cache: Path,
                models_yaml: Path) -> pd.DataFrame:
    import yaml
    entries = yaml.safe_load(models_yaml.read_text()) or []
    families = {e.get("display_name", e["id"]): e.get("family", "Other") for e in entries}
    hf_to_name = {e["id"]: e.get("display_name", e["id"]) for e in entries}

    df = load_gap(f_tests)
    df["family"] = df["model"].map(families).fillna("Other")

    fert = pd.read_parquet(fertility)[["model", "language", "fertility_vs_en", "tokens_per_char"]]
    df = df.merge(fert, on=["model", "language"], how="left")

    flu = pd.read_parquet(fluency)[["model", "language", "match_rate"]].copy()
    flu["model"] = flu["model"].map(lambda m: hf_to_name.get(m, m))
    df = df.merge(flu.rename(columns={"match_rate": "lid_match_rate"}),
                  on=["model", "language"], how="left")

    if not competence.exists():
        raise FileNotFoundError(f"{competence} not found; run experiments.run_competence "
                                f"for each model, then experiments.score_competence")
    comp = pd.read_parquet(competence)
    comp = comp[["model", "language", "belebele_letter_mass"] + COMPETENCE_MEASURES].copy()
    comp["model"] = comp["model"].map(lambda m: hf_to_name.get(m, m))
    df = df.merge(comp, on=["model", "language"], how="left")
    if df[measure].isna().all():
        raise ValueError(f"no {measure} values for any cell in {competence}")
    # A model with no benchmark run at all (e.g. too large for the GPU) leaves the
    # whole analysis, not just the competence fits, so every level -- cell fits,
    # language means, placebo -- describes the same set of models.
    unscored = sorted(set(df["model"]) - set(df.loc[df[measure].notna(), "model"]))
    if unscored:
        print(f"dropping {len(unscored)} models without {measure}: {', '.join(unscored)}")
        df = df[~df["model"].isin(unscored)].reset_index(drop=True)
    df["competence"] = df[measure]
    df["competence_measure"] = measure

    wiki = wikipedia_articles(wiki_cache)
    df["log_wiki"] = df["language"].map(lambda l: np.log10(wiki[l]))
    df = df.merge(morphology_from_flores(flores_dir), on="language", how="left")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

PREDICTORS = ["fertility_vs_en", "competence", "log_wiki", "ttr"]


def correlation_table(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Spearman rho of each predictor against the gap."""
    from scipy import stats
    rows = []
    for p in PREDICTORS:
        sub = df[["gap", p]].dropna()
        if len(sub) < 4:
            continue
        rho, pval = stats.spearmanr(sub[p], sub["gap"])
        rows.append({"level": level, "predictor": p, "n": len(sub),
                     "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows)


def mixed_model_coefs(df: pd.DataFrame, predictors: list[str],
                      outcome: str = "gap") -> pd.DataFrame:
    """Standardised coefficients with 95% CIs from the crossed-effects fit."""
    fit = _fit(df, predictors, outcome)[0]
    ci = fit.conf_int()
    rows = []
    for p in predictors:
        term = f"z_{p}"
        rows.append({"predictor": p, "coef": fit.params[term],
                     "lo": ci.loc[term, 0], "hi": ci.loc[term, 1],
                     "p_value": fit.pvalues[term]})
    return pd.DataFrame(rows)


def _fit(df: pd.DataFrame, predictors: list[str], outcome: str):
    """Shared fitting body: returns (fit, z-scored frame)."""
    import statsmodels.formula.api as smf
    sub = df.dropna(subset=[outcome] + predictors).copy()
    for p in predictors:
        sub[f"z_{p}"] = (sub[p] - sub[p].mean()) / sub[p].std()
    sub["_grp"] = 1
    formula = f"{outcome} ~ " + " + ".join(f"z_{p}" for p in predictors)
    fit = smf.mixedlm(
        formula, sub, groups="_grp", re_formula="0",
        vc_formula={"model": "0 + C(model)", "language": "0 + C(language)"},
    ).fit(method="lbfgs")
    return fit, sub


def fit_mixed_model(df: pd.DataFrame, predictors: list[str], outcome: str = "gap") -> str:
    """Cell-level model: outcome ~ predictors + (1 | model) + (1 | language).

    Random intercepts are *crossed*, not nested: every model answers in every
    language. This matters because log_wiki and ttr take only 13 distinct values
    repeated across 21 models -- with a model-only random intercept, those 13
    languages would be counted as 273 independent observations and the standard
    errors on the language-level predictors would be far too small. statsmodels
    expresses crossed effects as two variance components over one dummy group.

    Predictors are z-scored so coefficients are comparable to one another.
    """
    fit, sub = _fit(df, predictors, outcome)
    head = (f"n = {len(sub)} cells, {sub['model'].nunique()} models, "
            f"{sub['language'].nunique()} languages "
            f"(crossed random intercepts for model and language)")
    return f"{head}\n\n{fit.summary()}"


def placebo_language_level(lang: pd.DataFrame, predictors: list[str]) -> str:
    """OLS of human_intra on the language-level predictors, n = 13 languages."""
    import statsmodels.formula.api as smf
    sub = lang.dropna(subset=["human_intra"] + predictors).copy()
    for p in predictors:
        sub[f"z_{p}"] = (sub[p] - sub[p].mean()) / sub[p].std()
    formula = "human_intra ~ " + " + ".join(f"z_{p}" for p in predictors)
    return f"n = {len(sub)} languages\n\n{smf.ols(formula, sub).fit().summary()}"


def decomposition(lang: pd.DataFrame) -> pd.DataFrame:
    """Does a predictor move the gap via the LLM side or the human side?

    The gap is llm_intra minus human_intra, so a predictor that only tracks how
    varied the human responses happen to be in a language would be a measurement
    artifact rather than a fact about the models.
    """
    from scipy import stats
    rows = []
    for p in PREDICTORS:
        for outcome in ["gap", "llm_intra", "human_intra"]:
            sub = lang[[p, outcome]].dropna()
            rho, pval = stats.spearmanr(sub[p], sub[outcome])
            rows.append({"predictor": p, "outcome": outcome, "n": len(sub),
                         "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows)


def competence_diagnostics(df: pd.DataFrame, measure: str) -> str:
    """Is there enough within-model spread for the competence effect to be estimated?

    With a random intercept per model, the coefficient is identified mostly from
    how competence moves across languages *within* a model. A benchmark at its
    ceiling (or at chance) for most models leaves little of that to work with.
    """
    sub = df.dropna(subset=["competence"])
    within = sub["competence"] - sub.groupby("model")["competence"].transform("mean")
    lines = [f"measure = {measure}: {len(sub)} cells, {sub['model'].nunique()} models, "
             f"{sub['language'].nunique()} languages",
             f"within-model share of competence variance: {within.var() / sub['competence'].var():.3f}"]
    if measure == "belebele_acc":
        lines += [f"cells at ceiling (acc >= 0.95): {(sub['competence'] >= 0.95).sum()}",
                  f"cells near chance (acc <= 0.35; chance = 0.25): {(sub['competence'] <= 0.35).sum()}",
                  f"cells with letter mass < 0.5 (format not followed; acc understates "
                  f"comprehension): {(sub['belebele_letter_mass'] < 0.5).sum()}"]
    spread = (sub.groupby("model")["competence"].agg(["min", "max", "std"])
              .sort_values("std").round(3))
    return "\n".join(lines) + "\n\nper-model spread across languages:\n" + spread.to_string()


def proxy_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Cell-level Spearman of each independent measure against the old GlotLID proxy."""
    from scipy import stats
    rows = []
    for m in COMPETENCE_MEASURES:
        sub = df[[m, "lid_match_rate"]].dropna()
        if len(sub) < 4:
            continue
        rho, pval = stats.spearmanr(sub[m], sub["lid_match_rate"])
        rows.append({"measure": m, "n": len(sub), "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows)


def main(metrics_dir: Path, embed_model: str, ling_dir: Path, fluency: Path,
         competence: Path, measure: str, models_yaml: Path, out_dir: Path):
    slug = embed_model.replace("/", "_")
    df = build_table(metrics_dir / f"f_tests__{slug}.parquet",
                     ling_dir / "fertility.parquet", fluency, competence, measure,
                     ling_dir / "flores", ling_dir / "wikipedia.json", models_yaml)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "linguistic_table.parquet", index=False)
    print(f"wrote linguistic_table.parquet ({len(df)} cells)")

    lang = (df.groupby("language")
            .agg(gap=("gap", "mean"), fertility_vs_en=("fertility_vs_en", "mean"),
                 competence=("competence", "mean"), lid_match_rate=("lid_match_rate", "mean"),
                 log_wiki=("log_wiki", "first"),
                 ttr=("ttr", "first"), mean_word_len=("mean_word_len", "first"))
            .reset_index().sort_values("gap"))
    lang.to_parquet(out_dir / "language_summary.parquet", index=False)

    lang_full = df.groupby("language").agg(
        {p: "mean" for p in PREDICTORS} |
        {"gap": "mean", "llm_intra": "mean", "human_intra": "mean"}).reset_index()

    # No-morphology spec: ttr is undefined for zh and ja, so dropping it buys
    # back those two languages at the cost of one predictor.
    no_morph = [p for p in PREDICTORS if p != "ttr"]
    # Degenerate generations read as "diverse" to every diversity metric, so
    # re-fit on cells the language-ID check considers fluent. This filters the
    # outcome; GlotLID is not a predictor here.
    fluent = df[df["lid_match_rate"] >= 0.80]
    # The other independent measures, refit under spec B as a robustness check.
    alt_measures = [m for m in COMPETENCE_MEASURES if m != measure and df[m].notna().any()]

    corr = pd.concat([correlation_table(df, "cell"), correlation_table(lang, "language")])
    report = [
        f"Competence predictor: {measure} (held-out benchmark; not computed from the generations)",
        "",
        "=== Competence diagnostics: ceiling, floor, within-model spread ===",
        competence_diagnostics(df, measure), "",
        "=== Independent competence vs. the GlotLID match rate (cell-level Spearman) ===",
        proxy_agreement(df).round(4).to_string(index=False), "",
        "=== Spearman correlations with the homogenization gap ===",
        corr.to_string(index=False), "",
        "=== Predictor intercorrelations (Spearman, language level) ===",
        lang_full[PREDICTORS].corr(method="spearman").round(3).to_string(), "",
        "=== Gap decomposition: LLM side vs human side (language level) ===",
        "A predictor acting through human_intra would be an artifact of how varied",
        "the human responses happen to be, not a fact about the models.",
        decomposition(lang_full).round(4).to_string(index=False), "",
        "=== Language-level summary (sorted by gap) ===",
        lang.round(4).to_string(index=False), "",
        "=== Spec A: gap ~ all four predictors + (1 | model) ===",
        fit_mixed_model(df, PREDICTORS), "",
        "=== Spec B: drop ttr, recovering zh and ja ===",
        fit_mixed_model(df, no_morph), "",
        f"=== Spec C: spec B on fluent cells only (outcome-quality filter: "
        f"lid_match_rate >= 0.80; {len(df) - len(fluent)} of {len(df)} cells dropped) ===",
        fit_mixed_model(fluent, no_morph), "",
        "=== Spec D: outcome = llm_intra instead of the gap (spec B predictors) ===",
        fit_mixed_model(df, no_morph, outcome="llm_intra"), "",
        "=== Spec E: outcome = human_intra (placebo) ===",
        "human_intra is one value per language, identical for every model, so it is",
        "fitted at the language level (n=13) -- a cell-level fit would replicate 13",
        "numbers 21 times and report standard errors ~sqrt(21)x too small.",
        placebo_language_level(lang_full, no_morph), "",
        "=== Spec F: spec B + GlotLID match rate (does the old proxy add anything?) ===",
        "If lid_match_rate stays significant next to independent competence, it is",
        "tracking something other than competence -- e.g. language purity or",
        "formulaic output -- rather than standing in for it.",
        fit_mixed_model(df, no_morph + ["lid_match_rate"]),
    ]
    for m in alt_measures:
        report += ["", f"=== Spec G: spec B with competence = {m} ===",
                   fit_mixed_model(df.assign(competence=df[m]), no_morph)]
    text = "\n".join(report)
    (out_dir / "regression.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--ling-dir", type=Path, default=Path("results/linguistic"))
    ap.add_argument("--fluency", type=Path,
                    default=Path("results/fluency/fluency_summary__glotlid.parquet"))
    ap.add_argument("--competence", type=Path,
                    default=Path("results/competence/competence_summary.parquet"))
    ap.add_argument("--competence-measure", choices=COMPETENCE_MEASURES, default="belebele_acc")
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/linguistic"))
    args = ap.parse_args()
    main(args.metrics_dir, args.embed_model, args.ling_dir, args.fluency,
         args.competence, args.competence_measure, args.models_yaml, args.out_dir)
