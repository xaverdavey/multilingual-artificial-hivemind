"""Prompt-level paired tests for the homogenization gap, with multiple-testing control.

The F-tests in run_metrics are two-group one-way ANOVAs (equivalently pooled-
variance t-tests) whose statistical unit is the (language, prompt) cell: one
LLM number per prompt (mean pairwise cosine over that prompt's samples) against
one human number per prompt (mean pairwise cosine over the ~3 human answers).
That already collapses the many non-independent pairwise similarities inside a
prompt into one observation, but it ignores that the two sides are *paired* by
prompt and it applies no correction for the number of cells tested.

This script re-runs every comparison the paper reports as a paired, prompt-level
analysis:

  gap_p = llm_p - human_p                    one difference per prompt p

  * paired bootstrap 95% CI on the mean gap (resampling prompts);
  * for the pooled gap, a two-level cluster bootstrap (languages, then prompts);
  * Wilcoxon signed-rank test on gap_p (two-sided);
  * sign-flip permutation test on mean(gap_p) (two-sided, Monte Carlo);
  * Holm and Benjamini-Hochberg adjustment of the permutation p-values across
    every group tested at the same level (23 models, 13 languages, 299 cells,
    15 family pairs, 195 family-pair x language cells, ...).

The per-group means and standard errors over prompts of both sides are stored
too, so plot_paired_figures can redraw the paper's figures from this analysis.

Three comparisons, the first two mirroring run_metrics:
  intra   one LLM resampled vs. the humans who answered the same prompt
  cross   two different LLMs vs. two different humans on the same prompt;
          the LLM side is averaged over the distinct model pairs in the group
          *per prompt*, so each prompt again contributes one observation.
  ladder  one LLM resampled vs. two different LLMs (intra vs. inter, both
          pooled per prompt); the top rung of the same-model / different-models /
          different-humans ladder, which needs no human baseline.

Inputs are the per-prompt parquets written by run_metrics / run_lexical. The
human per-prompt intra-similarity for an embedding space is recomputed from the
human embeddings on disk (it is not saved by run_metrics).

Outputs (to --out-dir):
  prompt_tests__{slug}.parquet   one row per (comparison, level, group)
  prompt_tests__{slug}.txt       summary counts per level
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_metrics import load_human, load_config, mean_pairwise_cosine

N_BOOT = 10_000
N_PERM = 20_000
SEED = 0


# ---------------------------------------------------------------------------
# Human side
# ---------------------------------------------------------------------------

def human_intra_from_embeddings(embed_dir: Path, embed_model: str, any_slug: str) -> pd.DataFrame:
    """Per-(language, prompt) mean pairwise cosine over the human answers.

    Every model's `_human.npy` holds the same 1183 human answers; any one works.
    """
    emb, idx = load_human(any_slug, embed_model, embed_dir)
    rows = []
    for (lang, pid), grp in idx.reset_index(drop=True).groupby(["language", "prompt_id"]):
        rows.append({"language": lang, "prompt_id": pid,
                     "human": mean_pairwise_cosine(emb[grp.index.to_numpy()])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests on one vector of paired differences
# ---------------------------------------------------------------------------

def _paired_tests(d: np.ndarray, rng: np.random.Generator) -> dict:
    from scipy import stats
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n}
    mean = float(d.mean())
    boot = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # sign-flip permutation: under H0 the sign of each prompt's difference is exchangeable
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    perm = (signs * d).mean(axis=1)
    p_perm = (np.sum(np.abs(perm) >= abs(mean)) + 1) / (N_PERM + 1)
    try:
        p_wilcoxon = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:  # all differences zero
        p_wilcoxon = 1.0
    p_paired_t = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue) if d.std() > 0 else 1.0
    return {"n": n, "mean_gap": mean, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_perm": float(p_perm), "p_wilcoxon": p_wilcoxon, "p_paired_t": p_paired_t,
            "frac_prompts_positive": float((d > 0).mean())}


def _side_stats(pf: pd.DataFrame) -> dict:
    """Means and standard errors over prompts of the two sides, for the figures."""
    ok = pf[["llm", "human"]].dropna()
    n = len(ok)
    if n < 2:
        return {}
    return {"mean_llm": float(ok["llm"].mean()), "mean_human": float(ok["human"].mean()),
            "se_llm": float(ok["llm"].std(ddof=1) / np.sqrt(n)),
            "se_human": float(ok["human"].std(ddof=1) / np.sqrt(n))}


def _cluster_bootstrap(df: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """Two-level bootstrap of the pooled mean gap: languages, then prompts within."""
    df = df.dropna(subset=["gap"])
    groups = {lang: sub["gap"].to_numpy() for lang, sub in df.groupby("language")}
    langs = list(groups)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked = rng.choice(langs, size=len(langs), replace=True)
        vals = [rng.choice(groups[l], size=len(groups[l]), replace=True) for l in picked]
        out[b] = np.concatenate(vals).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def _adjust(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Holm and BH adjustment of `col` within each (comparison, level)."""
    from statsmodels.stats.multitest import multipletests
    df = df.copy()
    df[f"{col}_holm"] = np.nan
    df[f"{col}_bh"] = np.nan
    for _, ix in df.groupby(["comparison", "level"]).groups.items():
        p = df.loc[ix, col].to_numpy()
        ok = np.isfinite(p)
        if ok.sum() == 0:
            continue
        df.loc[ix[ok], f"{col}_holm"] = multipletests(p[ok], method="holm")[1]
        df.loc[ix[ok], f"{col}_bh"] = multipletests(p[ok], method="fdr_bh")[1]
    return df


# ---------------------------------------------------------------------------
# Assemble paired frames
# ---------------------------------------------------------------------------

def intra_frames(intra: pd.DataFrame, human: pd.DataFrame, families: dict[str, str]):
    """Yield (level, group, language, paired frame with columns gap/language)."""
    intra = intra.assign(family=intra["model"].map(families).fillna("Other"))
    key = ["language", "prompt_id"]

    def paired(sub: pd.DataFrame) -> pd.DataFrame:
        llm = sub.groupby(key)["intra_sim"].mean().rename("llm").reset_index()
        m = llm.merge(human, on=key, how="inner")
        return m.assign(gap=m["llm"] - m["human"])

    yield "general", "all", None, paired(intra)
    for fam, sub in intra.groupby("family"):
        yield "family", fam, None, paired(sub)
    for model, sub in intra.groupby("model"):
        yield "model", model, None, paired(sub)
    for lang, sub in intra.groupby("language"):
        yield "language", lang, lang, paired(sub)
    for (model, lang), sub in intra.groupby(["model", "language"]):
        yield "model_language", model, lang, paired(sub)


def cross_frames(inter: pd.DataFrame, human: pd.DataFrame, families: dict[str, str]):
    inter = inter.assign(fam_a=inter["model_a"].map(families).fillna("Other"),
                         fam_b=inter["model_b"].map(families).fillna("Other"))
    lo = inter[["fam_a", "fam_b"]].min(axis=1)
    hi = inter[["fam_a", "fam_b"]].max(axis=1)
    inter = inter.assign(family_pair=lo + " || " + hi)
    key = ["language", "prompt_id"]

    def paired(sub: pd.DataFrame) -> pd.DataFrame:
        llm = sub.groupby(key)["inter_sim"].mean().rename("llm").reset_index()
        m = llm.merge(human, on=key, how="inner")
        return m.assign(gap=m["llm"] - m["human"])

    yield "general", "all", None, paired(inter)
    for lang, sub in inter.groupby("language"):
        yield "language", lang, lang, paired(sub)
    for model in sorted(set(inter["model_a"]) | set(inter["model_b"])):
        sub = inter[(inter["model_a"] == model) | (inter["model_b"] == model)]
        yield "model", model, None, paired(sub)
    for pair, sub in inter.groupby("family_pair"):
        yield "family_pair", pair, None, paired(sub)
    for (pair, lang), sub in inter.groupby(["family_pair", "language"]):
        yield "family_pair_language", pair, lang, paired(sub)


def ladder_frames(intra: pd.DataFrame, inter: pd.DataFrame):
    """Top rung of the ladder: one model resampled vs. two different models, per prompt.

    Needs no human baseline: LLM_p is the intra-similarity pooled over models and
    Human_p's slot is taken by the inter-similarity pooled over distinct model pairs.
    """
    key = ["language", "prompt_id"]
    same = intra.groupby(key)["intra_sim"].mean().rename("llm").reset_index()
    diff = inter.groupby(key)["inter_sim"].mean().rename("human").reset_index()  # column name kept for _paired_tests
    m = same.merge(diff, on=key, how="inner")
    m = m.assign(gap=m["llm"] - m["human"])
    yield "general", "all", None, m
    for lang, sub in m.groupby("language"):
        yield "language", lang, lang, sub


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(intra: pd.DataFrame, inter: pd.DataFrame | None, human: pd.DataFrame,
        families: dict[str, str], slug: str, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    sources = [("intra", intra_frames(intra, human, families))]
    if inter is not None:
        sources.append(("cross", cross_frames(inter, human, families)))
        sources.append(("ladder", ladder_frames(intra, inter)))
    for comparison, frames in sources:
        for level, group, lang, pf in frames:
            r = {"comparison": comparison, "level": level, "group": group, "language": lang}
            r.update(_side_stats(pf))
            r.update(_paired_tests(pf["gap"].to_numpy(), rng))
            if level == "general":
                r["cluster_ci_lo"], r["cluster_ci_hi"] = _cluster_bootstrap(pf, rng)
            rows.append(r)
    df = pd.DataFrame(rows)
    df = _adjust(df, "p_perm")
    df = _adjust(df, "p_wilcoxon")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"prompt_tests__{slug}.parquet", index=False)

    lines = [f"=== Prompt-level paired tests: {slug} ===",
             f"bootstrap B={N_BOOT}, permutation B={N_PERM}, seed={SEED}", ""]
    for (comparison, level), sub in df.groupby(["comparison", "level"], sort=False):
        k = len(sub)
        lines.append(f"[{comparison} / {level}]  groups={k}")
        if level == "general":
            g = sub.iloc[0]
            lines.append(f"  mean gap {g['mean_gap']:+.4f}  prompt-bootstrap 95% CI "
                         f"[{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]  language-cluster CI "
                         f"[{g['cluster_ci_lo']:+.4f}, {g['cluster_ci_hi']:+.4f}]")
            lines.append(f"  p_perm {g['p_perm']:.2e}  p_wilcoxon {g['p_wilcoxon']:.2e}  "
                         f"p_paired_t {g['p_paired_t']:.2e}  prompts with gap>0: "
                         f"{g['frac_prompts_positive']:.1%} of n={int(g['n'])}")
            continue
        lines.append(f"  gap>0: {(sub['mean_gap'] > 0).sum()}/{k}   "
                     f"CI excludes 0 (positive): {((sub['ci_lo'] > 0)).sum()}/{k}")
        lines.append(f"  perm p<0.05: {(sub['p_perm'] < 0.05).sum()}/{k}   "
                     f"Holm<0.05: {(sub['p_perm_holm'] < 0.05).sum()}/{k}   "
                     f"BH<0.05: {(sub['p_perm_bh'] < 0.05).sum()}/{k}")
        lines.append(f"  wilcoxon p<0.05: {(sub['p_wilcoxon'] < 0.05).sum()}/{k}   "
                     f"Holm<0.05: {(sub['p_wilcoxon_holm'] < 0.05).sum()}/{k}   "
                     f"BH<0.05: {(sub['p_wilcoxon_bh'] < 0.05).sum()}/{k}")
        neg = sub[sub["mean_gap"] <= 0]
        if len(neg):
            lines.append("  non-positive groups: " + ", ".join(
                f"{r.group}" + (f"/{r.language}" if r.language else "") + f" ({r.mean_gap:+.3f})"
                for r in neg.itertuples()))
        ns = sub[(sub["mean_gap"] > 0) & (sub["p_perm_holm"] >= 0.05)]
        if len(ns) and level not in ("model_language", "family_pair_language"):
            lines.append("  positive but not Holm-significant: " + ", ".join(
                f"{r.group}" + (f"/{r.language}" if r.language else "") +
                f" ({r.mean_gap:+.3f}, p_holm={r.p_perm_holm:.3f})" for r in ns.itertuples()))
        lines.append("")
    text = "\n".join(lines)
    (out_dir / f"prompt_tests__{slug}.txt").write_text(text)
    print(text)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--lexical", default=None,
                    help="instead of an embedding space, use results/lexical/*__{lexical}.parquet "
                         "(e.g. char4gram or char4gram_trunc500)")
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/prompt_tests"))
    args = ap.parse_args()

    _, families = load_config(args.models_yaml)

    if args.lexical:
        slug = args.lexical
        d = Path("results/lexical")
        intra = pd.read_parquet(d / f"intra__{slug}.parquet")
        inter = pd.read_parquet(d / f"inter__{slug}.parquet")
        human = (pd.read_parquet(d / f"human_intra__{slug}.parquet")
                 .rename(columns={"intra_sim": "human"})[["language", "prompt_id", "human"]])
    else:
        slug = args.embed_model.replace("/", "_")
        intra = pd.read_parquet(args.metrics_dir / f"intra__{slug}.parquet")
        inter = pd.read_parquet(args.metrics_dir / f"inter__{slug}.parquet")
        any_slug = sorted(p.name[: -len(f"__{slug}_human.npy")]
                          for p in args.embed_dir.glob(f"*__{slug}_human.npy"))[0]
        human = human_intra_from_embeddings(args.embed_dir, args.embed_model, any_slug)

    run(intra, inter, human, families, slug, args.out_dir)


if __name__ == "__main__":
    main()
