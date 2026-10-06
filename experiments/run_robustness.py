"""Compare the homogenization gap across embedding spaces and an embedder-free metric.

The diversity metrics in this project are read off an embedding space, and any
embedding space carries the geometry of the corpus it was trained on. This script
recomputes the same gap under every metric available on disk and reports whether
the conclusions move.

Sources are discovered automatically:
  results/metrics/f_tests__{slug}.parquet   one per embedding model that has been
                                            through run_metrics
  results/lexical/intra__{tag}.parquet      character n-gram Jaccard (no embedder)

For each source it derives the per (model, language) gap -- LLM intra-similarity
minus human intra-similarity on the same prompts -- and reports:
  * the pooled LLM / human / gap means
  * how many of the 13 languages show a positive gap
  * rank and sign agreement with the primary metric across all 299 cells

Outputs (to --out-dir):
  robustness_by_language.csv   per-language gap under every metric
  robustness_summary.csv       one row per metric
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Readable labels for the on-disk slugs.
LABELS = {
    "text-embedding-3-small": "text-embedding-3-small (OpenAI, 1536d)",
    "BAAI_bge-m3": "BGE-M3 (XLM-R, 1024d)",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B (1024d)",
    "char4gram": "char 4-gram Jaccard (no embedder)",
    "char4gram_trunc500": "char 4-gram Jaccard, 500-char cap",
    "BAAI_bge-m3_trunc500_s10": "BGE-M3, 500-char cap (10 samples/prompt)",
}
PRIMARY = "text-embedding-3-small"


def gap_from_f_tests(path: Path) -> pd.DataFrame:
    f = pd.read_parquet(path)
    ml = f[f["level"] == "model_language"]
    return pd.DataFrame({
        "model": ml["group"].to_numpy(), "language": ml["language"].to_numpy(),
        "llm_intra": ml["mean_a"].to_numpy(), "human_intra": ml["mean_b"].to_numpy(),
    }).assign(gap=lambda d: d["llm_intra"] - d["human_intra"])


def gap_from_lexical(intra_path: Path, human_path: Path) -> pd.DataFrame:
    intra = pd.read_parquet(intra_path)
    human = pd.read_parquet(human_path).groupby("language")["intra_sim"].mean()
    g = (intra.groupby(["model", "language"])["intra_sim"].mean()
         .rename("llm_intra").reset_index())
    g["human_intra"] = g["language"].map(human)
    g["gap"] = g["llm_intra"] - g["human_intra"]
    return g


def discover(metrics_dir: Path, lexical_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(metrics_dir.glob("f_tests__*.parquet")):
        out[p.stem.replace("f_tests__", "")] = gap_from_f_tests(p)
    for p in sorted(lexical_dir.glob("intra__*.parquet")):
        tag = p.stem.replace("intra__", "")
        human = lexical_dir / f"human_intra__{tag}.parquet"
        if human.exists():
            out[tag] = gap_from_lexical(p, human)
    return out


def summarise(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from scipy import stats
    base = sources.get(PRIMARY)
    rows = []
    for slug, df in sources.items():
        by_lang = df.groupby("language")["gap"].mean()
        row = {
            "metric": LABELS.get(slug, slug), "slug": slug, "n_cells": len(df),
            "llm_intra": df["llm_intra"].mean(), "human_intra": df["human_intra"].mean(),
            "gap": df["gap"].mean(),
            "langs_positive": int((by_lang > 0).sum()), "n_langs": len(by_lang),
            "cells_positive": int((df["gap"] > 0).sum()),
        }
        if base is not None and slug != PRIMARY:
            m = base.merge(df, on=["model", "language"], suffixes=("_b", "_x"))
            row["spearman_vs_primary"] = stats.spearmanr(m["gap_b"], m["gap_x"]).statistic
            row["sign_agree_vs_primary"] = float(
                (np.sign(m["gap_b"]) == np.sign(m["gap_x"])).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(metrics_dir: Path, lexical_dir: Path, out_dir: Path):
    sources = discover(metrics_dir, lexical_dir)
    print(f"metrics found ({len(sources)}): {', '.join(sources)}\n")

    by_lang = pd.DataFrame({LABELS.get(s, s): d.groupby("language")["gap"].mean()
                            for s, d in sources.items()})
    if PRIMARY in sources:
        by_lang = by_lang.sort_values(LABELS[PRIMARY])
    summary = summarise(sources)

    out_dir.mkdir(parents=True, exist_ok=True)
    by_lang.to_csv(out_dir / "robustness_by_language.csv")
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)

    print("=== per-language gap ===")
    print(by_lang.round(4).to_string())
    print("\n=== summary ===")
    print(summary.drop(columns=["slug"]).round(4).to_string(index=False))
    print(f"\nwrote {out_dir}/robustness_{{by_language.csv,summary.csv,table.tex}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument("--lexical-dir", type=Path, default=Path("results/lexical"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/robustness"))
    args = ap.parse_args()
    main(args.metrics_dir, args.lexical_dir, args.out_dir)
