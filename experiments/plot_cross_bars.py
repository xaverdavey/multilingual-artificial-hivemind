"""Cross-respondent F-test bar panels with a shared human baseline line.

Replaces the per-model and per-family-pair panels of the paper's Figure 2,
where the Human<->Human comparison does not vary along the x axis: repeating
the same red bar 23 times is redundant, so those panels draw the human side
as one dashed horizontal line (with a +/-SEM band) instead.

The per-model level is not in f_tests_cross__*.parquet; it is recomputed here
from the same per-(pair, prompt) data (inter__*.parquet) and the same human
per-prompt cross-respondent values, using run_metrics' own F-test helper.
The family-pair rows are validated against the stored parquet.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_metrics import (
    load_human, compute_intra, _f_test_row, _sig_marker,
)
from experiments.plot_pca_grid import model_slugs
from experiments.plot_respondent_matrix import family_of

EMBED_MODEL = "text-embedding-3-small"


def human_cross_values(embed_dir: Path) -> np.ndarray:
    """Per-(language, prompt) mean pairwise cosine across the ~3 human authors."""
    slug = model_slugs(embed_dir)[0]
    emb, idx = load_human(slug, EMBED_MODEL, embed_dir)
    return compute_intra("Human", emb, idx)["intra_sim"].dropna().to_numpy()


def per_model_rows(inter: pd.DataFrame, human: np.ndarray) -> pd.DataFrame:
    rows = []
    for m in sorted(set(inter["model_a"]) | set(inter["model_b"])):
        vals = inter.loc[(inter["model_a"] == m) | (inter["model_b"] == m),
                         "inter_sim"].dropna().to_numpy()
        rows.append(_f_test_row("LLM_cross_model vs Human_cross_respondent",
                                "model", m, "llm", vals, "human", human))
    return pd.DataFrame([r for r in rows if r is not None])


def per_family_pair_rows(inter: pd.DataFrame, human: np.ndarray,
                         families: dict[str, str]) -> pd.DataFrame:
    fam_a = inter["model_a"].map(families).fillna("Other")
    fam_b = inter["model_b"].map(families).fillna("Other")
    pair = np.minimum(fam_a, fam_b) + " || " + np.maximum(fam_a, fam_b)
    rows = []
    for p, sub in inter.assign(pair=pair).groupby("pair"):
        rows.append(_f_test_row("LLM_cross_model vs Human_cross_respondent",
                                "family_pair", p, "llm",
                                sub["inter_sim"].dropna().to_numpy(),
                                "human", human))
    return pd.DataFrame([r for r in rows if r is not None])


def draw(sub: pd.DataFrame, out: Path, figsize: tuple, label_pt: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = sub.sort_values("mean_a")
    x = np.arange(1, len(sub) + 1)  # x=0 is the Human bar
    se_a = (sub["std_a"] / np.sqrt(sub["n_a"])).to_numpy()
    h_mean = float(sub["mean_b"].iloc[0])
    h_sem = float(sub["std_b"].iloc[0] / np.sqrt(sub["n_b"].iloc[0]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([0], [h_mean], 0.62, yerr=[h_sem], capsize=2,
           color="crimson", label=r"Human$\leftrightarrow$Human")
    ax.bar(x, sub["mean_a"], 0.62, yerr=se_a, capsize=2,
           color="steelblue", label=r"LLM$\leftrightarrow$LLM")
    ax.axhline(h_mean, color="crimson", ls="--", lw=1.4, alpha=0.8, zorder=0)
    for xi, (_, r) in zip(x, sub.iterrows()):
        star = _sig_marker(r["p_means"])
        if star:
            ax.text(xi, r["mean_a"] + se_a[int(xi) - 1] + 0.006, star,
                    ha="center", va="bottom", fontsize=label_pt)
    ax.set_xticks(np.arange(len(sub) + 1))
    ax.set_xticklabels(["Human"] + sub["group"].tolist(),
                       rotation=45, ha="right", fontsize=label_pt)
    ax.tick_params(axis="y", labelsize=label_pt)
    ax.set_ylabel("Mean Cross-Respondent Cosine", fontsize=label_pt + 2)
    ax.set_ylim(0, max(sub["mean_a"] + se_a) + 0.06)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color="steelblue"),
               plt.Rectangle((0, 0), 1, 1, color="crimson")]
    fig.legend(handles, [r"LLM$\leftrightarrow$LLM", r"Human$\leftrightarrow$Human"],
               loc="upper center", ncol=2, fontsize=label_pt + 2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = ap.parse_args()

    inter = pd.read_parquet(args.metrics_dir / f"inter__{EMBED_MODEL}.parquet")
    human = human_cross_values(args.embed_dir)
    print(f"human cross-respondent: n={len(human)}, mean={human.mean():.4f}")

    fams = family_of(Path("experiments/models.yaml"))
    fp = per_family_pair_rows(inter, human, fams)

    # validate against the stored parquet before trusting the recomputation
    stored = pd.read_parquet(args.metrics_dir / f"f_tests_cross__{EMBED_MODEL}.parquet")
    stored = stored[stored["level"] == "family_pair"].set_index("group")
    merged = fp.set_index("group").join(stored, rsuffix="_stored").dropna(subset=["mean_a", "mean_a_stored"])
    err = (merged["mean_a"] - merged["mean_a_stored"]).abs().max()
    print(f"family-pair validation: {len(merged)} pairs, max |mean_a| diff = {err:.2e}")
    assert err < 1e-9, "recomputed family-pair means do not match stored F-tests"

    pm = per_model_rows(inter, human)
    draw(pm, args.out_dir / f"f_tests_cross_bars_model__{EMBED_MODEL}.png",
         figsize=(12.8, 6.4), label_pt=13)
    draw(fp, args.out_dir / f"f_tests_cross_bars_family_pair__{EMBED_MODEL}.png",
         figsize=(9.3, 6.9), label_pt=13)


if __name__ == "__main__":
    main()
