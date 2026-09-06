"""Figure for the linguistic-property analysis.

Two stacked panels, drawn at one-column width so no LaTeX downscaling is needed:

  (a) cell-level scatter of the homogenization gap against model competence,
      one point per (model, language) -- the level the effect actually lives
      at. Plotted per language it would vanish: the language-level Spearman is
      non-significant, because competence varies far more across models within
      a language than it does between languages.
  (b) standardised coefficients with 95% CIs from the crossed-effects model,
      fitted separately in each embedding space -- shows which properties
      survive both when all four compete and when the embedder changes.

Panel (b) is the claim; panel (a) is the evidence a reader can check by eye.
Panel (a) uses the primary embedder only; three overlapping clouds of 299
points are unreadable, and the cross-embedder comparison is panel (b)'s job.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRETTY = {
    "lid_match_rate": "Model competence\n(GlotLID match rate)",
    "log_wiki": "Resource level\n(log$_{10}$ Wikipedia articles)",
    "fertility_vs_en": "Tokenization fertility\n(vs. English, FLORES-200)",
    "ttr": "Morphological complexity\n(type-token ratio)",
}
ORDER = ["lid_match_rate", "log_wiki", "fertility_vs_en", "ttr"]

# Results directory -> series label for each extra embedding space.
ALT_LABELS = {"linguistic_bge": "BGE-M3", "linguistic_qwen": "Qwen3-Emb"}

# Three-series categorical palette; validated (worst CVD dE 20.9 protan).
SERIES_COLORS = ["#2b6cb0", "#c05621", "#6b46c1"]


def main(table: Path, out_path: Path, width_in: float, alt_tables: list[Path] | None = None):
    import matplotlib.pyplot as plt
    from scipy import stats
    from experiments.run_linguistic import mixed_model_coefs, PREDICTORS

    tables = {"OpenAI": table}
    for alt in alt_tables or []:
        if Path(alt).exists():
            tables[ALT_LABELS.get(Path(alt).parent.name, Path(alt).parent.name)] = Path(alt)
    df = pd.read_parquet(table)
    cells = df.dropna(subset=["gap", "lid_match_rate"])
    fits = {name: mixed_model_coefs(pd.read_parquet(t), PREDICTORS).set_index("predictor").loc[ORDER]
            for name, t in tables.items()}

    def style(ax):
        ax.tick_params(labelsize=6.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.7)
            ax.spines[side].set_color("#888888")

    # --- (a) gap vs model competence, one point per (model, language) --------
    # No on-plot title: the stats printed below belong in the LaTeX subcaption.
    fig, ax1 = plt.subplots(figsize=(width_in, width_in * 0.82))
    ax1.scatter(cells["lid_match_rate"], cells["gap"], s=7, color=SERIES_COLORS[0],
                alpha=0.34, linewidth=0, zorder=3)
    slope, intercept, r, p_, _ = stats.linregress(cells["lid_match_rate"], cells["gap"])
    xs = np.linspace(cells["lid_match_rate"].min(), cells["lid_match_rate"].max(), 20)
    ax1.plot(xs, intercept + slope * xs, color="#c0392b", lw=1.2, zorder=4)
    rho, prho = stats.spearmanr(cells["lid_match_rate"], cells["gap"])
    ax1.set_xlabel("GlotLID match rate", fontsize=7.5)
    ax1.set_ylabel("homogenization gap\n(LLM $-$ human intra-sim)", fontsize=7.5)
    ax1.grid(alpha=0.22, lw=0.5)
    style(ax1)
    a_path = out_path.with_name(out_path.stem + "_competence.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a_path}")

    # --- (b) standardised coefficients, one series per embedding space ------
    fig, ax2 = plt.subplots(figsize=(width_in, width_in * 0.82))
    base = np.arange(len(ORDER))[::-1].astype(float)
    ax2.axvline(0, color="#999999", lw=0.8, zorder=1)
    offsets = np.linspace(0.17, -0.17, len(fits)) if len(fits) > 1 else [0.0]
    for (name, f), dy, col in zip(fits.items(), offsets, SERIES_COLORS):
        y = base + dy
        sig = (f["p_value"] < 0.05).to_numpy()
        ax2.hlines(y, f["lo"], f["hi"], color=col, lw=1.5, zorder=2)
        # filled = significant at 0.05, hollow = not: significance is not colour-alone
        ax2.scatter(f["coef"][sig], y[sig], s=26, color=col, zorder=3, label=name)
        ax2.scatter(f["coef"][~sig], y[~sig], s=26, facecolor="white",
                    edgecolor=col, linewidth=1.1, zorder=3)
    ax2.set_yticks(base)
    ax2.set_yticklabels([PRETTY[p] for p in ORDER], fontsize=6.5)
    ax2.set_xlabel("standardised coefficient (95% CI)", fontsize=7.5)
    if len(fits) > 1:
        ax2.legend(fontsize=6, frameon=False, loc="upper left", handletextpad=0.3,
                   borderaxespad=0.2)
    ax2.grid(axis="x", alpha=0.22, lw=0.5)
    style(ax2)
    b_path = out_path.with_name(out_path.stem + "_coefs.png")
    fig.savefig(b_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {b_path}")

    ptxt = "p < 0.001" if prho < 1e-3 else f"p = {prho:.3f}"
    print(f"\nfor the subcaption of (a): Spearman rho = {rho:.2f}, {ptxt}, n = {len(cells)} cells")
    for name, f in fits.items():
        print(f"\n-- {name} --"); print(f.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path,
                    default=Path("results/linguistic/linguistic_table.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/plots/linguistic_gap.png"))
    ap.add_argument("--width-in", type=float, default=3.15)
    ap.add_argument("--alt-tables", type=Path, nargs="*",
                    default=[Path("results/linguistic_bge/linguistic_table.parquet"),
                             Path("results/linguistic_qwen/linguistic_table.parquet")],
                    help="Further embedding spaces to overlay in panel (b); skipped if absent")
    args = ap.parse_args()
    main(args.table, args.out, args.width_in, args.alt_tables)
