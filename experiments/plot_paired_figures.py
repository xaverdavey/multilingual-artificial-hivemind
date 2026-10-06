"""Redraw the paper's bar and heatmap figures from the paired prompt-level analysis.

The original figures took their significance stars and error bars from the
unpaired F-tests in run_metrics. For the intra-model comparison that test sees
one value per prompt on each side, which is fine; for the inter-model comparison
its LLM side kept one value per (model pair, prompt), so every prompt entered
253 times and the stars and whiskers were far too optimistic.

This script draws the same panels from results/prompt_tests (run_prompt_tests):
  * bar heights and heatmap cells are means over prompts of the per-prompt values,
    with the LLM side averaged over the models or model pairs in the group first;
  * whiskers are standard errors over prompts on both sides;
  * stars are Holm-adjusted Wilcoxon signed-rank p-values on the per-prompt
    differences, adjusted across all groups at the same level:
    * p < 0.05, ** p < 0.01, *** p < 0.001 (two-sided; direction from the bars).

Outputs (to --out-dir), named like the paper's graphs_updated/ and graphs/ files:
  f_tests_bars_{general,family,language,model}__{slug}.png
  f_tests_cross_bars_{general,model,language,family_pair}__{slug}.png
  f_tests_heatmap_model_transposed__{slug}.png
  f_tests_cross_heatmap__{slug}.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BLUE, RED = "steelblue", "crimson"
P_COL = "p_wilcoxon_holm"


def sig(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _save(fig, out: Path, dpi: int = 150) -> None:
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _legend(fig, labels, pt: float) -> None:
    import matplotlib.pyplot as plt
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=RED)]
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=pt + 2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))


def paired_bars(sub: pd.DataFrame, out: Path, figsize, pt: float,
                ylabel: str, labels) -> None:
    """Two bars per group (LLM, Human), both with whiskers over prompts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = sub.sort_values("mean_llm")
    x = np.arange(len(sub))
    w = 0.4
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, sub["mean_llm"], w, yerr=sub["se_llm"], capsize=2, color=BLUE)
    ax.bar(x + w / 2, sub["mean_human"], w, yerr=sub["se_human"], capsize=2, color=RED)
    tops = np.maximum(sub["mean_llm"] + sub["se_llm"], sub["mean_human"] + sub["se_human"]).to_numpy()
    for xi, top, r in zip(x, tops, sub.itertuples()):
        star = sig(getattr(r, P_COL))
        if star:
            ax.text(xi, top + 0.008, star, ha="center", va="bottom", fontsize=pt)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["group"], rotation=45, ha="right", fontsize=pt)
    ax.tick_params(axis="y", labelsize=pt)
    ax.set_ylabel(ylabel, fontsize=pt + 2)
    ax.set_ylim(0, float(tops.max()) + 0.09)
    ax.grid(axis="y", alpha=0.25)
    _legend(fig, labels, pt)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out)


def baseline_bars(sub: pd.DataFrame, out: Path, figsize, pt: float,
                  ylabel: str, labels) -> None:
    """One LLM bar per group; the human side, constant along the axis, is drawn
    once as the first bar with a dashed reference line at its height."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = sub.sort_values("mean_llm")
    x = np.arange(1, len(sub) + 1)
    h_mean = float(sub["mean_human"].iloc[0])
    h_se = float(sub["se_human"].iloc[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([0], [h_mean], 0.62, yerr=[h_se], capsize=2, color=RED)
    ax.bar(x, sub["mean_llm"], 0.62, yerr=sub["se_llm"], capsize=2, color=BLUE)
    ax.axhline(h_mean, color=RED, ls="--", lw=1.4, alpha=0.8, zorder=0)
    tops = (sub["mean_llm"] + sub["se_llm"]).to_numpy()
    for xi, top, r in zip(x, tops, sub.itertuples()):
        star = sig(getattr(r, P_COL))
        if star:
            ax.text(xi, top + 0.006, star, ha="center", va="bottom", fontsize=pt)
    ax.set_xticks(np.arange(len(sub) + 1))
    ax.set_xticklabels(["Human"] + sub["group"].tolist(), rotation=45, ha="right", fontsize=pt)
    ax.tick_params(axis="y", labelsize=pt)
    ax.set_ylabel(ylabel, fontsize=pt + 2)
    ax.set_ylim(0, max(float(tops.max()), h_mean + h_se) + 0.06)
    ax.grid(axis="y", alpha=0.25)
    _legend(fig, labels, pt)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out)


def heatmap(sub: pd.DataFrame, out: Path, rows: str, cols: str, figsize,
            cbar_label: str, tick_pt: float, ann_pt: float, cbar_pt: float,
            title: str | None = None, dpi: int = 150) -> None:
    """Rows and columns ordered by their mean effect; cells annotated `effect star`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eff = sub.pivot(index=rows, columns=cols, values="mean_gap")
    pv = sub.pivot(index=rows, columns=cols, values=P_COL)
    row_order = eff.mean(axis=1).sort_values().index.tolist()
    col_order = eff.mean(axis=0).sort_values().index.tolist()
    eff = eff.reindex(index=row_order, columns=col_order)
    pv = pv.reindex(index=row_order, columns=col_order)

    n_rows, n_cols = eff.shape
    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(np.abs(eff.values)))
    im = ax.imshow(eff.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=cbar_pt)
    cb.ax.tick_params(labelsize=cbar_pt)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(eff.columns, rotation=45, ha="right", fontsize=tick_pt)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(eff.index, fontsize=tick_pt)
    for i in range(n_rows):
        for j in range(n_cols):
            e = eff.values[i, j]
            if not np.isfinite(e):
                continue
            color = "white" if abs(e) > vmax * 0.55 else "black"
            ax.text(j, i, f"{e:+.2f}{sig(pv.values[i, j])}", ha="center", va="center",
                    fontsize=ann_pt, color=color)
    if title:
        ax.set_title(title, fontsize=tick_pt + 2)
    fig.tight_layout()
    _save(fig, out, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default="text-embedding-3-small")
    ap.add_argument("--prompt-tests-dir", type=Path, default=Path("results/prompt_tests"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/plots/paired"))
    args = ap.parse_args()

    df = pd.read_parquet(args.prompt_tests_dir / f"prompt_tests__{args.slug}.parquet")
    need = {"mean_llm", "mean_human", "se_llm", "se_human", P_COL}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{missing} not in the prompt-tests parquet; rerun run_prompt_tests")

    def pick(comparison: str, level: str) -> pd.DataFrame:
        sub = df[(df["comparison"] == comparison) & (df["level"] == level)]
        if sub.empty:
            raise SystemExit(f"no rows for {comparison}/{level}")
        return sub

    out, slug = args.out_dir, args.slug
    intra_lab = ("LLM", "Human")
    cross_lab = (r"LLM$\leftrightarrow$LLM", r"Human$\leftrightarrow$Human")
    y_intra, y_cross = "Mean Intra-Similarity", "Mean Cross-Respondent Cosine"

    paired_bars(pick("intra", "general"), out / f"f_tests_bars_general__{slug}.png",
                (4.1, 5.5), 13, y_intra, intra_lab)
    paired_bars(pick("intra", "family"), out / f"f_tests_bars_family__{slug}.png",
                (4.4, 5.5), 13, y_intra, intra_lab)
    paired_bars(pick("intra", "language"), out / f"f_tests_bars_language__{slug}.png",
                (8.1, 5.5), 13, y_intra, intra_lab)
    paired_bars(pick("intra", "model"), out / f"f_tests_bars_model__{slug}.png",
                (12.8, 6.7), 13, y_intra, intra_lab)

    paired_bars(pick("cross", "general"), out / f"f_tests_cross_bars_general__{slug}.png",
                (4.4, 5.35), 13, y_cross, cross_lab)
    paired_bars(pick("cross", "language"), out / f"f_tests_cross_bars_language__{slug}.png",
                (8.1, 5.2), 13, y_cross, cross_lab)
    baseline_bars(pick("cross", "model"), out / f"f_tests_cross_bars_model__{slug}.png",
                  (12.8, 6.4), 13, y_cross, cross_lab)
    baseline_bars(pick("cross", "family_pair"), out / f"f_tests_cross_bars_family_pair__{slug}.png",
                  (9.3, 6.9), 13, y_cross, cross_lab)

    heatmap(pick("intra", "model_language"), out / f"f_tests_heatmap_model_transposed__{slug}.png",
            rows="language", cols="group", figsize=(16.6, 8.3),
            cbar_label="Mean(LLM) − Mean(Human) Intra-Similarity  (red = LLM less diverse)",
            tick_pt=18, ann_pt=8.5, cbar_pt=12, dpi=300)
    heatmap(pick("cross", "family_pair_language"), out / f"f_tests_cross_heatmap__{slug}.png",
            rows="group", cols="language", figsize=(10.4, 9.3),
            cbar_label="mean(LLM↔LLM) − mean(Human↔Human)  (red = LLM hivemind effect)",
            tick_pt=11, ann_pt=8.5, cbar_pt=11,
            title="Inter-model gap: family pair × language  "
                  "(+ = LLMs agree more with each other than humans do)")


if __name__ == "__main__":
    main()
