"""Rebuild the respondent-respondent similarity matrix at a legible size.

Each row/column is a *respondent* to the shared prompt set: one of the 23 LLMs,
or the pooled human annotators. Cell (i, j) is the mean cosine similarity between
a response from respondent i and a response from respondent j to the same prompt,
averaged over every prompt in every language.

  off-diagonal   two different respondents answering the same prompt
  LLM diagonal   the same model resampled at temperature 1.0
  Human diagonal two different humans answering the same prompt (the ~3 OASST2
                 siblings), so it is an off-diagonal quantity semantically --
                 there is no "resample the same human" condition

Two renderings, because one figure cannot do both jobs:
  full    23 LLMs + Human, sized for a two-column figure* -- every model visible
  family  5 families + Human, sized for a one-column figure -- each cell averages
          the model pairs across two families, using only pairs of *distinct*
          models so every cell means "two different respondents"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HUMAN = "Human"


def family_of(models_yaml: Path) -> dict[str, str]:
    import yaml
    return {e.get("display_name", e["id"]): e.get("family", "Other")
            for e in (yaml.safe_load(models_yaml.read_text()) or [])}


def family_matrix(mat: pd.DataFrame, fam: dict[str, str]) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse the model matrix to families, using distinct-respondent pairs only.

    Returns (family matrix, mean same-model-resampled value per family).
    """
    groups: dict[str, list[str]] = {}
    for m in mat.index:
        groups.setdefault(HUMAN if m == HUMAN else fam.get(m, "Other"), []).append(m)
    names = [g for g in groups if g != HUMAN] + [HUMAN]

    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            block = mat.loc[groups[a], groups[b]].to_numpy(dtype=float)
            if a == b:
                # strict off-diagonal: two *different* respondents of that family
                vals = block[~np.eye(len(block), dtype=bool)] if len(block) > 1 else block.ravel()
            else:
                vals = block.ravel()
            out.loc[a, b] = np.nanmean(vals)

    self_sim = pd.Series(
        {g: np.nanmean(np.diag(mat.loc[groups[g], groups[g]].to_numpy(dtype=float)))
         for g in names},
        name="same_model_resampled")
    return out, self_sim


def draw(mat: pd.DataFrame, out_path: Path, width_in: float, cell_pt: float,
         label_pt: float, boundaries: list[int] | None = None,
         show_values: bool = True, flipped: bool = True) -> None:
    r"""Single-hue sequential heatmap, drawn at the width it will occupy in print.

    `width_in` is the real figure width in inches (ACL: ~3.15 for a one-column
    figure, ~6.3 for a two-column figure*). Rendering at final size and including
    with width=\linewidth means no LaTeX downscaling, so the point sizes below are
    the point sizes a reader actually gets -- the original figure was authored
    large and then scaled to 0.85\linewidth, which is why it became unreadable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vals = mat.to_numpy(dtype=float)
    n = len(mat)
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.94))
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    im = ax.imshow(vals, vmin=vmin, vmax=vmax, cmap="Blues", aspect="equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
    cbar.set_label("Mean Cosine Similarity", fontsize=label_pt)
    cbar.ax.tick_params(labelsize=max(4.0, label_pt - 1))

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=label_pt)
    ax.set_yticklabels(mat.index, fontsize=label_pt)
    # the Human row/col is the comparison the figure exists to make
    for lbl in (ax.get_xticklabels(), ax.get_yticklabels()):
        for t in lbl:
            if t.get_text() == HUMAN:
                t.set_fontweight("bold")

    mid = vmin + 0.62 * (vmax - vmin)
    if show_values:
        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=cell_pt, color="white" if v >= mid else "#1a1a1a")

    # family separators + a heavier rule isolating Human; rows may be reversed
    # relative to columns, so horizontal rules mirror the vertical ones
    for b in (boundaries or []):
        ax.axvline(b - 0.5, color="white", lw=1.6)
        ax.axhline((n - b if flipped else b) - 0.5, color="white", lw=1.6)
    hr = list(mat.index).index(HUMAN)
    hc = list(mat.columns).index(HUMAN)
    ax.add_patch(Rectangle((-0.5, hr - 0.5), n, 1, fill=False, ec="#c0392b", lw=1.8, zorder=5))
    ax.add_patch(Rectangle((hc - 0.5, -0.5), 1, n, fill=False, ec="#c0392b", lw=1.8, zorder=5))

    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(csv: Path, models_yaml: Path, out_dir: Path, slug: str):
    mat = pd.read_csv(csv, index_col=0)
    fam = family_of(models_yaml)

    # order models by family so the block structure is visible; Human last
    models = [m for m in mat.index if m != HUMAN]
    order, boundaries = [], []
    for f in dict.fromkeys(fam.get(m, "Other") for m in models):
        order += [m for m in models if fam.get(m, "Other") == f]
        boundaries.append(len(order))
    order.append(HUMAN)
    # rows reversed relative to columns: self-pairs run bottom-left to
    # top-right, the x = y diagonal in axis convention
    full = mat.loc[order[::-1], order]

    # two-column figure*: every model, values at 5pt (legible in print, crisp on screen)
    draw(full, out_dir / f"respondent_matrix_full__{slug}.png",
         width_in=6.3, cell_pt=6.0, label_pt=7.0, boundaries=boundaries)

    famm, self_sim = family_matrix(mat, fam)
    # one-column figure: family summary, comfortably legible
    draw(famm.loc[famm.index[::-1]], out_dir / f"respondent_matrix_family__{slug}.png",
         width_in=3.15, cell_pt=10.0, label_pt=9.0)

    print("\n=== family matrix (x100) ===")
    print((famm * 100).round(1).to_string())
    print("\n=== same model resampled (x100) ===")
    print((self_sim * 100).round(1).to_string())
    print(f"\nHuman-Human {famm.loc[HUMAN, HUMAN]*100:.1f}   "
          f"lowest LLM-LLM cell {(famm.drop(index=HUMAN, columns=HUMAN).to_numpy().min())*100:.1f}   "
          f"max LLM-Human {famm.loc[HUMAN].drop(HUMAN).max()*100:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = ap.parse_args()
    slug = args.embed_model.replace("/", "_")
    main(args.metrics_dir / f"corr_matrix__{slug}.csv", args.models_yaml, args.out_dir, slug)
