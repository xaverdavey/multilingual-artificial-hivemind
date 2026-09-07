"""Per-language PCA grid over every language, drawn at final print size.

Appendix companion to the paper's Figure 1 (four representative languages):
the same projection for all 13 languages. Each panel fits a fresh 2-component
PCA on that language's pooled points (all 23 LLMs' generations + the human
answers), then draws LLM points as a blue cloud and human answers as red x
markers, matching Figure 1's style.
"""

import argparse
from pathlib import Path

import numpy as np

from experiments.run_metrics import load_model, load_human

EMBED_MODEL = "text-embedding-3-small"


def model_slugs(embed_dir: Path) -> list[str]:
    suffix = f"__{EMBED_MODEL}_llm.npy"
    return sorted(p.name[: -len(suffix)] for p in embed_dir.glob(f"*{suffix}"))


def draw(ax, llm_xy: np.ndarray, human_xy: np.ndarray, lang: str) -> None:
    ax.scatter(llm_xy[:, 0], llm_xy[:, 1], s=0.4, alpha=0.25,
               color="steelblue", linewidths=0, rasterized=True)
    ax.scatter(human_xy[:, 0], human_xy[:, 1], s=4, marker="x",
               color="crimson", linewidths=0.5)
    ax.set_title(lang, fontsize=7, pad=2)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out", type=Path,
                    default=Path(f"results/plots/pca_grid__{EMBED_MODEL}.png"))
    ap.add_argument("--width-in", type=float, default=5.5,
                    help="figure width in inches (NeurIPS text width)")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slugs = model_slugs(args.embed_dir)
    print(f"{len(slugs)} models found")
    loaded = [load_model(s, EMBED_MODEL, args.embed_dir) for s in slugs]
    human_emb, human_idx = load_human(slugs[0], EMBED_MODEL, args.embed_dir)

    langs = sorted(human_idx["language"].unique())
    rows = -(-len(langs) // args.cols)
    panel = args.width_in / args.cols
    fig, axes = plt.subplots(rows, args.cols,
                             figsize=(args.width_in, rows * panel * 0.9))

    for ax, lang in zip(axes.flat, langs):
        parts = [emb[(idx["language"] == lang).to_numpy()] for emb, idx in loaded]
        h_mask = (human_idx["language"] == lang).to_numpy()
        parts.append(human_emb[h_mask])
        pca = PCA(n_components=2, random_state=0).fit(np.vstack(parts))
        llm_xy = pca.transform(np.vstack(parts[:-1]))
        human_xy = pca.transform(human_emb[h_mask])
        draw(ax, llm_xy, human_xy, lang)
        var = pca.explained_variance_ratio_
        print(f"{lang}: llm n={len(llm_xy)}, human n={h_mask.sum()}, "
              f"var={var[0]:.1%}+{var[1]:.1%}")

    for ax in axes.flat[len(langs):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="steelblue", ms=3, label="LLM"),
        plt.Line2D([], [], marker="x", ls="", color="crimson", ms=4, label="Human"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8, w_pad=0.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=350)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
