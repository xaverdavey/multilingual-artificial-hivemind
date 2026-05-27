"""Compute intra- and inter-model similarity metrics from saved embeddings.

Config JSON maps display name → file slug (the prefix used in embeddings/):
    {"Qwen 0.5B": "Qwen_Qwen2.5-0.5B-Instruct", "GPT-4o": "gpt-4o-2024-11", ...}

Metrics computed for all models in the config:
  Intra-model similarity (N values, one per model):
    Per prompt: mean pairwise cosine similarity across all samples from that model.
    Reported as mean ± std over all prompts.

  Inter-model similarity (N-choose-2 values, one per model pair):
    Per prompt: E_{a~A, b~B}[cos(a,b)] — the expected cosine between a random sample
    from model A and a random sample from model B.  Computed exactly as
    dot(sum_A, sum_B) / (K_A * K_B), which is the closed-form of the bootstrap mean.
    Reported as mean ± std over all prompts.

Outputs (written to --out-dir):
  intra__{embed_slug}.parquet    per (model, language, prompt_id)
  inter__{embed_slug}.parquet    per (model_a, model_b, language, prompt_id)
  summary__{embed_slug}.json     aggregate scalars
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def mean_pairwise_cosine(emb: np.ndarray) -> float:
    """Mean cosine similarity over all i<j pairs for L2-normalised rows.

    Uses identity: mean_cos = (||sum||^2 - n) / (n*(n-1))  — O(n*d) not O(n^2*d).
    """
    n = len(emb)
    if n < 2:
        return float("nan")
    s = emb.sum(axis=0)
    return float((np.dot(s, s) - n) / (n * (n - 1)))


def expected_cross_cosine(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """E_{a~A, b~B}[cos(a,b)] for L2-normalised embeddings.

    Closed form of the bootstrap mean: dot(sum_A, sum_B) / (K_A * K_B).
    Sampling one embedding uniformly from A and one from B and averaging
    converges to this value.
    """
    return float(np.dot(emb_a.sum(axis=0), emb_b.sum(axis=0)) / (len(emb_a) * len(emb_b)))


# ---------------------------------------------------------------------------
# Per-prompt computations
# ---------------------------------------------------------------------------

def _build_lookup(idx: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """Map (language, prompt_id) -> row-index array for fast slicing."""
    return {key: grp.index.to_numpy() for key, grp in idx.groupby(["language", "prompt_id"])}


def compute_intra(name: str, emb: np.ndarray, idx: pd.DataFrame) -> pd.DataFrame:
    idx = idx.reset_index(drop=True)
    rows = []
    for (lang, prompt_id), row_ids in _build_lookup(idx).items():
        rows.append({
            "model": name,
            "language": lang,
            "prompt_id": prompt_id,
            "intra_sim": mean_pairwise_cosine(emb[row_ids]),
            "n_samples": len(row_ids),
        })
    return pd.DataFrame(rows)


def compute_inter_pair(
    name_a: str, emb_a: np.ndarray, idx_a: pd.DataFrame,
    name_b: str, emb_b: np.ndarray, idx_b: pd.DataFrame,
) -> pd.DataFrame:
    idx_a = idx_a.reset_index(drop=True)
    idx_b = idx_b.reset_index(drop=True)
    map_a = _build_lookup(idx_a)
    map_b = _build_lookup(idx_b)
    rows = []
    for key in sorted(set(map_a) & set(map_b)):
        lang, prompt_id = key
        rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "language": lang,
            "prompt_id": prompt_id,
            "inter_sim": expected_cross_cosine(emb_a[map_a[key]], emb_b[map_b[key]]),
        })
    return pd.DataFrame(rows)


def build_corr_matrix(
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]],
    intra_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    human_emb: np.ndarray,
    human_idx: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Build (N+1)×(N+1) mean-cosine similarity matrix for all LLMs + Human.

    Reuses already-computed intra_df / inter_df for LLM entries; only computes
    the human diagonal and human-vs-LLM row/col from scratch.
    """
    names = list(loaded.keys())
    all_names = names + ["Human"]
    n = len(all_names)
    matrix = np.full((n, n), np.nan)

    # LLM diagonal from intra_df
    for i, name in enumerate(names):
        vals = intra_df.loc[intra_df["model"] == name, "intra_sim"].dropna()
        matrix[i, i] = float(vals.mean())

    # Human diagonal
    human_intra = compute_intra("Human", human_emb, human_idx)
    matrix[-1, -1] = float(human_intra["intra_sim"].dropna().mean())

    # LLM-LLM off-diagonal from inter_df (already computed, just look up)
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                continue
            mask = (
                ((inter_df["model_a"] == na) & (inter_df["model_b"] == nb)) |
                ((inter_df["model_a"] == nb) & (inter_df["model_b"] == na))
            )
            matrix[i, j] = matrix[j, i] = float(inter_df.loc[mask, "inter_sim"].dropna().mean())

    # Human vs each LLM
    h_i = n - 1
    for i, (name, (emb, idx)) in enumerate(loaded.items()):
        pair_df = compute_inter_pair("Human", human_emb, human_idx, name, emb, idx)
        val = float(pair_df["inter_sim"].dropna().mean())
        matrix[h_i, i] = matrix[i, h_i] = val

    return all_names, matrix


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_model(
    slug: str, embed_model: str, embed_dir: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load (and L2-normalise) LLM embeddings + index for one model slug."""
    embed_slug = embed_model.replace("/", "_")
    stem = f"{slug}__{embed_slug}"
    npy_path = embed_dir / f"{stem}_llm.npy"
    idx_path = embed_dir / f"{stem}_llm_index.parquet"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {npy_path}\n"
            f"Run: python -m experiments.run_embeddings --model {slug!r} --embed-model {embed_model}"
        )
    emb = np.load(npy_path)
    idx = pd.read_parquet(idx_path)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    return emb, idx


def load_human(
    slug: str, embed_model: str, embed_dir: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load (and L2-normalise) human reference embeddings + index for one slug."""
    embed_slug = embed_model.replace("/", "_")
    stem = f"{slug}__{embed_slug}"
    emb = np.load(embed_dir / f"{stem}_human.npy")
    idx = pd.read_parquet(embed_dir / f"{stem}_human_index.parquet")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    return emb, idx


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _model_colors(n: int) -> list:
    import matplotlib.cm as cm
    palette = list(cm.get_cmap("tab20").colors) + list(cm.get_cmap("tab20b").colors)
    return [palette[i % len(palette)] for i in range(n)]


def plot_pca_by_model(
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]],
    embed_slug: str,
    plot_dir: Path,
    human_emb: np.ndarray | None = None,
    human_idx: pd.DataFrame | None = None,
) -> None:
    """One PNG per language: all models + Human colored by name, legend outside."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    include_human = human_emb is not None and human_idx is not None
    n_traces = len(loaded) + (1 if include_human else 0)
    colors = _model_colors(n_traces)
    all_langs = sorted({lang for _, idx in loaded.values() for lang in idx["language"].unique()})

    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating per-language PCA (by model) — {len(all_langs)} languages...")

    for lang in all_langs:
        parts = [
            emb[(idx["language"] == lang).to_numpy()]
            for _, (emb, idx) in loaded.items()
            if (idx["language"] == lang).any()
        ]
        if include_human:
            h_mask = (human_idx["language"] == lang).to_numpy()
            parts.append(human_emb[h_mask])
        pca = PCA(n_components=2, random_state=0).fit(np.vstack(parts))

        fig, ax = plt.subplots(figsize=(10, 7))
        for i, (name, (emb, idx)) in enumerate(loaded.items()):
            mask = (idx["language"] == lang).to_numpy()
            if not mask.any():
                continue
            xy = pca.transform(emb[mask])
            ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.35, color=colors[i], label=name)

        if include_human:
            xy_h = pca.transform(human_emb[h_mask])
            ax.scatter(xy_h[:, 0], xy_h[:, 1], s=20, marker="x", alpha=1.0,
                       color="black", linewidths=1.0, label="Human")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"PCA by model — {lang}")
        ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1),
                  loc="upper left", borderaxespad=0)
        plt.tight_layout()
        out_path = plot_dir / f"pca_by_model__{embed_slug}__{lang}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path}")

print("hi")

def plot_corr_matrix(
    labels: list[str],
    matrix: np.ndarray,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """Static heatmap of the agent-agent similarity matrix."""
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.75), max(8, n * 0.7)))
    vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean cosine similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if not np.isnan(v):
                pct = int(round(v * 100))
                txt_color = "white" if v >= 0.80 else "black"
                ax.text(j, i, f"{pct}", ha="center", va="center",
                        fontsize=16, color=txt_color)
    ax.set_title("Agent-Agent Similarity Matrix (mean cosine, averaged over all prompts & languages)")
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"corr_matrix__{embed_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_human_vs_llm(
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]],
    human_emb: np.ndarray,
    human_idx: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """One PNG with per-language subplots: pooled LLM cloud vs human reference points."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    all_langs = sorted({lang for _, idx in loaded.values() for lang in idx["language"].unique()})
    ncols = 4
    nrows = math.ceil(len(all_langs) / ncols)

    all_llm_emb = np.vstack([emb for emb, _ in loaded.values()])
    all_llm_lang = pd.concat(
        [idx["language"].reset_index(drop=True) for _, idx in loaded.values()],
        ignore_index=True,
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = axes.flatten()

    print(f"\nGenerating human vs LLM PCA ({len(all_langs)} languages)...")

    for i, lang in enumerate(all_langs):
        ax = axes[i]
        llm_mask = (all_llm_lang == lang).to_numpy()
        h_mask = (human_idx["language"] == lang).to_numpy()

        pca = PCA(n_components=2, random_state=0).fit(
            np.vstack([all_llm_emb[llm_mask], human_emb[h_mask]])
        )
        xy_llm = pca.transform(all_llm_emb[llm_mask])
        xy_h = pca.transform(human_emb[h_mask])

        ax.scatter(xy_llm[:, 0], xy_llm[:, 1], s=3, alpha=0.2, color="steelblue",
                   label=f"LLM (n={llm_mask.sum()})")
        ax.scatter(xy_h[:, 0], xy_h[:, 1], s=40, marker="x", color="crimson",
                   linewidths=1.2, label=f"human (n={h_mask.sum()})")
        ax.set_title(lang, fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc="best")

    for j in range(len(all_langs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-language PCA: all LLMs (pooled) vs human responses", fontsize=13)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"pca_human_vs_llm__{embed_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Interactive plots (Plotly) — requires --raw-parquet
# ---------------------------------------------------------------------------

def _wrap_html(text: str, width: int = 80) -> str:
    """Wrap text at word boundaries, inserting <br> tags for Plotly hover."""
    import textwrap
    return "<br>".join(textwrap.wrap(str(text), width=width))


def _hex_colors(n: int) -> list[str]:
    import matplotlib.colors as mcolors
    return [mcolors.to_hex(c) for c in _model_colors(n)]


def _llm_text_for_idx(slug: str, sub_idx: pd.DataFrame, gen_df: pd.DataFrame) -> tuple[list, list]:
    """Return (prompts, responses) aligned with sub_idx rows, joined from gen_df."""
    model_id = next((m for m in gen_df["model"].unique() if m.replace("/", "_") == slug), None)
    if model_id is None:
        n = len(sub_idx)
        return [""] * n, [""] * n
    model_gen = gen_df[gen_df["model"] == model_id][
        ["prompt_id", "sample_idx", "prompt_text", "response_text"]
    ]
    merged = sub_idx.reset_index(drop=True).merge(model_gen, on=["prompt_id", "sample_idx"], how="left")
    return (
        merged["prompt_text"].fillna("").str[:300].apply(_wrap_html).tolist(),
        merged["response_text"].fillna("").str[:400].apply(_wrap_html).tolist(),
    )


def _human_text_df(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Build (prompt_id, language, human_idx, prompt_text, response_text) for human refs."""
    return (
        gen_df.drop_duplicates("prompt_id")[
            ["prompt_id", "language", "prompt_text", "human_responses"]
        ]
        .explode("human_responses")
        .rename(columns={"human_responses": "response_text"})
        .assign(human_idx=lambda d: d.groupby("prompt_id").cumcount())
        .reset_index(drop=True)
    )


def plot_corr_matrix_html(
    labels: list[str],
    matrix: np.ndarray,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """Interactive Plotly heatmap of the agent-agent similarity matrix."""
    import plotly.graph_objects as go

    n = len(labels)
    fig = go.Figure(go.Heatmap(
        z=matrix.tolist(),
        x=labels,
        y=labels,
        colorscale="Blues",
        zmin=float(np.nanmin(matrix)), zmax=float(np.nanmax(matrix)),
        text=[[f"{int(round(matrix[i, j] * 100))}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>similarity: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Agent-Agent Similarity Matrix (mean cosine, averaged over all prompts & languages)",
        width=max(800, n * 42),
        height=max(700, n * 40),
        xaxis=dict(tickangle=45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"corr_matrix_interactive__{embed_slug}.html"
    fig.write_html(str(out_path))
    print(f"  {out_path}")


def plot_pca_by_model_html(
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]],
    config: dict[str, str],
    human_emb: np.ndarray,
    human_idx: pd.DataFrame,
    gen_df: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """One HTML per language: all models + Human as interactive colored traces."""
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    names = list(loaded.keys()) + ["Human"]
    colors = _hex_colors(len(names))
    ht = _human_text_df(gen_df)
    all_langs = sorted({lang for _, idx in loaded.values() for lang in idx["language"].unique()})

    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating interactive PCA by model HTML — {len(all_langs)} languages...")

    for lang in all_langs:
        llm_parts = [
            emb[(idx["language"] == lang).to_numpy()]
            for _, (emb, idx) in loaded.items()
            if (idx["language"] == lang).any()
        ]
        h_mask = (human_idx["language"] == lang).to_numpy()
        pca = PCA(n_components=2, random_state=0).fit(
            np.vstack(llm_parts + [human_emb[h_mask]])
        )

        fig = go.Figure()
        for i, (name, (emb, idx)) in enumerate(loaded.items()):
            mask = (idx["language"] == lang).to_numpy()
            if not mask.any():
                continue
            xy = pca.transform(emb[mask])
            sub_idx = idx[mask].reset_index(drop=True)
            prompts, responses = _llm_text_for_idx(config[name], sub_idx, gen_df)
            fig.add_trace(go.Scatter(
                x=xy[:, 0], y=xy[:, 1],
                mode="markers",
                name=name,
                marker=dict(size=4, color=colors[i], opacity=0.5),
                customdata=list(zip(sub_idx["prompt_id"], prompts, responses)),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "prompt_id: %{customdata[0]}<br><br>"
                    "<b>Prompt:</b> %{customdata[1]}<br><br>"
                    "<b>Response:</b> %{customdata[2]}"
                    "<extra></extra>"
                ),
            ))

        xy_h = pca.transform(human_emb[h_mask])
        sub_h = human_idx[h_mask].reset_index(drop=True)
        h_merged = sub_h.merge(
            ht[["prompt_id", "human_idx", "prompt_text", "response_text"]],
            on=["prompt_id", "human_idx"], how="left",
        )
        fig.add_trace(go.Scatter(
            x=xy_h[:, 0], y=xy_h[:, 1],
            mode="markers",
            name="Human",
            marker=dict(size=8, symbol="x", color=colors[-1], opacity=0.9,
                        line=dict(width=1.5)),
            customdata=list(zip(
                sub_h["prompt_id"],
                h_merged["prompt_text"].fillna("").str[:300].apply(_wrap_html),
                h_merged["response_text"].fillna("").str[:400].apply(_wrap_html),
            )),
            hovertemplate=(
                "<b>Human</b><br>"
                "prompt_id: %{customdata[0]}<br><br>"
                "<b>Prompt:</b> %{customdata[1]}<br><br>"
                "<b>Response:</b> %{customdata[2]}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=f"PCA by model — {lang}",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
            width=1100, height=700,
        )
        out_path = plot_dir / f"pca_by_model_interactive__{embed_slug}__{lang}.html"
        fig.write_html(str(out_path))
        print(f"  {out_path}")


def plot_human_vs_llm_html(
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]],
    config: dict[str, str],
    human_emb: np.ndarray,
    human_idx: pd.DataFrame,
    gen_df: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """One HTML per language: pooled LLM cloud vs human, hover shows text + model."""
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    ht = _human_text_df(gen_df)

    # precompute pooled LLM arrays with per-row text — done once, reused per language
    llm_emb_parts, llm_lang_parts = [], []
    model_labels, prompt_ids, prompts_pool, responses_pool = [], [], [], []
    for name, (emb, idx) in loaded.items():
        llm_emb_parts.append(emb)
        llm_lang_parts.append(idx["language"].reset_index(drop=True))
        model_labels.extend([name] * len(idx))
        p, r = _llm_text_for_idx(config[name], idx, gen_df)
        prompt_ids.extend(idx["prompt_id"].tolist())
        prompts_pool.extend(p)
        responses_pool.extend(r)

    all_llm_emb = np.vstack(llm_emb_parts)
    all_llm_lang = pd.concat(llm_lang_parts, ignore_index=True).to_numpy()
    model_labels = np.array(model_labels)
    prompt_ids = np.array(prompt_ids)
    prompts_pool = np.array(prompts_pool)
    responses_pool = np.array(responses_pool)

    all_langs = sorted({lang for _, idx in loaded.values() for lang in idx["language"].unique()})
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating interactive human vs LLM PCA — {len(all_langs)} languages...")

    for lang in all_langs:
        llm_mask = all_llm_lang == lang
        h_mask = (human_idx["language"] == lang).to_numpy()

        pca = PCA(n_components=2, random_state=0).fit(
            np.vstack([all_llm_emb[llm_mask], human_emb[h_mask]])
        )
        xy_llm = pca.transform(all_llm_emb[llm_mask])
        xy_h = pca.transform(human_emb[h_mask])

        sub_h = human_idx[h_mask].reset_index(drop=True)
        h_merged = sub_h.merge(
            ht[["prompt_id", "human_idx", "prompt_text", "response_text"]],
            on=["prompt_id", "human_idx"], how="left",
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xy_llm[:, 0], y=xy_llm[:, 1],
            mode="markers",
            name="LLM (pooled)",
            marker=dict(size=3, color="steelblue", opacity=0.25),
            customdata=list(zip(
                model_labels[llm_mask],
                prompt_ids[llm_mask],
                prompts_pool[llm_mask],
                responses_pool[llm_mask],
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "prompt_id: %{customdata[1]}<br><br>"
                "<b>Prompt:</b> %{customdata[2]}<br><br>"
                "<b>Response:</b> %{customdata[3]}"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=xy_h[:, 0], y=xy_h[:, 1],
            mode="markers",
            name="Human",
            marker=dict(size=8, symbol="x", color="crimson", opacity=0.9,
                        line=dict(width=1.5)),
            customdata=list(zip(
                sub_h["prompt_id"],
                h_merged["prompt_text"].fillna("").str[:300].apply(_wrap_html),
                h_merged["response_text"].fillna("").str[:400].apply(_wrap_html),
            )),
            hovertemplate=(
                "<b>Human</b><br>"
                "prompt_id: %{customdata[0]}<br><br>"
                "<b>Prompt:</b> %{customdata[1]}<br><br>"
                "<b>Response:</b> %{customdata[2]}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=f"Human vs LLM (pooled) — {lang}",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
            width=1000, height=650,
        )
        out_path = plot_dir / f"pca_human_vs_llm_interactive__{embed_slug}__{lang}.html"
        fig.write_html(str(out_path))
        print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Path, embed_model: str, embed_dir: Path, out_dir: Path,
         plot_dir: Path | None = None, raw_parquet: Path | None = None):
    config: dict[str, str] = json.loads(config_path.read_text())
    if not config:
        raise ValueError("Config is empty — add at least one model.")

    print(f"Models ({len(config)}) with embed_model={embed_model}:")
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name, slug in config.items():
        print(f"  {name!r}  ←  {slug}")
        loaded[name] = load_model(slug, embed_model, embed_dir)

    t0 = time.time()
    embed_slug = embed_model.replace("/", "_")
    names = list(loaded.keys())

    # --- Intra-model ---
    print("\nComputing intra-model similarities...")
    intra_df = pd.concat(
        [compute_intra(name, emb, idx) for name, (emb, idx) in loaded.items()],
        ignore_index=True,
    )

    # --- Inter-model (pairwise) ---
    pairs = list(itertools.combinations(names, 2))
    print(f"Computing inter-model similarities ({len(pairs)} pair(s))...")
    if pairs:
        inter_df = pd.concat(
            [
                compute_inter_pair(
                    a, *loaded[a],
                    b, *loaded[b],
                )
                for a, b in pairs
            ],
            ignore_index=True,
        )
    else:
        inter_df = pd.DataFrame(columns=["model_a", "model_b", "language", "prompt_id", "inter_sim"])

    # --- Save parquets ---
    out_dir.mkdir(parents=True, exist_ok=True)
    intra_path = out_dir / f"intra__{embed_slug}.parquet"
    inter_path = out_dir / f"inter__{embed_slug}.parquet"
    intra_df.to_parquet(intra_path, index=False)
    inter_df.to_parquet(inter_path, index=False)

    # --- Aggregate summary ---
    summary: dict = {"embed_model": embed_model, "intra_sim": {}, "inter_sim": {}}

    print("\n=== Intra-model similarity ===")
    print(f"  (mean pairwise cosine within each model's samples, averaged over {intra_df['prompt_id'].nunique()} prompts)")
    for name in names:
        vals = intra_df.loc[intra_df["model"] == name, "intra_sim"].dropna()
        mean, std = float(vals.mean()), float(vals.std())
        summary["intra_sim"][name] = {"mean": round(mean, 6), "std": round(std, 6)}
        print(f"  {name:<40s}  {mean:.4f} ± {std:.4f}")

    if pairs:
        print("\n=== Inter-model similarity (pairwise) ===")
        print("  (expected cross-model cosine, averaged over prompts)")
        for a, b in pairs:
            mask = (inter_df["model_a"] == a) & (inter_df["model_b"] == b)
            vals = inter_df.loc[mask, "inter_sim"].dropna()
            mean, std = float(vals.mean()), float(vals.std())
            key = f"{a}||{b}"
            summary["inter_sim"][key] = {"mean": round(mean, 6), "std": round(std, 6)}
            print(f"  {a} vs {b}:  {mean:.4f} ± {std:.4f}")

    summary_path = out_dir / f"summary__{embed_slug}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    elapsed = round(time.time() - t0, 1)
    print(f"\nWrote:")
    print(f"  {intra_path}  ({len(intra_df)} rows)")
    print(f"  {inter_path}  ({len(inter_df)} rows)")
    print(f"  {summary_path}")
    print(f"Elapsed: {elapsed}s")

    if plot_dir is not None:
        first_slug = next(iter(config.values()))
        human_emb, human_idx = load_human(first_slug, embed_model, embed_dir)

        print("\nBuilding agent-agent similarity matrix...")
        corr_labels, corr_matrix = build_corr_matrix(
            loaded, intra_df, inter_df, human_emb, human_idx
        )
        corr_csv = out_dir / f"corr_matrix__{embed_slug}.csv"
        pd.DataFrame(corr_matrix, index=corr_labels, columns=corr_labels).to_csv(corr_csv)
        print(f"  {corr_csv}")
        plot_corr_matrix(corr_labels, corr_matrix, embed_slug, plot_dir)
        plot_pca_by_model(loaded, embed_slug, plot_dir, human_emb, human_idx)
        plot_human_vs_llm(loaded, human_emb, human_idx, embed_slug, plot_dir)
        if raw_parquet is not None:
            gen_df = pd.read_parquet(raw_parquet)
            plot_corr_matrix_html(corr_labels, corr_matrix, embed_slug, plot_dir)
            plot_pca_by_model_html(loaded, config, human_emb, human_idx, gen_df, embed_slug, plot_dir)
            plot_human_vs_llm_html(loaded, config, human_emb, human_idx, gen_df, embed_slug, plot_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help='JSON file mapping display name → file slug, e.g. {"Qwen 0.5B": "Qwen_Qwen2.5-0.5B-Instruct"}')
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--raw-parquet", type=Path, default=None,
                    help="Path to all_generations.parquet; enables interactive Plotly HTML plots")
    args = ap.parse_args()
    main(args.config, args.embed_model, args.embed_dir, args.out_dir,
         plot_dir=None if args.no_plots else args.plot_dir,
         raw_parquet=args.raw_parquet)
