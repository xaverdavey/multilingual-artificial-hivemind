"""Compute intra- and inter-model similarity metrics from saved embeddings.

Reads the canonical model registry at experiments/models.yaml (id, display_name,
family per entry; embedding slug = id.replace("/", "_")). Legacy JSON configs
({name: slug} or {name: {slug, family}}) are still accepted.

Metrics computed for all models in the config:
  Intra-model similarity (N values, one per model):
    Per prompt: mean pairwise cosine similarity across all samples from that model.
    Reported as mean ± std over all prompts.

  Inter-model similarity (N-choose-2 values, one per model pair):
    Per prompt: E_{a~A, b~B}[cos(a,b)] — the expected cosine between a random sample
    from model A and a random sample from model B.  Computed exactly as
    dot(sum_A, sum_B) / (K_A * K_B), which is the closed-form of the bootstrap mean.
    Reported as mean ± std over all prompts.

  F-tests — three separate tests, each saved to its own parquet:
    1. LLM-intra vs Human-intra: at six levels
       (general, family, model, language, family_language, model_language).
       "Is an LLM less internally diverse than humans on the same prompt?"
    2. LLM-cross-model vs Human-cross-respondent: at three levels
       (general, language, family_pair).
       "Do different LLMs cluster more tightly with each other than humans
       cluster with each other?" (the cross-model hivemind component).
    3. Pairwise family intra-diversity: one row per (family A, family B) pair.
       "Are families A and B significantly different from each other in
       within-prompt diversity?"
    Each row reports one-way ANOVA F-test (means) and a variance F-test.

Outputs (written to --out-dir):
  intra__{embed_slug}.parquet                 per (model, language, prompt_id)
  inter__{embed_slug}.parquet                 per (model_a, model_b, language, prompt_id)
  f_tests__{embed_slug}.parquet               LLM-vs-Human intra, by level/group
  f_tests_cross__{embed_slug}.parquet         LLM-cross-model vs Human-cross-respondent
  f_tests_family_pairwise__{embed_slug}.parquet  family A vs family B intra-diversity
  summary__{embed_slug}.json                  aggregate scalars
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
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse model registry. Returns (slugs, families) keyed by display name.

    Canonical format is models.yaml: a top-level list of entries with
    ``id``, ``display_name``, and ``family``; the on-disk embedding slug is
    derived as ``id.replace("/", "_")``. JSON is still accepted for legacy
    configs (``{name: slug}`` or ``{name: {slug, family}}``).
    """
    text = config_path.read_text()
    slugs: dict[str, str] = {}
    families: dict[str, str] = {}

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        for entry in (yaml.safe_load(text) or []):
            model_id = entry["id"]
            slug = model_id.replace("/", "_")
            name = entry.get("display_name", slug)
            slugs[name] = slug
            families[name] = entry.get("family", "Other")
    else:
        raw = json.loads(text)
        for name, val in raw.items():
            if isinstance(val, str):
                slugs[name], families[name] = val, "Other"
            else:
                slugs[name] = val["slug"]
                families[name] = val.get("family", "Other")

    if not slugs:
        raise ValueError(f"No models found in {config_path}")
    return slugs, families


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
# F-tests: small, scrutiny-friendly helpers + per-test functions.
# Conventions across this section:
#   - "intra_sim" = mean pairwise cosine within one (model|human, lang, prompt).
#                   Lower = more diverse output.
#   - "inter_sim" = expected cross-model cosine for one (model_a, model_b,
#                   lang, prompt) — "two different LLMs answering same prompt".
#   - Each F-test row has two sides labelled `a` and `b`. The column schema
#     (n_a, n_b, mean_a, mean_b, std_a, std_b, F_means, p_means, F_var, p_var)
#     is constant; `label_a` / `label_b` describe what those sides are.
# ---------------------------------------------------------------------------

def _variance_f_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided F-test of equality of variances. Returns (F, p).

    Convention: F = max(var_a, var_b) / min(var_a, var_b) so F >= 1; df come
    from whichever sample is on top; p = 2 * P(F > F_obs).
    """
    from scipy import stats
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if va <= 0 or vb <= 0:
        return float("nan"), float("nan")
    if va >= vb:
        f, df1, df2 = va / vb, len(a) - 1, len(b) - 1
    else:
        f, df1, df2 = vb / va, len(b) - 1, len(a) - 1
    return f, min(2.0 * float(stats.f.sf(f, df1, df2)), 1.0)


def _f_test_row(
    comparison: str,
    level: str,
    group: str,
    label_a: str, a: np.ndarray,
    label_b: str, b: np.ndarray,
    language: str | None = None,
) -> dict | None:
    """One-way ANOVA F-test on means + F-test on variances. Returns one row dict.

    Returns None when either side has <2 finite samples (test undefined).
    """
    from scipy import stats
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    anova = stats.f_oneway(a, b)
    f_var, p_var = _variance_f_test(a, b)
    return {
        "comparison": comparison,
        "level": level,
        "group": group,
        "language": language,
        "label_a": label_a,
        "label_b": label_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std(ddof=1)),
        "std_b": float(b.std(ddof=1)),
        "F_means": float(anova.statistic),
        "p_means": float(anova.pvalue),
        "F_var": f_var,
        "p_var": p_var,
    }


def _pool_per_cell(df: pd.DataFrame) -> np.ndarray:
    """Average intra_sim across whatever rows are in `df`, grouped by
    (language, prompt_id). Returns one value per cell. Use when you want
    one LLM observation per prompt regardless of how many models contributed.
    """
    return df.groupby(["language", "prompt_id"])["intra_sim"].mean().to_numpy()


# --- Test 1: LLM intra-sim vs Human intra-sim ----------------------------

def compute_f_tests(
    intra_df: pd.DataFrame,
    human_intra_df: pd.DataFrame,
    families: dict[str, str],
) -> pd.DataFrame:
    """F-tests: LLM(s) vs Human within-prompt intra-similarity, six levels.

    Levels: general, family, model, language, family_language, model_language.
    For pooled levels the LLM side is averaged per (language, prompt_id) cell
    so each cell contributes one LLM observation (1:1 with the human cell).
    """
    intra_df = intra_df.assign(family=intra_df["model"].map(families).fillna("Other"))
    human_all = human_intra_df["intra_sim"].dropna().to_numpy()
    human_by_lang = {
        lang: sub["intra_sim"].dropna().to_numpy()
        for lang, sub in human_intra_df.groupby("language")
    }

    rows: list[dict | None] = []
    def row(level, group, llm, human, language=None):
        rows.append(_f_test_row(
            "LLM_intra vs Human_intra", level, group,
            "llm", llm, "human", human, language=language,
        ))

    # general — pool every LLM observation per cell.
    row("general", "all", _pool_per_cell(intra_df), human_all)

    # family — pool the family's models per cell.
    for fam, sub in intra_df.groupby("family"):
        row("family", fam, _pool_per_cell(sub), human_all)

    # model — already one obs per cell for that single model.
    for model, sub in intra_df.groupby("model"):
        row("model", model, sub["intra_sim"].to_numpy(), human_all)

    # language — pool all models per prompt within the language.
    for lang in sorted(intra_df["language"].unique()):
        llm = _pool_per_cell(intra_df[intra_df["language"] == lang])
        row("language", lang, llm, human_by_lang.get(lang, np.array([])), language=lang)

    # family x language — pool that family's models within that language.
    for (fam, lang), sub in intra_df.groupby(["family", "language"]):
        row("family_language", fam, _pool_per_cell(sub),
            human_by_lang.get(lang, np.array([])), language=lang)

    # model x language — one obs per prompt for (model, language).
    for (model, lang), sub in intra_df.groupby(["model", "language"]):
        row("model_language", model, sub["intra_sim"].to_numpy(),
            human_by_lang.get(lang, np.array([])), language=lang)

    return pd.DataFrame([r for r in rows if r is not None])


# --- Test 2: LLM cross-model cosine vs Human cross-respondent cosine ------

def compute_cross_model_f_tests(
    inter_df: pd.DataFrame,
    human_intra_df: pd.DataFrame,
    families: dict[str, str],
) -> pd.DataFrame:
    """F-tests: LLM cross-model cosine vs Human cross-respondent cosine.

    Both sides answer "two different respondents answer the same prompt — how
    similar are their answers?" so the units match: LLM side comes from
    `inter_df.inter_sim` (E[cos(a~A, b~B)] for distinct model pairs A, B);
    Human side is `human_intra_df.intra_sim` (mean pairwise cosine across the
    ~3 humans who answered the prompt). Rejecting => LLMs cluster more tightly
    with each other than humans do with each other (the cross-model hivemind
    component).
    """
    inter = inter_df.assign(
        fam_a=inter_df["model_a"].map(families).fillna("Other"),
        fam_b=inter_df["model_b"].map(families).fillna("Other"),
    )
    # canonical family-pair label, order-insensitive ("Qwen3 || Qwen3" or "Gemma-3 || Qwen3")
    pair_lo = inter[["fam_a", "fam_b"]].min(axis=1)
    pair_hi = inter[["fam_a", "fam_b"]].max(axis=1)
    inter = inter.assign(family_pair=pair_lo + " || " + pair_hi)

    human_all = human_intra_df["intra_sim"].dropna().to_numpy()
    human_by_lang = {
        lang: sub["intra_sim"].dropna().to_numpy()
        for lang, sub in human_intra_df.groupby("language")
    }

    rows: list[dict | None] = []
    def row(level, group, llm, human, language=None):
        rows.append(_f_test_row(
            "LLM_cross_model vs Human_cross_respondent", level, group,
            "llm", llm, "human", human, language=language,
        ))

    # general — all cross-model cosines pooled.
    row("general", "all", inter["inter_sim"].dropna().to_numpy(), human_all)

    # language — cross-model cosines restricted to that language.
    for lang in sorted(inter["language"].unique()):
        llm = inter.loc[inter["language"] == lang, "inter_sim"].dropna().to_numpy()
        row("language", lang, llm, human_by_lang.get(lang, np.array([])), language=lang)

    # family_pair — same-family pairs (Qwen3||Qwen3) and cross-family pairs
    # (Gemma-3||Qwen3); each labelled "fam_lo || fam_hi".
    for pair, sub in inter.groupby("family_pair"):
        row("family_pair", pair, sub["inter_sim"].dropna().to_numpy(), human_all)

    # family_pair x language — drives the cross-model heatmap.
    for (pair, lang), sub in inter.groupby(["family_pair", "language"]):
        row("family_pair_language", pair,
            sub["inter_sim"].dropna().to_numpy(),
            human_by_lang.get(lang, np.array([])), language=lang)

    return pd.DataFrame([r for r in rows if r is not None])


# --- Test 3: family-vs-family pairwise diversity --------------------------

def compute_family_pairwise_f_tests(
    intra_df: pd.DataFrame,
    families: dict[str, str],
) -> pd.DataFrame:
    """F-tests: pairwise family-vs-family intra-similarity comparison.

    For each pair of families (A, B), compares family A's pooled intra_sim
    distribution (one value per (lang, prompt_id) cell, averaged across A's
    models) against B's pooled distribution. Rejecting => the two families
    have different mean intra-diversity.
    """
    import itertools
    intra_df = intra_df.assign(family=intra_df["model"].map(families).fillna("Other"))
    fam_values = {
        fam: _pool_per_cell(sub)
        for fam, sub in intra_df.groupby("family")
    }

    rows: list[dict | None] = []
    for fa, fb in itertools.combinations(sorted(fam_values), 2):
        rows.append(_f_test_row(
            "Family_intra vs Family_intra", "family_pair", f"{fa} || {fb}",
            fa, fam_values[fa], fb, fam_values[fb],
        ))
    return pd.DataFrame([r for r in rows if r is not None])


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
# F-test plots
# ---------------------------------------------------------------------------

def _sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""


def plot_f_test_bars(
    f_tests_df: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """4-panel bar chart for the LLM-intra-vs-Human-intra test.

    One panel per level (general / family / language / model). For each group
    in a panel, two side-by-side bars show the LLM and Human means with SEM
    error caps; ANOVA significance stars (`*` p<0.05, `**` p<0.01, `***` p<1e-4)
    sit above the taller bar. Bars sort by ascending mean_a within each panel.
    """
    import matplotlib.pyplot as plt

    levels = [
        ("general", "General (all LLMs pooled)"),
        ("family",  "Per family"),
        ("language", "Per language"),
        ("model",   "Per model"),
    ]
    panels = [(lvl, ttl, f_tests_df[f_tests_df["level"] == lvl].sort_values("mean_a"))
              for lvl, ttl in levels]
    panels = [(lvl, ttl, sub) for lvl, ttl, sub in panels if not sub.empty]
    if not panels:
        return

    widths = [max(2, len(sub)) for _, _, sub in panels]
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(sum(widths) * 0.55 + 2 * len(panels), 5.5),
        gridspec_kw={"width_ratios": widths},
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (level, title, sub) in zip(axes, panels):
        x = np.arange(len(sub))
        w = 0.4
        # SEM = std / sqrt(n) — uncertainty of the means being F-tested.
        se_a = (sub["std_a"] / np.sqrt(sub["n_a"])).to_numpy()
        se_b = (sub["std_b"] / np.sqrt(sub["n_b"])).to_numpy()
        ax.bar(x - w/2, sub["mean_a"], w, yerr=se_a, capsize=2,
               color="steelblue", label="LLM")
        ax.bar(x + w/2, sub["mean_b"], w, yerr=se_b, capsize=2,
               color="crimson", label="Human")
        for xi, (_, r) in zip(x, sub.iterrows()):
            top = max(r["mean_a"] + se_a[int(xi)], r["mean_b"] + se_b[int(xi)])
            star = _sig_marker(r["p_means"])
            if star:
                ax.text(xi, top + 0.005, star, ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("mean intra-sim" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("F-test: LLM vs Human intra-similarity  (lower = more diverse;  *p<0.05  **p<0.01  ***p<1e-4)",
                 fontsize=11)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"f_tests_bars__{embed_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def _plot_effect_heatmap(
    sub: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    """Heatmap helper: cells = `effect` column, annotated with `effect ± star`.

    `sub` must have columns: group, language, effect, p_means. Rows/cols are
    reordered by mean effect for at-a-glance pattern detection.
    """
    import matplotlib.pyplot as plt

    pivot_eff = sub.pivot(index="group", columns="language", values="effect")
    pivot_p = sub.pivot(index="group", columns="language", values="p_means")
    lang_order = pivot_eff.mean(axis=0).sort_values().index.tolist()
    grp_order = pivot_eff.mean(axis=1).sort_values().index.tolist()
    pivot_eff = pivot_eff.reindex(index=grp_order, columns=lang_order)
    pivot_p = pivot_p.reindex(index=grp_order, columns=lang_order)

    n_rows, n_cols = pivot_eff.shape
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 0.8), max(4, n_rows * 0.55) + 1))
    vmax = float(np.nanmax(np.abs(pivot_eff.values)))
    im = ax.imshow(pivot_eff.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot_eff.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot_eff.index, fontsize=9)
    for i in range(n_rows):
        for j in range(n_cols):
            eff = pivot_eff.values[i, j]
            p = pivot_p.values[i, j]
            if not np.isfinite(eff):
                continue
            star = _sig_marker(p) if np.isfinite(p) else ""
            color = "white" if abs(eff) > vmax * 0.55 else "black"
            ax.text(j, i, f"{eff:+.2f}{star}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_f_test_heatmap_model(
    f_tests_df: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """Model × language heatmap. Cell = LLM − Human intra-sim.

    Reads `model_language` rows from the LLM-vs-Human intra-sim F-test.
    """
    mod = f_tests_df[f_tests_df["level"] == "model_language"].copy()
    if mod.empty:
        return
    mod["effect"] = mod["mean_a"] - mod["mean_b"]
    _plot_effect_heatmap(
        mod, "F-test effect: model × language  (+ = LLM less diverse than humans)",
        "mean(LLM) − mean(Human) intra-sim  (red = LLM less diverse)",
        plot_dir / f"f_tests_heatmap_model__{embed_slug}.png",
    )


# --- Plots for Test 2: LLM cross-model vs Human cross-respondent ----------

def plot_cross_model_bars(
    cross_df: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """Bars: LLM cross-model cosine vs Human cross-respondent cosine, per group.

    Two panels: `general` (one pair of bars) and `language` (13 pairs, one
    per language). Significance stars from `p_means`. Rejecting => LLMs
    cluster more tightly with each other than humans do with each other.
    """
    import matplotlib.pyplot as plt

    panels = []
    for lvl, ttl in [("general", "General"), ("language", "Per language")]:
        sub = cross_df[cross_df["level"] == lvl].sort_values("mean_a")
        if not sub.empty:
            panels.append((ttl, sub))
    if not panels:
        return

    widths = [max(2, len(sub)) for _, sub in panels]
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(sum(widths) * 0.7 + 2 * len(panels), 5.0),
        gridspec_kw={"width_ratios": widths},
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, sub) in zip(axes, panels):
        x = np.arange(len(sub))
        w = 0.4
        se_a = (sub["std_a"] / np.sqrt(sub["n_a"])).to_numpy()
        se_b = (sub["std_b"] / np.sqrt(sub["n_b"])).to_numpy()
        ax.bar(x - w/2, sub["mean_a"], w, yerr=se_a, capsize=2,
               color="seagreen", label="LLM↔LLM")
        ax.bar(x + w/2, sub["mean_b"], w, yerr=se_b, capsize=2,
               color="crimson", label="Human↔Human")
        for xi, (_, r) in zip(x, sub.iterrows()):
            top = max(r["mean_a"] + se_a[int(xi)], r["mean_b"] + se_b[int(xi)])
            star = _sig_marker(r["p_means"])
            if star:
                ax.text(xi, top + 0.005, star, ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("mean cosine between two respondents" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("F-test: LLM↔LLM vs Human↔Human cosine on the same prompt  "
                 "(higher LLM bar ⇒ hivemind effect)", fontsize=11)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"f_tests_cross_bars__{embed_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_cross_model_heatmap(
    cross_df: pd.DataFrame,
    embed_slug: str,
    plot_dir: Path,
) -> None:
    """Family-pair × language heatmap of the cross-model hivemind effect.

    Cell value = mean(LLM↔LLM cosine for that family-pair in that language)
                 − mean(Human↔Human cosine in that language).
    Positive (red) ⇒ the family-pair clusters more tightly than humans do
    in that language. Reads `family_pair_language` rows from cross-model parquet.
    """
    sub = cross_df[cross_df["level"] == "family_pair_language"].copy()
    if sub.empty:
        return
    sub["effect"] = sub["mean_a"] - sub["mean_b"]
    _plot_effect_heatmap(
        sub, "F-test effect: family-pair × language  "
             "(+ = LLMs cluster more than humans do in that language)",
        "mean(LLM↔LLM) − mean(Human↔Human)  (red = LLM hivemind effect)",
        plot_dir / f"f_tests_cross_heatmap__{embed_slug}.png",
    )


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
    config, families = load_config(config_path)

    print(f"Models ({len(config)}) with embed_model={embed_model}:")
    loaded: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name, slug in config.items():
        print(f"  {name!r}  ←  {slug}  [family: {families[name]}]")
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

    # --- F-tests ---
    first_slug = next(iter(config.values()))
    human_emb, human_idx = load_human(first_slug, embed_model, embed_dir)
    human_intra = compute_intra("Human", human_emb, human_idx)

    print("\nRunning F-tests (LLM vs Human intra-similarity)...")
    f_tests_df = compute_f_tests(intra_df, human_intra, families)
    f_tests_path = out_dir / f"f_tests__{embed_slug}.parquet"
    f_tests_df.to_parquet(f_tests_path, index=False)

    print("Running cross-model F-tests (LLM↔LLM vs Human↔Human)...")
    cross_df = compute_cross_model_f_tests(inter_df, human_intra, families)
    cross_path = out_dir / f"f_tests_cross__{embed_slug}.parquet"
    cross_df.to_parquet(cross_path, index=False)

    print("Running pairwise-family F-tests (family A intra vs family B intra)...")
    family_pairwise_df = compute_family_pairwise_f_tests(intra_df, families)
    family_pairwise_path = out_dir / f"f_tests_family_pairwise__{embed_slug}.parquet"
    family_pairwise_df.to_parquet(family_pairwise_path, index=False)

    # --- Aggregate summary ---
    summary: dict = {
        "embed_model": embed_model,
        "intra_sim": {}, "inter_sim": {},
        "f_tests": {}, "f_tests_cross_model": {}, "f_tests_family_pairwise": [],
    }

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

    # --- F-test reporting ---
    def _print_table(df: pd.DataFrame, a_lbl: str, b_lbl: str) -> None:
        print(f"    {'group':<32s} {f'mean({a_lbl})':>10s} {f'mean({b_lbl})':>10s} "
              f"{'F_means':>10s} {'p_means':>10s} {'F_var':>10s} {'p_var':>10s}")
        for _, r in df.iterrows():
            print(f"    {r['group']:<32s} {r['mean_a']:>10.4f} {r['mean_b']:>10.4f} "
                  f"{r['F_means']:>10.3f} {r['p_means']:>10.2e} "
                  f"{r['F_var']:>10.3f} {r['p_var']:>10.2e}")

    print("\n=== F-tests: LLM vs Human intra-similarity ===")
    print("  (lower intra-sim = more diverse output)")
    for level in ("general", "family", "language", "model"):
        sub = f_tests_df[f_tests_df["level"] == level]
        if sub.empty:
            continue
        print(f"\n  [{level}]")
        _print_table(sub, "LLM", "H")
        summary["f_tests"][level] = [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in sub.to_dict(orient="records")
        ]

    print("\n=== F-tests: LLM↔LLM cosine vs Human↔Human cosine (cross-respondent) ===")
    print("  (higher LLM mean ⇒ LLMs cluster more tightly with each other than humans do)")
    for level in ("general", "language", "family_pair"):
        sub = cross_df[cross_df["level"] == level]
        if sub.empty:
            continue
        print(f"\n  [{level}]")
        _print_table(sub, "LLM↔LLM", "H↔H")
        summary["f_tests_cross_model"][level] = [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in sub.to_dict(orient="records")
        ]

    print("\n=== F-tests: pairwise family intra-diversity ===")
    print("  (mean(A) − mean(B); positive ⇒ family A is less diverse than family B)")
    if not family_pairwise_df.empty:
        print(f"    {'pair':<32s} {'mean(A)':>10s} {'mean(B)':>10s} "
              f"{'eff(A−B)':>10s} {'F_means':>10s} {'p_means':>10s}")
        for _, r in family_pairwise_df.iterrows():
            print(f"    {r['group']:<32s} {r['mean_a']:>10.4f} {r['mean_b']:>10.4f} "
                  f"{r['mean_a'] - r['mean_b']:>+10.4f} "
                  f"{r['F_means']:>10.3f} {r['p_means']:>10.2e}")
        summary["f_tests_family_pairwise"] = [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in family_pairwise_df.to_dict(orient="records")
        ]

    summary_path = out_dir / f"summary__{embed_slug}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    elapsed = round(time.time() - t0, 1)
    print(f"\nWrote:")
    print(f"  {intra_path}  ({len(intra_df)} rows)")
    print(f"  {inter_path}  ({len(inter_df)} rows)")
    print(f"  {f_tests_path}  ({len(f_tests_df)} rows)")
    print(f"  {cross_path}  ({len(cross_df)} rows)")
    print(f"  {family_pairwise_path}  ({len(family_pairwise_df)} rows)")
    print(f"  {summary_path}")
    print(f"Elapsed: {elapsed}s")

    if plot_dir is not None:
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
        print("\nPlotting F-test results...")
        plot_f_test_bars(f_tests_df, embed_slug, plot_dir)
        plot_f_test_heatmap_model(f_tests_df, embed_slug, plot_dir)
        plot_cross_model_bars(cross_df, embed_slug, plot_dir)
        plot_cross_model_heatmap(cross_df, embed_slug, plot_dir)
        if raw_parquet is not None:
            gen_df = pd.read_parquet(raw_parquet)
            plot_corr_matrix_html(corr_labels, corr_matrix, embed_slug, plot_dir)
            plot_pca_by_model_html(loaded, config, human_emb, human_idx, gen_df, embed_slug, plot_dir)
            plot_human_vs_llm_html(loaded, config, human_emb, human_idx, gen_df, embed_slug, plot_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "models.yaml",
                    help="Path to model registry (models.yaml, default). "
                         "Legacy JSON name->slug maps are also accepted.")
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
