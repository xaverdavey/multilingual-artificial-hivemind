"""Score per-response language conformance using GlotLID, then summarise by model x language.

GlotLID is a fastText classifier covering ~2000 languages. For each LLM response we
record:
  pred_lang   - top-1 language label predicted by GlotLID
  pred_conf   - confidence of that top-1 prediction
  target_conf - probability GlotLID assigns to the target language (even when not top-1)
  match       - whether top-1 == target language

We aggregate to per (model, language) means and plot a heatmap. This is a *fluency
proxy* — it catches the most damaging failure modes (language drift, mode-collapsed
repetition that scrambles the LID signal, mixed-script gibberish) but does not score
grammatical correctness directly. Pair with the F-test plots when interpreting low-resource
language results.

Outputs:
  results/fluency/fluency__glotlid.parquet        per-response rows
  results/fluency/fluency_summary__glotlid.parquet  per (model, language) means
  results/plots/fluency_heatmap__glotlid.png        model x language heatmap
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Reasoning-block stripping. In this dataset only Qwen3 dense models (0.6B-32B) emit
# `<think>...</think>` blocks, and they do so in 100% of responses. ~0.5% are
# truncated (opening tag but no closing tag) when reasoning exceeds max_tokens.
# We strip closed blocks first, then anything from a dangling `<think>` to EOS.
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_DANGLING_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove `<think>...</think>` reasoning blocks; handle truncated (unclosed) ones.

    A truncated block (`<think>` with no matching close tag) means the LLM ran out
    of tokens before finishing reasoning — there is no answer, so we drop everything
    from `<think>` to end-of-string and the response becomes empty.
    """
    if not text:
        return text
    cleaned = _THINK_CLOSED_RE.sub("", text)
    cleaned = _THINK_DANGLING_RE.sub("", cleaned)
    return cleaned.strip()


# GlotLID uses ISO 639-3 + ISO 15924 (script) labels. For each project language,
# we keep one *primary* label (used to compute target_conf) and a set of close
# variant labels we still treat as a "match" (used to compute the binary match flag).
# Variants address GlotLID's fine-grained dialect splits, where genuinely-fluent
# text in the target language sometimes triggers a closely-related label
# (e.g., Swiss German for de, Hiragana-only for ja, Wu/Cantonese for zh).
LID_PRIMARY = {
    "ca": "cat_Latn", "de": "deu_Latn", "en": "eng_Latn", "es": "spa_Latn",
    "eu": "eus_Latn", "fr": "fra_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "pl": "pol_Latn", "pt-BR": "por_Latn", "ru": "rus_Cyrl", "vi": "vie_Latn",
    "zh": "cmn_Hani",   # Mandarin in Han script — GlotLID has no zho_Hans label.
}
LID_VARIANTS = {
    "ca": {"cat_Latn"},
    "de": {"deu_Latn", "gsw_Latn"},                # gsw = Swiss German Alemannic
    "en": {"eng_Latn"},
    "es": {"spa_Latn"},
    "eu": {"eus_Latn"},
    "fr": {"fra_Latn"},
    "it": {"ita_Latn"},
    "ja": {"jpn_Jpan", "und_Hira"},                # Hiragana-only is legit Japanese
    "pl": {"pol_Latn"},
    "pt-BR": {"por_Latn"},
    "ru": {"rus_Cyrl"},
    "vi": {"vie_Latn"},
    "zh": {"cmn_Hani", "wuu_Hani", "yue_Hani"},    # Wu, Cantonese — same script family
}


def load_glotlid_model():
    """Download (cached) the GlotLID v3 model from HuggingFace and load it."""
    from huggingface_hub import hf_hub_download
    import fasttext
    path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model_v3.bin")
    return fasttext.load_model(path)


def score_responses(texts: list[str], targets: list[str], model) -> pd.DataFrame:
    """Run GlotLID on each text. Returns a DataFrame aligned with `texts`.

    Calls the underlying C predictor (``model.f.predict``) directly to avoid a
    NumPy-2.0 incompatibility in fasttext 0.9.3's Python wrapper.

    `match` is True when top-1 prediction is in LID_VARIANTS[target] (variant-aware);
    `target_conf` is the probability GlotLID assigned to LID_PRIMARY[target].
    """
    pred_lang, pred_conf, target_conf = [], [], []
    primary_labels = [f"__label__{LID_PRIMARY[t]}" for t in targets]
    n_labels = len(model.get_labels())  # ~2000; ask for all so target is always found

    for text, primary_label in zip(texts, primary_labels):
        clean = (text or "").strip().replace("\n", " ")
        if not clean:
            pred_lang.append(None); pred_conf.append(0.0); target_conf.append(0.0)
            continue
        predictions = model.f.predict(clean, n_labels, 0.0, "strict")
        top_prob, top_label = predictions[0]
        pred_lang.append(top_label.replace("__label__", ""))
        pred_conf.append(float(top_prob))
        tc = 0.0
        for prob, label in predictions:
            if label == primary_label:
                tc = float(prob)
                break
        target_conf.append(tc)

    df = pd.DataFrame({"pred_lang": pred_lang, "pred_conf": pred_conf, "target_conf": target_conf})
    df["match"] = [pl in LID_VARIANTS.get(t, set()) for pl, t in zip(pred_lang, targets)]
    return df


def aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (model, language) mean match rate + mean target confidence."""
    grp = scored.groupby(["model", "language"], dropna=False)
    summary = grp.agg(
        n=("match", "size"),
        match_rate=("match", "mean"),
        mean_target_conf=("target_conf", "mean"),
        mean_pred_conf=("pred_conf", "mean"),
    ).reset_index()
    return summary


def plot_heatmap(summary: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    """Model x language heatmap. Cell = summary[value_col]."""
    import matplotlib.pyplot as plt

    pivot = summary.pivot(index="model", columns="language", values=value_col)
    # order rows by mean across languages, columns by mean across models
    pivot = pivot.reindex(
        index=pivot.mean(axis=1).sort_values().index,
        columns=pivot.mean(axis=0).sort_values().index,
    )
    n_rows, n_cols = pivot.shape

    fig, ax = plt.subplots(figsize=(max(7, n_cols * 0.8), max(5, n_rows * 0.45) + 1))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=value_col)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(n_rows):
        for j in range(n_cols):
            v = pivot.values[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if v < 0.35 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def main(raw_parquet: Path, out_dir: Path, plot_dir: Path) -> None:
    print(f"Loading {raw_parquet}...")
    df = pd.read_parquet(raw_parquet, columns=["model", "language", "prompt_id",
                                                "sample_idx", "response_text"])
    print(f"  {len(df):,} rows; {df['model'].nunique()} models; {df['language'].nunique()} languages")

    print("Loading GlotLID...")
    glotlid = load_glotlid_model()

    print("Stripping reasoning blocks (Qwen3 dense `<think>...</think>`)...")
    raw = df["response_text"].fillna("").tolist()
    stripped = [strip_think_blocks(t) for t in raw]
    n_had_think = sum(1 for r, s in zip(raw, stripped) if r != s)
    n_empty_after = sum(1 for s in stripped if not s.strip())
    print(f"  {n_had_think:,} responses had a <think> block ({100*n_had_think/len(raw):.1f}%)")
    print(f"  {n_empty_after:,} are empty after stripping (truncated reasoning, no answer emitted)")

    print(f"Scoring {len(df):,} stripped responses with GlotLID...")
    t0 = time.time()
    scored = score_responses(stripped, df["language"].tolist(), glotlid)
    print(f"  scored in {time.time() - t0:.1f}s")

    full = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    full["had_think"] = [r != s for r, s in zip(raw, stripped)]
    full["empty_after_strip"] = [not s.strip() for s in stripped]
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "fluency__glotlid.parquet"
    full.to_parquet(full_path, index=False)
    print(f"  wrote {full_path}  ({len(full):,} rows)")

    summary = aggregate(full)
    summary_path = out_dir / "fluency_summary__glotlid.parquet"
    summary.to_parquet(summary_path, index=False)
    print(f"  wrote {summary_path}  ({len(summary)} (model, language) rows)")

    print("\n=== Mean target-language match rate, by language (averaged across models) ===")
    by_lang = summary.groupby("language")["match_rate"].mean().sort_values()
    for lang, mr in by_lang.items():
        print(f"  {lang:<8s}  {mr:.3f}")

    print("\nPlotting fluency heatmap...")
    plot_heatmap(
        summary, "match_rate", plot_dir / "fluency_heatmap__glotlid.png",
        "GlotLID match rate per (model, language)  "
        "— fraction of samples whose top-1 predicted language is the target",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-parquet", type=Path, default=Path("results/all_generations.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/fluency"))
    ap.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    args = ap.parse_args()
    main(args.raw_parquet, args.out_dir, args.plot_dir)
