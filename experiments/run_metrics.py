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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Path, embed_model: str, embed_dir: Path, out_dir: Path):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help='JSON file mapping display name → file slug, e.g. {"Qwen 0.5B": "Qwen_Qwen2.5-0.5B-Instruct"}')
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    args = ap.parse_args()
    main(args.config, args.embed_model, args.embed_dir, args.out_dir)
