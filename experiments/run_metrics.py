"""Compute per-cell diversity metrics from saved embeddings.

Reads all (gen_model, embed_model) pairs matching --embed-model from
results/embeddings/ and writes:
  results/metrics/per_cell__{embed_model}.parquet

One row per (model × lang × prompt_id) with intra_sim, human_sim, sim_gap,
inter_sim, and the embed_model used — so ablation runs never overwrite each other.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def mean_pairwise_cosine(emb: np.ndarray) -> float:
    """Mean cosine similarity over all i<j pairs for L2-normalized rows.

    Uses the identity: mean_cos = (||sum||^2 - n) / (n*(n-1))
    which is O(n*d) instead of O(n^2*d).
    """
    n = len(emb)
    if n < 2:
        return float("nan")
    s = emb.sum(axis=0)
    return float((np.dot(s, s) - n) / (n * (n - 1)))


def compute_intra(emb: np.ndarray, idx: pd.DataFrame) -> pd.DataFrame:
    """Per (model, language, prompt_id) intra_sim."""
    idx = idx.reset_index(drop=True)
    rows = []
    for (model, lang, prompt_id), grp in idx.groupby(["model", "language", "prompt_id"]):
        rows.append({
            "model": model, "language": lang, "prompt_id": prompt_id,
            "intra_sim": mean_pairwise_cosine(emb[grp.index.to_numpy()]),
            "n_samples": len(grp),
        })
    return pd.DataFrame(rows)


def compute_human(emb: np.ndarray, idx: pd.DataFrame) -> pd.DataFrame:
    """Per (language, prompt_id) human_sim."""
    idx = idx.reset_index(drop=True)
    rows = []
    for (lang, prompt_id), grp in idx.groupby(["language", "prompt_id"]):
        rows.append({
            "language": lang, "prompt_id": prompt_id,
            "human_sim": mean_pairwise_cosine(emb[grp.index.to_numpy()]),
            "n_human": len(grp),
        })
    return pd.DataFrame(rows)


def compute_inter(all_llm: dict[str, tuple[np.ndarray, pd.DataFrame]]) -> pd.DataFrame:
    """Per (language, prompt_id) inter_sim.

    For each sample_idx slice, compute mean pairwise cosine across all models;
    report the mean over slices. Vectorized: uses ||sum||^2 identity per slice.
    """
    parts = []
    for model, (emb, idx) in all_llm.items():
        df = idx.reset_index(drop=True).copy()
        df["_row"] = np.arange(len(df))
        df["_model"] = model
        parts.append(df)
    lookup = pd.concat(parts, ignore_index=True)

    result = []
    for (lang, prompt_id), grp in lookup.groupby(["language", "prompt_id"]):
        try:
            pivot = grp.pivot(index="sample_idx", columns="_model", values="_row")
        except ValueError:
            continue
        models = pivot.columns.tolist()
        n_models = len(models)
        if n_models < 2:
            result.append({"language": lang, "prompt_id": prompt_id, "inter_sim": float("nan")})
            continue

        # stack to (n_samples, n_models, D)
        arrays = [all_llm[m][0][pivot[m].dropna().astype(int).to_numpy()] for m in models]
        min_len = min(len(a) for a in arrays)
        batch = np.stack([a[:min_len] for a in arrays], axis=1)  # (n_samples, n_models, D)

        sums = batch.sum(axis=1)                                   # (n_samples, D)
        gram_sums = np.einsum("nd,nd->n", sums, sums)             # (n_samples,)
        slice_sims = (gram_sums - n_models) / (n_models * (n_models - 1))

        result.append({
            "language": lang,
            "prompt_id": prompt_id,
            "inter_sim": float(slice_sims.mean()),
        })
    return pd.DataFrame(result)


def main(embed_model: str, embed_dir: Path, out_dir: Path):
    slug = embed_model.replace("/", "_")
    meta_files = sorted(embed_dir.glob(f"*__{slug}.meta.json"))
    if not meta_files:
        raise FileNotFoundError(
            f"No embeddings found for embed_model={embed_model} in {embed_dir}\n"
            f"Run: python -m experiments.run_embeddings --embed-model {embed_model}"
        )
    print(f"Found {len(meta_files)} model(s) embedded with {embed_model}")

    all_llm: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    human_emb, human_idx = None, None

    for mf in meta_files:
        meta = json.loads(mf.read_text())
        gen_model = meta["gen_model"]
        file_slug = mf.stem

        llm_emb = np.load(embed_dir / f"{file_slug}_llm.npy")
        llm_idx = pd.read_parquet(embed_dir / f"{file_slug}_llm_index.parquet")
        llm_idx["model"] = gen_model
        all_llm[gen_model] = (llm_emb, llm_idx)

        if human_emb is None:
            human_emb = np.load(embed_dir / f"{file_slug}_human.npy")
            human_idx = pd.read_parquet(embed_dir / f"{file_slug}_human_index.parquet")

    t0 = time.time()

    print("Computing intra_sim ...")
    intra_df = pd.concat(
        [compute_intra(emb, idx) for emb, idx in all_llm.values()],
        ignore_index=True,
    )

    print("Computing human_sim ...")
    human_df = compute_human(human_emb, human_idx)

    print("Computing inter_sim ...")
    inter_df = compute_inter(all_llm)

    result = (
        intra_df
        .merge(human_df, on=["language", "prompt_id"], how="left")
        .merge(inter_df, on=["language", "prompt_id"], how="left")
    )
    result["sim_gap"] = result["intra_sim"] - result["human_sim"]
    result["embed_model"] = embed_model

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_cell__{slug}.parquet"
    result.to_parquet(out_path, index=False)

    elapsed = round(time.time() - t0, 1)
    print(f"wrote {len(result)} rows to {out_path}  ({elapsed}s)")
    print(result[["model", "language", "intra_sim", "human_sim", "sim_gap", "inter_sim"]].describe())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    args = ap.parse_args()
    main(args.embed_model, args.embed_dir, args.out_dir)
