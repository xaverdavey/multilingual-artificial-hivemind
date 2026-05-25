"""Embed generation outputs and save as numpy arrays.

Run once per (generation model, embedding model). Saves:
  results/embeddings/{gen_model}__{embed_model}_llm.npy      shape (N_llm, D)
  results/embeddings/{gen_model}__{embed_model}_human.npy    shape (N_human, D)
  results/embeddings/{gen_model}__{embed_model}_llm_index.parquet
  results/embeddings/{gen_model}__{embed_model}_human_index.parquet
  results/embeddings/{gen_model}__{embed_model}.meta.json

Ablation: pass --embed-model to any other model (HuggingFace or OpenAI).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
OPENAI_PREFIXES = ("text-embedding-", "text-ada-")


def detect_embed_backend(embed_model: str) -> str:
    if any(embed_model.startswith(p) for p in OPENAI_PREFIXES):
        return "openai"
    return "sentence-transformers"


def _embed_openai(texts: list[str], model: str, batch_size: int = 100) -> np.ndarray:
    from openai import OpenAI
    from tqdm import tqdm
    client = OpenAI()
    vecs = []
    batches = range(0, len(texts), batch_size)
    for i in tqdm(batches, desc="batches", unit="batch"):
        resp = client.embeddings.create(model=model, input=texts[i : i + batch_size])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def _embed_st(texts: list[str], model: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model)
    if "e5" in model.lower():
        texts = ["passage: " + t for t in texts]
    return enc.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def embed(texts: list[str], embed_model: str) -> np.ndarray:
    if detect_embed_backend(embed_model) == "openai":
        return _embed_openai(texts, embed_model)
    return _embed_st(texts, embed_model)


def _run_one(gen_model: str, df: pd.DataFrame, embed_model: str, out_dir: Path):
    """Embed and save outputs for a single generation model."""
    slug = f"{gen_model.replace('/', '_')}__{embed_model.replace('/', '_')}"

    if (out_dir / f"{slug}_llm.npy").exists():
        print(f"[{gen_model}] embeddings already exist, skipping")
        return

    mask = df["response_text"].fillna("").str.strip() != ""
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"[{gen_model}] dropping {n_dropped} empty response(s)")
        df = df[mask].reset_index(drop=True)

    print(f"[{gen_model}] embedding {len(df)} LLM responses with {embed_model} ...")
    t0 = time.time()
    llm_emb = embed(df["response_text"].tolist(), embed_model)

    human_df = (
        df.drop_duplicates("prompt_id")[["prompt_id", "language", "human_responses"]]
        .explode("human_responses")
        .rename(columns={"human_responses": "response_text"})
        .reset_index(drop=True)
    )
    human_df["human_idx"] = human_df.groupby("prompt_id").cumcount()

    print(f"[{gen_model}] embedding {len(human_df)} human responses ...")
    human_emb = embed(human_df["response_text"].tolist(), embed_model)
    elapsed = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{slug}_llm.npy", llm_emb)
    np.save(out_dir / f"{slug}_human.npy", human_emb)

    df[["prompt_id", "language", "sample_idx"]].to_parquet(
        out_dir / f"{slug}_llm_index.parquet", index=False)
    human_df[["prompt_id", "language", "human_idx"]].to_parquet(
        out_dir / f"{slug}_human_index.parquet", index=False)

    meta = {
        "gen_model": gen_model,
        "embed_model": embed_model,
        "embed_backend": detect_embed_backend(embed_model),
        "n_llm": int(len(llm_emb)),
        "n_human": int(len(human_emb)),
        "embedding_dim": int(llm_emb.shape[1]),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{gen_model}] saved to {out_dir}/{slug}_{{llm,human}}.npy  ({elapsed}s)")


def run(gen_parquet: Path, embed_model: str, out_dir: Path, model_filter: str | None = None):
    df = (pd.read_parquet(gen_parquet)
          .sort_values(["prompt_id", "sample_idx"])
          .reset_index(drop=True))

    if model_filter:
        df = df[df["model"] == model_filter]
        if df.empty:
            raise ValueError(f"Model {model_filter!r} not found in {gen_parquet}")

    for gen_model, model_df in df.groupby("model"):
        _run_one(gen_model, model_df.reset_index(drop=True), embed_model, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id to embed; omit to process all parquets in --raw-dir")
    ap.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="Embedding model (OpenAI or HuggingFace sentence-transformers)")
    args = ap.parse_args()

    parquets = sorted(args.raw_dir.glob("*.parquet"))
    for p in parquets:
        run(p, args.embed_model, args.out_dir, model_filter=args.model or None)
