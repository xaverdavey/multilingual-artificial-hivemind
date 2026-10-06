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

# Some sentence-transformers models expect a task-prefix. Measuring diversity is a
# clustering-geometry question, so we use the clustering prompt where one is defined.
ST_PROMPT_NAME = {"google/embeddinggemma-300m": "Clustering"}

# Loaded models are reused across generation models -- run() embeds 23 x 2 text sets
# per sweep and reloading a 500M-param encoder each time dominates the runtime.
_ST_CACHE: dict[str, object] = {}
ST_HALF = False   # set by --fp16: load sentence-transformers encoders in float16
_HUMAN_CACHE: dict[tuple, np.ndarray] = {}   # human embeddings reused across generation models


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


def _load_st(model: str, max_seq_length: int | None):
    from sentence_transformers import SentenceTransformer
    if model not in _ST_CACHE:
        import torch
        enc = SentenceTransformer(model, model_kwargs={"torch_dtype": torch.float16} if ST_HALF else None)
        if max_seq_length:
            enc.max_seq_length = max_seq_length
        print(f"loaded {model} on {enc.device} (max_seq_length={enc.max_seq_length})")
        _ST_CACHE[model] = enc
    return _ST_CACHE[model]


def _embed_st(texts: list[str], model: str, batch_size: int,
              max_seq_length: int | None) -> np.ndarray:
    enc = _load_st(model, max_seq_length)
    if "e5" in model.lower():
        texts = ["passage: " + t for t in texts]
    kwargs = {}
    if model in ST_PROMPT_NAME:
        kwargs["prompt_name"] = ST_PROMPT_NAME[model]
    out = enc.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=True, **kwargs)
    # Apple MPS keeps freed blocks cached; release them between calls or a long
    # sweep runs out of GPU memory after a few models.
    import gc, torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return np.asarray(out, dtype=np.float32)


def embed(texts: list[str], embed_model: str, batch_size: int | None = None,
          max_seq_length: int | None = None) -> np.ndarray:
    if detect_embed_backend(embed_model) == "openai":
        return _embed_openai(texts, embed_model, batch_size or 100)
    return _embed_st(texts, embed_model, batch_size or 32, max_seq_length)


def _run_one(gen_model: str, df: pd.DataFrame, embed_model: str, out_dir: Path,
             batch_size: int | None = None, max_seq_length: int | None = None,
             truncate_chars: int | None = None, samples_per_prompt: int | None = None):
    """Embed and save outputs for a single generation model.

    With `truncate_chars`, every LLM and human response is cut to its first N
    characters before embedding (the length control of the lexical metric,
    applied in embedding space) and the embed-model slug gets a `_trunc{N}`
    suffix, so run_metrics / run_prompt_tests see it as a separate space.
    """
    embed_slug = (embed_model.replace('/', '_') + (f"_trunc{truncate_chars}" if truncate_chars else "")
                  + (f"_s{samples_per_prompt}" if samples_per_prompt else ""))
    slug = f"{gen_model.replace('/', '_')}__{embed_slug}"
    if samples_per_prompt:
        df = df[df["sample_idx"] < samples_per_prompt].reset_index(drop=True)
    if truncate_chars:
        df = df.assign(
            response_text=df["response_text"].fillna("").str.slice(0, truncate_chars),
            human_responses=df["human_responses"].apply(
                lambda hs: [str(h)[:truncate_chars] for h in (list(hs) if hs is not None else [])]))

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
    llm_emb = embed(df["response_text"].tolist(), embed_model, batch_size, max_seq_length)

    human_df = (
        df.drop_duplicates("prompt_id")[["prompt_id", "language", "human_responses"]]
        .explode("human_responses")
        .rename(columns={"human_responses": "response_text"})
        .reset_index(drop=True)
    )
    human_df["human_idx"] = human_df.groupby("prompt_id").cumcount()

    # The human answers are the same 1183 texts for every generation model, so embed
    # them once per (embedder, truncation) and reuse within the sweep.
    human_texts = human_df["response_text"].tolist()
    hkey = (embed_slug, tuple(human_texts))
    if hkey in _HUMAN_CACHE:
        print(f"[{gen_model}] reusing cached human embeddings")
        human_emb = _HUMAN_CACHE[hkey]
    else:
        print(f"[{gen_model}] embedding {len(human_df)} human responses ...")
        human_emb = embed(human_texts, embed_model, batch_size, max_seq_length)
        _HUMAN_CACHE.clear(); _HUMAN_CACHE[hkey] = human_emb
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
        "max_seq_length": (int(_ST_CACHE[embed_model].max_seq_length)
                           if embed_model in _ST_CACHE else None),
        "prompt_name": ST_PROMPT_NAME.get(embed_model),
        "truncate_chars": truncate_chars,
        "samples_per_prompt": samples_per_prompt,
        "fp16": ST_HALF,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{gen_model}] saved to {out_dir}/{slug}_{{llm,human}}.npy  ({elapsed}s)")


def run(gen_parquet: Path, embed_model: str, out_dir: Path, model_filter: str | None = None,
        batch_size: int | None = None, max_seq_length: int | None = None,
        truncate_chars: int | None = None, samples_per_prompt: int | None = None):
    df = (pd.read_parquet(gen_parquet)
          .sort_values(["prompt_id", "sample_idx"])
          .reset_index(drop=True))

    if model_filter:
        df = df[df["model"] == model_filter]
        if df.empty:
            raise ValueError(f"Model {model_filter!r} not found in {gen_parquet}")

    for gen_model, model_df in df.groupby("model"):
        _run_one(gen_model, model_df.reset_index(drop=True), embed_model, out_dir,
                 batch_size, max_seq_length, truncate_chars, samples_per_prompt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id to embed; omit to process all parquets in --raw-dir")
    ap.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="Embedding model (OpenAI or HuggingFace sentence-transformers)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Encode batch size (default: 100 OpenAI, 32 sentence-transformers)")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Override the encoder's token limit; omit to use its native default")
    ap.add_argument("--gen-parquet", type=Path, default=None,
                    help="A single consolidated generations parquet (e.g. results/all_generations.parquet) "
                         "to use instead of the per-model files in --raw-dir")
    ap.add_argument("--truncate-chars", type=int, default=None,
                    help="Cut every response to its first N characters before embedding (length control)")
    ap.add_argument("--samples-per-prompt", type=int, default=None,
                    help="Keep only sample_idx < N per prompt (deterministic subset; slug gets _sN)")
    ap.add_argument("--fp16", action="store_true", help="Load sentence-transformers encoders in float16")
    args = ap.parse_args()
    ST_HALF = args.fp16

    parquets = [args.gen_parquet] if args.gen_parquet else sorted(args.raw_dir.glob("*.parquet"))
    for p in parquets:
        run(p, args.embed_model, args.out_dir, model_filter=args.model or None,
            batch_size=args.batch_size, max_seq_length=args.max_seq_length,
            truncate_chars=args.truncate_chars, samples_per_prompt=args.samples_per_prompt)
