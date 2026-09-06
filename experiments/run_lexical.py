"""Embedder-free diversity: mean pairwise character n-gram Jaccard similarity.

Same estimator shape as the embedding metric in run_metrics (a mean over i<j
pairs within one (model, language, prompt) cell), so the outputs share that
module's schema and feed straight into its F-tests and plots. The point is that
no learned embedding space is involved at all: if the hivemind effect survives
here, it is not an artifact of embedding geometry.

Character n-grams rather than word n-grams because zh and ja are not
whitespace-delimited; character n-grams are defined identically across all 13
languages.

Pairwise rather than pooled (the usual distinct-n) because cells hold 50 LLM
samples but only ~3 human responses, and pooled n-gram ratios shrink as the
pool grows -- a pooled metric would report a large effect from sample count
alone. A mean over pairs is invariant to how many samples a cell holds.

Outputs (to --out-dir), where {tag} is the n-gram size plus any truncation:
  intra__{tag}.parquet         per (model, language, prompt_id)
  human_intra__{tag}.parquet   per (language, prompt_id)
  inter__{tag}.parquet         per (model_a, model_b, language, prompt_id)
"""

import argparse
import itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

HASH_BASE = np.uint64(1000003)

# (model, language, prompt_id) -> list of response texts. Populated in the parent
# before the worker pool forks, so children inherit it copy-on-write and only the
# small cell keys travel through the pool queue instead of ~1GB of text per pass.
# Requires the "fork" start method, which _imap requests explicitly.
_CELLS: dict[tuple, list[str]] = {}


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def ngram_hashes(text: str, n: int) -> np.ndarray:
    """Sorted unique rolling hashes of every character n-gram in `text`.

    A sorted unique array is the n-gram *set*; representing it this way lets
    the pairwise intersections below run in numpy rather than Python loops.
    """
    codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    if len(codes) < n:
        return np.empty(0, dtype=np.uint64)
    tail = len(codes) - n + 1
    h = np.zeros(tail, dtype=np.uint64)
    for k in range(n):
        h = h * HASH_BASE + codes[k : tail + k]
    return np.unique(h)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """|A n B| / |A u B| for two sorted unique hash arrays."""
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else float("nan")


def mean_pairwise_jaccard(grams: list[np.ndarray]) -> float:
    """Mean Jaccard over all i<j pairs. Mirrors mean_pairwise_cosine."""
    if len(grams) < 2:
        return float("nan")
    vals = [jaccard(grams[i], grams[j])
            for i, j in itertools.combinations(range(len(grams)), 2)]
    return float(np.nanmean(vals))


def mean_cross_jaccard(grams_a: list[np.ndarray], grams_b: list[np.ndarray]) -> float:
    """Mean Jaccard over all (a, b) pairs. Mirrors expected_cross_cosine."""
    if not grams_a or not grams_b:
        return float("nan")
    vals = [jaccard(a, b) for a in grams_a for b in grams_b]
    return float(np.nanmean(vals))


# ---------------------------------------------------------------------------
# Cell workers (module level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------

def _prep(texts, n: int, truncate: int | None) -> list[np.ndarray]:
    if truncate:
        texts = [t[:truncate] for t in texts]
    return [ngram_hashes(t, n) for t in texts]


def _intra_cell(job):
    key, n, truncate = job
    texts = _CELLS[key]
    return key, mean_pairwise_jaccard(_prep(texts, n, truncate)), len(texts)


def _inter_cell(job):
    (ma, mb, lang, pid), n, truncate = job
    sim = mean_cross_jaccard(_prep(_CELLS[(ma, lang, pid)], n, truncate),
                             _prep(_CELLS[(mb, lang, pid)], n, truncate))
    return (ma, mb, lang, pid), sim


def _imap(fn, jobs, workers: int, desc: str):
    from tqdm import tqdm
    with mp.get_context("fork").Pool(workers) as pool:
        return list(tqdm(pool.imap_unordered(fn, jobs, chunksize=16),
                         total=len(jobs), desc=desc, unit="cell"))


# ---------------------------------------------------------------------------
# Metric drivers
# ---------------------------------------------------------------------------

def compute_intra(df: pd.DataFrame, n: int, truncate, workers: int) -> pd.DataFrame:
    _CELLS.clear()
    for key, sub in df.groupby(["model", "language", "prompt_id"]):
        _CELLS[key] = sub["response_text"].tolist()
    jobs = [(key, n, truncate) for key in _CELLS]
    rows = _imap(_intra_cell, jobs, workers, "intra")
    return pd.DataFrame([
        {"model": k[0], "language": k[1], "prompt_id": k[2],
         "intra_sim": sim, "n_samples": n_s}
        for k, sim, n_s in rows
    ]).sort_values(["model", "language", "prompt_id"]).reset_index(drop=True)


def compute_human_intra(df: pd.DataFrame, n: int, truncate, workers: int) -> pd.DataFrame:
    human = (df.drop_duplicates("prompt_id")[["prompt_id", "language", "human_responses"]]
             .reset_index(drop=True))
    _CELLS.clear()
    for r in human.itertuples():
        _CELLS[(r.language, r.prompt_id)] = list(r.human_responses)
    jobs = [(key, n, truncate) for key in _CELLS]
    rows = _imap(_intra_cell, jobs, workers, "human")
    return pd.DataFrame([
        {"model": "Human", "language": k[0], "prompt_id": k[1],
         "intra_sim": sim, "n_samples": n_s}
        for k, sim, n_s in rows
    ]).sort_values(["language", "prompt_id"]).reset_index(drop=True)


def compute_inter(df: pd.DataFrame, n: int, truncate, workers: int,
                  inter_samples: int) -> pd.DataFrame:
    """Cross-model Jaccard, subsampled to `inter_samples` per model per prompt.

    Full 50x50 cross products over all 253 model pairs is ~247M Jaccards; the
    mean over a k x k subsample estimates the same quantity, and k=10 keeps the
    job to ~10M while leaving 100 pairs behind every cell mean.
    """
    sub = df[df["sample_idx"] < inter_samples]
    _CELLS.clear()
    for key, g in sub.groupby(["model", "language", "prompt_id"]):
        _CELLS[key] = g["response_text"].tolist()
    models = sorted(df["model"].unique())
    cells = sorted({(lang, pid) for _, lang, pid in _CELLS})

    jobs = [((ma, mb, lang, pid), n, truncate)
            for ma, mb in itertools.combinations(models, 2)
            for lang, pid in cells
            if (ma, lang, pid) in _CELLS and (mb, lang, pid) in _CELLS]
    rows = _imap(_inter_cell, jobs, workers, "inter")
    return pd.DataFrame([
        {"model_a": k[0], "model_b": k[1], "language": k[2], "prompt_id": k[3],
         "inter_sim": sim}
        for k, sim in rows
    ]).sort_values(["model_a", "model_b", "language", "prompt_id"]).reset_index(drop=True)


def display_names(models_yaml: Path) -> dict[str, str]:
    """HF id -> display_name, so outputs key the same way run_metrics does."""
    import yaml
    return {e["id"]: e.get("display_name", e["id"])
            for e in (yaml.safe_load(models_yaml.read_text()) or [])}


def main(raw_parquet: Path, out_dir: Path, models_yaml: Path, n: int,
         truncate: int | None, workers: int, inter_samples: int, skip_inter: bool):
    df = pd.read_parquet(raw_parquet, columns=["model", "language", "prompt_id",
                                               "sample_idx", "response_text",
                                               "human_responses"])
    df["response_text"] = df["response_text"].fillna("")
    names = display_names(models_yaml)
    df["model"] = df["model"].map(lambda m: names.get(m, m))
    print(f"{len(df):,} rows; {df['model'].nunique()} models; {df['language'].nunique()} languages")

    tag = f"char{n}gram" + (f"_trunc{truncate}" if truncate else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"tag={tag}  workers={workers}  truncate={truncate}")

    intra = compute_intra(df, n, truncate, workers)
    intra.to_parquet(out_dir / f"intra__{tag}.parquet", index=False)
    print(f"  wrote intra__{tag}.parquet ({len(intra)} rows), mean {intra['intra_sim'].mean():.4f}")

    human = compute_human_intra(df, n, truncate, workers)
    human.to_parquet(out_dir / f"human_intra__{tag}.parquet", index=False)
    print(f"  wrote human_intra__{tag}.parquet ({len(human)} rows), mean {human['intra_sim'].mean():.4f}")

    if not skip_inter:
        inter = compute_inter(df, n, truncate, workers, inter_samples)
        inter.to_parquet(out_dir / f"inter__{tag}.parquet", index=False)
        print(f"  wrote inter__{tag}.parquet ({len(inter)} rows), mean {inter['inter_sim'].mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-parquet", type=Path, default=Path("results/all_generations.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/lexical"))
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("-n", "--ngram", type=int, default=4, help="character n-gram size")
    ap.add_argument("--truncate", type=int, default=None,
                    help="truncate every response to this many characters (length control)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inter-samples", type=int, default=10,
                    help="samples per model per prompt for the cross-model metric")
    ap.add_argument("--skip-inter", action="store_true")
    args = ap.parse_args()
    main(args.raw_parquet, args.out_dir, args.models_yaml, args.ngram,
         args.truncate, args.workers, args.inter_samples, args.skip_inter)
