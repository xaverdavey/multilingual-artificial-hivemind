"""Pick the OASST2 prompt set used across all models.

Filters root prompter messages with >=3 sibling assistant responses, then samples
K per language with a fixed seed so every model sees the same prompts.
Open-endedness filtering is a separate downstream step.

Output: experiments/oasst2_prompts.parquet — one row per selected prompt with
the human responses embedded as a list, ready to feed both the generation
script and the analysis pipeline.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

LANGS = ["en", "es", "ru", "zh", "de", "fr", "ca", "pt-BR", "it", "ja", "eu", "pl", "vi"]
DEFAULT_K = 30
MIN_SIBLINGS = 3
SEED = 42

PARQUET_URL = "https://huggingface.co/api/datasets/OpenAssistant/oasst2/parquet/default/train/0.parquet"


def main(out_path: Path, parquet_path: Path | None, k: int):
    src = parquet_path or PARQUET_URL
    df = pd.read_parquet(src)
    df = df[(~df["deleted"]) & df["review_result"]]

    assistants = df[df["role"] == "assistant"]
    sibling_counts = assistants.groupby("parent_id").size()
    siblings_by_parent = assistants.groupby("parent_id")["text"].apply(list)

    roots = df[(df["role"] == "prompter") & (df["parent_id"].isnull())].copy()
    roots["n_responses"] = roots["message_id"].map(sibling_counts).fillna(0).astype(int)
    roots["human_responses"] = roots["message_id"].map(siblings_by_parent)

    pool_sizes = {
        lang: int(((roots["lang"] == lang) & (roots["n_responses"] >= MIN_SIBLINGS)).sum())
        for lang in LANGS
    }
    print("Eligible prompts per language (n_responses >= {}):".format(MIN_SIBLINGS))
    for lang in LANGS:
        print(f"  {lang}: {pool_sizes[lang]}")
    print(f"Bottleneck: {min(pool_sizes, key=pool_sizes.get)} ({min(pool_sizes.values())})")
    print(f"Requested K={k}")

    selected = []
    rng = random.Random(SEED)
    for lang in LANGS:
        pool = roots[(roots["lang"] == lang) & (roots["n_responses"] >= MIN_SIBLINGS)]
        ids = sorted(pool["message_id"].tolist())  # deterministic order
        if len(ids) < k:
            raise RuntimeError(f"{lang}: only {len(ids)} prompts meet criteria, need {k}")
        picked = rng.sample(ids, k)
        selected.append(pool[pool["message_id"].isin(picked)])

    out = pd.concat(selected)[
        ["message_id", "lang", "text", "human_responses", "n_responses"]
    ].rename(columns={"message_id": "prompt_id", "lang": "language", "text": "prompt_text"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"wrote {len(out)} prompts to {out_path}")
    print(out.groupby("language").size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "oasst2_prompts.parquet")
    ap.add_argument("--parquet", type=Path, default=None,
                    help="Local OASST2 parquet path; default downloads from HF")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="Prompts per language (must be <= smallest eligible pool)")
    args = ap.parse_args()
    main(args.out, args.parquet, args.k)
