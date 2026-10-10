"""Save the per-prompt human baseline for every embedding space on disk.

run_prompt_tests.py recomputes the human per-(language, prompt) mean pairwise
cosine from the human embeddings, which are too large for git. This script
writes that series to results/metrics/human_intra__{embed_slug}.parquet
(columns: language, prompt_id, intra_sim, n_answers), mirroring the
human_intra__*.parquet files that run_lexical.py already emits, so the released
metrics carry both sides of every paired comparison.

Run from the repo root:  python -m experiments.export_human_intra
"""
import argparse
from pathlib import Path

import pandas as pd

from experiments.run_metrics import load_human, mean_pairwise_cosine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Every model's `_human.npy` holds the same human answers; take the first per embedding space.
    first_model: dict[str, str] = {}
    for p in sorted(args.embed_dir.glob("*__*_human.npy")):
        model_slug, embed_slug = p.name[: -len("_human.npy")].split("__", 1)
        first_model.setdefault(embed_slug, model_slug)

    for embed_slug, model_slug in first_model.items():
        emb, idx = load_human(model_slug, embed_slug, args.embed_dir)
        rows = []
        for (lang, pid), grp in idx.reset_index(drop=True).groupby(["language", "prompt_id"]):
            rows.append({"language": lang, "prompt_id": pid,
                         "intra_sim": mean_pairwise_cosine(emb[grp.index.to_numpy()]),
                         "n_answers": int(len(grp))})
        df = pd.DataFrame(rows)
        out = args.out_dir / f"human_intra__{embed_slug}.parquet"
        df.to_parquet(out, index=False)
        print(f"{out}: {len(df)} (language, prompt) cells, mean intra_sim {df.intra_sim.mean():.4f}")


if __name__ == "__main__":
    main()
