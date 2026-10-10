"""Aggregate run_competence outputs into one competence score per (model, language).

Reads every results/competence/raw/{model}.parquet and reports, per cell:

  belebele_acc          accuracy over the 900 parallel questions (chance 0.25)
  belebele_letter_mass  mean probability the model puts on A-D at all. Low mass
                        means the model is not following the answer format, so
                        its accuracy understates comprehension -- check this
                        before reading the smallest models' scores.
  flores_chrf           corpus chrF++ (sacrebleu) of the en -> xx translations
  flores_comet          mean segment COMET (Unbabel/wmt22-comet-da), only with
                        --comet; needs a GPU and the unbabel-comet package

chrF and COMET are not on a common scale across languages (chrF in particular
depends on the script), so the downstream regression keeps a random intercept
per language; what the predictor contributes is model-to-model variation
within a language plus within-model variation across languages.

COMET segment scores are cached per model under results/competence/comet/, so
re-running after adding a model only scores the new one.

Output: results/competence/competence_summary.parquet  (model = HF id)
"""

import argparse
from pathlib import Path

import pandas as pd

COMET_MODEL = "Unbabel/wmt22-comet-da"


def chrf(group: pd.DataFrame) -> float:
    import sacrebleu
    hyps = group["prediction"].fillna("").tolist()
    return sacrebleu.corpus_chrf(hyps, [group["reference"].tolist()], word_order=2).score


def comet_scores(flores: pd.DataFrame, cache: Path, batch_size: int) -> pd.Series:
    """Segment COMET for every FLORES row, reusing the cache where it is complete."""
    if cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(flores) and (cached["item_id"].values == flores["item_id"].values).all():
            return pd.Series(cached["comet"].values, index=flores.index)

    from comet import download_model, load_from_checkpoint
    scorer = load_from_checkpoint(download_model(COMET_MODEL))
    data = [{"src": s, "mt": m or "", "ref": r}
            for s, m, r in zip(flores["source"], flores["prediction"], flores["reference"])]
    scores = scorer.predict(data, batch_size=batch_size, gpus=1, progress_bar=False).scores

    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"language": flores["language"].values, "item_id": flores["item_id"].values,
                  "comet": scores}).to_parquet(cache, index=False)
    return pd.Series(scores, index=flores.index)


def summarise(raw: pd.DataFrame, comet_cache: Path | None, batch_size: int) -> pd.DataFrame:
    model = raw["model"].iloc[0]
    bele = (raw[raw["task"] == "belebele"].groupby("language")
            .agg(belebele_acc=("correct", "mean"), belebele_letter_mass=("letter_mass", "mean"),
                 belebele_n=("correct", "size")))

    flores = raw[raw["task"] == "flores"].reset_index(drop=True)
    fl = pd.DataFrame({"flores_chrf": {lang: chrf(g) for lang, g in flores.groupby("language")},
                       "flores_n": flores.groupby("language").size()})
    if comet_cache is not None:
        flores["comet"] = comet_scores(flores, comet_cache, batch_size)
        fl["flores_comet"] = flores.groupby("language")["comet"].mean()

    out = bele.join(fl, how="left").rename_axis("language").reset_index()
    out.insert(0, "model", model)
    return out


def main(raw_dir: Path, out_path: Path, comet: bool, batch_size: int):
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No competence outputs in {raw_dir}; run experiments.run_competence first")

    frames = []
    for f in files:
        cache = raw_dir.parent / "comet" / f.name if comet else None
        frames.append(summarise(pd.read_parquet(f), cache, batch_size))
        print(f"scored {f.stem}")
    df = pd.concat(frames, ignore_index=True)
    if "flores_comet" not in df:
        df["flores_comet"] = float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} cells, {df['model'].nunique()} models)")
    print(df.pivot(index="model", columns="language", values="belebele_acc").round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("results/competence/raw"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/competence/competence_summary.parquet"))
    ap.add_argument("--comet", action="store_true", help="Also score FLORES with COMET (GPU)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    main(args.raw_dir, args.out, args.comet, args.batch_size)
