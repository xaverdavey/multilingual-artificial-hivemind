"""Tokenization fertility per (model tokenizer, language) on FLORES-200 parallel text.

Fertility is measured on parallel text rather than on the models' own generations
on purpose: a model that answers repetitively in some language would otherwise
produce a "fertility" that reflects its output distribution instead of its
tokenizer. FLORES-200 devtest is 1012 sentences carrying identical content in
every language, so token counts are directly comparable across the 13 languages.

Reported per (model, language):
  tokens_per_sentence  mean subword tokens over the 1012 parallel sentences
  fertility_vs_en      tokens_per_sentence divided by the same model's English
                       value -- how many more tokens this tokenizer spends to say
                       the same thing in this language (the tokenization premium)
  tokens_per_char      subword tokens per character (script-sensitive; zh/ja pack
                       far more content per character, so compare within a script)

Output: results/linguistic/fertility.parquet  (one row per model x language)
"""

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# Project language code -> FLORES-200 code. FLORES por_Latn is Brazilian
# Portuguese, and OASST2 zh is Simplified, so zho_Hans is the right variant.
FLORES_CODE = {
    "en": "eng_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "de": "deu_Latn", "fr": "fra_Latn", "ca": "cat_Latn", "pt-BR": "por_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "eu": "eus_Latn", "pl": "pol_Latn",
    "vi": "vie_Latn",
}


def load_flores(cache_dir: Path) -> dict[str, list[str]]:
    """Return {project_lang_code: [1012 parallel sentences]}, downloading once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in FLORES_CODE.values() if not (cache_dir / f"{c}.devtest").exists()]
    if missing:
        print(f"downloading FLORES-200 ({len(missing)} languages missing) ...")
        raw = urllib.request.urlopen(FLORES_URL, timeout=600).read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            for code in FLORES_CODE.values():
                member = tar.extractfile(f"flores200_dataset/devtest/{code}.devtest")
                (cache_dir / f"{code}.devtest").write_bytes(member.read())

    out = {}
    for lang, code in FLORES_CODE.items():
        text = (cache_dir / f"{code}.devtest").read_text(encoding="utf-8")
        out[lang] = [s for s in text.splitlines() if s.strip()]
    sizes = {len(v) for v in out.values()}
    if len(sizes) != 1:
        raise ValueError(f"FLORES files are not parallel: line counts {sizes}")
    print(f"FLORES-200 devtest: {sizes.pop()} parallel sentences x {len(out)} languages")
    return out


def model_entries(models_yaml: Path) -> list[tuple[str, str, str]]:
    """(hf_id, display_name, family) for every active entry in the registry."""
    import yaml
    return [(e["id"], e.get("display_name", e["id"]), e.get("family", "Other"))
            for e in (yaml.safe_load(models_yaml.read_text()) or [])]


def fertility_for_tokenizer(tok, sentences: dict[str, list[str]]) -> dict[str, dict]:
    """Token / char counts for one tokenizer across every language."""
    stats = {}
    for lang, sents in sentences.items():
        enc = tok(sents, add_special_tokens=False)["input_ids"]
        n_tok = sum(len(ids) for ids in enc)
        n_chr = sum(len(s) for s in sents)
        stats[lang] = {
            "tokens_per_sentence": n_tok / len(sents),
            "tokens_per_char": n_tok / n_chr,
        }
    return stats


def main(models_yaml: Path, cache_dir: Path, out_dir: Path):
    from transformers import AutoTokenizer

    sentences = load_flores(cache_dir)
    rows, failed = [], []

    for hf_id, name, family in model_entries(models_yaml):
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception as exc:
            failed.append((hf_id, f"{type(exc).__name__}: {str(exc)[:100]}"))
            continue
        stats = fertility_for_tokenizer(tok, sentences)
        en_tps = stats["en"]["tokens_per_sentence"]
        for lang, s in stats.items():
            rows.append({
                "model": name, "hf_id": hf_id, "family": family, "language": lang,
                "tokens_per_sentence": s["tokens_per_sentence"],
                "tokens_per_char": s["tokens_per_char"],
                "fertility_vs_en": s["tokens_per_sentence"] / en_tps,
                "vocab_size": len(tok),
            })
        print(f"  {name:<24s} vocab {len(tok):>7,}  en {en_tps:6.1f} tok/sent  "
              f"max premium {max(s['tokens_per_sentence'] for s in stats.values())/en_tps:.2f}x")

    if failed:
        print("\nTokenizers that could not be loaded:")
        for hf_id, err in failed:
            print(f"  {hf_id}: {err}")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "fertility.parquet", index=False)
    print(f"\nwrote {out_dir/'fertility.parquet'} ({len(df)} rows, "
          f"{df['model'].nunique()} models x {df['language'].nunique()} languages)")

    print("\n=== fertility_vs_en, averaged across models ===")
    by_lang = df.groupby("language")["fertility_vs_en"].mean().sort_values()
    for lang, v in by_lang.items():
        print(f"  {lang:<6s} {v:.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("--cache-dir", type=Path, default=Path("results/linguistic/flores"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/linguistic"))
    args = ap.parse_args()
    main(args.models_yaml, args.cache_dir, args.out_dir)
