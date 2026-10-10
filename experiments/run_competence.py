"""Independent multilingual competence per (model, language): Belebele + FLORES+.

The GlotLID match rate (run_fluency) is computed on the same generations the
similarity scores come from, so using it as a competence predictor of the gap is
circular. This script measures competence on held-out benchmarks that never see
the OASST2 prompts or the sampled responses:

  belebele  Reading comprehension (Bandarkar et al. 2024): 900 four-way multiple-
            choice questions per language over FLORES passages. Fully parallel --
            the same questions in every language -- so a model's accuracy in two
            languages differs because of the language, not the items. Scored
            zero-shot from the next-token distribution over the answer letters:
            no sampling and no answer parsing. Chance is 0.25.
  flores    English -> target translation of the 1012 FLORES+ devtest sentences,
            greedy. Tests *writing* in the target language, which Belebele cannot:
            a model that can only produce generic text in a language scores
            badly here. Undefined for en. Scored by score_competence.

Thinking is disabled (enable_thinking=False; ignored by templates without the
switch), so the Qwen3 dense models answer directly. This measures the model's
non-reasoning competence, which is what a benchmark score conventionally means.

FLORES+ is gated on Hugging Face: accept its terms once on the dataset page, then
the HF_TOKEN used for the sweep is enough.

Output: results/competence/raw/{model}.parquet  one row per (task, language, item)
        plus a .meta.json sidecar, matching run_generation.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd

from experiments.run_fluency import strip_think_blocks

# Every decode here is greedy or a single argmax, so the sampler kernel is never
# exercised; skipping FlashInfer's sampler avoids a JIT compile that fails on
# machines whose CUDA toolkit lacks cicc (seen on an sm_120 workstation).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Project language code -> Belebele / FLORES+ code. Both por_Latn are Brazilian
# Portuguese. Belebele keeps the FLORES-200 Chinese label; FLORES+ renamed it.
BELEBELE_CODE = {
    "en": "eng_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "de": "deu_Latn", "fr": "fra_Latn", "ca": "cat_Latn", "pt-BR": "por_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "eu": "eus_Latn", "pl": "pol_Latn",
    "vi": "vie_Latn",
}
FLORES_PLUS_CODE = BELEBELE_CODE | {"zh": "cmn_Hans"}

LANG_NAME = {
    "en": "English", "es": "Spanish", "ru": "Russian", "zh": "Simplified Chinese",
    "de": "German", "fr": "French", "ca": "Catalan", "pt-BR": "Brazilian Portuguese",
    "it": "Italian", "ja": "Japanese", "eu": "Basque", "pl": "Polish",
    "vi": "Vietnamese",
}

LETTERS = ["A", "B", "C", "D"]
# vLLM's default max_logprobs; the four letters are virtually always in the top 20
# for an instruction-tuned model that follows the format.
TOP_LOGPROBS = 20
FLORES_MAX_TOKENS = 256

# The zero-shot instruction Bandarkar et al. use for instruction-tuned models.
BELEBELE_TEMPLATE = (
    "Given the following passage, query, and answer choices, output the letter "
    "corresponding to the correct answer.\n"
    "###\nPassage:\n{passage}\n###\nQuery:\n{question}\n###\nChoices:\n"
    "(A) {a1}\n(B) {a2}\n(C) {a3}\n(D) {a4}\n###\nAnswer:"
)
FLORES_TEMPLATE = ("Translate the following sentence from English to {lang}. "
                   "Reply with the translation only.\n\n{src}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_belebele(langs: list[str]) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    frames = []
    for lang in langs:
        path = hf_hub_download("facebook/belebele", f"data/{BELEBELE_CODE[lang]}.jsonl",
                               repo_type="dataset")
        df = pd.DataFrame(_read_jsonl(path))
        df["language"] = lang
        df["item_id"] = df["link"] + "#" + df["question_number"].astype(str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    per_lang = out.groupby("language")["item_id"].apply(frozenset)
    if per_lang.nunique() != 1:
        raise ValueError("Belebele item ids differ across languages; the set is not parallel")
    return out


def load_flores_plus(langs: list[str], split: str = "devtest") -> pd.DataFrame:
    """Aligned (English source, target reference) pairs for every non-English lang."""
    from huggingface_hub import hf_hub_download

    def read(code: str) -> pd.DataFrame:
        path = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{code}.jsonl",
                               repo_type="dataset")
        df = pd.DataFrame(_read_jsonl(path))
        missing = {"id", "text"} - set(df.columns)
        if missing:
            raise KeyError(f"FLORES+ {code}: no {missing} field (has {list(df.columns)})")
        return df[["id", "text"]].sort_values("id").reset_index(drop=True)

    src = read(FLORES_PLUS_CODE["en"])
    frames = []
    for lang in langs:
        if lang == "en":
            continue
        ref = read(FLORES_PLUS_CODE[lang])
        if not src["id"].equals(ref["id"]):
            raise ValueError(f"FLORES+ {lang} is not aligned with English by sentence id")
        frames.append(pd.DataFrame({"language": lang, "item_id": src["id"].astype(str),
                                    "source": src["text"], "reference": ref["text"]}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def letter_probs(top_logprobs: dict) -> dict[str, float]:
    """Probability of each answer letter at the first generated position.

    Several vocabulary entries can render as the same letter ("A", " A"); take the
    most probable of them rather than summing, so a tokenizer with many surface
    variants is not credited with extra mass.
    """
    probs = dict.fromkeys(LETTERS, 0.0)
    for lp in top_logprobs.values():
        tok = (lp.decoded_token or "").strip().strip("(").strip(")")
        if tok in probs:
            probs[tok] = max(probs[tok], math.exp(lp.logprob))
    return probs


QUOTE_PAIRS = [('"', '"'), ("“", "”"), ("«", "»"), ("「", "」")]


def clean_translation(text: str) -> str:
    """Drop reasoning blocks and one pair of wrapping quotes; nothing more.

    Chatter such as "Here is the translation:" is left in on purpose: failing to
    follow the instruction is part of what the metric should penalise.
    """
    text = strip_think_blocks(text or "").strip()
    for open_q, close_q in QUOTE_PAIRS:
        if len(text) > 1 and text.startswith(open_q) and text.endswith(close_q):
            return text[1:-1].strip()
    return text


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(model: str, langs: list[str], gpu_mem_util: float, flores_limit: int | None):
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    bele = load_belebele(langs)
    flores = load_flores_plus(langs)
    if flores_limit:
        flores = flores.groupby("language", sort=False).head(flores_limit).reset_index(drop=True)

    llm = LLM(model=model, gpu_memory_utilization=gpu_mem_util,
              tensor_parallel_size=max(1, torch.cuda.device_count()),
              max_logprobs=TOP_LOGPROBS)
    chat_kwargs = {"enable_thinking": False}

    def chat(prompts: list[str], params):
        return llm.chat([[{"role": "user", "content": p}] for p in prompts], params,
                        chat_template_kwargs=chat_kwargs)

    t0 = time.time()
    rows = []

    bele_prompts = [BELEBELE_TEMPLATE.format(passage=r.flores_passage, question=r.question,
                                             a1=r.mc_answer1, a2=r.mc_answer2,
                                             a3=r.mc_answer3, a4=r.mc_answer4)
                    for r in bele.itertuples(index=False)]
    outs = chat(bele_prompts, SamplingParams(max_tokens=1, temperature=0.0,
                                             logprobs=TOP_LOGPROBS))
    for r, out in zip(bele.itertuples(index=False), outs):
        probs = letter_probs(out.outputs[0].logprobs[0])
        mass = sum(probs.values())
        pred = max(probs, key=probs.get) if mass > 0 else None
        gold = LETTERS[int(r.correct_answer_num) - 1]
        rows.append({"model": model, "task": "belebele", "language": r.language,
                     "item_id": r.item_id, "prediction": pred, "gold": gold,
                     "correct": pred == gold, "letter_mass": mass})

    flores_prompts = [FLORES_TEMPLATE.format(lang=LANG_NAME[r.language], src=r.source)
                      for r in flores.itertuples(index=False)]
    outs = chat(flores_prompts, SamplingParams(max_tokens=FLORES_MAX_TOKENS, temperature=0.0))
    for r, out in zip(flores.itertuples(index=False), outs):
        gen = out.outputs[0]
        rows.append({"model": model, "task": "flores", "language": r.language,
                     "item_id": r.item_id, "source": r.source, "reference": r.reference,
                     "prediction": clean_translation(gen.text),
                     "n_tokens": len(gen.token_ids), "finish_reason": gen.finish_reason})
    elapsed = time.time() - t0

    cfg = llm.llm_engine.model_config
    meta = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "dtype": str(cfg.dtype),
        "quantization": cfg.quantization,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "chat_template_kwargs": chat_kwargs,
        "belebele": {"n_items_per_lang": len(bele) // len(langs),
                     "decoding": "argmax over A-D at token 1", "top_logprobs": TOP_LOGPROBS},
        "flores": {"split": "devtest", "direction": "en->xx", "decoding": "greedy",
                   "max_tokens": FLORES_MAX_TOKENS, "limit_per_lang": flores_limit},
        "elapsed_seconds": round(elapsed, 1),
    }
    return rows, meta


def main(model: str, out_dir: Path, gpu_mem_util: float, flores_limit: int | None):
    langs = list(BELEBELE_CODE)
    rows, meta_extra = run(model, langs, gpu_mem_util, flores_limit)

    out_path = out_dir / f"{model.replace('/', '_')}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")

    bele = df[df["task"] == "belebele"]
    print(bele.groupby("language").agg(acc=("correct", "mean"),
                                       letter_mass=("letter_mass", "mean")).round(3).to_string())

    meta = {"model": model, "languages": langs, "n_rows": len(df),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **meta_extra}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id (vLLM backend only)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "results" / "competence" / "raw")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--flores-limit", type=int, default=None,
                    help="Translate only the first N devtest sentences per language (smoke tests)")
    args = ap.parse_args()
    main(args.model, args.out_dir, args.gpu_mem_util, args.flores_limit)
