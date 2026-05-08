"""Sanity check: does the prose-completion prefix "A metaphor for time is:" reproduce
the chat-format river-mode result from the preliminary investigation?

If chat-format and prose-completion give similar diversity and similar river-mode
counts, the prose-continuation framework subsumes the prompt-based one as a special
case (short prefix), validating the methodology pivot in `RQ1_experimental_setup.md`.

Run on Fenrir. Defaults to Qwen2.5-7B-Instruct for speed; swap to 72B-FP8 for exact
preliminary-work replication.
"""

import re
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 20

# Mode keywords per language. River-family is the primary preliminary finding;
# weaver/sculptor are the deeper-stack modes also reported.
MODES = {
    "en": {
        "river": [r"\briver\b", r"\bstream\b", r"\bwater\b", r"\bflow", r"\bcurrent\b"],
        "weaver": [r"\bweav", r"\btapestry\b", r"\bthread", r"\bloom\b"],
        "sculptor": [r"\bsculpt", r"\bchisel", r"\bmarble\b", r"\bstone\b"],
    },
    "es": {
        "river": [r"\brío\b", r"\bcorriente\b", r"\bagua\b", r"\bflu(ye|ir|ido)"],
        "weaver": [r"\btejed", r"\btapiz\b", r"\bhilo", r"\btelar\b"],
        "sculptor": [r"\besculp", r"\bcincel", r"\bmármol\b"],
    },
}

PROMPTS = {
    "en": {
        "chat": "Write a one-sentence metaphor about time.",
        "chat_complete": "Please complete the following sentence in a single sentence: \"A metaphor for time is:\"",
        "prose": "A metaphor for time is:",
    },
    "es": {
        "chat": "Escribe una metáfora de una sola frase sobre el tiempo.",
        "chat_complete": "Por favor completa la siguiente frase con una sola frase: \"Una metáfora del tiempo es:\"",
        "prose": "Una metáfora del tiempo es:",
    },
}


def diversity(texts, embedder):
    e = embedder.encode(texts, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(len(texts), k=1)]
    return 1 - sims.mean()


def mode_counts(texts, modes_for_lang):
    counts = {m: 0 for m in modes_for_lang}
    for t in texts:
        tl = t.lower()
        for mode_name, patterns in modes_for_lang.items():
            if any(re.search(p, tl) for p in patterns):
                counts[mode_name] += 1
    return counts


def run(llm, tokenizer, embedder, lang):
    sp_chat = SamplingParams(n=N, max_tokens=128, top_p=0.95, temperature=1.0)
    sp_prose = SamplingParams(n=N, max_tokens=64, top_p=0.95, temperature=1.0,
                              stop=["\n"])

    # 1) chat (direct imperative, control) — replicates preliminary work
    chat_out = llm.chat(
        [{"role": "user", "content": PROMPTS[lang]["chat"]}], sp_chat
    )
    chat_resps = [o.text.strip() for o in chat_out[0].outputs]

    # 2) chat_complete (chat-wrapped completion task)
    cc_out = llm.chat(
        [{"role": "user", "content": PROMPTS[lang]["chat_complete"]}], sp_chat
    )
    cc_resps = [o.text.strip() for o in cc_out[0].outputs]

    # 3) assistant_prefill (imperative task + assistant continues its own stub)
    msgs = [
        {"role": "user", "content": PROMPTS[lang]["chat"]},
        {"role": "assistant", "content": PROMPTS[lang]["prose"]},
    ]
    prefill_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, continue_final_message=True
    )
    pf_out = llm.generate(prefill_prompt, sp_prose)
    pf_resps = [PROMPTS[lang]["prose"] + o.text for o in pf_out[0].outputs]

    # 4) prose (raw, no chat template)
    prose_out = llm.generate(PROMPTS[lang]["prose"], sp_prose)
    prose_resps = [PROMPTS[lang]["prose"] + o.text for o in prose_out[0].outputs]

    results = {}
    for name, resps in [("chat", chat_resps), ("chat_complete", cc_resps),
                        ("prefill", pf_resps), ("prose", prose_resps)]:
        results[name] = {
            "div": diversity(resps, embedder),
            "modes": mode_counts(resps, MODES[lang]),
            "samples": resps,
        }

    print(f"\n{'='*60}\n[{lang}] Model: {MODEL}\n{'='*60}")
    for name in ["chat", "chat_complete", "prefill", "prose"]:
        r = results[name]
        print(f"\n--- {name}")
        print(f"diversity = {r['div']:.4f}   modes (n={N}): {r['modes']}")
        for s in r["samples"][:5]:
            print(f"  - {s}")

    return {"lang": lang, "results": results}


if __name__ == "__main__":
    llm = LLM(model=MODEL, gpu_memory_utilization=0.4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    results = [run(llm, tokenizer, embedder, lang) for lang in ["en", "es"]]

    print(f"\n{'='*60}\nSUMMARY (diversity / river-mode count out of {N})\n{'='*60}")
    cols = ["chat", "chat_complete", "prefill", "prose"]
    header = f"{'lang':<4}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    for r in results:
        cells = []
        for name in cols:
            x = r["results"][name]
            cells.append(f"{x['div']:.4f} / {x['modes']['river']:>2d}")
        line = f"{r['lang']:<4}" + "".join(f"{c:>16}" for c in cells)
        print(line)
