"""Validate that the assistant-prefill mechanism works for natural multi-sentence
prose (not just a 5-word designed stub).

Tests three conditions on a Wikipedia-style prose chunk that ends mid-thought:
  1. prefill         — chat template w/ user "continue" instruction + assistant prefill
  2. user_msg        — chat template w/ prose in user message, asking model to continue
  3. raw_prose       — llm.generate(prose), no chat template at all

What we want to see in (1):
  - Continuations are coherent (extend the passage naturally, no frame-breaks).
  - Continuations are not all memorized (diversity > ~0).
  - No off-task quiz-format collapse like the short-stub raw_prose case.

If (1) passes, the unified methodology (chat-template prefill for instruct models,
arbitrary prose content) is validated and we can commit to it in
RQ1_experimental_setup.md.
"""

import re
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 20

# Prose ending mid-thought to invite natural continuation. Wikipedia-style;
# memorization risk acknowledged — if continuations all match the real article
# verbatim, that is itself a finding (memorization, not Hivemind).
PREFIX_EN = (
    "Mount Vesuvius is a stratovolcano located on the Gulf of Naples in "
    "Campania, Italy, about 9 km east of Naples and a short distance from "
    "the shore. It is one of several volcanoes which form the Campanian "
    "volcanic arc. Vesuvius consists of a large cone partially encircled "
    "by the steep rim of a summit caldera, caused by the collapse of an "
    "earlier and originally much higher structure. The eruption of Mount "
    "Vesuvius in AD 79 was one of the deadliest volcanic eruptions in "
    "European history, with"
)

USER_INSTR = "Continue the following passage in 2-3 sentences."


def diversity(texts, embedder):
    e = embedder.encode(texts, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(len(texts), k=1)]
    return 1 - sims.mean()


def coherence_check(texts):
    """Heuristic flags for likely off-task responses."""
    flags = {
        "has_multiple_choice": sum(1 for t in texts if re.search(r"^\s*[A-D]\)", t, re.M) or re.search(r"^\s*\([A-D]\)", t, re.M)),
        "very_short": sum(1 for t in texts if len(t.strip()) < 30),
        "starts_with_meta": sum(1 for t in texts if re.match(r"^\s*(Sure|Here|Of course|I would|Let me)", t)),
    }
    return flags


def run(llm, tokenizer, embedder):
    sp = SamplingParams(n=N, max_tokens=200, top_p=0.95, temperature=1.0,
                        stop=["<|im_end|>"])

    # 1) prefill
    msgs = [
        {"role": "user", "content": USER_INSTR},
        {"role": "assistant", "content": PREFIX_EN},
    ]
    prefill_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, continue_final_message=True
    )
    pf_out = llm.generate(prefill_prompt, sp)
    pf_continuations = [o.text for o in pf_out[0].outputs]
    pf_full = [PREFIX_EN + c for c in pf_continuations]

    # 2) user_msg — prose in user content, model generates from scratch
    um_msgs = [{"role": "user", "content": f"{USER_INSTR}\n\n{PREFIX_EN}"}]
    um_out = llm.chat(um_msgs, sp)
    um_resps = [o.text.strip() for o in um_out[0].outputs]

    # 3) raw_prose — no chat template, instruct model on raw prefix
    raw_out = llm.generate(PREFIX_EN, sp)
    raw_continuations = [o.text for o in raw_out[0].outputs]
    raw_full = [PREFIX_EN + c for c in raw_continuations]

    print(f"\n{'='*70}\nModel: {MODEL}\nPrefix length: ~{len(PREFIX_EN.split())} words\n{'='*70}")

    for name, full, only_continuation in [
        ("prefill (continuation only)", pf_full, pf_continuations),
        ("user_msg (whole response)", um_resps, um_resps),
        ("raw_prose (continuation only)", raw_full, raw_continuations),
    ]:
        div = diversity(full if "user_msg" in name else only_continuation, embedder)
        flags = coherence_check(only_continuation)
        print(f"\n--- {name}")
        print(f"diversity = {div:.4f}")
        print(f"off-task flags: {flags}")
        print(f"first 4 continuations:")
        for c in only_continuation[:4]:
            preview = c[:200].replace("\n", " ")
            print(f"  - {preview}{'…' if len(c) > 200 else ''}")


if __name__ == "__main__":
    llm = LLM(model=MODEL, gpu_memory_utilization=0.4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    run(llm, tokenizer, embedder)
