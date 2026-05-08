"""Find the prefix-length threshold above which raw prose continuation works on
an instruct model. Vary length on truncations of a single Wikipedia passage so
content is held constant; include the original quiz-like stub as a reference
point for whether the issue is length alone or length × ambiguity.
"""

import re
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 20

WIKI = (
    "Mount Vesuvius is a stratovolcano located on the Gulf of Naples in "
    "Campania, Italy, about 9 km east of Naples and a short distance from "
    "the shore. It is one of several volcanoes which form the Campanian "
    "volcanic arc. Vesuvius consists of a large cone partially encircled "
    "by the steep rim of a summit caldera, caused by the collapse of an "
    "earlier and originally much higher structure. The eruption of Mount "
    "Vesuvius in AD 79 was one of the deadliest volcanic eruptions in "
    "European history, with"
)

def truncate_words(text, n):
    return " ".join(text.split()[:n])

PREFIXES = {
    "stub_quiz (5w)": "A metaphor for time is:",
    "wiki_5w":  truncate_words(WIKI, 5),
    "wiki_10w": truncate_words(WIKI, 10),
    "wiki_20w": truncate_words(WIKI, 20),
    "wiki_40w": truncate_words(WIKI, 40),
    "wiki_80w": truncate_words(WIKI, 80),
}


def diversity(texts, embedder):
    e = embedder.encode(texts, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(len(texts), k=1)]
    return 1 - sims.mean()


def off_task_flags(continuations):
    """Heuristics for likely break-frame outputs."""
    return {
        "multi_choice": sum(1 for c in continuations
                            if re.search(r"(?:^|\n)\s*[A-D][.)\]]", c) or
                               re.search(r"\b[A-D]\)\s+\w", c)),
        "very_short":   sum(1 for c in continuations if len(c.strip()) < 30),
        "fill_blank":   sum(1 for c in continuations if "_____" in c or "______" in c),
        "starts_meta":  sum(1 for c in continuations
                            if re.match(r"^\s*(Sure|Here|Of course|I would|Let me|To answer)", c)),
    }


def run_one(llm, embedder, name, prefix):
    sp = SamplingParams(n=N, max_tokens=120, top_p=0.95, temperature=1.0,
                        stop=["\n\n"])
    out = llm.generate(prefix, sp)
    cont = [o.text for o in out[0].outputs]
    div = diversity(cont, embedder)
    flags = off_task_flags(cont)
    total_flagged = sum(v for k, v in flags.items() if k != "very_short")
    print(f"\n--- {name}  ({len(prefix.split())}w prefix)")
    print(f"prefix: {prefix!r}")
    print(f"diversity = {div:.4f}   off-task flags: {flags}   any-flag total: {total_flagged}/{N}")
    print(f"sample continuations:")
    for c in cont[:3]:
        prev = c[:160].replace("\n", " ⏎ ")
        print(f"  - {prev}{'…' if len(c) > 160 else ''}")


def main():
    llm = LLM(model=MODEL, gpu_memory_utilization=0.4)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    print(f"\n{'='*70}\nModel: {MODEL}   raw llm.generate (no chat template)\n{'='*70}")
    for name, prefix in PREFIXES.items():
        run_one(llm, embedder, name, prefix)


if __name__ == "__main__":
    main()
