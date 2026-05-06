import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")


def diversity(prompt, n=20):
    out = llm.chat(
        [{"role": "user", "content": prompt}],
        SamplingParams(n=n, max_tokens=256, top_p=0.95, temperature=1.0),
    )
    resps = [o.text for o in out[0].outputs]
    e = embedder.encode(resps, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(n, k=1)]
    return 1 - sims.mean(), resps


if __name__ == "__main__":
    score, resps = diversity("Write a one-sentence metaphor about time.")
    print(f"diversity = {score:.4f}")
    for r in resps[:3]:
        print("-", r.strip())
