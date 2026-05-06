"""Diversity test on the same prompt translated into multiple languages.

Same metric as diversity.py (1 - mean pairwise cosine similarity in
multilingual-e5-small space), but per language. Tests whether the Hivemind
effect generalizes across languages or is English-specific.
"""
import json
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer


PROMPTS = {
    "English":    "Write a one-sentence metaphor about time.",
    "Spanish":    "Escribe una metáfora de una sola oración sobre el tiempo.",
    "French":     "Écris une métaphore d'une phrase sur le temps.",
    "German":     "Schreibe eine Metapher in einem Satz über die Zeit.",
    "Portuguese": "Escreva uma metáfora de uma só frase sobre o tempo.",
    "Italian":    "Scrivi una metafora di una sola frase sul tempo.",
    "Mandarin":   "用一句话写一个关于时间的隐喻。",
    "Japanese":   "時間についての比喩を一文で書いてください。",
    "Russian":    "Напиши метафору о времени одним предложением.",
    "Hindi":      "समय के बारे में एक वाक्य में रूपक लिखें।",
    "Arabic":     "اكتب استعارة عن الوقت في جملة واحدة.",
    "Indonesian": "Tulislah metafora satu kalimat tentang waktu.",
    "Swedish":    "Skriv en metafor om tid i en enda mening.",
}


def diversity(resps, embedder):
    e = embedder.encode(resps, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(len(resps), k=1)]
    return float(1 - sims.mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="multilingual_results.json")
    args = ap.parse_args()

    n = args.n
    llm = LLM(model=args.model)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    params = SamplingParams(n=n, max_tokens=256, top_p=0.95, temperature=1.0, seed=0)

    results = {}
    for lang, prompt in PROMPTS.items():
        out = llm.chat([{"role": "user", "content": prompt}], params)
        resps = [o.text for o in out[0].outputs]
        d = diversity(resps, embedder)
        results[lang] = {"diversity": d, "samples": resps[:3]}
        print(f"{lang:11s} diversity={d:.4f}")
        for r in resps[:2]:
            print(f"   - {r.strip()[:140]}")

    print("\n=== Sorted by diversity ===")
    for lang, r in sorted(results.items(), key=lambda kv: kv[1]["diversity"]):
        print(f"  {lang:11s} {r['diversity']:.4f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
