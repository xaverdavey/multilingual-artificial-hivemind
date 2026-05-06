"""Multilingual diversity test with explicit demand for uniqueness/originality.

Tests whether asking for novelty in the prompt itself unlocks diversity, or
whether the model still falls into universal stock metaphors despite being
told not to.
"""
import json
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer


PROMPTS = {
    "English":    "Write a unique and unusual one-sentence metaphor about time. Avoid common or clichéd comparisons.",
    "Spanish":    "Escribe una metáfora única e inusual sobre el tiempo en una sola oración. Evita las comparaciones comunes o trilladas.",
    "French":     "Écris une métaphore unique et inhabituelle sur le temps en une seule phrase. Évite les comparaisons banales ou clichées.",
    "German":     "Schreibe eine einzigartige und ungewöhnliche Metapher über die Zeit in einem Satz. Vermeide gewöhnliche oder abgedroschene Vergleiche.",
    "Portuguese": "Escreva uma metáfora única e incomum sobre o tempo em uma só frase. Evite comparações comuns ou clichês.",
    "Italian":    "Scrivi una metafora unica e insolita sul tempo in una sola frase. Evita paragoni comuni o triti.",
    "Mandarin":   "用一句话写一个独特而不寻常的关于时间的隐喻。避免常见或陈词滥调的比喻。",
    "Japanese":   "時間についての独特で珍しい比喩を一文で書いてください。ありきたりな表現や使い古された比喩は避けてください。",
    "Russian":    "Напиши уникальную и необычную метафору о времени одним предложением. Избегай распространённых или клишированных сравнений.",
    "Hindi":      "समय के बारे में एक अनोखा और असामान्य रूपक एक वाक्य में लिखें। सामान्य या घिसे-पिटे तुलनाओं से बचें।",
    "Arabic":     "اكتب استعارة فريدة وغير مألوفة عن الوقت في جملة واحدة. تجنب المقارنات الشائعة أو المبتذلة.",
    "Indonesian": "Tulislah metafora yang unik dan tidak biasa tentang waktu dalam satu kalimat. Hindari perbandingan umum atau klise.",
    "Swedish":    "Skriv en unik och ovanlig metafor om tid i en enda mening. Undvik vanliga eller klichéartade jämförelser.",
}


def diversity(resps, embedder):
    e = embedder.encode(resps, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(len(resps), k=1)]
    return float(1 - sims.mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="multilingual_creative_results.json")
    args = ap.parse_args()

    llm = LLM(model=args.model)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    params = SamplingParams(n=args.n, max_tokens=256, top_p=0.95, temperature=1.0, seed=0)

    results = {}
    for lang, prompt in PROMPTS.items():
        out = llm.chat([{"role": "user", "content": prompt}], params)
        resps = [o.text for o in out[0].outputs]
        d = diversity(resps, embedder)
        results[lang] = {"diversity": d, "samples": resps[:5]}
        print(f"{lang:11s} diversity={d:.4f}")
        for r in resps[:3]:
            print(f"   - {r.strip()[:140]}")

    print("\n=== Sorted by diversity ===")
    for lang, r in sorted(results.items(), key=lambda kv: kv[1]["diversity"]):
        print(f"  {lang:11s} {r['diversity']:.4f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
