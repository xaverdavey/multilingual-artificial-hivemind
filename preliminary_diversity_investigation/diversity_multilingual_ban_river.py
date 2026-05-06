"""Multilingual diversity test with explicit ban on river/water metaphors.

If banning the dominant mode just reveals a 2nd-best mode (sand, hourglass, etc.),
the Hivemind has a hierarchy. If diversity jumps substantially, the river
metaphor was a uniquely strong attractor.
"""
import json
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer


PROMPTS = {
    "English":    "Write a one-sentence metaphor about time. Do not use a river or water metaphor.",
    "Spanish":    "Escribe una metáfora de una sola oración sobre el tiempo. No uses una metáfora de río o agua.",
    "French":     "Écris une métaphore d'une phrase sur le temps. N'utilise pas une métaphore de rivière ou d'eau.",
    "German":     "Schreibe eine Metapher in einem Satz über die Zeit. Verwende keine Fluss- oder Wassermetapher.",
    "Portuguese": "Escreva uma metáfora de uma só frase sobre o tempo. Não use uma metáfora de rio ou água.",
    "Italian":    "Scrivi una metafora di una sola frase sul tempo. Non usare una metafora di fiume o acqua.",
    "Mandarin":   "用一句话写一个关于时间的隐喻。不要使用河流或水的隐喻。",
    "Japanese":   "時間についての比喩を一文で書いてください。川や水の比喩を使わないでください。",
    "Russian":    "Напиши метафору о времени одним предложением. Не используй метафору реки или воды.",
    "Hindi":      "समय के बारे में एक वाक्य में रूपक लिखें। नदी या पानी के रूपक का उपयोग न करें।",
    "Arabic":     "اكتب استعارة عن الوقت في جملة واحدة. لا تستخدم استعارة النهر أو الماء.",
    "Indonesian": "Tulislah metafora satu kalimat tentang waktu. Jangan gunakan metafora sungai atau air.",
    "Swedish":    "Skriv en metafor om tid i en enda mening. Använd inte en metafor om flod eller vatten.",
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
    ap.add_argument("--out", default="multilingual_ban_river_results.json")
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
