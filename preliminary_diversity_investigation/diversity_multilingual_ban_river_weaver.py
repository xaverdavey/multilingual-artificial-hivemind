"""Multilingual diversity test with double ban: river/water AND weaver/loom/tapestry/thread.

Tests whether peeling back a second layer of the universal mode hierarchy
unlocks genuine spread or just reveals a third universal mode.
"""
import json
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer


PROMPTS = {
    "English":    "Write a one-sentence metaphor about time. Do not use a river, water, weaver, loom, tapestry, or thread metaphor.",
    "Spanish":    "Escribe una metáfora de una sola oración sobre el tiempo. No uses una metáfora de río, agua, tejedor, telar, tapiz o hilo.",
    "French":     "Écris une métaphore d'une phrase sur le temps. N'utilise pas une métaphore de rivière, d'eau, de tisserand, de métier à tisser, de tapisserie ou de fil.",
    "German":     "Schreibe eine Metapher in einem Satz über die Zeit. Verwende keine Fluss-, Wasser-, Weber-, Webstuhl-, Wandteppich- oder Fadenmetapher.",
    "Portuguese": "Escreva uma metáfora de uma só frase sobre o tempo. Não use uma metáfora de rio, água, tecelão, tear, tapeçaria ou fio.",
    "Italian":    "Scrivi una metafora di una sola frase sul tempo. Non usare una metafora di fiume, acqua, tessitore, telaio, arazzo o filo.",
    "Mandarin":   "用一句话写一个关于时间的隐喻。不要使用河流、水、织工、织布机、挂毯或线的隐喻。",
    "Japanese":   "時間についての比喩を一文で書いてください。川、水、織り手、織機、タペストリー、糸の比喩を使わないでください。",
    "Russian":    "Напиши метафору о времени одним предложением. Не используй метафору реки, воды, ткача, ткацкого станка, гобелена или нити.",
    "Hindi":      "समय के बारे में एक वाक्य में रूपक लिखें। नदी, पानी, बुनकर, करघा, टेपेस्ट्री या धागे के रूपक का उपयोग न करें।",
    "Arabic":     "اكتب استعارة عن الوقت في جملة واحدة. لا تستخدم استعارة النهر أو الماء أو النساج أو النول أو النسيج أو الخيط.",
    "Indonesian": "Tulislah metafora satu kalimat tentang waktu. Jangan gunakan metafora sungai, air, penenun, alat tenun, permadani, atau benang.",
    "Swedish":    "Skriv en metafor om tid i en enda mening. Använd inte en metafor om flod, vatten, vävare, vävstol, gobeläng eller tråd.",
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
    ap.add_argument("--out", default="multilingual_ban_river_weaver_results.json")
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
