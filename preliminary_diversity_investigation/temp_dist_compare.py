"""Compare temperature distribution families matched on (median, 95th percentile).

For each family (beta, lognormal, power_law) we numerically solve for shape parameters
that hit the same median and p95, then sample n temperatures per generation. We compare
against fixed-T baselines on the same prompts. Diversity = 1 - mean pairwise cosine
similarity of multilingual-e5-small embeddings of the n responses.
"""
import argparse
import json
import numpy as np
from scipy import optimize, stats
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer


PROMPTS = [
    "Write a one-sentence metaphor about time.",
    "Generate a joke about electric vehicles.",
    "Write a pun about peanut.",
    "Name one meaning of life.",
    "Write a paragraph about how the internet shaped society.",
    "Create a slogan for a cosmetic bag.",
    "Write a short story about a colorful toad on an adventure in 50 words.",
    "Suggest a creative name for a coffee shop in a small town.",
    "Describe the smell of rain in one sentence.",
    "Invent a fictional holiday and describe how people celebrate it.",
    "Brainstorm three uses for a paperclip.",
    "Write the opening line of a mystery novel.",
    "Describe an alien planet's weather in two sentences.",
    "Compose a haiku about coffee.",
    "Write a tagline for a public library.",
    "Imagine a new flavor of ice cream and describe it.",
    "Write a riddle whose answer is a clock.",
    "Suggest a name for a fictional band.",
    "Describe what silence sounds like.",
    "Write a thank-you note from a houseplant to its owner.",
]


def fit_beta(med, p95, lo=0.3, hi=2.0):
    tm, tp = (med - lo) / (hi - lo), (p95 - lo) / (hi - lo)
    res = optimize.minimize(
        lambda x: (stats.beta.ppf(0.5, *np.exp(x)) - tm) ** 2
        + (stats.beta.ppf(0.95, *np.exp(x)) - tp) ** 2,
        [0.0, 0.0],
    )
    a, b = np.exp(res.x)
    return (lambda n, rng: lo + (hi - lo) * rng.beta(a, b, size=n)), {"alpha": float(a), "beta": float(b)}


def fit_lognormal(med, p95, lo=0.3, hi=2.0):
    def loss(x):
        mu, sigma = x[0], np.exp(x[1])
        return (np.exp(mu) - med) ** 2 + (np.exp(mu + sigma * stats.norm.ppf(0.95)) - p95) ** 2

    res = optimize.minimize(loss, [np.log(med), np.log(0.5)])
    mu, sigma = res.x[0], np.exp(res.x[1])

    def sample(n, rng):
        return np.clip(np.exp(rng.normal(mu, sigma, size=n)), lo, hi)

    return sample, {"mu": float(mu), "sigma": float(sigma)}


def fit_power_law(med, p95, hi=2.0):
    """Bounded Pareto on [t_min, hi] with PDF ∝ t^(-alpha)."""

    def q(p, alpha, t_min):
        term = t_min ** (1 - alpha) + p * (hi ** (1 - alpha) - t_min ** (1 - alpha))
        return term ** (1 / (1 - alpha))

    def loss(x):
        alpha = 1 + np.exp(x[0])
        t_min = 0.05 + np.exp(x[1])
        if t_min >= hi - 0.05:
            return 1e6
        return (q(0.5, alpha, t_min) - med) ** 2 + (q(0.95, alpha, t_min) - p95) ** 2

    res = optimize.minimize(loss, [np.log(2.0), np.log(0.3)])
    alpha = 1 + np.exp(res.x[0])
    t_min = 0.05 + np.exp(res.x[1])

    def sample(n, rng):
        u = rng.uniform(size=n)
        term = t_min ** (1 - alpha) + u * (hi ** (1 - alpha) - t_min ** (1 - alpha))
        return term ** (1 / (1 - alpha))

    return sample, {"alpha": float(alpha), "t_min": float(t_min)}


def diversity(resps, embedder):
    e = embedder.encode(resps, normalize_embeddings=True)
    sims = (e @ e.T)[np.triu_indices(len(resps), k=1)]
    return float(1 - sims.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=20, help="generations per (prompt, condition)")
    ap.add_argument("--median", type=float, default=0.9)
    ap.add_argument("--p95", type=float, default=1.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="temp_dist_results.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    llm = LLM(model=args.model)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")

    families = {
        "beta": fit_beta(args.median, args.p95),
        "lognormal": fit_lognormal(args.median, args.p95),
        "power_law": fit_power_law(args.median, args.p95),
    }
    fixed = {"fixed_T=0.9": 0.9, "fixed_T=1.0": 1.0}

    print("\n=== Realized quantiles for fitted distributions ===")
    for name, (sampler, params) in families.items():
        s = sampler(20000, np.random.default_rng(0))
        print(f"{name:10s} params={params}  median={np.median(s):.3f} p95={np.quantile(s, 0.95):.3f} mean={s.mean():.3f}")

    results = {k: [] for k in list(families) + list(fixed)}
    for i, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]

        for fname, T in fixed.items():
            out = llm.chat(
                msgs,
                SamplingParams(n=args.n, max_tokens=256, top_p=0.95, temperature=T, seed=args.seed + i),
            )
            results[fname].append(diversity([o.text for o in out[0].outputs], embedder))

        for fname, (sampler, _) in families.items():
            temps = sampler(args.n, rng)
            params_list = [
                SamplingParams(n=1, max_tokens=256, top_p=0.95, temperature=float(t), seed=args.seed + i * 1000 + k)
                for k, t in enumerate(temps)
            ]
            outs = llm.chat([msgs] * args.n, params_list)
            results[fname].append(diversity([o.outputs[0].text for o in outs], embedder))

        print(f"prompt {i+1}/{len(PROMPTS)}: " + ", ".join(f"{k}={results[k][-1]:.3f}" for k in results))

    print("\n=== Summary across prompts (mean ± stderr) ===")
    for name, scores in results.items():
        m, s = float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))
        print(f"{name:20s}: {m:.4f} ± {s:.4f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
