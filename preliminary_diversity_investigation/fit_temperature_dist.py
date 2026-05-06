"""Fit a temperature distribution mu(T) by EM so the model's per-token mixture
distribution maximizes likelihood on a target corpus.

p_T(x_i | x_<i) ∝ p(x_i | x_<i)^(1/T)
p_mu(x_i) = Σ_T mu(T) * p_T(x_i)
maximize Σ_i log p_mu(x_i) over mu (a distribution on a finite T grid).

This is mixture-of-categorical mixture estimation. Standard EM, convex M-step.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def collect_log_p_T(model, tokenizer, texts, T_grid, max_tokens=512):
    """Return [N_tokens_total, K_temperatures] of log p_T(target | context)."""
    device = next(model.parameters()).device
    rows = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
        if ids.size(1) < 2:
            continue
        with torch.no_grad():
            logits = model(ids).logits.float()[:, :-1, :]   # predict next token
        targets = ids[:, 1:]
        per_T = []
        for T in T_grid:
            log_p = F.log_softmax(logits / T, dim=-1)
            per_T.append(log_p.gather(2, targets.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().numpy())
        rows.append(np.stack(per_T, axis=-1))
    return np.concatenate(rows, axis=0)


def fit_em(log_p_T, n_iter=100, tol=1e-6):
    N, K = log_p_T.shape
    w = np.ones(K) / K
    last_nll = None
    for it in range(n_iter):
        log_joint = log_p_T + np.log(w + 1e-12)
        log_norm = np.logaddexp.reduce(log_joint, axis=-1, keepdims=True)
        post = np.exp(log_joint - log_norm)
        w_new = post.mean(axis=0)
        w_new /= w_new.sum()
        nll = -log_norm.mean()
        change = np.abs(w_new - w).max()
        w = w_new
        if it % 5 == 0 or it == n_iter - 1:
            print(f"  iter {it:3d}: nll={nll:.4f} max_dw={change:.6f}")
        if last_nll is not None and abs(last_nll - nll) < tol:
            break
        last_nll = nll
    return w, float(nll)


def load_corpus(name, n_docs):
    if name == "poetry":
        ds = load_dataset("merve/poetry", split="train")
        return [r["content"] for r in ds.select(range(min(n_docs, len(ds))))]
    elif name == "wiki":
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        out = []
        for r in ds:
            if len(r["text"]) > 200:
                out.append(r["text"][:3000])
            if len(out) >= n_docs:
                break
        return out
    else:
        raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--corpus", choices=["poetry", "wiki"], default="poetry")
    ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--t-min", type=float, default=0.4)
    ap.add_argument("--t-max", type=float, default=2.0)
    ap.add_argument("--n-grid", type=int, default=17)
    ap.add_argument("--out", default="temp_dist_fit.json")
    args = ap.parse_args()

    T_grid = np.linspace(args.t_min, args.t_max, args.n_grid)
    print(f"T grid: {T_grid.round(3).tolist()}")

    print(f"Loading corpus '{args.corpus}'...")
    texts = load_corpus(args.corpus, args.n_docs)
    print(f"  {len(texts)} docs")

    print(f"Loading model '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda().eval()

    print("Collecting log p_T per token...")
    log_p_T = collect_log_p_T(model, tokenizer, texts, T_grid.tolist(), args.max_tokens)
    print(f"  shape: {log_p_T.shape}")

    print("Fitting EM...")
    w, final_nll = fit_em(log_p_T)

    print("\n=== Fitted temperature distribution ===")
    bar_w = max(w)
    for T, wt in zip(T_grid, w):
        bar = "█" * int(40 * wt / bar_w)
        print(f"  T={T:.2f}: w={wt:.4f}  {bar}")

    mean_T = (T_grid * w).sum()
    median_T = T_grid[np.searchsorted(np.cumsum(w), 0.5)]
    p95_T = T_grid[np.searchsorted(np.cumsum(w), 0.95)]
    print(f"\n  final NLL = {final_nll:.4f}")
    print(f"  mean T = {mean_T:.3f}, median T = {median_T:.3f}, p95 T = {p95_T:.3f}")

    with open(args.out, "w") as f:
        json.dump({
            "args": vars(args),
            "T_grid": T_grid.tolist(),
            "weights": w.tolist(),
            "final_nll": final_nll,
            "summary": {"mean": float(mean_T), "median": float(median_T), "p95": float(p95_T)},
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
