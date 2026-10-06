"""Emit the camera-ready appendix tables: prompt-level tests and model/reproducibility details.

Inputs
  results/prompt_tests/prompt_tests__{slug}.parquet   from run_prompt_tests
  results/all_generations.parquet                     raw generations (lengths, truncation)
  results/fluency/fluency__glotlid.parquet            think-block / empty-after-strip flags
  experiments/models.yaml                             ids, names, GPUs
  experiments/hub_meta/hf_{commits,dtypes}.json  commit lists and config dtypes fetched from the Hub
                                                 (see experiments/fetch_hub_meta.py)

Outputs (to --out-dir)
  tab_prompt_tests.tex, tab_models_repro.tex, models_repro.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GEN_CUTOFF = "2026-05-27"   # generation finished before the embedding run of 2026-05-24..26

METRICS = [("text-embedding-3-small", "text-embedding-3-small"),
           ("char4gram", "char 4-gram Jaccard"),
           ("char4gram_trunc500", "Jaccard, 500-char cap")]
LEVELS = [("intra", "family", "Intra-model: per family"),
          ("intra", "model", "Intra-model: per model"),
          ("intra", "language", "Intra-model: per language"),
          ("intra", "model_language", "Intra-model: per (model, language)"),
          ("cross", "language", "Inter-model: per language"),
          ("cross", "model", "Inter-model: per model"),
          ("cross", "family_pair", "Inter-model: per family pair"),
          ("cross", "family_pair_language", "Inter-model: per (family pair, language)"),
          ("ladder", "language", "Same vs.\\ different models: per language")]


def prompt_tests_table(pt_dir: Path) -> str:
    rows = []
    for slug, label in METRICS:
        df = pd.read_parquet(pt_dir / f"prompt_tests__{slug}.parquet")
        first = True
        for comp, level, name in LEVELS:
            sub = df[(df.comparison == comp) & (df.level == level)]
            k = len(sub)
            rows.append((label if first else "", name, k,
                         int((sub.mean_gap > 0).sum()),
                         int((sub.ci_lo > 0).sum()),
                         int(((sub.mean_gap > 0) & (sub.p_perm_holm < 0.05)).sum()),
                         int(((sub.mean_gap > 0) & (sub.p_wilcoxon_holm < 0.05)).sum()),
                         int(((sub.mean_gap > 0) & (sub.p_perm_bh < 0.05)).sum())))
            first = False
        rows.append(None)
    body = []
    for r in rows:
        if r is None:
            body.append(r"\midrule")
            continue
        body.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]} \\\\")
    if body[-1] == r"\midrule":
        body.pop()
    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{\textbf{Paired prompt-level tests of the homogenization gap, with multiple-comparison correction.} "
        r"Each group is tested on one LLM-minus-human difference per prompt ($n = 390$ prompts for per-model and "
        r"per-family-pair groups, $30$ otherwise). Columns count the groups at that level ($k$) whose mean gap is positive, "
        r"whose 95\% prompt-bootstrap interval lies above zero, and that are positive and significant at $0.05$ after Holm "
        r"correction of the sign-flip permutation and Wilcoxon signed-rank $p$-values, and after Benjamini--Hochberg "
        r"correction of the permutation $p$-values, each corrected across the $k$ groups of its row (a group that is "
        r"significant in the reverse direction, such as Basque in the inter-model comparison, is not counted). "
        r"The last row of each block tests the top rung of the ladder in Table~\ref{tab:robustness}, one model "
        r"resampled against two different models, which needs no human baseline. Section~\ref{sec:embedding_robustness} "
        r"discusses the lexical inter-model rows, where character overlap between two different models is no "
        r"higher than between two different humans.}",
        r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"\textbf{Metric} & \textbf{Comparison / level} & $k$ & \textbf{gap}$>0$ & \textbf{CI}$>0$ & "
        r"\textbf{perm.\ Holm} & \textbf{Wilc.\ Holm} & \textbf{perm.\ BH} \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\label{tab:prompt_tests}", r"\end{table*}", ""])


def models_table(models_yaml: Path, commits: dict, dtypes: dict, gen: Path, fluency: Path):
    entries = yaml.safe_load(models_yaml.read_text())
    g = pd.read_parquet(gen, columns=["model", "response_text", "finish_reason"])
    g["nchar"] = g.response_text.str.len()
    g["think"] = g.response_text.str.contains("<think>", regex=False)
    g["empty"] = g.response_text.fillna("").str.strip() == ""
    agg = g.groupby("model").agg(nchar=("nchar", "mean"), trunc=("finish_reason", lambda s: (s == "length").mean()),
                                 think=("think", "mean"), n_empty=("empty", "sum"))
    fl = pd.read_parquet(fluency, columns=["model", "empty_after_strip", "had_think"])
    dangling = fl.groupby("model")["empty_after_strip"].sum()
    rows = []
    for e in entries:
        rid, name = e["id"], e["display_name"]
        c = commits.get(rid)
        if isinstance(c, list):
            ok = [x for x in c if x["date"][:10] <= GEN_CUTOFF]
            rev = ok[0]["sha"][:8] if ok else "---"
        else:
            rev = "gated"
        d = dtypes.get(rid, {})
        prec = ("FP8" if d.get("quant") == "fp8" else {"bfloat16": "bf16", "float16": "fp16"}.get(d.get("dtype"), "---"))
        gpus = e.get("gpus", "")
        kind, _, n = gpus.partition(":")
        gpu = {"A40": "A40 48\\,GB", "A100fat": "A100 80\\,GB"}.get(kind, kind)
        a = agg.loc[rid]
        rows.append({"model": name, "hf_id": rid, "revision": rev, "precision": prec,
                     "gpus": f"{n}$\\times$ {gpu}", "mean_chars": round(float(a.nchar)),
                     "trunc_pct": round(100 * float(a.trunc), 1), "think": "yes" if a.think > 0.5 else "no",
                     "n_empty": int(a["n_empty"]), "dangling_think": int(dangling.get(rid, 0))})
    df = pd.DataFrame(rows)
    body = []
    for r in df.itertuples():
        hf = r.hf_id.replace("_", r"\_")
        body.append(f"{r.model} & \\texttt{{{hf}}} & \\texttt{{{r.revision}}} & {r.precision} & {r.gpus} & "
                    f"{r.mean_chars:,} & {r.trunc_pct:.1f} & {r.think} \\\\")
    tex = "\n".join([
        r"\begin{table*}[t]",
        r"\caption{\textbf{The 23 open-weight model variants and how each was run.} Revision is the Hugging Face commit "
        r"that was the repository head when generation ran. Precision is the dtype declared in the checkpoint's config, "
        r"which vLLM serves by default. Chars is the mean response length in "
        r"characters, reasoning block included where present; Trunc.\ is the share of responses cut off by the 2048-token "
        r"limit; Think marks checkpoints whose responses carry a \texttt{<think>} reasoning block.}",
        r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llllrrrl}", r"\toprule",
        r"\textbf{Model} & \textbf{Checkpoint} & \textbf{Rev.} & \textbf{Prec.} & \textbf{GPUs} & "
        r"\textbf{Chars} & \textbf{Trunc.\ \%} & \textbf{Think} \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\label{tab:model-families}", r"\end{table*}", ""])
    return tex, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", type=Path, default=Path("results/prompt_tests"))
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("--commits", type=Path, default=Path("experiments/hub_meta/hf_commits.json"))
    ap.add_argument("--dtypes", type=Path, default=Path("experiments/hub_meta/hf_dtypes.json"))
    ap.add_argument("--gen", type=Path, default=Path("results/all_generations.parquet"))
    ap.add_argument("--fluency", type=Path, default=Path("results/fluency/fluency__glotlid.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tab_prompt_tests.tex").write_text(prompt_tests_table(args.pt_dir))
    tex, df = models_table(args.models_yaml, json.load(open(args.commits)), json.load(open(args.dtypes)),
                           args.gen, args.fluency)
    (args.out_dir / "tab_models_repro.tex").write_text(tex)
    df.to_csv(args.out_dir / "models_repro.csv", index=False)
    print(df.to_string())
    print("total empty:", df.n_empty.sum(), " total dangling-think (Qwen3 dense):", df[df.think == "yes"].dangling_think.sum())


if __name__ == "__main__":
    main()
