"""Emit the appendix LaTeX tables from the analysis outputs.

Regenerating these from the parquets keeps the paper in sync with the data
rather than relying on numbers transcribed by hand. Writes one .tex per table
into --out-dir, each meant to be \\input from the appendix.
"""

import argparse
from pathlib import Path

import pandas as pd

LANG_ORDER = ["en", "de", "fr", "es", "it", "pt-BR", "ca", "pl", "ru", "vi", "zh", "ja", "eu"]

# Results directory -> column label for each extra embedding space.
ALT_LABELS = {"linguistic_bge": "BGE-M3", "linguistic_qwen": "Qwen3-Emb"}


def _wrap(body: str, header: str, colspec: str, caption: str, label: str,
          star: bool = False, size: str = "small", colsep_pt: float | None = None) -> str:
    """colsep_pt tightens inter-column padding; wide tables overflow \textwidth without it."""
    env = "table*" if star else "table"
    # Paper convention: caption above the tabular, first sentence in bold.
    if not caption.startswith("\\textbf"):
        title, sep, rest = caption.partition(". ")
        caption = f"\\textbf{{{title}.}} {rest}" if sep else caption
    return "\n".join([
        f"\\begin{{{env}}}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\{size}",
        *([f"\\setlength{{\\tabcolsep}}{{{colsep_pt}pt}}"] if colsep_pt else []),
        f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule",
        body, r"\bottomrule", r"\end{tabular}",
        f"\\label{{{label}}}", f"\\end{{{env}}}",
    ])


# Full metric names overrun \textwidth as column headings.
METRIC_SHORT = {"text-embedding-3-small": "OpenAI", "BGE-M3": "BGE-M3",
                "Qwen3-Embedding-0.6B": "Qwen3-Emb",
                "char 4-gram Jaccard": "Jaccard", "char 4-gram Jaccard, 500-char cap": "Jacc.\\,500c",
                "BGE-M3, 500-char cap (10 samples/prompt)": "BGE\\,500c"}


# slug -> per-(model pair, prompt) inter-similarity parquet, for the "different models" rung
INTER_PATHS = {
    "text-embedding-3-small": Path("results/metrics/inter__text-embedding-3-small.parquet"),
    "BAAI_bge-m3": Path("results/metrics/inter__BAAI_bge-m3.parquet"),
    "Qwen_Qwen3-Embedding-0.6B": Path("results/metrics/inter__Qwen_Qwen3-Embedding-0.6B.parquet"),
    "char4gram": Path("results/lexical/inter__char4gram.parquet"),
    "char4gram_trunc500": Path("results/lexical/inter__char4gram_trunc500.parquet"),
    "BAAI_bge-m3_trunc500_s10": Path("results/metrics/inter__BAAI_bge-m3_trunc500_s10.parquet"),
}
PRIMARY_SLUG = "text-embedding-3-small"


def robustness_summary(path: Path) -> str:
    """The similarity ladder (same model / different models / different humans) per metric.

    Same-model and different-humans values come from run_robustness' summary CSV;
    the different-models rung is the mean inter-similarity over every (model pair,
    prompt) row, which is what the cross-model F-test in run_metrics pools too.
    """
    df = pd.read_csv(path)
    df["inter"] = df["slug"].map(
        lambda sl: float(pd.read_parquet(INTER_PATHS[sl], columns=["inter_sim"])["inter_sim"].mean()))
    df = pd.concat([df[df["slug"] == PRIMARY_SLUG], df[df["slug"] != PRIMARY_SLUG]])
    body = "\n".join(
        f"{r['metric'].split(' (')[0]} & {r['llm_intra']:.3f} & {r['inter']:.3f} & {r['human_intra']:.3f} & "
        f"{r['langs_positive']}/{r['n_langs']} & "
        + ("---" if pd.isna(r.get("spearman_vs_primary")) else f"{r['spearman_vs_primary']:.2f}")
        + r" \\" for _, r in df.iterrows())
    return _wrap(body,
                 r"\textbf{Similarity metric} & \textbf{Same model} & \textbf{Diff.\ models} & "
                 r"\textbf{Diff.\ humans} & \textbf{Langs.} & \textbf{$\rho$} \\",
                 "@{}lccccc@{}",
                 "\\textbf{The similarity ladder under three embedding spaces and an embedder-free "
                 "lexical metric.} Mean similarity between two answers to the same prompt when they come "
                 "from the same model resampled, from two different models, and from two different humans, "
                 "averaged over all prompts, models and model pairs. \\textbf{Langs.} counts languages in "
                 "which the same-model value exceeds the human value; $\\rho$ is the Spearman correlation of "
                 "the per-cell same-model-minus-human gap against the primary metric. The ordering holds in "
                 "every embedding space; under the lexical metric the different-models rung falls to the "
                 "human level. The BGE-M3 500-char row re-embeds every response cut to its first 500 "
                 "characters, on 10 of the 50 samples per prompt.",
                 "tab:robustness", colsep_pt=4)


def robustness_by_language(path: Path) -> str:
    df = pd.read_csv(path, index_col=0)
    short = {c: METRIC_SHORT.get(c.split(" (")[0], c.split(" (")[0]) for c in df.columns}
    df = df.rename(columns=short).reindex(LANG_ORDER)
    cols = list(df.columns)
    header = r"\textbf{Lang.} & " + " & ".join(f"\\textbf{{{c}}}" for c in cols) + r" \\"
    body = "\n".join(
        f"\\texttt{{{lang}}} & " + " & ".join(f"${df.loc[lang, c]:+.3f}$" for c in cols) + r" \\"
        for lang in df.index)
    return _wrap(body, header, "l" + "c" * len(cols),
                 "Per-language homogenization gap (LLM minus human intra-similarity) under "
                 "each similarity metric, averaged over the 23 models. Every language shows a "
                 "positive gap under every metric.",
                 "tab:robustness_by_language", star=True, colsep_pt=3.5)


def fertility_table(path: Path) -> str:
    df = pd.read_parquet(path)
    # fertility is a property of the tokenizer, so collapse to one row per family
    piv = (df.groupby(["family", "language"])["fertility_vs_en"].mean()
           .unstack("language").reindex(columns=LANG_ORDER))
    header = r"\textbf{Family} & " + " & ".join(f"\\texttt{{{l}}}" for l in piv.columns) + r" \\"
    body = "\n".join(
        f"{fam} & " + " & ".join(
            "---" if pd.isna(v) else f"{v:.2f}" for v in piv.loc[fam]) + r" \\"
        for fam in piv.index)
    return _wrap(body, header, "l" + "c" * len(piv.columns),
                 "Tokenization fertility on FLORES-200 devtest. We report subword tokens each family's "
                 "tokenizer spends per parallel sentence, relative to its own English count. "
                 "Basque and Polish cost the most; Chinese the least.",
                 "tab:fertility", star=True, colsep_pt=4)


def regression_table(tables: dict[str, Path]) -> str:
    """Predictors x (embedding space x specification) coefficient table."""
    from experiments.run_linguistic import mixed_model_coefs, PREDICTORS
    no_morph = [p for p in PREDICTORS if p != "ttr"]
    pretty = {"lid_match_rate": "Target-lang.\\ fidelity", "log_wiki": "Resource level",
              "fertility_vs_en": "Tokenization fertility", "ttr": "Morph. complexity"}

    fits: dict[tuple[str, str], pd.DataFrame] = {}
    for emb, path in tables.items():
        df = pd.read_parquet(path)
        for spec, preds, d in [("A", PREDICTORS, df), ("B", no_morph, df),
                               ("C", no_morph, df[df["lid_match_rate"] >= 0.80])]:
            fits[(emb, spec)] = mixed_model_coefs(d, preds).set_index("predictor")

    embs = list(tables)
    order = ["lid_match_rate", "log_wiki", "fertility_vs_en", "ttr"]
    header = (r"\textbf{Space} & \textbf{Spec} & "
              + " & ".join(f"\\textbf{{{pretty[p]}}}" for p in order) + r" \\")

    def cell(f, pred):
        if pred not in f.index:
            return "---"
        r = f.loc[pred]
        star = "^{***}" if r["p_value"] < .001 else ("^{*}" if r["p_value"] < .05 else "")
        return f"${r['coef']:+.3f}{star}$"

    rows = []
    for i, emb in enumerate(embs):
        if i:
            rows.append(r"\midrule")
        for j, spec in enumerate(("A", "B", "C")):
            f = fits[(emb, spec)]
            label = emb if j == 0 else ""
            rows.append(f"{label} & {spec} & "
                        + " & ".join(cell(f, p) for p in order) + r" \\")

    return _wrap("\n".join(rows), header, "ll" + "c" * len(order),
                 "Standardised coefficients from the crossed-effects model of the "
                 "homogenization gap, fitted in each embedding space. Spec A uses all four "
                 "predictors on the 11 languages for which morphology is defined (253 cells; the "
                 "type-token ratio is undefined for \\texttt{zh}/\\texttt{ja}); B drops morphology "
                 "to recover all 13 languages (299 cells); C repeats B on the 277 cells with a "
                 "GlotLID match rate $\\geq 0.80$, confirming the effect is not driven by degenerate "
                 "generation. Target-language fidelity (the competence proxy) is significant in every "
                 "specification and all three embedding spaces. Resource level reaches $p<0.05$ only "
                 "under \\texttt{text-embedding-3-small}, and tokenization fertility only in "
                 "specification C under two embedders; neither is a robust finding. "
                 "$^{*}p<0.05$, $^{***}p<0.001$.",
                 "tab:regression", star=True, colsep_pt=3.5)


def main(robustness_csv: Path, fertility: Path, ling_table: Path, out_dir: Path,
         alt_ling_tables: list[Path] | None = None):
    ling_tables = {"OpenAI": ling_table}
    for alt in alt_ling_tables or []:
        if alt.exists():
            ling_tables[ALT_LABELS.get(alt.parent.name, alt.parent.name)] = alt
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in [
        ("tab_robustness", robustness_summary(robustness_csv.parent / "robustness_summary.csv")),
        ("tab_robustness_by_language", robustness_by_language(robustness_csv)),
        ("tab_fertility", fertility_table(fertility)),
        ("tab_regression", regression_table(ling_tables)),
    ]:
        (out_dir / f"{name}.tex").write_text(text + "\n")
        print(f"wrote {out_dir/name}.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness-csv", type=Path,
                    default=Path("results/robustness/robustness_by_language.csv"))
    ap.add_argument("--fertility", type=Path, default=Path("results/linguistic/fertility.parquet"))
    ap.add_argument("--ling-table", type=Path,
                    default=Path("results/linguistic/linguistic_table.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    ap.add_argument("--alt-ling-tables", type=Path, nargs="*",
                    default=[Path("results/linguistic_bge/linguistic_table.parquet"),
                             Path("results/linguistic_qwen/linguistic_table.parquet")])
    args = ap.parse_args()
    main(args.robustness_csv, args.fertility, args.ling_table, args.out_dir,
         args.alt_ling_tables)
