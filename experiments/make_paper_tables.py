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
    return "\n".join([
        f"\\begin{{{env}}}[t]", r"\centering", f"\\{size}",
        *([f"\\setlength{{\\tabcolsep}}{{{colsep_pt}pt}}"] if colsep_pt else []),
        f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule",
        body, r"\bottomrule", r"\end{tabular}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}",
    ])


# Full metric names overrun \textwidth as column headings.
METRIC_SHORT = {"text-embedding-3-small": "OpenAI", "BGE-M3": "BGE-M3",
                "Qwen3-Embedding-0.6B": "Qwen3-Emb",
                "char 4-gram Jaccard": "Jaccard", "char 4-gram Jaccard, 500-char cap": "Jacc.\\,500c"}


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
                 "Tokenization fertility on FLORES-200 devtest: subword tokens each family's "
                 "tokenizer spends per parallel sentence, relative to its own English count. "
                 "Aya Expanse is absent because its tokenizer repository is gated. Basque and "
                 "Polish cost the most; Chinese the least.",
                 "tab:fertility", star=True, colsep_pt=4)


def regression_table(tables: dict[str, Path]) -> str:
    """Predictors x (embedding space x specification) coefficient table."""
    from experiments.run_linguistic import mixed_model_coefs, PREDICTORS
    no_morph = [p for p in PREDICTORS if p != "ttr"]
    pretty = {"lid_match_rate": "Model competence", "log_wiki": "Resource level",
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
                 "predictors (11 languages; morphology is undefined for \\texttt{zh}/"
                 "\\texttt{ja}); B drops morphology to recover all 13; C repeats B on the "
                 "cells with a GlotLID match rate $\\geq 0.80$, confirming the effect is not "
                 "driven by degenerate generation. Model competence is significant in every "
                 "specification and all three embedding spaces; resource level is significant "
                 "only under \\texttt{text-embedding-3-small}, and is therefore a property of "
                 "that embedding geometry rather than a robust finding. "
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
