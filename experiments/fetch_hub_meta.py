"""Snapshot Hugging Face metadata for every model in models.yaml (plus the embedders and GlotLID).

Writes experiments/hub_meta/hf_commits.json (full commit history per repo, newest first) and
hf_dtypes.json (torch_dtype / quantization / architectures from config.json). Gated repos the
token cannot read are recorded with an "error" entry. make_repro_tables reads both files.
"""

import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

EXTRA = ["BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B", "cis-lmu/glotlid"]


def main(models_yaml: Path, out_dir: Path) -> None:
    api = HfApi()
    ids = [e["id"] for e in yaml.safe_load(models_yaml.read_text())] + EXTRA
    commits, dtypes = {}, {}
    for rid in ids:
        try:
            commits[rid] = [{"sha": c.commit_id, "date": c.created_at.isoformat(), "title": c.title}
                            for c in api.list_repo_commits(rid)]
        except Exception as ex:  # gated or missing
            commits[rid] = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
        if rid in EXTRA:
            continue
        try:
            cfg = json.load(open(hf_hub_download(rid, "config.json")))
            tc = cfg.get("text_config", {})
            q = cfg.get("quantization_config") or {}
            dtypes[rid] = {"dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tc.get("torch_dtype"),
                           "quant": q.get("quant_method") or q.get("fmt") or "",
                           "arch": cfg.get("architectures"),
                           "max_pos": cfg.get("max_position_embeddings") or tc.get("max_position_embeddings")}
        except Exception as ex:
            dtypes[rid] = {"error": str(ex)[:120]}
        print(rid, "ok" if isinstance(commits[rid], list) else "gated")
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(commits, open(out_dir / "hf_commits.json", "w"), indent=1)
    json.dump(dtypes, open(out_dir / "hf_dtypes.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-yaml", type=Path, default=Path("experiments/models.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/hub_meta"))
    args = ap.parse_args()
    main(args.models_yaml, args.out_dir)
