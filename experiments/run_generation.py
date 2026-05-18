"""Run one model over the full prompt set; save raw generations to parquet.

Designed for one SLURM job per model. vLLM batches across prompts internally;
we just hand it the full list.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
import vllm
from vllm import LLM, SamplingParams

SEED = 42
JIANG = SamplingParams(n=50, temperature=1.0, top_p=0.9, max_tokens=2048, seed=SEED)


def load_prompt_pool(prompts_dir: Path) -> pd.DataFrame:
    """Concatenate OASST2 + Aya-eval + designed-stubs prompt files into one frame.
    Each file must have columns: prompt_id, language, prompt_text; human_responses optional.
    """
    frames = []
    for f in sorted(prompts_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["dataset"] = f.stem
        if "human_responses" not in df.columns:
            df["human_responses"] = None
        frames.append(df[["prompt_id", "language", "prompt_text", "human_responses", "dataset"]])
    return pd.concat(frames, ignore_index=True)


def prompt_set_sha256(prompts_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(prompts_dir.glob("*.parquet")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main(model: str, prompts_dir: Path, out_dir: Path, gpu_mem_util: float):
    prompts = load_prompt_pool(prompts_dir)
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem_util)

    messages = [[{"role": "user", "content": t}] for t in prompts["prompt_text"]]
    t0 = time.time()
    outputs = llm.chat(messages, JIANG)  # vLLM handles batching internally
    elapsed = time.time() - t0

    rows = []
    for prompt_row, out in zip(prompts.itertuples(index=False), outputs):
        for i, gen in enumerate(out.outputs):
            rows.append({
                "model": model,
                "dataset": prompt_row.dataset,
                "language": prompt_row.language,
                "prompt_id": prompt_row.prompt_id,
                "prompt_text": prompt_row.prompt_text,
                "human_responses": prompt_row.human_responses,
                "sample_idx": i,
                "response_text": gen.text,
                "n_tokens": len(gen.token_ids),
                "finish_reason": gen.finish_reason,
            })

    out_path = out_dir / f"{model.replace('/', '_')}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote {len(rows)} rows to {out_path}")

    cfg = llm.llm_engine.model_config
    meta = {
        "model": model,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "dtype": str(cfg.dtype),
        "quantization": cfg.quantization,
        "max_model_len": cfg.max_model_len,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_utilization": gpu_mem_util,
        "sampling": {"n": JIANG.n, "temperature": JIANG.temperature,
                     "top_p": JIANG.top_p, "max_tokens": JIANG.max_tokens, "seed": JIANG.seed},
        "prompt_set_sha256": prompt_set_sha256(prompts_dir),
        "n_prompts": len(prompts),
        "n_rows": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote metadata to {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-72B-Instruct-FP8")
    ap.add_argument("--prompts-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "raw")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9,
                    help="Fraction of GPU memory vLLM may use (lower on shared GPUs)")
    args = ap.parse_args()
    main(args.model, args.prompts_dir, args.out_dir, args.gpu_mem_util)
