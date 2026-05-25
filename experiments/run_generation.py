"""Run one model over the full prompt set; save raw generations to parquet.

Designed for one SLURM job per model. Backend is auto-detected from the model
name: claude-* → Anthropic API, gpt-*/o*-* → OpenAI API, gemini-* → Google
Gemini API, everything else → vLLM.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SEED = 42
N_SAMPLES = 50
TEMPERATURE = 1.0
TOP_P = 0.9
MAX_TOKENS = 2048


def detect_backend(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if model.startswith("gemini-"):
        return "gemini"
    return "vllm"


def load_prompt_pool(prompts_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(prompts_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["dataset"] = f.stem
        if "human_responses" not in df.columns:
            df["human_responses"] = None
        frames.append(df[["prompt_id", "language", "prompt_text", "human_responses", "dataset"]])
    if not frames:
        raise FileNotFoundError(
            f"No prompt parquets found in {prompts_dir}\n"
            f"Run first: python -m experiments.select_oasst2_prompts"
        )
    return pd.concat(frames, ignore_index=True)


def prompt_set_sha256(prompts_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(prompts_dir.glob("*.parquet")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _row(model: str, prompt_row, sample_idx: int, text: str, n_tokens, finish_reason) -> dict:
    return {
        "model": model,
        "dataset": prompt_row.dataset,
        "language": prompt_row.language,
        "prompt_id": prompt_row.prompt_id,
        "prompt_text": prompt_row.prompt_text,
        "human_responses": prompt_row.human_responses,
        "sample_idx": sample_idx,
        "response_text": text,
        "n_tokens": n_tokens,
        "finish_reason": finish_reason,
    }


def run_vllm(model: str, prompts: pd.DataFrame, gpu_mem_util: float) -> tuple[list[dict], dict]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    sampling = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                              max_tokens=MAX_TOKENS, seed=SEED)
    # Shard across whatever GPUs SLURM gave us; vLLM defaults to TP=1 otherwise.
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem_util,
              tensor_parallel_size=max(1, torch.cuda.device_count()))
    messages = [[{"role": "user", "content": t}] for t in prompts["prompt_text"]]
    t0 = time.time()
    outputs = llm.chat(messages, sampling)
    elapsed = time.time() - t0

    rows = []
    for prompt_row, out in zip(prompts.itertuples(index=False), outputs):
        for i, gen in enumerate(out.outputs):
            rows.append(_row(model, prompt_row, i, gen.text, len(gen.token_ids), gen.finish_reason))

    cfg = llm.llm_engine.model_config
    meta_extra = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "dtype": str(cfg.dtype),
        "quantization": cfg.quantization,
        "max_model_len": cfg.max_model_len,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_utilization": gpu_mem_util,
        "elapsed_seconds": round(elapsed, 1),
    }
    return rows, meta_extra


def run_openai(model: str, prompts: pd.DataFrame) -> tuple[list[dict], dict]:
    from openai import OpenAI
    client = OpenAI()

    rows = []
    t0 = time.time()
    for prompt_row in prompts.itertuples(index=False):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_row.prompt_text}],
            n=N_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        for i, choice in enumerate(resp.choices):
            # per-choice token count unavailable when n>1; total is in resp.usage
            rows.append(_row(model, prompt_row, i, choice.message.content, None, choice.finish_reason))

    return rows, {"elapsed_seconds": round(time.time() - t0, 1)}


def run_anthropic(model: str, prompts: pd.DataFrame) -> tuple[list[dict], dict]:
    import anthropic
    client = anthropic.Anthropic()

    def _sample(prompt_row, idx: int) -> dict:
        msg = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[{"role": "user", "content": prompt_row.prompt_text}],
        )
        return _row(model, prompt_row, idx,
                    msg.content[0].text, msg.usage.output_tokens, msg.stop_reason)

    rows = []
    t0 = time.time()
    for prompt_row in prompts.itertuples(index=False):
        with ThreadPoolExecutor(max_workers=N_SAMPLES) as pool:
            futures = [pool.submit(_sample, prompt_row, i) for i in range(N_SAMPLES)]
            prompt_rows = [f.result() for f in as_completed(futures)]
        prompt_rows.sort(key=lambda r: r["sample_idx"])
        rows.extend(prompt_rows)

    return rows, {"elapsed_seconds": round(time.time() - t0, 1)}


def run_gemini(model: str, prompts: pd.DataFrame) -> tuple[list[dict], dict]:
    from google import genai
    from google.genai import types
    client = genai.Client()

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_TOKENS,
    )

    def _sample(prompt_row, idx: int) -> dict:
        resp = client.models.generate_content(
            model=model, contents=prompt_row.prompt_text, config=config,
        )
        text = resp.text or ""
        n_tok = resp.usage_metadata.candidates_token_count if resp.usage_metadata else None
        finish = resp.candidates[0].finish_reason.name if resp.candidates else None
        return _row(model, prompt_row, idx, text, n_tok, finish)

    rows = []
    t0 = time.time()
    for prompt_row in prompts.itertuples(index=False):
        with ThreadPoolExecutor(max_workers=N_SAMPLES) as pool:
            futures = [pool.submit(_sample, prompt_row, i) for i in range(N_SAMPLES)]
            prompt_rows = [f.result() for f in as_completed(futures)]
        prompt_rows.sort(key=lambda r: r["sample_idx"])
        rows.extend(prompt_rows)

    return rows, {"elapsed_seconds": round(time.time() - t0, 1)}


def main(model: str, prompts_dir: Path, out_dir: Path, gpu_mem_util: float):
    prompts = load_prompt_pool(prompts_dir)
    backend = detect_backend(model)

    if backend == "vllm":
        rows, meta_extra = run_vllm(model, prompts, gpu_mem_util)
    elif backend == "openai":
        rows, meta_extra = run_openai(model, prompts)
    elif backend == "gemini":
        rows, meta_extra = run_gemini(model, prompts)
    else:
        rows, meta_extra = run_anthropic(model, prompts)

    out_path = out_dir / f"{model.replace('/', '_')}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote {len(rows)} rows to {out_path}")

    meta = {
        "model": model,
        "backend": backend,
        "sampling": {"n": N_SAMPLES, "temperature": TEMPERATURE,
                     "top_p": TOP_P, "max_tokens": MAX_TOKENS, "seed": SEED},
        "prompt_set_sha256": prompt_set_sha256(prompts_dir),
        "n_prompts": len(prompts),
        "n_rows": len(rows),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **meta_extra,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote metadata to {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or API model name (e.g. claude-sonnet-4-6, gpt-5)")
    ap.add_argument("--prompts-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "results" / "raw")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9,
                    help="vLLM only: fraction of GPU memory to use")
    args = ap.parse_args()
    main(args.model, args.prompts_dir, args.out_dir, args.gpu_mem_util)
