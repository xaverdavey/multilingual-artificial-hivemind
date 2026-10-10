"""Sweep run_generation.py (or run_competence.py) over models with overlapping prefetch.

Downloads model N+1 while model N is running, then deletes model N's HF cache
once its run finishes. One subprocess per model so vLLM/CUDA state is isolated.

Models default to the `id` field of each entry in `experiments/models.yaml`.
Override with `--models id1 id2 ...`, or drop some with `--exclude`. Models whose
output parquet already exists are skipped, so an interrupted sweep resumes
where it stopped.

Single-GPU workstation run of the competence benchmarks, e.g.:
    python -m experiments.sweep_models --task competence --keep-going \
        --exclude Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
"""

import argparse
import subprocess
import sys
from pathlib import Path
from threading import Thread

import yaml
from huggingface_hub import scan_cache_dir, snapshot_download

DEFAULT_MODELS_FILE = Path(__file__).parent / "models.yaml"
RESULTS = Path(__file__).parent.parent / "results"
TASKS = {
    "generation": ("experiments.run_generation", RESULTS / "raw"),
    "competence": ("experiments.run_competence", RESULTS / "competence" / "raw"),
}


def load_models(path: Path) -> list[str]:
    """Return HF ids of all cluster-bound entries (entries with a `gpus` field).

    API entries lack `gpus`/`time` and are skipped — they don't go through SLURM.
    """
    return [m["id"] for m in yaml.safe_load(path.read_text()) if m.get("gpus")]


def prefetch(model: str) -> None:
    snapshot_download(model)


def purge(model: str) -> None:
    cache = scan_cache_dir()
    revs = [rev.commit_hash for r in cache.repos if r.repo_id == model for rev in r.revisions]
    if revs:
        cache.delete_revisions(*revs).execute()


def run(module: str, model: str, out_dir: Path, passthrough: list[str]) -> None:
    cmd = [sys.executable, "-m", module, "--model", model, "--out-dir", str(out_dir), *passthrough]
    subprocess.run(cmd, check=True)


def main(models: list[str], task: str, out_dir: Path | None, passthrough: list[str],
         keep: bool, keep_going: bool) -> None:
    module, default_out = TASKS[task]
    out_dir = out_dir or default_out
    done = [m for m in models if (out_dir / f"{m.replace('/', '_')}.parquet").exists()]
    models = [m for m in models if m not in done]
    if done:
        print(f"skipping {len(done)} models already in {out_dir}")
    if not models:
        return

    failed = []
    fetch = Thread(target=prefetch, args=(models[0],))
    fetch.start()
    for i, model in enumerate(models):
        fetch.join()
        if i + 1 < len(models):
            fetch = Thread(target=prefetch, args=(models[i + 1],))
            fetch.start()
        try:
            run(module, model, out_dir, passthrough)
        except subprocess.CalledProcessError:
            if not keep_going:
                raise
            failed.append(model)
            print(f"FAILED: {model} -- continuing")
        if not keep:
            purge(model)
    if failed:
        print(f"{len(failed)} models failed: {' '.join(failed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="HF model ids (overrides --models-file)")
    ap.add_argument("--models-file", type=Path, default=DEFAULT_MODELS_FILE,
                    help="YAML file with a top-level list of HF model ids")
    ap.add_argument("--exclude", nargs="+", default=[], help="HF model ids to leave out")
    ap.add_argument("--task", choices=TASKS, default="generation")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir passed to the task script (default: its own)")
    ap.add_argument("--keep", action="store_true", help="Don't delete model from HF cache after run")
    ap.add_argument("--keep-going", action="store_true",
                    help="Log a failed model and move on instead of stopping the sweep")
    args, passthrough = ap.parse_known_args()
    models = [m for m in (args.models or load_models(args.models_file)) if m not in args.exclude]
    main(models, args.task, args.out_dir, passthrough, args.keep, args.keep_going)
