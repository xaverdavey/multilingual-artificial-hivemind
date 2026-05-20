"""Sweep run_generation.py over a list of models with overlapping prefetch.

Downloads model N+1 while model N is running, then deletes model N's HF cache
once its run finishes. One subprocess per model so vLLM/CUDA state is isolated.

Models default to the `id` field of each entry in `experiments/models.yaml`.
Override with `--models id1 id2 ...`.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from threading import Thread

import yaml
from huggingface_hub import scan_cache_dir, snapshot_download

DEFAULT_MODELS_FILE = Path(__file__).parent / "models.yaml"


def load_models(path: Path) -> list[str]:
    return [m["id"] for m in yaml.safe_load(path.read_text())]


def prefetch(model: str) -> None:
    snapshot_download(model)


def purge(model: str) -> None:
    cache = scan_cache_dir()
    revs = [rev.commit_hash for r in cache.repos if r.repo_id == model for rev in r.revisions]
    if revs:
        cache.delete_revisions(*revs).execute()


def run(model: str, passthrough: list[str]) -> None:
    cmd = [sys.executable, "-m", "experiments.run_generation", "--model", model, *passthrough]
    subprocess.run(cmd, check=True)


def main(models: list[str], passthrough: list[str], keep: bool) -> None:
    fetch = Thread(target=prefetch, args=(models[0],))
    fetch.start()
    for i, model in enumerate(models):
        fetch.join()
        if i + 1 < len(models):
            fetch = Thread(target=prefetch, args=(models[i + 1],))
            fetch.start()
        run(model, passthrough)
        if not keep:
            purge(model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="HF model ids (overrides --models-file)")
    ap.add_argument("--models-file", type=Path, default=DEFAULT_MODELS_FILE,
                    help="YAML file with a top-level list of HF model ids")
    ap.add_argument("--keep", action="store_true", help="Don't delete model from HF cache after run")
    args, passthrough = ap.parse_known_args()
    models = args.models or load_models(args.models_file)
    main(models, passthrough, args.keep)
