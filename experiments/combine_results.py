"""Concatenate per-model parquets in results/raw/ into a single file.

Each per-model parquet already carries a `model` column (set in
run_generation.py), so distinguishing rows by model is automatic.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RAW = Path(__file__).parent.parent / "results" / "raw"
DEFAULT_OUT = Path(__file__).parent.parent / "results" / "all_generations.parquet"


def main(raw_dir: Path, out_path: Path) -> None:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquets under {raw_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"merged {len(files)} files -> {out_path} ({len(df)} rows)")
    print(df.groupby("model").size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    main(args.raw_dir, args.out)
