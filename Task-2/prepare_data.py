#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from task2.nga_similarity import prepare_similarity_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare NGA metadata for Task 2 similarity retrieval.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--portrait-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    output = prepare_similarity_metadata(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        portrait_only=args.portrait_only,
        max_rows=args.max_rows,
    )
    print(f"Wrote prepared metadata to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
