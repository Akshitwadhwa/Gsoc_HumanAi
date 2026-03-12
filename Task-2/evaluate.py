#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from task2.nga_similarity import evaluate_retrieval, load_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for Task 2.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--label-column", type=str, default="artist_name")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    metadata, embeddings = load_index(args.output_dir)
    metrics = evaluate_retrieval(metadata, embeddings, label_column=args.label_column, top_k=args.top_k)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
