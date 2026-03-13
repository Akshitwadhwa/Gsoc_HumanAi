#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from task2.nga_similarity import extract_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a similarity index from downloaded NGA images.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/processed/nga_similarity_metadata.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    metadata_out, embed_out = extract_embeddings(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"Saved metadata index to: {metadata_out}")
    print(f"Saved embeddings to: {embed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
