#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from task2.nga_similarity import download_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NGA open-access images for Task 2.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/processed/nga_similarity_metadata.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    downloaded = download_images(args.metadata_csv, args.image_dir, limit=args.limit)
    print(f"Downloaded {downloaded} images into: {args.image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
