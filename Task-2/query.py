#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from task2.nga_similarity import (
    encode_single_image,
    find_similar,
    load_index,
    load_index_info,
    write_retrieval_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query similar NGA paintings.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/query_results.csv"))
    args = parser.parse_args()

    metadata, embeddings = load_index(args.output_dir)
    index_info = load_index_info(args.output_dir)
    model_name = str(index_info.get("model_name", "resnet50"))
    if args.object_id:
        matches = metadata.index[metadata["objectid"] == args.object_id].tolist()
        if not matches:
            raise SystemExit(f"Object id {args.object_id} not found in index.")
        query_index = matches[0]
        query_embedding = embeddings[query_index]
        query_object_id = int(metadata.iloc[query_index]["objectid"])
    elif args.image_path is not None:
        query_index = None
        query_embedding = encode_single_image(args.image_path, model_name=model_name)
        query_object_id = -1
    else:
        raise SystemExit("Provide either --object-id or --image-path.")

    indices, scores = find_similar(embeddings, query_embedding, top_k=args.top_k, exclude_index=query_index)
    neighbors = []
    for rank, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist()), start=1):
        row = metadata.iloc[idx]
        neighbors.append(
            {
                "rank": rank,
                "objectid": int(row["objectid"]),
                "score": float(score),
                "title": row.get("title", ""),
                "artist_name": row.get("artist_name", ""),
                "classification": row.get("classification", ""),
            }
        )
    write_retrieval_results(args.output_csv, query_object_id, neighbors)

    print(f"Query object id: {query_object_id}")
    for item in neighbors:
        print(
            f"{item['rank']}. objectid={item['objectid']} score={item['score']:.4f} "
            f"artist={item['artist_name']} title={item['title']}"
        )
    print(f"Saved retrieval results to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
