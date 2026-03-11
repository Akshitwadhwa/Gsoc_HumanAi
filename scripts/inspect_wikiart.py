#!/usr/bin/env python3
"""Inspect WikiArt-style dataset CSV splits for Day 1 analysis."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CSV_CANDIDATE_NAMES = {
    "train.csv": "train",
    "training.csv": "train",
    "val.csv": "val",
    "valid.csv": "val",
    "validation.csv": "val",
    "test.csv": "test",
}

TASK_NAMES = {"artist", "genre", "style"}


@dataclass
class Record:
    task: str
    split: str
    csv_path: Path
    image_path: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing the WikiArt dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated Day 1 summaries.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check whether image files referenced in CSVs exist.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def infer_split(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if name in CSV_CANDIDATE_NAMES:
        return CSV_CANDIDATE_NAMES[name]

    stem = csv_path.stem.lower()
    for candidate_name, split in CSV_CANDIDATE_NAMES.items():
        candidate_stem = Path(candidate_name).stem
        if stem == candidate_stem or stem.endswith(f"_{candidate_stem}"):
            return split
    return stem


def infer_task(csv_path: Path) -> str:
    for part in reversed(csv_path.parts):
        normalized = normalize_name(part)
        for task in TASK_NAMES:
            if task in normalized:
                return task
    return "unknown"


def read_rows(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values = [value.strip() for value in row if value.strip()]
            if len(values) < 2:
                continue
            image_path = values[0]
            label = values[-1]
            lowered_image_path = image_path.lower()
            lowered_label = label.lower()
            if "path to image" in lowered_image_path or "groundtruth class" in lowered_label:
                continue
            if not image_path:
                continue
            rows.append((image_path, label))
    return rows


def collect_records(dataset_root: Path) -> list[Record]:
    records: list[Record] = []
    csv_paths = sorted(dataset_root.rglob("*.csv"))
    for csv_path in csv_paths:
        split = infer_split(csv_path)
        task = infer_task(csv_path)
        for image_path, label in read_rows(csv_path):
            records.append(
                Record(
                    task=task,
                    split=split,
                    csv_path=csv_path,
                    image_path=image_path,
                    label=label,
                )
            )
    return records


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def file_exists(dataset_root: Path, image_path: str) -> bool:
    candidates = [
        dataset_root / image_path,
        dataset_root / "wikiart" / image_path,
    ]
    return any(candidate.exists() for candidate in candidates)


def build_split_summary(records: list[Record]) -> list[tuple[str, str, int, int]]:
    grouped: dict[tuple[str, str], list[Record]] = {}
    for record in records:
        grouped.setdefault((record.task, record.split), []).append(record)

    rows: list[tuple[str, str, int, int]] = []
    for (task, split), task_records in sorted(grouped.items()):
        label_count = len({record.label for record in task_records})
        rows.append((task, split, len(task_records), label_count))
    return rows


def build_class_balance(records: list[Record]) -> list[tuple[str, str, str, int]]:
    grouped: dict[tuple[str, str], Counter[str]] = {}
    for record in records:
        grouped.setdefault((record.task, record.split), Counter())[record.label] += 1

    rows: list[tuple[str, str, str, int]] = []
    for (task, split), counts in sorted(grouped.items()):
        for label, count in counts.most_common():
            rows.append((task, split, label, count))
    return rows


def build_missing_files(records: list[Record], dataset_root: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for record in records:
        if not file_exists(dataset_root, record.image_path):
            rows.append((record.task, record.split, record.label, record.image_path))
    return rows


def summarize_balance(records: list[Record]) -> list[str]:
    lines: list[str] = []
    grouped: dict[tuple[str, str], Counter[str]] = {}
    for record in records:
        grouped.setdefault((record.task, record.split), Counter())[record.label] += 1

    for (task, split), counts in sorted(grouped.items()):
        class_sizes = sorted(counts.values(), reverse=True)
        if not class_sizes:
            continue
        max_size = class_sizes[0]
        min_size = class_sizes[-1]
        imbalance_ratio = max_size / min_size if min_size else float("inf")
        lines.append(
            f"- `{task}` / `{split}`: {len(class_sizes)} classes, "
            f"largest class = {max_size}, smallest class = {min_size}, "
            f"imbalance ratio = {imbalance_ratio:.2f}"
        )
    return lines


def write_markdown_summary(
    path: Path,
    dataset_root: Path,
    split_rows: list[tuple[str, str, int, int]],
    balance_lines: list[str],
    missing_count: int | None,
) -> None:
    lines = [
        "# Day 1 Dataset Summary",
        "",
        f"Dataset root: `{dataset_root}`",
        "",
        "## Split Summary",
        "",
        "| Task | Split | Samples | Unique Labels |",
        "|---|---:|---:|---:|",
    ]
    for task, split, samples, labels in split_rows:
        lines.append(f"| {task} | {split} | {samples} | {labels} |")

    lines.extend(["", "## Class Imbalance", ""])
    if balance_lines:
        lines.extend(balance_lines)
    else:
        lines.append("- No records found.")

    if missing_count is not None:
        lines.extend(["", "## Missing Files", "", f"- Missing referenced image files: {missing_count}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    records = collect_records(dataset_root)
    split_rows = build_split_summary(records)
    balance_rows = build_class_balance(records)

    write_csv(
        output_dir / "split_summary.csv",
        ["task", "split", "samples", "unique_labels"],
        split_rows,
    )
    write_csv(
        output_dir / "class_balance.csv",
        ["task", "split", "label", "count"],
        balance_rows,
    )

    missing_count = None
    if args.check_files:
        missing_rows = build_missing_files(records, dataset_root)
        missing_count = len(missing_rows)
        write_csv(
            output_dir / "missing_files.csv",
            ["task", "split", "label", "image_path"],
            missing_rows,
        )

    write_markdown_summary(
        output_dir / "dataset_summary.md",
        dataset_root,
        split_rows,
        summarize_balance(records),
        missing_count,
    )

    print(f"Wrote Day 1 outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
