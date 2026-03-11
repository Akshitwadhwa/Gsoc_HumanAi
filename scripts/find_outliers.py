#!/usr/bin/env python3
"""Generate outlier reports from per-image prediction exports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank likely WikiArt outliers from model predictions.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def int_value(row: dict[str, str], key: str) -> int:
    return int(row.get(key, "0") or 0)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]], top_n: int) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    tasks = sorted(
        key[:-8]
        for key in rows[0].keys()
        if key.endswith("_correct") and key != "all_correct"
    )
    error_rows = [row for row in rows if int_value(row, "all_correct") == 0]
    low_confidence_rows = sorted(rows, key=lambda row: float_value(row, "mean_confidence"))[:top_n]
    high_confidence_errors = sorted(
        error_rows,
        key=lambda row: (
            -float_value(row, "mean_confidence"),
            -sum(1 for task in tasks if int_value(row, f"{task}_correct") == 0),
            row["image_path"],
        ),
    )[:top_n]

    task_error_counts = Counter()
    for row in error_rows:
        for task in tasks:
            if int_value(row, f"{task}_correct") == 0:
                task_error_counts[task] += 1

    lines = [
        "# Outlier Summary",
        "",
        f"- Total samples scored: {len(rows)}",
        f"- Samples with at least one wrong task: {len(error_rows)}",
        f"- Top ranked rows per CSV: {top_n}",
        "",
        "## Task Error Counts",
        "",
    ]
    for task in tasks:
        lines.append(f"- `{task}` errors: {task_error_counts[task]}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `high_confidence_errors.csv` contains images the model classified incorrectly despite being confident. These are the strongest label/outlier candidates.",
            "- `low_confidence_samples.csv` contains images with the lowest average confidence across tasks. These are ambiguity candidates.",
        ]
    )
    return high_confidence_errors, low_confidence_rows, lines


def main() -> int:
    args = parse_args()
    rows = read_rows(args.predictions_csv)
    if not rows:
        raise SystemExit(f"No prediction rows found in: {args.predictions_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    high_confidence_errors, low_confidence_rows, summary_lines = build_summary(rows, args.top_n)

    fieldnames = list(rows[0].keys())
    write_csv(args.output_dir / "high_confidence_errors.csv", high_confidence_errors, fieldnames)
    write_csv(args.output_dir / "low_confidence_samples.csv", low_confidence_rows, fieldnames)
    (args.output_dir / "outlier_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote outlier report to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
