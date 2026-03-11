#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def slugify(value: str) -> str:
    return value.replace(",", "_").replace(" ", "_")


def build_audit_frame(group: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "image_path",
        "incorrect_tasks",
        "mean_confidence",
        "style_true_name",
        "style_pred_name",
        "genre_true_name",
        "genre_pred_name",
        "artist_true_name",
        "artist_pred_name",
    ]
    audit = group[columns].copy()
    audit.insert(0, "rank_in_group", range(1, len(audit) + 1))
    audit["verdict"] = ""
    audit["suggested_label"] = ""
    audit["notes"] = ""
    return audit


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def build_contact_sheet(
    group: pd.DataFrame,
    image_root: Path,
    output_path: Path,
    title: str,
    thumb_size: tuple[int, int] = (220, 220),
    columns: int = 4,
) -> None:
    if group.empty:
        return

    font = ImageFont.load_default()
    padding = 16
    label_height = 72
    title_height = 36
    rows = (len(group) + columns - 1) // columns
    width = padding + columns * (thumb_size[0] + padding)
    height = title_height + padding + rows * (thumb_size[1] + label_height + padding)
    sheet = Image.new("RGB", (width, height), color=(250, 248, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 8), title, fill=(20, 20, 20), font=font)

    for idx, row in enumerate(group.itertuples(index=False), start=1):
        col = (idx - 1) % columns
        row_idx = (idx - 1) // columns
        x = padding + col * (thumb_size[0] + padding)
        y = title_height + padding + row_idx * (thumb_size[1] + label_height + padding)

        image_path = image_root / row.image_path
        try:
            thumb = fit_image(image_path, thumb_size)
            sheet.paste(thumb, (x, y))
        except Exception:
            draw.rectangle((x, y, x + thumb_size[0], y + thumb_size[1]), outline=(180, 80, 80), width=2)
            draw.text((x + 8, y + 8), "missing", fill=(180, 80, 80), font=font)

        label = [
            f"{idx}. {row.image_path.split('/')[-1][:26]}",
            f"wrong: {row.incorrect_tasks}",
            f"conf: {float(row.mean_confidence):.3f}",
        ]
        text_y = y + thumb_size[1] + 4
        for line in label:
            draw.text((x, text_y), line, fill=(20, 20, 20), font=font)
            text_y += 14

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare grouped audit CSVs and contact sheets.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/wikiart"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    df = df.sort_values(["incorrect_tasks", "mean_confidence"], ascending=[True, False]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for task_pattern, group in df.groupby("incorrect_tasks", sort=True):
        group = group.reset_index(drop=True)
        slug = slugify(task_pattern)
        audit = build_audit_frame(group)
        csv_path = args.output_dir / f"manual_label_audit_{slug}.csv"
        audit.to_csv(csv_path, index=False)

        image_path = args.output_dir / f"contact_sheet_{slug}.jpg"
        build_contact_sheet(group, args.image_root, image_path, f"High-confidence outliers: {task_pattern}")
        summary_rows.append(
            {
                "incorrect_tasks": task_pattern,
                "rows": len(group),
                "audit_csv": csv_path.name,
                "contact_sheet": image_path.name,
            }
        )

    pd.DataFrame(summary_rows).sort_values("incorrect_tasks").to_csv(
        args.output_dir / "manual_label_audit_index.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
