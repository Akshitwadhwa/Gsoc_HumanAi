from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_TASKS = ("style", "genre", "artist")


@dataclass(frozen=True)
class Sample:
    image_path: str
    labels: dict[str, int]


def parse_class_names(path: Path) -> list[str]:
    class_names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            class_names.append(parts[1])
        else:
            class_names.append(line)
    return class_names


def read_manifest_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            values = [value.strip() for value in row if value.strip()]
            if len(values) < 2:
                continue
            image_path = values[0]
            label = values[-1]
            lowered_path = image_path.lower()
            lowered_label = label.lower()
            if "path to image" in lowered_path or "groundtruth class" in lowered_label:
                continue
            rows.append((image_path, int(label)))
    return rows


def build_multitask_samples(
    manifest_root: Path,
    split: str,
    tasks: tuple[str, ...],
) -> list[Sample]:
    per_task: dict[str, dict[str, int]] = {}
    for task in tasks:
        manifest_path = manifest_root / f"{task}_{split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        task_rows = read_manifest_rows(manifest_path)
        per_task[task] = {image_path: label for image_path, label in task_rows}

    shared_paths = set.intersection(*(set(task_rows) for task_rows in per_task.values()))
    if not shared_paths:
        raise ValueError(f"No overlapping samples found for split={split!r}, tasks={tasks!r}")

    return [
        Sample(
            image_path=image_path,
            labels={task: per_task[task][image_path] for task in tasks},
        )
        for image_path in sorted(shared_paths)
    ]


class WikiArtMultiTaskDataset(Dataset[tuple[object, dict[str, int], str]]):
    def __init__(
        self,
        image_root: str | Path,
        manifest_root: str | Path,
        split: str,
        tasks: tuple[str, ...] = DEFAULT_TASKS,
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.manifest_root = Path(manifest_root)
        self.split = split
        self.tasks = tasks
        self.transform = transform

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")
        if not self.manifest_root.exists():
            raise FileNotFoundError(f"Manifest root does not exist: {self.manifest_root}")

        self.class_names = {
            task: parse_class_names(self.manifest_root / f"{task}_class.txt") for task in self.tasks
        }
        self.samples = build_multitask_samples(self.manifest_root, self.split, self.tasks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, dict[str, int], str] | None:
        sample = self.samples[index]
        try:
            image = Image.open(self.image_root / sample.image_path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            return None
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.labels, sample.image_path

    def class_counts(self) -> dict[str, Counter[int]]:
        counts = {task: Counter() for task in self.tasks}
        for sample in self.samples:
            for task, label in sample.labels.items():
                counts[task][label] += 1
        return counts


def collate_valid_samples(batch: list[tuple[object, dict[str, int], str] | None]):
    valid_batch = [sample for sample in batch if sample is not None]
    if not valid_batch:
        return None
    return default_collate(valid_batch)
