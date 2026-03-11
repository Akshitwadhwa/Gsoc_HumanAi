from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import WikiArtMultiTaskDataset, collate_valid_samples
from src.model import MultiTaskClassifier


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-image predictions for a trained WikiArt model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/wikiart"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    tasks = tuple(checkpoint["tasks"])
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])

    dataset = WikiArtMultiTaskDataset(
        image_root=args.image_root,
        manifest_root=args.manifest_root,
        split=args.split,
        tasks=tasks,
        transform=build_eval_transform(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_valid_samples,
    )

    model = MultiTaskClassifier(
        backbone_name=checkpoint["backbone"],
        num_classes={task: len(class_names[task]) for task in tasks},
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        header = ["image_path"]
        for task in tasks:
            header.extend(
                [
                    f"{task}_true_index",
                    f"{task}_true_name",
                    f"{task}_pred_index",
                    f"{task}_pred_name",
                    f"{task}_confidence",
                    f"{task}_correct",
                ]
            )
        header.extend(["mean_confidence", "all_correct", "incorrect_tasks"])

        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                if batch is None:
                    continue
                images, labels, paths = batch
                images = images.to(device, non_blocking=True)
                labels = {task: target.to(device, non_blocking=True) for task, target in labels.items()}
                outputs = model(images)

                batch_predictions: dict[str, dict[str, torch.Tensor]] = {}
                for task, logits in outputs.items():
                    probabilities = torch.softmax(logits, dim=1)
                    confidences, predicted = probabilities.max(dim=1)
                    batch_predictions[task] = {
                        "predicted": predicted.cpu(),
                        "confidence": confidences.cpu(),
                        "true": labels[task].cpu(),
                    }

                for index, image_path in enumerate(paths):
                    row: dict[str, object] = {"image_path": image_path}
                    confidences: list[float] = []
                    incorrect_tasks: list[str] = []

                    for task in tasks:
                        true_index = int(batch_predictions[task]["true"][index].item())
                        pred_index = int(batch_predictions[task]["predicted"][index].item())
                        confidence = float(batch_predictions[task]["confidence"][index].item())
                        correct = int(true_index == pred_index)
                        if not correct:
                            incorrect_tasks.append(task)
                        confidences.append(confidence)

                        row[f"{task}_true_index"] = true_index
                        row[f"{task}_true_name"] = class_names[task][true_index]
                        row[f"{task}_pred_index"] = pred_index
                        row[f"{task}_pred_name"] = class_names[task][pred_index]
                        row[f"{task}_confidence"] = f"{confidence:.6f}"
                        row[f"{task}_correct"] = correct

                    mean_confidence = sum(confidences) / len(confidences)
                    row["mean_confidence"] = f"{mean_confidence:.6f}"
                    row["all_correct"] = int(not incorrect_tasks)
                    row["incorrect_tasks"] = ",".join(incorrect_tasks)
                    writer.writerow(row)
                if args.max_batches and batch_index >= args.max_batches:
                    break

    print(f"Saved predictions to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
