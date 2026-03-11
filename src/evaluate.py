from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import WikiArtMultiTaskDataset, collate_valid_samples
from src.metrics import mean_task_score, topk_accuracy
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
    parser = argparse.ArgumentParser(description="Evaluate a WikiArt CNN baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/wikiart"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=default_device())
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
    image_size = int(checkpoint["image_size"])
    class_names = checkpoint["class_names"]
    model = MultiTaskClassifier(
        backbone_name=checkpoint["backbone"],
        num_classes={task: len(class_names[task]) for task in tasks},
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

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

    loss_fns = {task: nn.CrossEntropyLoss() for task in tasks}
    total_loss = 0.0
    total_batches = 0
    running = {task: {"top1_sum": 0.0, "top5_sum": 0.0, "count": 0} for task in tasks}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels, _paths = batch
            images = images.to(device, non_blocking=True)
            labels = {task: target.to(device, non_blocking=True) for task, target in labels.items()}
            outputs = model(images)

            batch_loss = 0.0
            batch_size = images.shape[0]
            for task, logits in outputs.items():
                targets = labels[task]
                batch_loss += loss_fns[task](logits, targets)
                running[task]["top1_sum"] += topk_accuracy(logits, targets, k=1) * batch_size
                running[task]["top5_sum"] += topk_accuracy(logits, targets, k=5) * batch_size
                running[task]["count"] += batch_size

            total_loss += batch_loss.item()
            total_batches += 1

    metrics = {}
    for task, values in running.items():
        count = max(values["count"], 1)
        metrics[task] = {
            "top1": values["top1_sum"] / count,
            "top5": values["top5_sum"] / count,
        }

    result = {
        "split": args.split,
        "loss": total_loss / max(total_batches, 1),
        "metrics": metrics,
        "mean_top1": mean_task_score(metrics, "top1"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
