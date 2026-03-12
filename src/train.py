from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import DEFAULT_TASKS, WikiArtMultiTaskDataset, collate_valid_samples
from src.metrics import mean_task_score, topk_accuracy
from src.model import BACKBONE_CHOICES, MODEL_CHOICES, build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WikiArt multi-task classifier.")
    parser.add_argument("--image-root", type=Path, default=Path("data/wikiart"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default="cnn")
    parser.add_argument("--backbone", choices=list(BACKBONE_CHOICES), default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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


def build_loss_fns(
    train_dataset: WikiArtMultiTaskDataset,
    use_class_weights: bool,
    device: torch.device,
) -> dict[str, nn.Module]:
    loss_fns: dict[str, nn.Module] = {}
    class_counts = train_dataset.class_counts()
    for task in train_dataset.tasks:
        weights = None
        if use_class_weights:
            task_counts = class_counts[task]
            weights = torch.zeros(len(train_dataset.class_names[task]), dtype=torch.float32)
            total = sum(task_counts.values())
            for label, count in task_counts.items():
                weights[label] = total / max(count, 1)
            weights = weights / weights.mean()
            weights = weights.to(device)
        loss_fns[task] = nn.CrossEntropyLoss(weight=weights)
    return loss_fns


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fns: dict[str, nn.Module],
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, dict[str, dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    running = {
        task: {"top1_sum": 0.0, "top5_sum": 0.0, "count": 0}
        for task in loss_fns
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
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
            if max_batches and batch_index >= max_batches:
                break

    metrics = {}
    for task, task_running in running.items():
        count = max(task_running["count"], 1)
        metrics[task] = {
            "top1": task_running["top1_sum"] / count,
            "top5": task_running["top5_sum"] / count,
        }
    average_loss = total_loss / max(total_batches, 1)
    return average_loss, metrics


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    tasks = tuple(args.tasks)
    device = torch.device(args.device)

    train_dataset = WikiArtMultiTaskDataset(
        image_root=args.image_root,
        manifest_root=args.manifest_root,
        split="train",
        tasks=tasks,
        transform=build_train_transform(args.image_size),
    )
    val_dataset = WikiArtMultiTaskDataset(
        image_root=args.image_root,
        manifest_root=args.manifest_root,
        split="val",
        tasks=tasks,
        transform=build_eval_transform(args.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_valid_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_valid_samples,
    )

    num_classes = {task: len(train_dataset.class_names[task]) for task in tasks}
    model = build_model(
        model_name=args.model,
        backbone_name=args.backbone,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_layers=args.rnn_layers,
        bidirectional=not args.no_bidirectional,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device)
    loss_fns = build_loss_fns(train_dataset, use_class_weights=args.use_class_weights, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_score = -1.0
    best_checkpoint_path = args.output_dir / "best.pt"

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Model: {args.model}")
    print(f"Backbone: {args.backbone}")
    print(f"Tasks: {', '.join(tasks)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue
            images, labels, _paths = batch
            images = images.to(device, non_blocking=True)
            labels = {task: target.to(device, non_blocking=True) for task, target in labels.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = sum(loss_fns[task](outputs[task], labels[task]) for task in tasks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        train_loss = running_loss / max(batch_count, 1)
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            loss_fns,
            device,
            max_batches=args.max_val_batches,
        )
        mean_top1 = mean_task_score(val_metrics, "top1")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "mean_top1": mean_top1,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} mean_top1={mean_top1:.4f}"
        )
        for task, metrics in val_metrics.items():
            print(f"  {task}: top1={metrics['top1']:.4f} top5={metrics['top5']:.4f}")

        if mean_top1 > best_score:
            best_score = mean_top1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "backbone": args.backbone,
                    "tasks": list(tasks),
                    "class_names": train_dataset.class_names,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "rnn_hidden_size": args.rnn_hidden_size,
                    "rnn_layers": args.rnn_layers,
                    "bidirectional": not args.no_bidirectional,
                },
                best_checkpoint_path,
            )

    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint to: {best_checkpoint_path}")
    print(f"Saved training history to: {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
