from __future__ import annotations

import math

import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if targets.numel() == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def mean_task_score(metrics: dict[str, dict[str, float]], key: str) -> float:
    if not metrics:
        return math.nan
    return sum(task_metrics[key] for task_metrics in metrics.values()) / len(metrics)
