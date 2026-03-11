from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim

    if name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim

    raise ValueError(f"Unsupported backbone: {name}")


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: dict[str, int],
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(feature_dim, class_count)
                for task, class_count in sorted(num_classes.items())
            }
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.dropout(features)
        return {task: head(features) for task, head in self.heads.items()}
