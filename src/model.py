from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


BACKBONE_CHOICES = ("resnet50", "efficientnet_b0", "efficientnet_b2")
MODEL_CHOICES = ("cnn", "conv_recurrent")


@dataclass(frozen=True)
class BackboneSpec:
    feature_extractor: nn.Module
    feature_dim: int


def build_backbone(name: str, pretrained: bool) -> BackboneSpec:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        return BackboneSpec(feature_extractor=feature_extractor, feature_dim=2048)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        return BackboneSpec(feature_extractor=backbone.features, feature_dim=1280)

    if name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        return BackboneSpec(feature_extractor=backbone.features, feature_dim=1408)

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
        spec = build_backbone(backbone_name, pretrained=pretrained)
        self.model_name = "cnn"
        self.backbone_name = backbone_name
        self.backbone = spec.feature_extractor
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(spec.feature_dim, class_count)
                for task, class_count in sorted(num_classes.items())
            }
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)
        features = self.pool(feature_map).flatten(1)
        features = self.dropout(features)
        return {task: head(features) for task, head in self.heads.items()}


class ConvRecurrentClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: dict[str, int],
        pretrained: bool = True,
        dropout: float = 0.2,
        rnn_hidden_size: int = 256,
        rnn_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        spec = build_backbone(backbone_name, pretrained=pretrained)
        self.model_name = "conv_recurrent"
        self.backbone_name = backbone_name
        self.backbone = spec.feature_extractor
        self.sequence_norm = nn.LayerNorm(spec.feature_dim)
        self.recurrent = nn.GRU(
            input_size=spec.feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        recurrent_dim = rnn_hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(recurrent_dim, class_count)
                for task, class_count in sorted(num_classes.items())
            }
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)
        sequence = feature_map.flatten(2).transpose(1, 2).contiguous()
        sequence = self.sequence_norm(sequence)
        recurrent_features, _ = self.recurrent(sequence)
        pooled = recurrent_features.mean(dim=1)
        pooled = self.dropout(pooled)
        return {task: head(pooled) for task, head in self.heads.items()}


def build_model(
    model_name: str,
    backbone_name: str,
    num_classes: dict[str, int],
    pretrained: bool = True,
    dropout: float = 0.2,
    rnn_hidden_size: int = 256,
    rnn_layers: int = 1,
    bidirectional: bool = True,
) -> nn.Module:
    if model_name == "cnn":
        return MultiTaskClassifier(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if model_name == "conv_recurrent":
        return ConvRecurrentClassifier(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            rnn_hidden_size=rnn_hidden_size,
            rnn_layers=rnn_layers,
            bidirectional=bidirectional,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def build_model_from_checkpoint(
    checkpoint: dict[str, object],
    num_classes: dict[str, int],
    pretrained: bool = False,
) -> nn.Module:
    return build_model(
        model_name=str(checkpoint.get("model_name", "cnn")),
        backbone_name=str(checkpoint["backbone"]),
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=float(checkpoint.get("dropout", 0.2)),
        rnn_hidden_size=int(checkpoint.get("rnn_hidden_size", 256)),
        rnn_layers=int(checkpoint.get("rnn_layers", 1)),
        bidirectional=bool(checkpoint.get("bidirectional", True)),
    )
