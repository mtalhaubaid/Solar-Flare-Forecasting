"""Model definitions and model factory."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from . import config


class CustomCNN(nn.Module):
    """Small baseline CNN for quick sanity checks."""

    def __init__(self, num_outputs: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


class FocalLoss(nn.Module):
    """Binary focal loss for strongly imbalanced flare labels."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probabilities, 1 - probabilities)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


class CNNLSTM(nn.Module):
    """Temporal baseline for a sequence of image tensors shaped [B, T, C, H, W]."""

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = CustomCNN(num_outputs=feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)
        features = self.encoder(x).view(batch_size, time_steps, -1)
        _, (hidden, _) = self.lstm(features)
        return self.classifier(hidden[-1]).squeeze(1)


class PGMCTNet(nn.Module):
    """Physics-guided multi-channel temporal network prototype."""

    def __init__(self, num_channels: int = 3, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.channel_encoders = nn.ModuleList([CustomCNN(num_outputs=feature_dim) for _ in range(num_channels)])
        self.fusion = nn.Linear(num_channels * feature_dim, feature_dim)
        self.temporal = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_channels, height, width = x.shape
        fused_steps = []
        for step in range(time_steps):
            channel_features = []
            for channel_index, encoder in enumerate(self.channel_encoders):
                image = x[:, step, channel_index].unsqueeze(1).repeat(1, 3, 1, 1)
                channel_features.append(encoder(image))
            fused = torch.cat(channel_features, dim=1)
            fused_steps.append(torch.relu(self.fusion(fused)))
        sequence = torch.stack(fused_steps, dim=1)
        temporal_features, _ = self.temporal(sequence)
        weights = torch.softmax(self.attention(temporal_features), dim=1)
        pooled = (weights * temporal_features).sum(dim=1)
        return self.classifier(pooled).squeeze(1)


def _torchvision_weights(enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    weights_enum = getattr(models, enum_name, None)
    return weights_enum.DEFAULT if weights_enum else None


def get_resnet18(pretrained: bool = False) -> nn.Module:
    try:
        model = models.resnet18(weights=_torchvision_weights("ResNet18_Weights", pretrained))
    except TypeError:
        model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_efficientnet_b0(pretrained: bool = False) -> nn.Module:
    try:
        model = models.efficientnet_b0(weights=_torchvision_weights("EfficientNet_B0_Weights", pretrained))
    except TypeError:
        model = models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def get_convnext_tiny(pretrained: bool = False) -> nn.Module:
    try:
        model = models.convnext_tiny(weights=_torchvision_weights("ConvNeXt_Tiny_Weights", pretrained))
    except TypeError:
        model = models.convnext_tiny(pretrained=pretrained)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model


def get_model(name: str, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name == "custom_cnn":
        return CustomCNN()
    if name == "resnet18":
        return get_resnet18(pretrained=pretrained)
    if name == "efficientnet_b0":
        return get_efficientnet_b0(pretrained=pretrained)
    if name == "convnext_tiny":
        return get_convnext_tiny(pretrained=pretrained)
    if name == "cnn_lstm":
        return CNNLSTM()
    if name in {"pg_mctnet", "pg-mctnet"}:
        return PGMCTNet()
    raise ValueError(f"Unknown model '{name}'. Available first-stage models: {config.MODEL_NAMES}")

