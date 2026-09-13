from __future__ import annotations

import torch
from torch import nn
from torchvision import models


SUPPORTED_CLASSIFIERS = ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0")


def build_classifier(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture {architecture!r}; choose from {SUPPORTED_CLASSIFIERS}")
    return model


def build_segmenter(num_classes: int, pretrained: bool = True) -> nn.Module:
    """DeepLabV3-MobileNetV3 for pixel-level lesion segmentation."""
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None if not pretrained else models.MobileNet_V3_Large_Weights.DEFAULT,
    )
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, num_classes, 1)
    return model


def save_checkpoint(path, model: nn.Module, architecture: str, classes: list[str], image_size: int) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "classes": classes,
            "image_size": image_size,
        },
        path,
    )


def load_classifier(path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
