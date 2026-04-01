from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class MultiScaleBallDetector(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        backbone = backbone.lower()
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            net = resnet18(weights=weights)
            c2, c3, c4 = 64, 128, 256
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            net = resnet34(weights=weights)
            c2, c3, c4 = 64, 128, 256
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            c2, c3, c4 = 256, 512, 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ResNet backbone
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
        )  # 1/2
        self.pool = net.maxpool   # 1/4
        self.layer1 = net.layer1  # 1/4
        self.layer2 = net.layer2  # 1/8
        self.layer3 = net.layer3  # 1/16

        # FPN-style top-down fusion
        self.lateral1 = nn.Conv2d(c2, 64, kernel_size=1)
        self.lateral2 = nn.Conv2d(c3, 64, kernel_size=1)
        self.lateral3 = nn.Conv2d(c4, 64, kernel_size=1)

        self.smooth = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Detector heads
        self.head_heatmap = nn.Conv2d(64, 1, kernel_size=1)
        self.head_offset = nn.Conv2d(64, 2, kernel_size=1)
        self.head_log_var = nn.Conv2d(64, 2, kernel_size=1)

        # Keep visibility head for interface compatibility, but make it harmless.
        # If visible_weight: 0.0 in config, this branch will not affect training.
        self.head_visible = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x0 = self.stem(x)        # 1/2
        x0p = self.pool(x0)      # 1/4
        x1 = self.layer1(x0p)    # 1/4
        x2 = self.layer2(x1)     # 1/8
        x3 = self.layer3(x2)     # 1/16

        p3 = self.lateral3(x3)
        p2 = self.lateral2(x2) + F.interpolate(
            p3, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        p1 = self.lateral1(x1) + F.interpolate(
            p2, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )
        f = self.smooth(p1)

        return {
            "heatmap": torch.sigmoid(self.head_heatmap(f)),
            "offset": self.head_offset(f),
            "log_var": self.head_log_var(f),
            "visible_logit": self.head_visible(f),
        }


def decode_heatmap(heatmap: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    b, _, h, w = heatmap.shape

    flat_idx = heatmap.view(b, -1).argmax(dim=1)
    ys = (flat_idx // w).long()
    xs = (flat_idx % w).long()

    batch_idx = torch.arange(b, device=heatmap.device)
    off_x = offset[batch_idx, 0, ys, xs]
    off_y = offset[batch_idx, 1, ys, xs]

    xs_f = xs.float() + off_x
    ys_f = ys.float() + off_y

    xs_f = xs_f.clamp(0.0, w - 1.0)
    ys_f = ys_f.clamp(0.0, h - 1.0)

    return torch.stack([xs_f, ys_f], dim=1)