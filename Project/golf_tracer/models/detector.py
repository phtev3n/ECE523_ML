from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleBallDetector(nn.Module):
    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        # backbone arg retained for config compatibility
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)

        self.lateral1 = nn.Conv2d(64, 64, 1)
        self.lateral2 = nn.Conv2d(128, 64, 1)
        self.lateral3 = nn.Conv2d(256, 64, 1)

        self.smooth = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head_heatmap = nn.Conv2d(64, 1, 1)
        self.head_offset = nn.Conv2d(64, 2, 1)
        self.head_log_var = nn.Conv2d(64, 2, 1)
        self.head_visible = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x0 = self.stem(x)   # 1/2
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16

        p3 = self.lateral3(x3)
        p2 = self.lateral2(x2) + F.interpolate(p3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(x1) + F.interpolate(p2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
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
    ys = (flat_idx // w).float()
    xs = (flat_idx % w).float()
    off_x = offset[torch.arange(b), 0, ys.long(), xs.long()]
    off_y = offset[torch.arange(b), 1, ys.long(), xs.long()]
    return torch.stack([xs + off_x, ys + off_y], dim=1)
