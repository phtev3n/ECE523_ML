from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class MultiScaleBallDetector(nn.Module):
    """Single-frame golf ball detector using a ResNet backbone with FPN-style fusion.

    Architecture overview
    ---------------------
    A ResNet backbone extracts feature maps at three spatial scales (1/4, 1/8,
    1/16 of input resolution).  A lightweight Feature Pyramid Network (FPN)
    top-down pathway fuses these scales via lateral 1×1 projections and
    bilinear upsampling, then smooths the fused map with a 3×3 conv block.

    Four prediction heads branch from the shared fused feature map:
      heatmap      — Gaussian blob centred on the ball (CenterNet-style),
                     used for coarse localisation
      offset       — sub-pixel refinement within the peak heatmap cell
      log_var      — aleatoric uncertainty of the offset prediction; used by
                     the Kalman filter to weight measurement updates
      visible_logit — scalar logit indicating whether the ball is visible in
                      this frame; sigmoid of this is the visibility probability

    Using a heatmap + offset representation rather than direct (u, v) regression
    allows the loss to be spatially structured: the heatmap BCE encourages a
    smooth Gaussian response near the ball, while the offset head only needs to
    refine within a single cell, making the regression problem much easier.

    Backbone options
    ----------------
    resnet18 / resnet34  — lighter, faster; suitable for GPU-limited inference
    resnet50             — higher capacity; default for training on synthetic data

    Channel counts differ between ResNet-18/34 (basic blocks) and ResNet-50
    (bottleneck blocks), so lateral conv in-channels are set accordingly.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        backbone = backbone.lower()
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            net = resnet18(weights=weights)
            c2, c3, c4 = 64, 128, 256     # layer1/2/3 output channels
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            net = resnet34(weights=weights)
            c2, c3, c4 = 64, 128, 256
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            c2, c3, c4 = 256, 512, 1024   # bottleneck blocks have 4× channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # --- Backbone stages (stride noted relative to input) ---
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
        )  # stride 1/2
        self.pool = net.maxpool   # stride 1/4
        self.layer1 = net.layer1  # stride 1/4  (no additional downsampling)
        self.layer2 = net.layer2  # stride 1/8
        self.layer3 = net.layer3  # stride 1/16

        # --- FPN lateral projections: reduce each scale to 64 channels ---
        # This normalises channel counts before fusion so no scale dominates.
        self.lateral1 = nn.Conv2d(c2, 64, kernel_size=1)   # from layer1 (1/4)
        self.lateral2 = nn.Conv2d(c3, 64, kernel_size=1)   # from layer2 (1/8)
        self.lateral3 = nn.Conv2d(c4, 64, kernel_size=1)   # from layer3 (1/16)

        # Smooth the merged feature map to reduce checkerboard artefacts
        # introduced by bilinear upsampling
        self.smooth = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # --- Prediction heads (all operate at 1/4 input resolution) ---
        self.head_heatmap = nn.Conv2d(64, 1, kernel_size=1)   # single-channel Gaussian target
        self.head_offset  = nn.Conv2d(64, 2, kernel_size=1)   # (du, dv) sub-pixel offset
        self.head_log_var = nn.Conv2d(64, 2, kernel_size=1)   # log-variance per offset dim

        # Visibility head: global max-pool then classify visible / not-visible.
        # AdaptiveMaxPool aggregates the most active spatial location, which is
        # appropriate since we want to know whether *any* location looks like a ball.
        # Keep this head for interface compatibility; its loss weight can be set
        # to 0.0 in config to disable training when visibility labels are noisy.
        self.head_visible = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, H, W) — normalised RGB image batch

        Returns:
            dict with keys:
              "heatmap"      : (B, 1, H/4, W/4)  — sigmoid probability map
              "offset"       : (B, 2, H/4, W/4)  — raw (du, dv) sub-pixel offsets
              "log_var"      : (B, 2, H/4, W/4)  — log-variance of offset
              "visible_logit": (B, 1)             — raw visibility score (pre-sigmoid)
        """
        # Backbone: extract multi-scale features
        x0 = self.stem(x)        # (B, 64,  H/2,  W/2)
        x0p = self.pool(x0)      # (B, 64,  H/4,  W/4)
        x1 = self.layer1(x0p)    # (B, c2,  H/4,  W/4)
        x2 = self.layer2(x1)     # (B, c3,  H/8,  W/8)
        x3 = self.layer3(x2)     # (B, c4,  H/16, W/16)

        # FPN top-down pathway: start from coarsest scale and merge upward
        p3 = self.lateral3(x3)
        p2 = self.lateral2(x2) + F.interpolate(
            p3, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        p1 = self.lateral1(x1) + F.interpolate(
            p2, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )

        # Smooth the finest-scale fused map (output resolution: H/4 × W/4)
        f = self.smooth(p1)

        return {
            # Raw logit (pre-sigmoid) so bce_with_logits can be used safely under
            # AMP autocast.  Callers that need probabilities must apply sigmoid themselves.
            "heatmap_logit": self.head_heatmap(f),
            "offset":        self.head_offset(f),
            "log_var":       self.head_log_var(f),
            "visible_logit": self.head_visible(f),
        }


def decode_heatmap(heatmap: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Convert a heatmap + offset prediction to sub-pixel (u, v) coordinates.

    Step 1 — peak detection: find the argmax of the heatmap (the cell most
             likely to contain the ball centre).
    Step 2 — sub-pixel refinement: add the predicted fractional offset within
             that cell to get coordinates within the heatmap grid.

    The returned coordinates are in heatmap-pixel space (H/4 × W/4).  The
    caller must scale them up by (img_w / hm_w, img_h / hm_h) to get
    image-pixel coordinates.

    Args:
        heatmap: (B, 1, H, W) — sigmoid probability map
        offset:  (B, 2, H, W) — per-cell (du, dv) sub-pixel offsets

    Returns:
        (B, 2) — sub-pixel (x, y) = (u, v) positions in heatmap space
    """
    b, _, h, w = heatmap.shape

    # Flatten spatial dims and find peak cell index for each batch element
    flat_idx = heatmap.view(b, -1).argmax(dim=1)
    ys = (flat_idx // w).long()   # row of peak cell
    xs = (flat_idx  % w).long()   # column of peak cell

    # Read off the predicted sub-pixel offsets at the peak cell
    batch_idx = torch.arange(b, device=heatmap.device)
    off_x = offset[batch_idx, 0, ys, xs]
    off_y = offset[batch_idx, 1, ys, xs]

    # Add offset to integer cell position and clamp to valid heatmap range
    xs_f = (xs.float() + off_x).clamp(0.0, w - 1.0)
    ys_f = (ys.float() + off_y).clamp(0.0, h - 1.0)

    return torch.stack([xs_f, ys_f], dim=1)   # (B, 2): [u, v] per sample
