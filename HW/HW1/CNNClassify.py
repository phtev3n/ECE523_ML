#!/usr/bin/env python3
"""
ECE 523 HW1 - CNN classifier for CIFAR-10
Commands:
  python CNNclassify.py train
  python CNNclassify.py test xxx.png

Requirements satisfied:
- First conv layer: 32 filters, 5x5 kernel, stride 1
- Save model to folder named "model"
- Print testing accuracy each epoch during training
- Test command loads model, predicts xxx.png, and saves first conv visualizations to CONV_rslt.png
- Minimal prints (only required results)
"""

import os
import sys
import time
from typing import Tuple, List

import numpy as np
from PIL import Image


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


# -----------------------------
# Constants / configuration
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_cifar10.pth")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -----------------------------
# CNN Architecture
# -----------------------------
class CIFAR10CNN(nn.Module):
    """
    CNN tuned to exceed 75% on CIFAR-10 with modest training.
    First conv layer is constrained per HW:
      - out_channels = 32
      - kernel_size = 5
      - stride = 1
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # REQUIRED FIRST CONV LAYER (DO NOT CHANGE)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,   # keep spatial size 32x32
            bias=False
        )

        # Follow-on layers (allowed to change)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # More conv blocks
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32 -> 16
            nn.Dropout(0.20),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16 -> 8
            nn.Dropout(0.25),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8 -> 4
            nn.Dropout(0.30),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer explicitly separated so we can visualize it in test()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Data utilities
# -----------------------------
def get_transforms() -> Tuple[T.Compose, T.Compose]:
    """
    Training uses standard CIFAR-10 augmentation for accuracy.
    Testing uses only normalization.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return train_tf, test_tf


def make_loaders(batch_size: int = 128, num_workers: int = 1) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = get_transforms()

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    ds =  torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True
        )

    return train_loader, test_loader


# -----------------------------
# Training / evaluation
# -----------------------------
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)

@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Returns:
        avg_loss: average loss over dataset
        accuracy: percentage correct
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy

def train():
    # Device selection (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_loaders(batch_size=128, num_workers=2)

    model = CIFAR10CNN(num_classes=10).to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # Scheduler helps reach >=75% reliably
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training settings
    epochs = 35
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_acc = 0.0

    # Optional header line
    print("Loop  Train Loss  Train Acc%  Test Loss  Test Acc%")

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            # Accumulate training stats
            train_loss_sum += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        scheduler.step()

        # Final train stats for this epoch
        train_loss = train_loss_sum / max(1, train_total)
        train_acc = 100.0 * train_correct / max(1, train_total)

        # Test stats
        test_loss, test_acc = evaluate_metrics(model, test_loader, device, criterion)

        # Print required statistics
        print(
            f"{epoch:02d}    "
            f"{train_loss:.4f}      "
            f"{train_acc:6.2f}      "
            f"{test_loss:.4f}     "
            f"{test_acc:6.2f}"
        )

        # Save best model by test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CIFAR10_CLASSES,
                },
                MODEL_PATH
            )

# -----------------------------
# Testing: predict + visualize conv1 outputs
# -----------------------------
def _load_image_as_cifar_tensor(img_path: str) -> torch.Tensor:
    """
    Loads an image and applies the SAME preprocessing as CIFAR-10 test data.
    Output: tensor shape [1, 3, 32, 32]
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")

    # CIFAR images are 32x32; if external image, resize to 32x32.
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BILINEAR)

    _, test_tf = get_transforms()
    x = test_tf(img).unsqueeze(0)  # [1,3,32,32]
    return x


@torch.no_grad()
def _visualize_first_conv(model: CIFAR10CNN, x: torch.Tensor, out_path: str = "CONV_rslt.png") -> None:
    """
    Produces a 32-tile visualization of conv1 feature maps for a single input image.
    Saves as out_path. Uses PIL for visualization.
    """
    model.eval()

    # Get conv1 activation (before pooling; after conv only is fine for visualization)
    feats = model.conv1(x)  # [1, 32, 32, 32]
    feats = feats.squeeze(0).cpu().numpy()  # [32, 32, 32]

    # Normalize each feature map for display and convert to RGB using a simple colormap
    def gray_to_rgb_colormap(arr):
        # Simple jet-like colormap using PIL (not as fancy as cv2, but sufficient)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        # Apply a basic color mapping: map grayscale to RGB (blue to red)
        # We'll use the R channel for high values, B for low, G for mid
        r, g, b = img.split()
        r = arr.copy()
        g = (255 - np.abs(arr - 128)).astype(np.uint8)
        b = 255 - arr
        return Image.merge("RGB", (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))

    tiles = []
    for k in range(feats.shape[0]):
        fm = feats[k]
        fm = fm - fm.min()
        denom = (fm.max() + 1e-8)
        fm = fm / denom
        tile = gray_to_rgb_colormap(fm)
        tiles.append(tile)

    # Create an 8x4 grid (32 = 8 rows x 4 cols)
    rows, cols = 8, 4
    h, w = tiles[0].size
    grid_img = Image.new("RGB", (cols * h, rows * w))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            grid_img.paste(tiles[idx], (c * h, r * w))
            idx += 1

    grid_img.save(out_path)

@torch.no_grad()
def sanity_check_on_cifar(model, device):
    _, test_loader = make_loaders(batch_size=1, num_workers=2)
    model.eval()

    x, y = next(iter(test_loader))
    x = x.to(device)
    logits = model(x)
    pred = torch.argmax(logits, dim=1).item()

    print(f"Sanity Check - True: {CIFAR10_CLASSES[y.item()]}, Pred: {CIFAR10_CLASSES[pred]}")
    
def test(img_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python CNNclassify.py train"
        )

    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt.get("class_names", CIFAR10_CLASSES)

    model = CIFAR10CNN(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = _load_image_as_cifar_tensor(img_path).to(device)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    # REQUIRED: show prediction result (keep it minimal)
    print(f"Prediction: {pred_name}")

    # REQUIRED: save 32 feature map visualization of first conv layer
    _visualize_first_conv(model, x, out_path="CONV_rslt.png")


# -----------------------------
# CLI
# -----------------------------
def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python CNNclassify.py train | python CNNclassify.py test xxx.png")

    cmd = sys.argv[1].lower()

    if cmd == "train":
        train()
    elif cmd == "test":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: python CNNclassify.py test xxx.png")
        test(sys.argv[2])
    else:
        raise SystemExit("Unknown command. Use: train or test")


if __name__ == "__main__":
    main()