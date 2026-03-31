from __future__ import annotations
import json
from pathlib import Path
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def gaussian_2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return g.astype(np.float32)


def make_background(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 100, 40)
    horizon = int(h * 0.35)
    img[:horizon] = (170, 210, 235)
    for _ in range(12):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(horizon, h - 1)
        x2 = min(w - 1, x1 + random.randint(30, 140))
        y2 = min(h - 1, y1 + random.randint(5, 20))
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 80, 20), -1)
    return img


def simulate_ballistics(T: int, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    speed = random.uniform(45.0, 75.0)
    launch = math.radians(random.uniform(10.0, 28.0))
    side = math.radians(random.uniform(-8.0, 8.0))
    g = 9.81
    vx = speed * math.cos(launch) * math.cos(side)
    vy = speed * math.sin(launch)
    vz = speed * math.cos(launch) * math.sin(side) + 40.0
    pts = []
    for i in range(T):
        t = i * dt
        x = vx * t
        y = max(0.0, vy * t - 0.5 * g * t * t + 0.2)
        z = max(5.0, vz * t + 10.0)
        pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float32)


def project(points_xyz: np.ndarray, cam: dict) -> np.ndarray:
    z = np.clip(points_xyz[:, 2], 1e-3, None)
    u = cam["fx"] * points_xyz[:, 0] / z + cam["cx"]
    v = cam["fy"] * (cam["camera_height_m"] - points_xyz[:, 1]) / z + cam["cy"]
    return np.stack([u, v], axis=1).astype(np.float32)


class SyntheticGolfTrajectoryDataset(Dataset):
    def __init__(
        self,
        num_sequences: int = 200,
        sequence_length: int = 24,
        image_size=(256, 256),
        fps: float = 60.0,
        mode: str = "detector",
    ):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.h, self.w = image_size
        self.fps = fps
        self.mode = mode
        self.camera = {
            "fx": self.w * 1.5,
            "fy": self.h * 1.5,
            "cx": self.w * 0.5,
            "cy": self.h * 0.55,
            "camera_height_m": 1.3,
        }

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        T = self.sequence_length
        xyz = simulate_ballistics(T, self.fps)
        uv = project(xyz, self.camera)
        frames = []
        visible = []
        heatmaps = []
        offsets = []
        scale = 0.25
        out_h = int(self.h * scale)
        out_w = int(self.w * scale)

        for t in range(T):
            frame = make_background(self.h, self.w)
            u, v = uv[t]
            vis = 1.0
            if u < 0 or u >= self.w or v < 0 or v >= self.h:
                vis = 0.0
            if random.random() < 0.08:
                vis = 0.0
            if vis > 0.5:
                radius = random.randint(2, 4)
                cv2.circle(frame, (int(u), int(v)), radius, (245, 245, 245), -1)
                if random.random() < 0.4:
                    k = random.choice([3, 5, 7])
                    frame = cv2.GaussianBlur(frame, (k, k), 0)
            noise = np.random.normal(0, 5, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            hu = u * scale
            hv = v * scale
            hm = gaussian_2d(out_h, out_w, hu, hv, sigma=2.0) if vis > 0.5 else np.zeros((out_h, out_w), dtype=np.float32)
            cx = int(np.clip(round(hu), 0, out_w - 1))
            cy = int(np.clip(round(hv), 0, out_h - 1))
            off = np.zeros((2, out_h, out_w), dtype=np.float32)
            off[0, cy, cx] = hu - cx
            off[1, cy, cx] = hv - cy

            frames.append(frame)
            visible.append(vis)
            heatmaps.append(hm)
            offsets.append(off)

        frames = np.stack(frames)
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        uv_t = torch.from_numpy(uv).float()
        xyz_t = torch.from_numpy(xyz).float()
        visible_t = torch.tensor(visible, dtype=torch.float32)
        heatmaps_t = torch.from_numpy(np.stack(heatmaps))[:, None]
        offsets_t = torch.from_numpy(np.stack(offsets))
        eot = torch.zeros(T, dtype=torch.float32)
        eot[-1] = 1.0

        if self.mode == "detector":
            t = random.randint(0, T - 1)
            return {
                "image": frames_t[t],
                "uv": uv_t[t],
                "visible": visible_t[t:t+1],
                "heatmap": heatmaps_t[t],
                "offset": offsets_t[t],
            }

        features = torch.cat(
            [
                uv_t,
                visible_t[:, None],
                torch.zeros(T, 1),
                torch.linspace(0.0, 1.0, T)[:, None],
            ],
            dim=1,
        )
        return {
            "frames": frames_t,
            "uv": uv_t,
            "xyz": xyz_t,
            "visible": visible_t,
            "features": features,
            "eot": eot,
            "camera": self.camera,
        }


def export_synthetic_dataset(out_dir: str | Path, num_sequences: int = 100, sequence_length: int = 24, image_size=(256, 256), fps: float = 60.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = SyntheticGolfTrajectoryDataset(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        image_size=image_size,
        fps=fps,
        mode="trajectory",
    )
    for i in range(num_sequences):
        sample = ds[i]
        seq_dir = out_dir / f"seq_{i:04d}"
        frames_dir = seq_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        records = []
        frames = (sample["frames"].permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        for t, frame in enumerate(frames):
            cv2.imwrite(str(frames_dir / f"{t:06d}.png"), frame)
            records.append(
                {
                    "frame_index": t,
                    "visible": int(sample["visible"][t].item() > 0.5),
                    "uv": sample["uv"][t].tolist(),
                    "xyz": sample["xyz"][t].tolist(),
                }
            )
        with open(seq_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump({"fps": fps, "camera": sample["camera"], "frames": records}, f, indent=2)
