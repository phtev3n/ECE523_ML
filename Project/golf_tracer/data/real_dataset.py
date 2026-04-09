"""Dataset loader for real (or exported synthetic) golf sequences.

Each sequence lives in a directory with the layout:
    seq_XXXX/
        frames/
            000000.png
            000001.png
            ...
        annotations.json

annotations.json contains the pinhole camera intrinsics, per-frame 2D
positions (uv), 3D positions (xyz), and visibility flags.  When xyz is not
available (2D-only mode built by build_dataset.py --mode_2d_only) it is
filled with zeros — the trajectory model will still produce xyz predictions
but the recon3d_loss will be invalid and should not be used for evaluation.

This dataset supports both 'detector' mode (single-frame, returns one image
and its annotations) and 'trajectory' mode (full sequence, returns all frames
and the complete feature tensor for the LSTM).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from golf_tracer.utils.io import read_json


def gaussian_2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return g.astype(np.float32)


class RealGolfSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        sequence_length: int = 24,
        mode: str = "trajectory",
        detector_scale: float = 0.25,
        detector_sigma: float = 2.0,
    ):
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.mode = mode
        self.detector_scale = detector_scale
        self.detector_sigma = detector_sigma
        self.sequences = sorted([p for p in self.dataset_root.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.sequences)

    def _load_sequence(self, seq_dir: Path):
        meta = read_json(seq_dir / "annotations.json")
        frames_meta = meta["frames"]

        if len(frames_meta) == 0:
            raise ValueError(f"No frames found in {seq_dir}")

        # Truncate or pad to fixed sequence length
        if len(frames_meta) >= self.sequence_length:
            frames_meta = frames_meta[: self.sequence_length]
        else:
            last = frames_meta[-1]
            while len(frames_meta) < self.sequence_length:
                frames_meta.append(last.copy())

        frames = []
        uv = []
        xyz = []
        visible = []

        for item in frames_meta:
            img_path = seq_dir / "frames" / f"{item['frame_index']:06d}.png"
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not read frame: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frames.append(img)
            uv.append(item["uv"])
            xyz.append(item.get("xyz", [0.0, 0.0, 0.0]))
            visible.append(float(item["visible"]))

        frames = np.asarray(frames, dtype=np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        uv_t = torch.tensor(uv, dtype=torch.float32)
        xyz_t = torch.tensor(xyz, dtype=torch.float32)
        visible_t = torch.tensor(visible, dtype=torch.float32)

        eot = torch.zeros(self.sequence_length, dtype=torch.float32)
        eot[-1] = 1.0

        features = torch.cat(
            [
                uv_t,
                visible_t[:, None],
                torch.zeros(self.sequence_length, 1, dtype=torch.float32),
                torch.linspace(0.0, 1.0, self.sequence_length, dtype=torch.float32)[:, None],
            ],
            dim=1,
        )

        return meta, frames_t, uv_t, xyz_t, visible_t, eot, features

    def __getitem__(self, idx):
        seq_dir = self.sequences[idx]
        meta, frames_t, uv_t, xyz_t, visible_t, eot, features = self._load_sequence(seq_dir)

        if self.mode == "detector":
            visible_idx = torch.where(visible_t > 0.5)[0]
            hidden_idx = torch.where(visible_t <= 0.5)[0]

            if len(visible_idx) > 0 and (len(hidden_idx) == 0 or np.random.rand() < 0.5):
                t = int(visible_idx[np.random.randint(0, len(visible_idx))].item())
            else:
                t = int(hidden_idx[np.random.randint(0, len(hidden_idx))].item())

            h, w = frames_t.shape[-2:]
            out_h = int(round(h * self.detector_scale))
            out_w = int(round(w * self.detector_scale))

            heatmap = np.zeros((out_h, out_w), dtype=np.float32)
            offset = np.zeros((2, out_h, out_w), dtype=np.float32)
            uncertainty = np.zeros((2, out_h, out_w), dtype=np.float32)

            if visible_t[t] > 0.5:
                u, v = uv_t[t].tolist()
                hu = u * self.detector_scale
                hv = v * self.detector_scale

                # Match synthetic dataset target style: Gaussian heatmap
                heatmap = gaussian_2d(out_h, out_w, hu, hv, sigma=self.detector_sigma)

                # Use floor for cell assignment, then residual offset within that cell
                cx = int(np.clip(np.floor(hu), 0, out_w - 1))
                cy = int(np.clip(np.floor(hv), 0, out_h - 1))

                offset[0, cy, cx] = hu - cx
                offset[1, cy, cx] = hv - cy

                # Simple constant target uncertainty; optional
                uncertainty[:, cy, cx] = 1.0

            return {
                "image": frames_t[t],
                "uv": uv_t[t],
                "visible": visible_t[t:t + 1],
                "heatmap": torch.from_numpy(heatmap[None, ...]),
                "offset": torch.from_numpy(offset),
                "uncertainty": torch.from_numpy(uncertainty),
            }

        return {
            "frames": frames_t,
            "uv": uv_t,
            "xyz": xyz_t,
            "visible": visible_t,
            "features": features,
            "eot": eot,
            "camera": meta["camera"],
        }