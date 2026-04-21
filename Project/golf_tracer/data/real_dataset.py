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

In 'detector' mode the dataset enumerates every individual annotated frame
across all sequences (not one frame per sequence), so the effective dataset
size is num_sequences × frames_per_sequence.  Optional colour augmentation
is applied to each image when augment=True.
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


def _augment_image(img: np.ndarray) -> np.ndarray:
    """Apply colour augmentation to a uint8 HxWx3 BGR image (in-place safe)."""
    img = img.copy()
    # Brightness / contrast jitter
    alpha = np.random.uniform(0.75, 1.25)   # contrast
    beta  = np.random.uniform(-25.0, 25.0)  # brightness shift (0-255 scale)
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    # Saturation jitter in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.8, 1.2), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Gaussian noise
    noise = np.random.normal(0, 5.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


class RealGolfSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        sequence_length: int = 24,
        mode: str = "trajectory",
        detector_scale: float = 0.25,
        detector_sigma: float = 2.0,
        augment: bool = False,
        load_frames: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.mode = mode
        self.detector_scale = detector_scale
        self.detector_sigma = detector_sigma
        self.augment = augment
        # In trajectory mode the LSTM only needs UV features, not pixel data.
        # Setting load_frames=False skips disk I/O for 24×H×W images per sequence
        # (~94 MB each at 512×512), which prevents OOM on HPC nodes.
        self.load_frames = load_frames if mode == "trajectory" else True
        self.sequences = sorted([p for p in self.dataset_root.iterdir() if p.is_dir()])

        # In detector mode pre-build an index of (seq_dir, frame_meta) pairs
        # so __len__ returns the true number of individually labelled frames.
        if self.mode == "detector":
            self._det_index: list[tuple[Path, dict]] = []
            for seq_dir in self.sequences:
                meta = read_json(seq_dir / "annotations.json")
                for fm in meta["frames"]:
                    self._det_index.append((seq_dir, fm))

    def __len__(self):
        if self.mode == "detector":
            return len(self._det_index)
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
            if self.load_frames:
                img_path = seq_dir / "frames" / f"{item['frame_index']:06d}.png"
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"Could not read frame: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)

            uv.append(item["uv"])
            xyz.append(item.get("xyz", [0.0, 0.0, 0.0]))
            visible.append(float(item["visible"]))

        if self.load_frames:
            frames_arr = np.asarray(frames, dtype=np.float32) / 255.0
            frames_t = torch.from_numpy(frames_arr).permute(0, 3, 1, 2).contiguous()
        else:
            frames_t = torch.empty(0)
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
        if self.mode == "detector":
            seq_dir, fm = self._det_index[idx]
            img_path = seq_dir / "frames" / f"{fm['frame_index']:06d}.png"
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read frame: {img_path}")

            if self.augment:
                img_bgr = _augment_image(img_bgr)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

            is_visible = float(fm["visible"]) > 0.5
            uv_raw = fm.get("uv", [0.0, 0.0])
            uv_item = torch.tensor(uv_raw, dtype=torch.float32)
            vis_item = torch.tensor([float(fm["visible"])], dtype=torch.float32)

            out_h = int(round(h * self.detector_scale))
            out_w = int(round(w * self.detector_scale))

            heatmap = np.zeros((out_h, out_w), dtype=np.float32)
            offset = np.zeros((2, out_h, out_w), dtype=np.float32)
            uncertainty = np.zeros((2, out_h, out_w), dtype=np.float32)

            if is_visible:
                u, v = float(uv_raw[0]), float(uv_raw[1])
                hu = u * self.detector_scale
                hv = v * self.detector_scale
                heatmap = gaussian_2d(out_h, out_w, hu, hv, sigma=self.detector_sigma)
                cx = int(np.clip(np.floor(hu), 0, out_w - 1))
                cy = int(np.clip(np.floor(hv), 0, out_h - 1))
                offset[0, cy, cx] = hu - cx
                offset[1, cy, cx] = hv - cy
                uncertainty[:, cy, cx] = 1.0

            return {
                "image": img_t,
                "uv": uv_item,
                "visible": vis_item,
                "heatmap": torch.from_numpy(heatmap[None, ...]),
                "offset": torch.from_numpy(offset),
                "uncertainty": torch.from_numpy(uncertainty),
            }

        seq_dir = self.sequences[idx]
        meta, frames_t, uv_t, xyz_t, visible_t, eot, features = self._load_sequence(seq_dir)
        out = {
            "uv": uv_t,
            "xyz": xyz_t,
            "visible": visible_t,
            "features": features,
            "eot": eot,
            "camera": meta["camera"],
            "fps": float(meta.get("fps", 30.0)),
        }
        if self.load_frames:
            out["frames"] = frames_t
        return out