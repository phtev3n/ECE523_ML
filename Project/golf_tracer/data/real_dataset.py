from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from golf_tracer.utils.io import read_json


class RealGolfSequenceDataset(Dataset):
    def __init__(self, dataset_root: str | Path, sequence_length: int = 24, mode: str = "trajectory"):
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.mode = mode
        self.sequences = sorted([p for p in self.dataset_root.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_dir = self.sequences[idx]
        meta = read_json(seq_dir / "annotations.json")
        frames_meta = meta["frames"][: self.sequence_length]
        frames = []
        uv = []
        xyz = []
        visible = []
        for item in frames_meta:
            img = cv2.imread(str(seq_dir / "frames" / f"{item['frame_index']:06d}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            uv.append(item["uv"])
            xyz.append(item.get("xyz", [0.0, 0.0, 0.0]))
            visible.append(item["visible"])
        frames = np.asarray(frames, dtype=np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)
        uv_t = torch.tensor(uv, dtype=torch.float32)
        xyz_t = torch.tensor(xyz, dtype=torch.float32)
        visible_t = torch.tensor(visible, dtype=torch.float32)
        eot = torch.zeros(len(frames_meta), dtype=torch.float32)
        eot[-1] = 1.0
        features = torch.cat(
            [
                uv_t,
                visible_t[:, None],
                torch.zeros(len(frames_meta), 1),
                torch.linspace(0.0, 1.0, len(frames_meta))[:, None],
            ],
            dim=1,
        )
        if self.mode == "detector":
            t = np.random.randint(0, len(frames_meta))
            h, w = frames_t.shape[-2:]
            hs, ws = h // 4, w // 4
            heatmap = torch.zeros(1, hs, ws)
            offset = torch.zeros(2, hs, ws)
            if visible_t[t] > 0.5:
                u, v = uv_t[t]
                su = u * ws / w
                sv = v * hs / h
                cx = int(torch.clamp(torch.round(su), 0, ws - 1).item())
                cy = int(torch.clamp(torch.round(sv), 0, hs - 1).item())
                heatmap[0, cy, cx] = 1.0
                offset[0, cy, cx] = su - cx
                offset[1, cy, cx] = sv - cy
            return {
                "image": frames_t[t],
                "uv": uv_t[t],
                "visible": visible_t[t:t+1],
                "heatmap": heatmap,
                "offset": offset,
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
