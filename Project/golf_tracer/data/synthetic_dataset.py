from __future__ import annotations

import json
import math
import random
from pathlib import Path

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

    # --- Sky: clear blue / overcast / warm sunrise ---
    sky_type = random.randint(0, 2)
    if sky_type == 0:  # clear blue
        sky = (
            random.randint(160, 220),  # B
            random.randint(190, 235),  # G
            random.randint(80, 150),   # R
        )
    elif sky_type == 1:  # overcast grey
        v = random.randint(140, 210)
        sky = (v, v, v)
    else:  # warm sunrise / golden hour
        sky = (
            random.randint(80, 160),
            random.randint(120, 185),
            random.randint(180, 240),
        )

    # --- Grass: vary shade and saturation ---
    grass = (
        random.randint(15, 55),   # B
        random.randint(70, 140),  # G
        random.randint(15, 60),   # R
    )

    horizon = int(h * random.uniform(0.28, 0.48))
    img[:] = grass
    img[:horizon] = sky

    # Add subtle per-pixel noise to grass to break uniformity
    grass_region = img[horizon:].astype(np.int16)
    grass_noise = np.random.randint(-12, 13, grass_region.shape, dtype=np.int16)
    img[horizon:] = np.clip(grass_region + grass_noise, 0, 255).astype(np.uint8)

    # Grass stripe / mowing pattern (horizontal bands of slightly different shade)
    stripe_h = random.randint(8, 20)
    for row in range(horizon, h, stripe_h * 2):
        shade = random.randint(-8, 8)
        end = min(h, row + stripe_h)
        band = img[row:end].astype(np.int16)
        img[row:end] = np.clip(band + shade, 0, 255).astype(np.uint8)

    # Tree silhouettes near horizon
    for _ in range(random.randint(0, 5)):
        tx = random.randint(0, w - 1)
        tw = random.randint(8, 28)
        th = random.randint(15, 55)
        cv2.rectangle(img, (tx, horizon - th), (min(w - 1, tx + tw), horizon + 4),
                      (10, random.randint(45, 75), 10), -1)

    # Bunkers / fairway patches
    for _ in range(random.randint(6, 18)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(horizon, h - 1)
        x2 = min(w - 1, x1 + random.randint(20, 130))
        y2 = min(h - 1, y1 + random.randint(4, 22))
        shade = random.randint(-28, 12)
        col = tuple(max(0, min(255, c + shade)) for c in grass)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)

    return img


def simulate_ballistics(T: int, fps: float) -> tuple[np.ndarray, dict]:
    """Simulate a golf ball trajectory in camera-frame coordinates with Magnus force.

    Coordinate convention (matches project() pinhole model):
      x = lateral  (horizontal in image via u = fx*x/z + cx)
      y = height   (vertical in image via v = fy*(cam_h-y)/z + cy)
      z = depth    (ball flies away from camera; z must be positive)

    Magnus force is modelled as:
      a_magnus = K_MAGNUS * (omega × v)
    where omega = (omega_x, omega_y, 0) and K_MAGNUS absorbs air density,
    ball cross-section, and mass (empirically calibrated for a golf ball).

    Backspin (omega_x < 0, top of ball moves rearward) produces lift (+y).
    Sidespin (omega_y != 0) produces lateral drift (±x, draw/fade).

    Returns:
        xyz: (T, 3) float32 array of world positions
        spin: dict with keys backspin_rpm (float) and sidespin_rpm (float)
    """
    K_MAGNUS = 1.5e-4   # m⁻¹; empirically calibrated for a golf ball
    G = 9.81
    dt = 1.0 / fps

    speed = random.uniform(45.0, 75.0)
    launch = math.radians(random.uniform(10.0, 28.0))
    side = math.radians(random.uniform(-8.0, 8.0))

    backspin_rpm = random.uniform(1500.0, 5000.0)
    sidespin_rpm = random.uniform(-1000.0, 1000.0)

    # Angular velocity in rad/s; omega_x < 0 for backspin (top → rear)
    omega_x = -backspin_rpm * 2.0 * math.pi / 60.0
    omega_y = sidespin_rpm * 2.0 * math.pi / 60.0

    # Initial velocity components (same convention as original)
    vx = speed * math.cos(launch) * math.sin(side)
    vy = speed * math.sin(launch)
    vz = speed * math.cos(launch) * math.cos(side)

    x, y, z = 0.0, 0.2, 10.0

    pts = []
    for _ in range(T):
        pts.append([x, y, z])

        # Magnus acceleration: a = K * (omega × v)
        # omega = (omega_x, omega_y, 0), so cross product simplifies to:
        ax = K_MAGNUS * (omega_y * vz)                        # i: ωy·vz
        ay = -G + K_MAGNUS * (-omega_x * vz)                  # j: -ωx·vz
        az = K_MAGNUS * (omega_x * vy - omega_y * vx)         # k: ωx·vy - ωy·vx

        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x += vx * dt
        y = max(0.0, y + vy * dt)
        z += vz * dt

    spin = {"backspin_rpm": float(backspin_rpm), "sidespin_rpm": float(sidespin_rpm)}
    return np.asarray(pts, dtype=np.float32), spin


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
        detector_scale: float = 0.25,
        detector_sigma: float = 2.0,
        # Gaussian noise (pixels) added to UV in trajectory-mode features to
        # simulate detector error and close the train/inference domain gap.
        detector_noise_std: float = 0.0,
    ):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.h, self.w = image_size
        self.fps = fps
        self.mode = mode
        self.detector_scale = detector_scale
        self.detector_sigma = detector_sigma
        self.detector_noise_std = detector_noise_std

    def _sample_camera(self) -> dict:
        """Sample randomised camera intrinsics per sequence for diversity."""
        fx = self.w * random.uniform(1.2, 1.8)
        fy = self.h * random.uniform(1.2, 1.8)
        cx = self.w * random.uniform(0.44, 0.56)
        cy = self.h * random.uniform(0.48, 0.62)
        cam_h = random.uniform(0.8, 2.0)
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "camera_height_m": cam_h}

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        T = self.sequence_length
        camera = self._sample_camera()
        xyz, spin = simulate_ballistics(T, self.fps)
        uv = project(xyz, camera)

        frames = []
        visible = []
        heatmaps = []
        offsets = []
        uncertainties = []

        out_h = int(round(self.h * self.detector_scale))
        out_w = int(round(self.w * self.detector_scale))

        # Per-sequence random dropout rate (3–15 %) to vary visibility patterns
        dropout_rate = random.uniform(0.03, 0.15)

        for t in range(T):
            frame = make_background(self.h, self.w)
            u, v = uv[t]

            vis = 1.0
            if u < 0 or u >= self.w or v < 0 or v >= self.h:
                vis = 0.0
            if random.random() < dropout_rate:
                vis = 0.0

            if vis > 0.5:
                # Scale ball radius with depth so far-away balls look smaller
                z_depth = float(xyz[t, 2])
                base_r = random.uniform(3.0, 5.5)
                radius = max(1, int(round(base_r * 10.0 / max(z_depth, 5.0))))
                cv2.circle(frame, (int(round(u)), int(round(v))), radius, (245, 245, 245), -1)
                if random.random() < 0.4:
                    k = random.choice([3, 5, 7])
                    frame = cv2.GaussianBlur(frame, (k, k), 0)

            noise = np.random.normal(0, 5, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            hm = np.zeros((out_h, out_w), dtype=np.float32)
            off = np.zeros((2, out_h, out_w), dtype=np.float32)
            unc = np.zeros((2, out_h, out_w), dtype=np.float32)

            if vis > 0.5:
                hu = u * self.detector_scale
                hv = v * self.detector_scale

                hm = gaussian_2d(out_h, out_w, hu, hv, sigma=self.detector_sigma)

                cx_idx = int(np.clip(np.floor(hu), 0, out_w - 1))
                cy_idx = int(np.clip(np.floor(hv), 0, out_h - 1))

                off[0, cy_idx, cx_idx] = hu - cx_idx
                off[1, cy_idx, cx_idx] = hv - cy_idx
                unc[:, cy_idx, cx_idx] = 1.0

            frames.append(frame)
            visible.append(vis)
            heatmaps.append(hm)
            offsets.append(off)
            uncertainties.append(unc)

        frames = np.stack(frames)
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        uv_t = torch.from_numpy(uv).float()
        xyz_t = torch.from_numpy(xyz).float()
        visible_t = torch.tensor(visible, dtype=torch.float32)
        heatmaps_t = torch.from_numpy(np.stack(heatmaps))[:, None]
        offsets_t = torch.from_numpy(np.stack(offsets))
        uncertainties_t = torch.from_numpy(np.stack(uncertainties))

        eot = torch.zeros(T, dtype=torch.float32)
        eot[-1] = 1.0

        if self.mode == "detector":
            visible_idx = torch.where(visible_t > 0.5)[0]
            hidden_idx = torch.where(visible_t <= 0.5)[0]

            if len(visible_idx) > 0 and (len(hidden_idx) == 0 or np.random.rand() < 0.5):
                t = int(visible_idx[np.random.randint(0, len(visible_idx))].item())
            else:
                t = int(hidden_idx[np.random.randint(0, len(hidden_idx))].item())
            return {
                "image": frames_t[t],
                "uv": uv_t[t],
                "visible": visible_t[t:t + 1],
                "heatmap": heatmaps_t[t],
                "offset": offsets_t[t],
                "uncertainty": uncertainties_t[t],
            }

        # Trajectory mode: optionally corrupt UV to simulate detector noise so
        # the LSTM learns to handle imperfect 2-D inputs at inference time.
        if self.detector_noise_std > 0:
            uv_feat = uv_t + torch.randn_like(uv_t) * self.detector_noise_std
            vis_feat = torch.clamp(
                visible_t + torch.randn_like(visible_t) * 0.12, 0.0, 1.0
            )
        else:
            uv_feat = uv_t
            vis_feat = visible_t

        features = torch.cat(
            [
                uv_feat,
                vis_feat[:, None],
                torch.zeros(T, 1, dtype=torch.float32),
                torch.linspace(0.0, 1.0, T, dtype=torch.float32)[:, None],
            ],
            dim=1,
        )

        spin_t = torch.tensor(
            [spin["backspin_rpm"], spin["sidespin_rpm"]], dtype=torch.float32
        )

        return {
            "frames": frames_t,
            "uv": uv_t,
            "xyz": xyz_t,
            "visible": visible_t,
            "features": features,
            "eot": eot,
            "camera": camera,
            "spin": spin_t,
        }


def export_synthetic_dataset(
    out_dir: str | Path,
    num_sequences: int = 100,
    sequence_length: int = 24,
    image_size=(256, 256),
    fps: float = 60.0,
):
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

        spin_data = {
            "backspin_rpm": float(sample["spin"][0].item()),
            "sidespin_rpm": float(sample["spin"][1].item()),
        }
        with open(seq_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fps": fps,
                    "camera": sample["camera"],
                    "spin": spin_data,
                    "frames": records,
                },
                f,
                indent=2,
            )