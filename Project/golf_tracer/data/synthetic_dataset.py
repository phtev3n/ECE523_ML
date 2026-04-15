"""Synthetic golf ball dataset for training the detector and trajectory models.

Everything here is procedurally generated — no real video is required.  The
synthetic data is designed to be diverse enough that models trained on it
transfer to real footage without catastrophic failure, acting as a strong
pre-training baseline before fine-tuning on real sequences.

Key generation choices
----------------------
Backgrounds: randomised sky colour, horizon position, grass shade, mowing
  stripe pattern, tree silhouettes, and bunker patches.  This variety prevents
  the detector from over-fitting to a single background type.

Ball rendering: radius scales inversely with depth (z) so far-away balls
  appear smaller.  Occasional Gaussian blur simulates motion blur at high speed.

Ballistics: Magnus-force numerical integration with club-family-specific
  parameter ranges (Trackman-sourced).  Each sequence independently samples
  its club family, speed, launch angle, sidespin, and backspin.

Camera intrinsics: randomised per sequence (focal length ±40 %, principal
  point ±6 %, camera height 0.8–2.0 m) so the trajectory model learns a
  camera-invariant representation rather than memorising a fixed projection.

Visibility: random per-sequence dropout rate (3–15 %) plus out-of-frame
  clipping.  Frames where the ball is outside image bounds are always invisible.
"""
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
    """Generate a 2D Gaussian blob centred at (cx, cy) as the heatmap target.

    Using a soft Gaussian target (rather than a hard 1-at-peak label) gives
    the heatmap loss spatial smoothness: nearby cells receive partial credit,
    which encourages the network to produce peaked, well-localised responses
    rather than sparse single-pixel activations.
    """
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
      y = height   (vertical in image via v increases as y increases)
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

    # Trackman-sourced ranges per club family.
    # Sampling a club family first then drawing parameters within its range
    # produces a realistic distribution of shot shapes and distances rather
    # than an unrealistically uniform spread across driver and wedge extremes.
    #   (sp_lo, sp_hi m/s,  la_lo, la_hi °,  bs_lo, bs_hi rpm)
    #
    # Carry targets (approximate):
    #   Driver         235–300 yd  (215–274 m)   speed 73–83 m/s
    #   Fairway woods  180–235 yd  (165–215 m)   speed 58–72 m/s
    #   Long irons     170–200 yd  (155–183 m)   speed 52–62 m/s
    #   Mid irons      150–175 yd  (137–160 m)   speed 44–54 m/s
    #   Wedges          30–120 yd   (27–110 m)   speed 16–46 m/s
    #     (covers lob wedge 30 yd punch shots through pitching wedge full swings)
    CLUB_FAMILIES = [
        (73.0, 83.0,  8.0, 14.0,  2000.0,  3500.0),   # driver
        (58.0, 72.0, 10.0, 16.0,  3500.0,  5500.0),   # fairway woods
        (52.0, 62.0, 13.0, 18.0,  4000.0,  6000.0),   # long irons (3–5i)
        (44.0, 54.0, 16.0, 22.0,  5500.0,  8000.0),   # mid irons (6–8i)
        (16.0, 46.0, 22.0, 50.0,  7000.0, 12000.0),   # wedges (LW–PW, 30–120 yd)
    ]
    sp_lo, sp_hi, la_lo, la_hi, bs_lo, bs_hi = random.choice(CLUB_FAMILIES)

    speed = random.uniform(sp_lo, sp_hi)
    launch = math.radians(random.uniform(la_lo, la_hi))
    side = math.radians(random.uniform(-8.0, 8.0))

    backspin_rpm = random.uniform(bs_lo, bs_hi)
    sidespin_rpm = random.uniform(-1000.0, 1000.0)

    # Angular velocity in rad/s; omega_x < 0 for backspin (top → rear)
    omega_x = -backspin_rpm * 2.0 * math.pi / 60.0
    omega_y = sidespin_rpm * 2.0 * math.pi / 60.0

    # Initial velocity components (same convention as original)
    vx = speed * math.cos(launch) * math.sin(side)
    vy = speed * math.sin(launch)
    vz = speed * math.cos(launch) * math.cos(side)

    # Randomise initial camera-to-ball depth over a realistic range.
    # Fixed z=10m caused a systematic 2× speed overestimate on real recordings
    # where the camera is typically 2–4m from the ball at impact.  The real
    # camera distance was estimated at ~2.7m from the 52px/frame UV motion at
    # impact (z = fy * v_ball / uv_rate).  Sampling 2–12m ensures the LSTM
    # learns to infer depth from the UV arc shape rather than a fixed prior.
    z = random.uniform(2.0, 12.0)
    x, y = 0.0, 0.2

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
    """Project 3D camera-frame points to image-pixel UV.

    Convention matches the real dataset (frames extracted with --orient 180).
    A 180° rotation of the standard pinhole model gives:
      u_rot = (W-1) - (fx*x/z + cx)  →  u_rot = (W-1-cx) - fx*x/z
      v_rot = (H-1) - (fy*(cam_h-y)/z + cy)  →  v_rot = (H-1-cy) + fy*(y-cam_h)/z
    Both u and v increase in the OPPOSITE direction from the standard pinhole.
    In particular, v INCREASES as the ball rises.
    """
    z   = np.clip(points_xyz[:, 2], 1e-3, None)
    H   = cam.get("image_h", 512)
    W   = cam.get("image_w", 512)
    u   = (W - 1 - cam["cx"]) - cam["fx"] * points_xyz[:, 0] / z
    v   = (H - 1 - cam["cy"]) + cam["fy"] * (points_xyz[:, 1] - cam["camera_height_m"]) / z
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
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "camera_height_m": cam_h,
            "image_h": self.h,   # needed by project() for flipped convention
            "image_w": self.w,
        }

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

        # Feature vector layout (T, 5):
        #   col 0–1 : (u, v) pixel position (possibly noise-corrupted)
        #   col 2   : visibility probability (possibly noise-corrupted)
        #   col 3   : zeros placeholder for log-variance uncertainty
        #             (filled by the real detector at inference; zero here
        #              because the synthetic generator has no uncertainty model)
        #   col 4   : normalised frame index 0→1 (trajectory phase encoding)
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