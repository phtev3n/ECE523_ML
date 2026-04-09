from __future__ import annotations
import torch
import torch.nn as nn


class TrajectoryLifter(nn.Module):
    """LSTM-based model that lifts a 2D image-plane trajectory to 3D world coordinates.

    Architecture overview
    ---------------------
    Input (T, 5) per sequence:
      channel 0–1 : filtered (u, v) pixel position from the Kalman tracker
      channel 2   : visibility probability from the detector head (0–1)
      channel 3   : detector uncertainty (log-variance, normalised)
      channel 4   : normalised frame index (0 → 1) encoding temporal position

    A linear pre-projection expands each frame's 5-dim feature to hidden_size
    before being fed into a causal (forward-only) LSTM.  Bidirectional LSTM was
    tested but offered marginal gains while complicating real-time deployment.

    Three output heads branch from the LSTM hidden states:
      xyz_head    — per-frame 3D position (x, y, z) in camera-frame metres
      eot_head    — per-frame end-of-trajectory probability (sigmoid)
      spin_head   — sequence-level spin estimate [backspin, sidespin] in rpm,
                    computed from mean-pooled hidden states so the full
                    trajectory curvature informs the spin prediction rather
                    than just the last frame
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        # Project input features into LSTM dimensionality
        self.pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        # Causal LSTM: processes frames left-to-right, mimicking real-time tracking
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,   # dropout only between layers
            bidirectional=False,
        )

        # Per-frame 3D position head: hidden → 3 (x, y, z metres, camera-frame)
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 3),
        )

        # Per-frame end-of-trajectory head: hidden → scalar probability
        # Used to signal when the ball has landed or left the frame so the
        # pipeline can stop the tracer cleanly rather than running on a frozen
        # Kalman prediction.
        self.eot_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Sequence-level spin head: mean-pooled hidden → [backspin_rpm, sidespin_rpm]
        # Mean pooling aggregates curvature signal across all frames rather than
        # relying solely on the last hidden state.  Spin estimation requires
        # observing the Magnus-force deflection accumulated over the full clip;
        # this is only reliably estimable from clips ≥ ~90 frames at 60 fps.
        self.spin_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, T, input_size) — batch of frame-feature sequences

        Returns:
            dict with keys:
              "xyz"      : (B, T, 3)  per-frame 3D positions (metres)
              "eot_prob" : (B, T)     per-frame end-of-trajectory probability
              "spin"     : (B, 2)     sequence-level [backspin, sidespin] rpm
        """
        x = self.pre(x)               # (B, T, hidden)
        y, _ = self.lstm(x)           # (B, T, hidden)
        xyz = self.xyz_head(y)        # (B, T, 3)
        eot = self.eot_head(y).squeeze(-1)         # (B, T)
        spin = self.spin_head(y.mean(dim=1))        # (B, 2)
        return {"xyz": xyz, "eot_prob": eot, "spin": spin}
