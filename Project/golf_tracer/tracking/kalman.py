from __future__ import annotations
import numpy as np


class Kalman2D:
    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        process_var: float = 15.0,
        meas_var: float = 4.0,
        max_coast_frames: int = 8,
    ):
        self.dt = float(dt)
        self.max_coast_frames = int(max_coast_frames)
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = float(process_var) * np.eye(4, dtype=np.float32)
        self.R = float(meas_var) * np.eye(2, dtype=np.float32)
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.initialized = False
        self.coast_count = 0

    def init(self, uv):
        self.x[:, 0] = [uv[0], uv[1], 0.0, 0.0]
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = True
        self.coast_count = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2, 0].copy()

    def update(self, uv):
        z = np.array(uv, dtype=np.float32).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        self.coast_count = 0
        return self.x[:2, 0].copy()

    def miss(self):
        self.coast_count += 1
        return self.x[:2, 0].copy()

    @property
    def can_coast(self) -> bool:
        return self.coast_count < self.max_coast_frames
