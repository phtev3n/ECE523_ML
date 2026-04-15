from __future__ import annotations
import numpy as np


class Kalman2D:
    """Constant-velocity Kalman filter for tracking a 2D image-plane position.

    State vector: x = [u, v, du, dv]^T
      u, v   — pixel position (horizontal, vertical)
      du, dv — pixel velocity (pixels per frame)

    The filter smooths noisy detector outputs and allows the tracker to
    "coast" for a limited number of frames when the ball is occluded or
    the detector's visibility head reports low confidence.  This prevents
    re-initialisation on every missed detection, which would introduce
    discontinuities into the tracer trail.
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        process_var: float = 15.0,
        meas_var: float = 4.0,
        max_coast_frames: int = 8,
        gate_sigma: float = 0.0,
    ):
        """
        Args:
            dt:               Time step in seconds (1/fps).
            process_var:      Process noise variance (Q = process_var * I).
                              Higher values allow faster acceleration changes
                              at the cost of noisier estimates.
            meas_var:         Measurement noise variance (R = meas_var * I).
                              Should reflect expected detector pixel error.
            max_coast_frames: Maximum consecutive frames the filter will
                              propagate without a measurement update before
                              the track is considered lost and re-initialised.
            gate_sigma:       Innovation gate threshold in standard deviations.
                              A measurement is rejected (and the filter coasts)
                              if its Mahalanobis distance from the predicted
                              position exceeds this value.  Set to 0 to disable
                              gating (legacy behaviour).  A value of 3.0 rejects
                              measurements more than 3-sigma from the prediction,
                              which corresponds to chi2(2) = 9.0, capturing 99%
                              of true ball detections while rejecting gross
                              outliers (e.g. detector locked on the golfer body).
        """
        self.dt = float(dt)
        self.max_coast_frames = int(max_coast_frames)
        # gate_sigma=0 disables gating; >0 enables chi-squared innovation gate
        self._gate_chi2 = float(gate_sigma) ** 2 if gate_sigma > 0.0 else 0.0

        # State transition matrix: x_{t+1} = F * x_t
        # Models constant velocity: position += velocity * dt
        self.F = np.array(
            [
                [1, 0, dt, 0],   # u_{t+1} = u_t + du*dt
                [0, 1, 0, dt],   # v_{t+1} = v_t + dv*dt
                [0, 0, 1,  0],   # du unchanged
                [0, 0, 0,  1],   # dv unchanged
            ],
            dtype=np.float32,
        )

        # Observation matrix: measurement z = H * x (we observe position only)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance — isotropic across all state dimensions
        self.Q = float(process_var) * np.eye(4, dtype=np.float32)

        # Measurement noise covariance — isotropic across u and v
        self.R = float(meas_var) * np.eye(2, dtype=np.float32)

        # State estimate and error covariance; uninitialised until init() is called
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0   # large initial uncertainty

        self.initialized = False
        self.coast_count = 0   # frames elapsed since last measurement update

    def init(self, uv):
        """Initialise (or re-initialise) the filter at position uv with zero velocity."""
        self.x[:, 0] = [uv[0], uv[1], 0.0, 0.0]
        self.P = np.eye(4, dtype=np.float32) * 10.0   # moderate initial covariance
        self.initialized = True
        self.coast_count = 0

    def predict(self):
        """Propagate the state forward one time step (prior estimate)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2, 0].copy()   # return predicted (u, v)

    def gate(self, uv) -> bool:
        """Return True if the measurement passes the innovation gate.

        Computes the squared Mahalanobis distance between the measurement and
        the predicted position:
            d² = yᵀ · S⁻¹ · y    where y = z - H·x̂,  S = H·P·Hᵀ + R

        For a 2-DOF chi-squared distribution:
            gate_sigma=2 → d² < 4    (chi² ≈ 86% of inliers)
            gate_sigma=3 → d² < 9    (chi² ≈ 99% of inliers)

        Returns True (accept) when gate is disabled (gate_sigma=0) or d² ≤ threshold.
        """
        if self._gate_chi2 <= 0.0 or not self.initialized:
            return True
        z = np.array(uv, dtype=np.float32).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float(y.T @ np.linalg.inv(S) @ y)
        return d2 <= self._gate_chi2

    def update(self, uv):
        """Fuse a new measurement with the predicted state (posterior estimate).

        Implements the standard Kalman correction step:
          y = z - H*x        (innovation)
          S = H*P*H^T + R    (innovation covariance)
          K = P*H^T * S^-1   (Kalman gain)
          x = x + K*y        (state update)
          P = (I - K*H)*P    (covariance update)
        """
        z = np.array(uv, dtype=np.float32).reshape(2, 1)
        y = z - self.H @ self.x          # innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        self.coast_count = 0
        return self.x[:2, 0].copy()

    def miss(self):
        """Record a missed detection — propagate without updating.

        The predicted position from the last predict() call is used as-is.
        coast_count is incremented so the pipeline knows how long the track
        has been running without measurement support.
        """
        self.coast_count += 1
        return self.x[:2, 0].copy()

    @property
    def can_coast(self) -> bool:
        """True while the track can still propagate without a measurement."""
        return self.coast_count < self.max_coast_frames
