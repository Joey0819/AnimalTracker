"""
Kalman filter for bounding box tracking.

State vector: [x, y, s, r, dx, dy, ds]
  x, y = center position
  s = scale (area)
  r = aspect ratio (constant velocity model assumes r is constant)
  dx, dy = velocity
  ds = scale change rate

Fully implemented — no external dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["KalmanBoxTracker"]


class KalmanBoxTracker:
    """
    Kalman filter tracker for a single bounding box.

    Tracks the bounding box state using a constant-velocity model.
    Based on the SORT tracker's Kalman filter design.
    """

    _count = 0

    def __init__(self, bbox_xyxy: np.ndarray | tuple):
        """
        Initialize tracker with a bounding box.

        Args:
            bbox_xyxy: Bounding box (x1, y1, x2, y2).
        """
        bbox = np.asarray(bbox_xyxy, dtype=np.float64)

        # Convert [x1, y1, x2, y2] to [cx, cy, s, r]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h  # scale = area
        r = w / max(h, 1e-6)  # aspect ratio

        # State: [cx, cy, s, r, dcx, dcy, ds]
        self.x = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float64)

        # State transition matrix (constant velocity)
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1  # cx += dcx
        self.F[1, 5] = 1  # cy += dcy
        self.F[2, 6] = 1  # s += ds

        # Measurement matrix (we observe [cx, cy, s, r])
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Covariance matrices
        self.P = np.eye(7, dtype=np.float64) * 10  # State covariance
        self.P[4, 4] = 1000  # High uncertainty for velocities initially
        self.P[5, 5] = 1000
        self.P[6, 6] = 1000

        # Process noise
        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4, 4] = 0.01
        self.Q[5, 5] = 0.01
        self.Q[6, 6] = 0.0001
        self.Q[-1, -1] = 0.0001

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64)
        self.R[2, 2] = 10  # Scale measurement is noisier
        self.R[3, 3] = 10

        # Tracking stats
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

    def predict(self) -> np.ndarray:
        """
        Predict the next state and return predicted bbox.

        Returns:
            Predicted bounding box (x1, y1, x2, y2).
        """
        # Prevent negative area
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0

        # State prediction: x = F * x
        self.x = self.F @ self.x

        # Covariance prediction: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def update(self, bbox_xyxy: np.ndarray | tuple) -> None:
        """
        Update the state with a new measurement.

        Args:
            bbox_xyxy: Observed bounding box (x1, y1, x2, y2).
        """
        bbox = np.asarray(bbox_xyxy, dtype=np.float64)

        # Convert to measurement: [cx, cy, s, r]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([
            bbox[0] + w / 2.0,
            bbox[1] + h / 2.0,
            w * h,
            w / max(h, 1e-6),
        ])

        # Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * (z - H * x)
        y = z - self.H @ self.x  # Innovation
        self.x = self.x + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

        # Update tracking stats
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        """
        Get the current state as a bounding box.

        Returns:
            Bounding box (x1, y1, x2, y2).
        """
        cx, cy, s, r = self.x[:4]
        s = max(s, 1)  # Prevent negative area
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return np.array([
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ])

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity (dcx, dcy) in pixels/frame."""
        return self.x[4:6]
