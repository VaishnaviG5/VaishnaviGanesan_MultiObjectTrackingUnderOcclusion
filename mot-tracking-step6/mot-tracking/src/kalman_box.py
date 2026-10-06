"""Kalman filter for one tracked bounding box (the motion model used by SORT).

State (7 numbers):   x = [cx, cy, s, r, vx, vy, vs]
    cx, cy : box center
    s      : box area (w * h)
    r      : aspect ratio (w / h)         <- assumed constant, so no velocity for r
    vx, vy, vs : velocities of cx, cy, s (pixels per frame)

Measurement (4 numbers):  z = [cx, cy, s, r]   (what the detector gives us)

Motion model = constant velocity:  cx_next = cx + vx, and so on.
Why area + ratio instead of w + h? Area changes smoothly as a person walks
toward/away from the camera, and the ratio of a walking person stays ~constant.
"""
from __future__ import annotations

import numpy as np


def xywh_to_z(box) -> np.ndarray:
    """(x, y, w, h) top-left box -> measurement vector [cx, cy, s, r] (column)."""
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def state_to_xywh(state) -> np.ndarray:
    """State vector -> (x, y, w, h) top-left box."""
    cx, cy, s, r = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h])


class KalmanBoxTracker:
    def __init__(self, box, track_id: int, conf: float = 1.0):
        # F: how the state moves from one frame to the next (constant velocity)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        # H: which part of the state we can measure (cx, cy, s, r)
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        # R: measurement noise (area and ratio are noisier than the center)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        # P: state uncertainty. Velocities are completely unknown at the start -> large.
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        # Q: process noise (how much we allow the real motion to deviate from the model)
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = xywh_to_z(box)

        self.id = track_id
        self.conf = conf
        self.age = 0                 # frames since creation
        self.hits = 0                # total matched detections
        self.hit_streak = 0          # consecutive frames matched
        self.time_since_update = 0   # frames since last matched detection (0 = matched this frame)

    def predict(self) -> np.ndarray:
        """Advance the state one frame; return the predicted (x, y, w, h)."""
        if self.x[6, 0] + self.x[2, 0] <= 0:   # area would go negative -> stop shrinking
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:         # previous frame had no match: streak broken
            self.hit_streak = 0
        self.time_since_update += 1
        return state_to_xywh(self.x[:, 0])

    def update(self, box, conf: float = 1.0) -> None:
        """Correct the state with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf = conf
        z = xywh_to_z(box)
        y = z - self.H @ self.x                          # innovation: measurement vs prediction
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain: how much to trust the measurement
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def box(self) -> np.ndarray:
        """Current estimated (x, y, w, h)."""
        return state_to_xywh(self.x[:, 0])
