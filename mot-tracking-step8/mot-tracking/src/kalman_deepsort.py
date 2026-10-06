"""Kalman filter used by DeepSORT (Wojke et al., 2017).

State (8 numbers):  [cx, cy, a, h, vx, vy, va, vh]
    a = aspect ratio w/h, h = height  (note: different from SORT, which uses area and ratio)

The noise levels are expressed RELATIVE TO THE BOX HEIGHT (1/20 of h for position, 1/160 for
velocity). That makes the filter behave the same for a big person near the camera and a small
one far away, and it makes the Mahalanobis distance meaningful, which DeepSORT uses to
"gate" (rule out) impossible matches before looking at appearance.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

# 95% quantile of the chi-square distribution with N degrees of freedom (Mahalanobis gating)
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


def xywh_to_cahh(box) -> np.ndarray:
    """(x, y, w, h) top-left -> measurement (cx, cy, a, h)."""
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0, w / h, h], dtype=float)


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)                # constant-velocity model
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)          # we observe (cx, cy, a, h)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2,
               2 * self._std_weight_position * h, 10 * self._std_weight_velocity * h,
               10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        h = mean[3]
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2,
                   self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5,
                   self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """State -> measurement space (mean and covariance of the expected detection)."""
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1,
               self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), (covariance @ self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements):
        """Squared Mahalanobis distance between the track's predicted box and each measurement (K,4)."""
        mean, covariance = self.project(mean, covariance)
        chol = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False)
        return np.sum(z * z, axis=0)
