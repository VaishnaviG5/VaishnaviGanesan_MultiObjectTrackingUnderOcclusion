"""Bounding-box helpers. All boxes are (x, y, w, h) with (x, y) = top-left corner."""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) -> (N, M) matrix."""
    a, b = xywh_to_xyxy(a), xywh_to_xyxy(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)
