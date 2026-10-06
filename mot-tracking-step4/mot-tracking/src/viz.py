"""Small drawing helpers shared by later steps (tracking video, trajectories)."""
from __future__ import annotations

import cv2
import numpy as np


def id_color(track_id: int) -> tuple[int, int, int]:
    """A stable, visually distinct BGR color for each ID."""
    rng = np.random.default_rng(int(track_id) * 9973 + 7)
    return tuple(int(c) for c in rng.integers(60, 256, size=3))


def draw_boxes(img: np.ndarray, boxes, ids=None, thickness: int = 2) -> np.ndarray:
    """Draw (x, y, w, h) boxes, with optional ID labels. Returns a copy."""
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        tid = int(ids[i]) if ids is not None else i
        color = id_color(tid)
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(out, p1, p2, color, thickness)
        if ids is not None:
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
            cv2.putText(out, label, (p1[0] + 2, p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out
