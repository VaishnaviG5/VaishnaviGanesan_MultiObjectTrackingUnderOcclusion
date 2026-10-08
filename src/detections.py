"""Read/write detections in MOTChallenge text format.

One line per detection:
    frame, -1, x, y, w, h, conf, -1, -1, -1
(frame is 1-indexed, id is -1 because detections have no identity yet.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DET_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"]


def save_detections(rows, path: str | Path) -> None:
    """rows: iterable of (frame, x, y, w, h, conf)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, x, y, w, h, conf in rows:
            f.write(f"{frame},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")


def load_detections(path: str | Path, min_conf: float = 0.0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/detect.py first.")
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=DET_COLUMNS)
    df = pd.read_csv(path, header=None, names=DET_COLUMNS)
    return df[df["conf"] >= min_conf]


def detections_by_frame(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """frame -> array of shape (n, 5): x, y, w, h, conf."""
    out = {}
    for frame, g in df.groupby("frame"):
        out[int(frame)] = g[["x", "y", "w", "h", "conf"]].to_numpy()
    return out
