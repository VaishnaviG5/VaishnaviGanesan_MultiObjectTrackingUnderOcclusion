"""Drawing code for Step 8: boxes + IDs + fading trails for video, and static trajectory plots."""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from src.viz import draw_boxes, id_color


def anchor_point(rows: np.ndarray, point: str = "feet") -> np.ndarray:
    """(M,2) positions of boxes (x,y,w,h,...): 'feet' = bottom centre (where the person stands), or 'center'."""
    x, y, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    return np.column_stack([x + w / 2, y + h if point == "feet" else y + h / 2])


class TrailRenderer:
    """Keeps the recent path of every track ID and draws it with a fade: new = strong, old = faint."""

    def __init__(self, trail: int = 30, point: str = "feet", buckets: int = 5):
        self.trail, self.point, self.buckets = trail, point, buckets
        self.hist: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    def update(self, frame: int, rows: np.ndarray) -> None:
        if len(rows):
            for tid, (px, py) in zip(rows[:, 4].astype(int), anchor_point(rows, self.point)):
                self.hist[int(tid)].append((frame, float(px), float(py)))
        for tid in list(self.hist):                               # forget points older than the trail length
            self.hist[tid] = [p for p in self.hist[tid] if frame - p[0] <= self.trail]
            if not self.hist[tid]:
                del self.hist[tid]

    def draw(self, img: np.ndarray, frame: int, scale: float = 1.0) -> np.ndarray:
        """Draw all trails onto (a copy of) img. Coordinates are multiplied by `scale`."""
        segs = [[] for _ in range(self.buckets)]                  # segments grouped by age
        for tid, pts in self.hist.items():
            for (f0, x0, y0), (f1, x1, y1) in zip(pts[:-1], pts[1:]):
                b = min(int((frame - f1) / max(self.trail, 1) * self.buckets), self.buckets - 1)
                thick = 1 if f1 - f0 > 1 else max(1, 3 - b // 2)    # thin line across a gap in tracking
                segs[b].append((tid, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), thick))
        out = img.copy()
        for b, group in enumerate(segs):
            if not group:
                continue
            alpha = 1.0 - 0.85 * (b + 0.5) / self.buckets
            layer = out.copy()
            for tid, p0, p1, thick in group:
                cv2.line(layer, p0, p1, id_color(tid), thick, cv2.LINE_AA)
            out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
        return out


def draw_overlay_text(img: np.ndarray, text: str, org=(12, 30), scale: float = 0.8) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(img: np.ndarray, rows: np.ndarray, trails: TrailRenderer, frame: int, label: str,
                 scale: float = 1.0) -> np.ndarray:
    """One output frame: image (resized by `scale`), fading trails, boxes with IDs, caption."""
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = trails.draw(img, frame, scale)
    if len(rows):
        out = draw_boxes(out, rows[:, :4] * scale, ids=rows[:, 4])
        for (px, py), tid in zip(anchor_point(rows, trails.point) * scale, rows[:, 4].astype(int)):
            cv2.circle(out, (int(px), int(py)), 4, id_color(tid), -1, cv2.LINE_AA)
    draw_overlay_text(out, f"{label}   frame {frame}   tracks {len(rows)}")
    return out


def hstack_same_height(frames: list[np.ndarray]) -> np.ndarray:
    h = min(f.shape[0] for f in frames)
    fs = [f if f.shape[0] == h else cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]
    return np.hstack(fs)


# ------------------------------------------------------------------ static trajectories
def trajectories(df, min_len: int = 15, point: str = "feet", start: int | None = None, end: int | None = None):
    """df with columns frame,id,x,y,w,h -> {id: (N,2) array of positions ordered by frame}, long enough only."""
    d = df
    if start is not None:
        d = d[d["frame"] >= start]
    if end is not None:
        d = d[d["frame"] <= end]
    out = {}
    for tid, g in d.sort_values("frame").groupby("id"):
        if len(g) >= min_len:
            out[int(tid)] = anchor_point(g[["x", "y", "w", "h"]].to_numpy(dtype=float), point)
    return out


def plot_trajectories(ax, background_bgr, trajs: dict, title: str) -> None:
    """Draw trajectories (start = circle, end = cross) over a dimmed background image."""
    ax.imshow(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB), alpha=0.45)
    for tid, pts in trajs.items():
        b, g, r = id_color(tid)
        c = (r / 255, g / 255, b / 255)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.6, alpha=0.95)
        ax.scatter(pts[0, 0], pts[0, 1], s=22, color=c, edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=26, color=c, marker="X", edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_title(f"{title}: {len(trajs)} trajectories", fontsize=10)
    ax.axis("off")


# ------------------------------------------------------------------ clip suggestions
def busiest_windows(per_frame: dict[int, float], length: int, top: int = 3, n_frames: int | None = None):
    """Non-overlapping windows [start, start+length) with the highest total of per_frame values."""
    n = n_frames or (max(per_frame) if per_frame else 0)
    vals = np.array([per_frame.get(f, 0.0) for f in range(1, n + 1)])
    if n < length:
        return []
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    scores = csum[length:] - csum[:-length]                       # score of window starting at frame i+1
    chosen, taken = [], np.zeros(len(scores), dtype=bool)
    for i in np.argsort(-scores, kind="stable"):
        if taken[i]:
            continue
        chosen.append((int(i) + 1, float(scores[i])))
        taken[max(0, i - length + 1): i + length] = True
        if len(chosen) == top:
            break
    return chosen
