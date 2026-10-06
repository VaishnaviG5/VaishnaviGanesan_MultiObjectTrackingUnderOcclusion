"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Per frame:
  1. PREDICT   move every existing track forward with its Kalman filter
  2. ASSOCIATE match detections to predicted boxes (cost = IoU, Hungarian algorithm);
               reject matches with IoU below `iou_thr`
  3. UPDATE    matched tracks are corrected with their detection;
               unmatched detections start new tracks;
               unmatched tracks keep "coasting" on their prediction
  4. DELETE    tracks unmatched for more than `max_age` frames are removed

Occlusion handling: a person hidden for fewer than `max_age` frames keeps their
track alive; when they reappear near the predicted position, the detection is
matched back to the same track and the ID is preserved.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.boxes import iou_matrix
from src.kalman_box import KalmanBoxTracker


def associate(det_boxes: np.ndarray, trk_boxes: np.ndarray, iou_thr: float):
    """Match detections to tracks. Returns (matches, unmatched_det_idx, unmatched_trk_idx)."""
    if len(trk_boxes) == 0:
        return np.empty((0, 2), dtype=int), list(range(len(det_boxes))), []
    if len(det_boxes) == 0:
        return np.empty((0, 2), dtype=int), [], list(range(len(trk_boxes)))

    iou = iou_matrix(det_boxes, trk_boxes)           # (num_dets, num_tracks)
    rows, cols = linear_sum_assignment(-iou)         # maximize total IoU

    matches = []
    for d, t in zip(rows, cols):
        if iou[d, t] >= iou_thr:                     # too little overlap: not the same person
            matches.append((d, t))
    matched_d = {d for d, _ in matches}
    matched_t = {t for _, t in matches}
    unmatched_d = [d for d in range(len(det_boxes)) if d not in matched_d]
    unmatched_t = [t for t in range(len(trk_boxes)) if t not in matched_t]
    return np.array(matches, dtype=int).reshape(-1, 2), unmatched_d, unmatched_t


class Sort:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_thr: float = 0.3):
        self.max_age = max_age      # frames a lost track survives before deletion
        self.min_hits = min_hits    # matched frames needed before a track is reported
        self.iou_thr = iou_thr      # minimum IoU for a detection-track match
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1           # per-instance counter -> IDs restart for every sequence

    def update(self, dets: np.ndarray) -> np.ndarray:
        """Process one frame.

        dets: array (N, 5) of [x, y, w, h, conf]  (N may be 0; call this EVERY frame
              so that lost tracks keep aging).
        Returns array (M, 6) of [x, y, w, h, track_id, conf] for tracks matched this frame.
        """
        self.frame_count += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)

        # 1. PREDICT
        preds = []
        for t in self.trackers:
            preds.append(t.predict())
        trk_boxes = np.array(preds).reshape(-1, 4)
        ok = np.isfinite(trk_boxes).all(axis=1)      # drop tracks whose state blew up
        self.trackers = [t for t, good in zip(self.trackers, ok) if good]
        trk_boxes = trk_boxes[ok]

        # 2. ASSOCIATE
        matches, unmatched_d, _ = associate(dets[:, :4], trk_boxes, self.iou_thr)

        # 3. UPDATE
        for d, t in matches:
            self.trackers[t].update(dets[d, :4], dets[d, 4])
        for d in unmatched_d:
            self.trackers.append(KalmanBoxTracker(dets[d, :4], self._next_id, dets[d, 4]))
            self._next_id += 1

        # report: only tracks matched THIS frame and confirmed (enough hits).
        # During the first `min_hits` frames of a video we report everything (nothing is confirmed yet).
        out = []
        for t in self.trackers:
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if t.time_since_update < 1 and confirmed:
                x, y, w, h = t.box()
                out.append([x, y, w, h, t.id, t.conf])

        # 4. DELETE tracks lost for too long
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return np.array(out).reshape(-1, 6)
