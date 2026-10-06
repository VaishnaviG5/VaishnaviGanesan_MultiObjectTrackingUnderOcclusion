"""DeepSORT-style tracker: SORT's motion model + appearance embeddings.

What is added on top of SORT (src/sort.py):

1. APPEARANCE. Every detection comes with an embedding vector (from a re-ID network or a
   color histogram). Each track keeps a gallery of its recent embeddings. The "cost" of
   matching a detection to a track is the smallest cosine distance to anything in the gallery.
2. GATING. A match is only allowed if the detection is physically plausible given the track's
   Kalman prediction (squared Mahalanobis distance below the chi-square 95% limit).
3. MATCHING CASCADE. Confirmed tracks are matched in order of how recently they were seen
   (tracks seen 1 frame ago first, then 2 frames ago, ...). This stops an old, uncertain track
   from stealing a detection that belongs to a recently seen track.
4. IoU FALLBACK. Detections still unmatched are matched, by IoU, to tentative tracks and to
   confirmed tracks that were lost only 1 frame ago.
5. TRACK STATES. tentative -> confirmed after `n_init` hits (a tentative track that misses once
   is deleted); a confirmed track is deleted after `max_age` frames without a match. A confirmed
   track is reported again immediately after re-matching (unlike SORT's min_hits streak).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.boxes import iou_matrix
from src.kalman_deepsort import CHI2INV95, KalmanFilter, xywh_to_cahh

INFTY_COST = 1e5
TENTATIVE, CONFIRMED, DELETED = 1, 2, 3


class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature, conf, budget):
        self.mean, self.covariance = mean, covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TENTATIVE
        self.features = [feature]          # appearance gallery
        self.conf = conf
        self._n_init, self._max_age, self._budget = n_init, max_age, budget

    def to_xywh(self) -> np.ndarray:
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h])

    def predict(self, kf: KalmanFilter) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, box, feature, conf) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, xywh_to_cahh(box))
        self.features.append(feature)
        self.features = self.features[-self._budget:]
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0
        if self.state == TENTATIVE and self.hits >= self._n_init:
            self.state = CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TENTATIVE:
            self.state = DELETED
        elif self.time_since_update > self._max_age:
            self.state = DELETED

    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED


def min_cost_matching(cost_fn, max_distance, tracks, track_indices, det_indices):
    """Hungarian matching on cost_fn(track_indices, det_indices); pairs above max_distance are rejected."""
    if len(det_indices) == 0 or len(track_indices) == 0:
        return [], list(track_indices), list(det_indices)
    cost = cost_fn(track_indices, det_indices)
    cost[cost > max_distance] = max_distance + 1e-5
    rows, cols = linear_sum_assignment(cost)
    matches, matched_t, matched_d = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_distance:
            matches.append((track_indices[r], det_indices[c]))
            matched_t.add(track_indices[r])
            matched_d.add(det_indices[c])
    unmatched_t = [t for t in track_indices if t not in matched_t]
    unmatched_d = [d for d in det_indices if d not in matched_d]
    return matches, unmatched_t, unmatched_d


class DeepSort:
    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.2,
                 max_iou_distance: float = 0.7, nn_budget: int = 100, use_gating: bool = True,
                 output_predicted: bool = False):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance      # 0.7 means "IoU must be at least 0.3"
        self.nn_budget = nn_budget
        self.use_gating = use_gating
        self.output_predicted = output_predicted      # also report tracks lost this very frame (predicted box)
        self.kf = KalmanFilter()
        self.tracks: list[Track] = []
        self._next_id = 1

    # ---- cost functions ----------------------------------------------------
    def _appearance_cost(self, track_indices, det_indices, dets, feats):
        det_feats = feats[det_indices]
        cost = np.zeros((len(track_indices), len(det_indices)))
        for r, ti in enumerate(track_indices):
            gallery = np.asarray(self.tracks[ti].features)
            cost[r] = (1.0 - gallery @ det_feats.T).min(axis=0)        # best match in the gallery
        if self.use_gating:
            meas = np.array([xywh_to_cahh(dets[d, :4]) for d in det_indices])
            for r, ti in enumerate(track_indices):
                t = self.tracks[ti]
                d2 = self.kf.gating_distance(t.mean, t.covariance, meas)
                cost[r, d2 > CHI2INV95[4]] = INFTY_COST                # physically implausible
        return cost

    def _iou_cost(self, track_indices, det_indices, dets):
        cost = np.zeros((len(track_indices), len(det_indices)))
        det_boxes = dets[det_indices, :4]
        for r, ti in enumerate(track_indices):
            t = self.tracks[ti]
            if t.time_since_update > 1:
                cost[r, :] = INFTY_COST
                continue
            cost[r] = 1.0 - iou_matrix(t.to_xywh()[None, :], det_boxes)[0]
        return cost

    # ---- association -------------------------------------------------------
    def _match(self, dets, feats):
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # (a) matching cascade for confirmed tracks: most recently seen first
        matches_a, unmatched_d = [], list(range(len(dets)))
        for level in range(self.max_age):
            if not unmatched_d:
                break
            level_tracks = [k for k in confirmed if self.tracks[k].time_since_update == 1 + level]
            if not level_tracks:
                continue
            m, _, unmatched_d = min_cost_matching(
                lambda ti, di: self._appearance_cost(ti, di, dets, feats),
                self.max_cosine_distance, self.tracks, level_tracks, unmatched_d)
            matches_a += m
        matched_conf = {k for k, _ in matches_a}
        unmatched_conf = [k for k in confirmed if k not in matched_conf]

        # (b) IoU matching: tentative tracks + confirmed tracks lost only 1 frame ago
        iou_candidates = unconfirmed + [k for k in unmatched_conf if self.tracks[k].time_since_update == 1]
        left_over = [k for k in unmatched_conf if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_b, unmatched_d = min_cost_matching(
            lambda ti, di: self._iou_cost(ti, di, dets),
            self.max_iou_distance, self.tracks, iou_candidates, unmatched_d)

        return matches_a + matches_b, left_over + unmatched_b, unmatched_d

    # ---- one frame ---------------------------------------------------------
    def update(self, dets: np.ndarray, feats: np.ndarray) -> np.ndarray:
        """dets: (N,5) [x,y,w,h,conf];  feats: (N,D) L2-normalised embeddings.

        Call every frame (also with N=0). Returns (M,6) [x,y,w,h,track_id,conf].
        """
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        feats = np.asarray(feats, dtype=float)
        feats = feats.reshape(len(dets), -1) if len(dets) else np.zeros((0, 1))   # empty frame
        ok = (dets[:, 2] > 0) & (dets[:, 3] > 0)
        dets, feats = dets[ok], feats[ok]

        for t in self.tracks:
            t.predict(self.kf)

        matches, unmatched_tracks, unmatched_dets = self._match(dets, feats)

        for ti, di in matches:
            self.tracks[ti].update(self.kf, dets[di, :4], feats[di], dets[di, 4])
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()
        for di in unmatched_dets:
            mean, cov = self.kf.initiate(xywh_to_cahh(dets[di, :4]))
            self.tracks.append(Track(mean, cov, self._next_id, self.n_init, self.max_age,
                                     feats[di], dets[di, 4], self.nn_budget))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.state != DELETED]

        max_gap = 1 if self.output_predicted else 0
        out = []
        for t in self.tracks:
            if t.is_confirmed() and t.time_since_update <= max_gap:
                x, y, w, h = t.to_xywh()
                out.append([x, y, w, h, t.track_id, t.conf])
        return np.array(out).reshape(-1, 6)
