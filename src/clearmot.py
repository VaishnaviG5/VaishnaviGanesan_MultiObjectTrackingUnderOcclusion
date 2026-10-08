"""A small, self-contained MOT evaluator (CLEAR-MOT metrics + IDF1) with ID-switch events.

Why in the repo? It lets us run many experiments in seconds (Step 4 ablations) and gives us
the list of ID-switch events we need for the failure analysis (Step 7). The OFFICIAL numbers
come from TrackEval in Step 6; this evaluator is used for fast iteration and cross-checking.

Definitions (IoU threshold 0.5 for a GT-hypothesis match):
  FN    ground-truth person with no matched hypothesis in a frame
  FP    hypothesis box with no matched ground truth
  IDSW  a GT person is matched to a DIFFERENT track ID than the one it was last matched to
  MOTA  1 - (FN + FP + IDSW) / num_gt_boxes          (can be negative)
  IDF1  2*IDTP / (num_gt_boxes + num_hyp_boxes), where IDTP comes from the best GLOBAL
        one-to-one assignment of GT identities to track IDs (rewards consistent identities)
  Frag  a GT person is tracked, then lost, then tracked again
  MT/ML mostly tracked (> 80% of its frames matched) / mostly lost (< 20%)

Frame-level matching follows CLEAR-MOT: first keep last frame's pairings if still valid,
then match the rest with the Hungarian algorithm.

Simplification vs. the official MOT17 protocol: hypotheses that land on "ignore" ground truth
(static person, person on vehicle, distractor, reflection) are removed before scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.boxes import iou_matrix

IGNORE_CLASSES = {2, 7, 8, 12}
TRACK_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"]
XYWH = ["x", "y", "w", "h"]


def load_tracks(path) -> pd.DataFrame:
    """Read a tracker result file (MOT format)."""
    try:
        df = pd.read_csv(path, header=None, names=TRACK_COLUMNS)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=TRACK_COLUMNS)
    return df


def results_to_df(results: dict) -> pd.DataFrame:
    """{frame: array (M,6) of x,y,w,h,id,conf} -> DataFrame in TRACK_COLUMNS layout."""
    rows = []
    for frame in sorted(results):
        for x, y, w, h, tid, conf in results[frame]:
            rows.append((frame, int(tid), x, y, w, h, conf, -1, -1, -1))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _match(iou: np.ndarray, thr: float):
    """Hungarian matching on IoU; only pairs with IoU >= thr are allowed."""
    if iou.size == 0:
        return []
    cost = np.where(iou >= thr, 1.0 - iou, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] < 1e6]


def finalize(m: dict) -> dict:
    """Compute rate metrics from raw counts (also used for the OVERALL row)."""
    tp, fp, fn = m["TP"], m["FP"], m["FN"]
    m["MOTA"] = 1.0 - (fn + fp + m["IDSW"]) / max(m["num_gt"], 1)
    m["IDF1"] = 2.0 * m["IDTP"] / max(m["num_gt"] + m["num_hyp"], 1)
    m["Precision"] = tp / max(tp + fp, 1)
    m["Recall"] = tp / max(tp + fn, 1)
    return m


def aggregate(rows: list[dict], name: str = "OVERALL") -> dict:
    """Combine per-sequence results: sum the counts, then recompute rates."""
    keys = ["num_gt", "num_hyp", "TP", "FP", "FN", "IDSW", "Frag", "MT", "ML",
            "num_gt_ids", "num_hyp_ids", "IDTP"]
    out = {"seq": name, **{k: int(sum(r[k] for r in rows)) for k in keys}}
    return finalize(out)


def evaluate_tracks(seq, hyp: pd.DataFrame, iou_thr: float = 0.5, seq_name: str | None = None):
    """Score tracker output `hyp` (DataFrame with TRACK_COLUMNS) against seq's ground truth.

    Returns (metrics_dict, idsw_events_dataframe).
    `seq` only needs a .load_gt(pedestrians_only=False) method.
    """
    gt_all = seq.load_gt(pedestrians_only=False)
    gt_groups = {int(f): g for f, g in gt_all.groupby("frame")}
    hyp_groups = {int(f): g for f, g in hyp.groupby("frame")} if len(hyp) else {}
    empty_gt, empty_h = gt_all.iloc[0:0], hyp.iloc[0:0]
    n_frames = int(max(max(gt_groups, default=0), max(hyp_groups, default=0)))

    last_hyp: dict[int, int] = {}                 # gt id -> track id it was last matched to
    last_frame: dict[int, int] = {}               # gt id -> frame it was last matched
    hyp_last_gt: dict[int, int] = {}              # track id -> gt id it was last matched to
    gt_frames: dict[int, int] = {}                # gt id -> frames present
    gt_tracked: dict[int, int] = {}               # gt id -> frames matched
    ever_tracked: dict[int, bool] = {}
    last_status: dict[int, bool] = {}
    hyp_counts: dict[int, int] = {}               # track id -> boxes (after ignore filtering)
    overlap: dict[tuple[int, int], int] = {}      # (gt id, track id) -> frames with IoU >= thr
    events = []
    tp = fp = fn = idsw = frag = num_gt = num_hyp = 0

    for f in range(1, n_frames + 1):
        g = gt_groups.get(f, empty_gt)
        ped = g[(g["cls"] == 1) & (g["conf"] == 1)]
        h = hyp_groups.get(f, empty_h)
        h_boxes = h[XYWH].to_numpy(dtype=float)
        h_ids = h["id"].to_numpy(dtype=int)

        # drop hypotheses that sit on ignore regions (matched to a distractor-type GT box)
        if len(h) and len(g):
            g_cls = g["cls"].to_numpy()
            pairs = _match(iou_matrix(h_boxes, g[XYWH].to_numpy(dtype=float)), iou_thr)
            drop = {i for i, j in pairs if g_cls[j] in IGNORE_CLASSES}
            if drop:
                keep = [i for i in range(len(h)) if i not in drop]
                h_boxes, h_ids = h_boxes[keep], h_ids[keep]

        gt_ids = ped["id"].to_numpy(dtype=int)
        gt_boxes = ped[XYWH].to_numpy(dtype=float)
        num_gt += len(gt_ids)
        num_hyp += len(h_ids)
        for hid in h_ids:
            hyp_counts[int(hid)] = hyp_counts.get(int(hid), 0) + 1

        iou = iou_matrix(gt_boxes, h_boxes)       # (G, H)

        # IDF1 bookkeeping: every (gt, track) pair that overlaps enough in this frame
        for i, j in np.argwhere(iou >= iou_thr):
            key = (int(gt_ids[i]), int(h_ids[j]))
            overlap[key] = overlap.get(key, 0) + 1

        # CLEAR-MOT step 1: keep last pairings that are still valid
        h_index = {int(hid): j for j, hid in enumerate(h_ids)}
        matches, used_g, used_h = [], set(), set()
        for i, gid in enumerate(gt_ids):
            prev = last_hyp.get(int(gid))
            j = h_index.get(prev) if prev is not None else None
            if j is not None and iou[i, j] >= iou_thr:
                matches.append((i, j)); used_g.add(i); used_h.add(j)
        # step 2: Hungarian on whatever is left
        rem_g = [i for i in range(len(gt_ids)) if i not in used_g]
        rem_h = [j for j in range(len(h_ids)) if j not in used_h]
        if rem_g and rem_h:
            for a, b in _match(iou[np.ix_(rem_g, rem_h)], iou_thr):
                matches.append((rem_g[a], rem_h[b]))

        tp += len(matches)
        fn += len(gt_ids) - len(matches)
        fp += len(h_ids) - len(matches)

        matched_g = set()
        for i, j in matches:
            gid, hid = int(gt_ids[i]), int(h_ids[j])
            matched_g.add(i)
            if gid in last_hyp and last_hyp[gid] != hid:
                idsw += 1
                events.append(dict(frame=f, gt_id=gid, prev_track=last_hyp[gid], new_track=hid,
                                   last_matched_frame=last_frame[gid],
                                   new_track_prev_gt=hyp_last_gt.get(hid, -1)))
            last_hyp[gid] = hid
            last_frame[gid] = f
            hyp_last_gt[hid] = gid

        for i, gid in enumerate(gt_ids):
            gid = int(gid)
            gt_frames[gid] = gt_frames.get(gid, 0) + 1
            if i in matched_g:
                gt_tracked[gid] = gt_tracked.get(gid, 0) + 1
                if ever_tracked.get(gid) and last_status.get(gid) is False:
                    frag += 1
                ever_tracked[gid] = True
                last_status[gid] = True
            else:
                last_status[gid] = False

    # IDF1: best global one-to-one assignment of GT identities to track IDs
    g_list, h_list = sorted(gt_frames), sorted(hyp_counts)
    idtp = 0
    if g_list and h_list:
        gi = {gid: k for k, gid in enumerate(g_list)}
        hi = {hid: k for k, hid in enumerate(h_list)}
        M = np.zeros((len(g_list), len(h_list)))
        for (gid, hid), c in overlap.items():
            M[gi[gid], hi[hid]] = c
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())

    ratios = [gt_tracked.get(gid, 0) / gt_frames[gid] for gid in g_list]
    metrics = finalize(dict(
        seq=seq_name or getattr(getattr(seq, "info", None), "name", "seq"),
        num_gt=num_gt, num_hyp=num_hyp, TP=tp, FP=fp, FN=fn, IDSW=idsw, Frag=frag,
        MT=sum(r > 0.8 for r in ratios), ML=sum(r < 0.2 for r in ratios),   # same cut-offs as TrackEval
        num_gt_ids=len(g_list), num_hyp_ids=len(h_list), IDTP=idtp,
    ))
    events_df = pd.DataFrame(events, columns=["frame", "gt_id", "prev_track", "new_track",
                                          "last_matched_frame", "new_track_prev_gt"])
    return metrics, events_df
