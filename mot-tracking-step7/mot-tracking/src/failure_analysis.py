"""Why do ID switches happen? Classify every ID switch by mechanism and by likely cause.

Mechanism (from the evaluator's event log)
  swap            the new track ID was previously following a DIFFERENT person -> the ID moved between people
  revert          the new track ID was previously following the SAME person (ID flip-flop A -> B -> A)
  respawn_gap     brand-new ID after the person was unmatched for >= 1 frame (tracker lost them, re-found as new)
  respawn_nogap   brand-new ID although the person was matched in the previous frame

Cause flags (evaluated over the window from `pre` frames before the person was last tracked until the switch)
  occlusion           GT visibility < vis_thr, or box overlaps another person/occluder with IoU >= occ_iou
  fast_motion         box centre moves faster than the `fast_pct` percentile of that sequence's speeds
                      (speed = centre displacement per frame / box height; includes camera motion)
  similar_appearance  (swaps only, needs embeddings) the two people involved look more alike than
                      `sim_pct`% of neighbouring pairs of people in the same sequences

Primary cause = first flag that applies in `priority` order, else "other" (typically missed
detections or detector flicker with no occlusion / fast motion / look-alike nearby).
Flags are also reported without priority because causes can overlap.

To avoid misleading conclusions, event rates are compared with BASE RATES: how often an arbitrary
GT person-frame is in the same context (e.g. "occluded"). A cause only "explains" switches if it
is over-represented among them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.boxes import iou_matrix

CAUSES = ("occlusion", "similar_appearance", "fast_motion")
OTHER = "other"
OCCLUDER_CLASSES = (2, 7)          # person on vehicle, static person (they hide pedestrians too)
XYWH = ["x", "y", "w", "h"]


class GTFeatures:
    """Per-box context features for every ground-truth pedestrian box of ONE sequence."""

    def __init__(self, gt_all: pd.DataFrame, vis_thr=0.5, occ_iou=0.3, fast_pct=90.0):
        ped = gt_all[(gt_all["cls"] == 1) & (gt_all["conf"] == 1)]
        occl = gt_all[gt_all["cls"].isin(OCCLUDER_CLASSES)]
        rows = pd.concat([ped.assign(is_ped=True), occl.assign(is_ped=False)], ignore_index=True)

        max_iou = np.zeros(len(rows))
        boxes_all = rows[XYWH].to_numpy(dtype=float)
        for _, idx in rows.groupby("frame").indices.items():
            if len(idx) < 2:
                continue
            m = iou_matrix(boxes_all[idx], boxes_all[idx])
            np.fill_diagonal(m, 0.0)                     # a box does not occlude itself
            max_iou[idx] = m.max(axis=1)
        rows["max_iou_other"] = max_iou

        df = rows[rows["is_ped"]].sort_values(["id", "frame"]).reset_index(drop=True)
        cx, cy = df["x"] + df["w"] / 2, df["y"] + df["h"] / 2
        g = df.groupby("id")
        dframe = df["frame"] - g["frame"].shift()
        dist = np.hypot(cx - cx.groupby(df["id"]).shift(), cy - cy.groupby(df["id"]).shift())
        speed = (dist / df["h"].groupby(df["id"]).shift()).where(dframe == 1)
        df["speed"] = speed

        self.fast_thr = float(np.nanpercentile(speed.to_numpy(), fast_pct)) if speed.notna().any() else np.inf
        df["occ"] = (df["vis"] < vis_thr) | (df["max_iou_other"] >= occ_iou)
        df["fast"] = df["speed"] > self.fast_thr
        self.df = df
        self.by_id = {int(i): sub.reset_index(drop=True) for i, sub in df.groupby("id")}

    def window(self, gid: int, start: int, end: int) -> dict:
        """Context of one person over frames [start, end]."""
        sub = self.by_id.get(int(gid))
        if sub is None:
            return dict(occ=False, fast=False, min_vis=np.nan, max_iou=np.nan, max_speed=np.nan)
        w = sub[(sub["frame"] >= start) & (sub["frame"] <= end)]
        if w.empty:
            return dict(occ=False, fast=False, min_vis=np.nan, max_iou=np.nan, max_speed=np.nan)
        return dict(occ=bool(w["occ"].any()), fast=bool(w["fast"].any()), min_vis=float(w["vis"].min()),
                    max_iou=float(w["max_iou_other"].max()),
                    max_speed=float(w["speed"].max()) if w["speed"].notna().any() else np.nan)

    def base_counts(self, window_len: int):
        """(num boxes, boxes whose last `window_len` frames contain occlusion, ... fast motion)."""
        n = len(self.df)
        occ = self.df.groupby("id")["occ"].transform(lambda s: s.astype(float).rolling(window_len, min_periods=1).max())
        fast = self.df.groupby("id")["fast"].transform(lambda s: s.astype(float).rolling(window_len, min_periods=1).max())
        return n, int((occ > 0).sum()), int((fast > 0).sum())


class EmbeddingIndex:
    """Embeddings of detections that were matched to ground-truth identities."""

    def __init__(self, frame, box, feat, labels):
        self.frame, self.box, self.labels = np.asarray(frame), np.asarray(box, float), np.asarray(labels)
        f = np.asarray(feat, dtype=np.float32)
        self.feat = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        self.by_gid = {}
        for gid in np.unique(self.labels[self.labels >= 0]):
            idx = np.where(self.labels == gid)[0]
            order = idx[np.argsort(self.frame[idx], kind="stable")]
            self.by_gid[int(gid)] = (self.frame[order], self.feat[order])

    def mean(self, gid: int, center: int, half: int):
        item = self.by_gid.get(int(gid))
        if item is None:
            return None
        frames, feats = item
        sel = (frames >= center - half) & (frames <= center + half)
        if not sel.any():
            return None
        m = feats[sel].mean(axis=0)
        return m / max(np.linalg.norm(m), 1e-12)

    def distance(self, g: int, o: int, center: int, half: int = 15):
        a, b = self.mean(g, center, half), self.mean(o, center, half)
        return np.nan if a is None or b is None else float(1.0 - a @ b)

    def baseline(self, half: int = 15, k: int = 1500, rng=None, min_pairs: int = 20) -> np.ndarray:
        """Distances between pairs of people that are NEIGHBOURS in some frame (their boxes touch or
        overlap): swaps happen between such people, and overlapping crops share pixels, so a fair
        baseline must come from the same kind of pair. Falls back to any co-visible pair."""
        rng = rng or np.random.default_rng(0)
        touching, anypair = [], []
        for f in np.unique(self.frame):
            idx = np.where((self.frame == f) & (self.labels >= 0))[0]
            if len(idx) < 2:
                continue
            m = iou_matrix(self.box[idx], self.box[idx])
            ii, jj = np.triu_indices(len(idx), k=1)
            for a, b in zip(ii, jj):
                if self.labels[idx[a]] == self.labels[idx[b]]:
                    continue
                item = (int(f), int(self.labels[idx[a]]), int(self.labels[idx[b]]))
                anypair.append(item)
                if m[a, b] > 0:
                    touching.append(item)
        pool = touching if len(touching) >= min_pairs else anypair
        if not pool:
            return np.array([])
        pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        d = [self.distance(pool[i][1], pool[i][2], pool[i][0], half) for i in pick]
        return np.array([x for x in d if not np.isnan(x)])


def annotate_events(events: pd.DataFrame, feats: GTFeatures, pre: int = 5,
                    emb: EmbeddingIndex | None = None, sim_thr: float | None = None, half: int = 15,
                    priority=("occlusion", "similar_appearance", "fast_motion")) -> pd.DataFrame:
    """Add mechanism, context flags and primary cause to the evaluator's ID-switch events."""
    rows = []
    for e in events.itertuples(index=False):
        gap = int(e.frame - e.last_matched_frame - 1)
        if e.new_track_prev_gt == e.gt_id:
            mech, other = "revert", -1
        elif e.new_track_prev_gt >= 0:
            mech, other = "swap", int(e.new_track_prev_gt)
        else:
            mech, other = ("respawn_gap" if gap > 0 else "respawn_nogap"), -1

        w = feats.window(e.gt_id, int(e.last_matched_frame) - pre, int(e.frame))
        sim_dist, similar = np.nan, False
        if mech == "swap" and emb is not None:
            sim_dist = emb.distance(e.gt_id, other, int(e.frame), half)
            similar = bool(not np.isnan(sim_dist) and sim_thr is not None and sim_dist <= sim_thr)

        flags = {"occlusion": w["occ"], "similar_appearance": similar, "fast_motion": w["fast"]}
        primary = next((c for c in priority if flags[c]), OTHER)
        rows.append(dict(
            frame=int(e.frame), gt_id=int(e.gt_id), prev_track=int(e.prev_track), new_track=int(e.new_track),
            last_matched_frame=int(e.last_matched_frame), gap=gap, mechanism=mech, other_gt=other,
            min_vis=w["min_vis"], max_iou_other=w["max_iou"], max_speed=w["max_speed"], sim_dist=sim_dist,
            occlusion=bool(w["occ"]), fast_motion=bool(w["fast"]), similar_appearance=similar, primary=primary))
    return pd.DataFrame(rows)


def summarize(ann: pd.DataFrame, base_occ_rate: float, base_fast_rate: float) -> dict:
    """Tables for the write-up (all computed from the annotated events)."""
    n = max(len(ann), 1)
    mech = ann["mechanism"].value_counts().rename_axis("mechanism").reset_index(name="count")
    mech["percent"] = (100 * mech["count"] / n).round(1)

    primary = (ann["primary"].value_counts().reindex(list(CAUSES) + [OTHER], fill_value=0)
               .rename_axis("primary_cause").reset_index(name="count"))
    primary["percent"] = (100 * primary["count"] / n).round(1)

    flags = pd.DataFrame([dict(flag=c, count=int(ann[c].sum()), percent=round(100 * ann[c].sum() / n, 1))
                          for c in CAUSES])
    lift = pd.DataFrame([
        dict(context="occlusion", switches_pct=round(100 * ann["occlusion"].mean(), 1) if len(ann) else 0.0,
             all_gt_boxes_pct=round(100 * base_occ_rate, 1)),
        dict(context="fast_motion", switches_pct=round(100 * ann["fast_motion"].mean(), 1) if len(ann) else 0.0,
             all_gt_boxes_pct=round(100 * base_fast_rate, 1)),
    ])
    lift["lift"] = (lift["switches_pct"] / lift["all_gt_boxes_pct"].replace(0, np.nan)).round(2)
    return dict(mechanism=mech, primary=primary, flags=flags, lift=lift)
