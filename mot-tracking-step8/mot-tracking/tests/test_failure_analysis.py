"""Tests for the failure analysis on a synthetic scene where we know why each ID switch happens.

Scene (200 frames):
  person 1  walks; person 2 passes in front of them in frames 60-69 (occluded, vis 0.2);
            the tracker loses person 1 and re-finds them under a NEW id         -> occlusion
  person 3  moves slowly, then jumps 40 px/frame in frames 120-124;
            the tracker loses them and starts a new id                          -> fast motion
  persons 4,5  stand side by side, look alike; the tracker swaps their ids at 150 -> similar appearance
  persons 6,7  stand side by side, look different; ids swapped at 170              -> other (no cause)

Run with:  python tests/test_failure_analysis.py     (or: python -m pytest tests/ -q)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clearmot import TRACK_COLUMNS, evaluate_tracks  # noqa: E402
from src.failure_analysis import EmbeddingIndex, GTFeatures, annotate_events, summarize  # noqa: E402

N, W, H = 200, 50, 130
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


class FakeSeq:
    def __init__(self, gt):
        self.gt = pd.DataFrame(gt, columns=GT_COLS)

    def load_gt(self, pedestrians_only=True):
        return self.gt


def build_scene():
    gt, hyp = [], []
    for f in range(1, N + 1):
        pos = {
            1: (100 + f, 100, 0.2 if 60 <= f < 70 else 1.0),
            2: ((100 + f + 8, 105, 0.9) if 60 <= f < 70 else (800, 300, 1.0)),
            3: (300 + f + (40 * (f - 119) - (f - 119) if 120 <= f <= 124 else 0) + (200 - 5 if f > 124 else 0), 250, 1.0),
            4: (100 + 0.5 * f, 420, 1.0), 5: (160 + 0.5 * f, 420, 1.0),
            6: (600 + 0.5 * f, 100, 1.0), 7: (660 + 0.5 * f, 100, 1.0),
        }
        for pid, (x, y, vis) in pos.items():
            gt.append((f, pid, x, y, W, H, 1, 1, vis))
        track_id = {1: 11 if f < 60 else 21, 2: 12, 3: 13 if f < 120 else 23,
                    4: 14 if f < 150 else 15, 5: 15 if f < 150 else 14,
                    6: 16 if f < 170 else 17, 7: 17 if f < 170 else 16}
        lost = {1: 60 <= f < 70, 3: 120 <= f < 125}
        for pid, (x, y, vis) in pos.items():
            if not lost.get(pid, False):
                hyp.append((f, track_id[pid], x, y, W, H, 0.9, -1, -1, -1))
    return FakeSeq(gt), pd.DataFrame(hyp, columns=TRACK_COLUMNS)


def build_embeddings(seq):
    rng = np.random.default_rng(0)
    base = {p: rng.normal(size=16) for p in range(1, 8)}
    base[5] = base[4] + rng.normal(0, 0.02, 16)          # look-alikes
    base[7] = -base[6] + rng.normal(0, 0.02, 16)         # very different
    rows = seq.gt
    feats = np.stack([base[int(p)] + rng.normal(0, 0.02, 16) for p in rows["id"]])
    return EmbeddingIndex(rows["frame"].to_numpy(), rows[["x", "y", "w", "h"]].to_numpy(),
                          feats, rows["id"].to_numpy())


def analyse(sim_pct=10):
    seq, hyp = build_scene()
    metrics, events = evaluate_tracks(seq, hyp)
    feats = GTFeatures(seq.load_gt(False), vis_thr=0.5, occ_iou=0.3, fast_pct=90)
    emb = build_embeddings(seq)
    thr = float(np.percentile(emb.baseline(), sim_pct))
    ann = annotate_events(events, feats, pre=5, emb=emb, sim_thr=thr)
    return metrics, ann, feats, emb


def test_six_switches_found():
    metrics, ann, *_ = analyse()
    assert metrics["IDSW"] == 6 and len(ann) == 6
    assert sorted(ann["frame"]) == [70, 125, 150, 150, 170, 170]


def test_mechanisms():
    _, ann, *_ = analyse()
    by = {(r.gt_id): r for r in ann.itertuples()}
    assert by[1].mechanism == "respawn_gap" and by[1].gap == 10
    assert by[3].mechanism == "respawn_gap" and by[3].gap == 5
    for g, other in ((4, 5), (5, 4), (6, 7), (7, 6)):
        assert by[g].mechanism == "swap" and by[g].other_gt == other and by[g].gap == 0


def test_primary_causes():
    _, ann, *_ = analyse()
    by = {r.gt_id: r.primary for r in ann.itertuples()}
    assert by[1] == "occlusion"
    assert by[3] == "fast_motion"
    assert by[4] == by[5] == "similar_appearance"
    assert by[6] == by[7] == "other"


def test_flags_are_not_exclusive_and_base_rates_sane():
    _, ann, feats, _ = analyse()
    assert ann.loc[ann.gt_id == 1, "occlusion"].item() and not ann.loc[ann.gt_id == 1, "fast_motion"].item()
    n, occ, fast = feats.base_counts(window_len=16)
    assert 0 < occ / n < 0.2 and 0 < fast / n < 0.2          # rare contexts in this scene
    s = summarize(ann, occ / n, fast / n)
    assert s["primary"].set_index("primary_cause")["count"].to_dict() == {
        "occlusion": 1, "similar_appearance": 2, "fast_motion": 1, "other": 2}
    lift = s["lift"].set_index("context")["lift"]
    assert lift["occlusion"] > 1.0                               # occlusion is over-represented among switches


def test_without_embeddings_similarity_is_skipped():
    seq, hyp = build_scene()
    _, events = evaluate_tracks(seq, hyp)
    feats = GTFeatures(seq.load_gt(False))
    ann = annotate_events(events, feats)
    assert not ann["similar_appearance"].any() and (ann["primary"] != "similar_appearance").all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
