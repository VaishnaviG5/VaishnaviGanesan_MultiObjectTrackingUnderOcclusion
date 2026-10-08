"""Sanity tests for the SORT tracker on synthetic data.

Run with either:
    python tests/test_sort.py
    python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kalman_box import KalmanBoxTracker  # noqa: E402
from src.sort import Sort  # noqa: E402

RNG = np.random.default_rng(0)


def person(x, y, w=50, h=120, noise=1.0, conf=0.9):
    j = RNG.normal(0, noise, 4)
    return [x + j[0], y + j[1], w + j[2], h + j[3], conf]


def run(frames_dets, **kw):
    """frames_dets: list (per frame) of lists of [x, y, w, h, conf]. Returns {frame: array}."""
    sort = Sort(**kw)
    return {i + 1: sort.update(np.array(d).reshape(-1, 5)) for i, d in enumerate(frames_dets)}


def ids_in(res, frames):
    return {int(r[4]) for f in frames for r in res[f]}


def test_kalman_learns_velocity():
    kf = KalmanBoxTracker([100, 100, 50, 100], 1)
    for k in range(1, 15):                       # person moving +5 px/frame in x
        kf.predict()
        kf.update([100 + 5 * k, 100, 50, 100])
    pred = kf.predict()
    assert abs(pred[0] - (100 + 5 * 15)) < 3, f"predicted x={pred[0]:.1f}, expected ~175"


def test_single_person_keeps_one_id():
    frames = [[person(100 + 4 * f, 200)] for f in range(60)]
    res = run(frames)
    assert len(ids_in(res, range(1, 61))) == 1


def test_short_occlusion_keeps_same_id():
    # one person walks right; no detections for 8 frames (occluded) in the middle
    frames = []
    for f in range(60):
        frames.append([] if 25 <= f < 33 else [person(100 + 4 * f, 200)])
    res = run(frames, max_age=30)
    assert len(ids_in(res, range(1, 61))) == 1, "ID should survive an 8-frame gap"
    assert all(len(res[f]) == 0 for f in range(27, 33)), "no output while occluded (plain SORT)"


def test_long_gap_creates_new_id():
    frames = []
    for f in range(100):
        frames.append([] if 20 <= f < 70 else [person(100 + 2 * f, 200)])
    res = run(frames, max_age=30)
    before, after = ids_in(res, range(1, 21)), ids_in(res, range(75, 101))
    assert before and after and before.isdisjoint(after), "gap (50) > max_age (30): new ID expected"


def test_two_people_passing_keep_ids():
    # A walks right, B walks left, at different heights so boxes overlap only partly
    frames = []
    for f in range(80):
        frames.append([person(50 + 6 * f, 150), person(530 - 6 * f, 190)])
    res = run(frames)
    a = {int(r[4]) for f in range(1, 10) for r in res[f] if r[0] < 200}   # left-hand person early on
    a_late = {int(r[4]) for f in range(70, 81) for r in res[f] if r[0] > 400}  # same person at the right late
    assert len(ids_in(res, range(1, 81))) == 2
    assert a == a_late, f"ID swapped during crossing: early {a}, late {a_late}"


def test_tracks_deleted_after_max_age():
    sort = Sort(max_age=5)
    for f in range(5):
        sort.update(np.array([person(100, 100)]))
    for f in range(10):
        sort.update(np.zeros((0, 5)))
    assert len(sort.trackers) == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
