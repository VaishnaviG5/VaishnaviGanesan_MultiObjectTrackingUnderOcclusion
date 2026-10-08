"""Sanity tests for the DeepSORT-style tracker on synthetic data.

Run with:  python tests/test_deepsort.py     (or: python -m pytest tests/ -q)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deepsort import DeepSort  # noqa: E402
from src.sort import Sort  # noqa: E402

RNG = np.random.default_rng(0)
DIM = 16


def feat(k, noise=0.05):
    """A fake embedding for 'person k': mostly axis k, plus a little noise, unit length."""
    v = np.zeros(DIM)
    v[k] = 1.0
    v = v + RNG.normal(0, noise, DIM)
    return v / np.linalg.norm(v)


def det(x, y=200, w=50, h=130, noise=0.7):
    j = RNG.normal(0, noise, 4)
    return [x + j[0], y + j[1], w + j[2], h + j[3], 0.9]


def run_deep(frames, **kw):
    """frames: list of lists of (box5, feature)."""
    tr = DeepSort(**kw)
    out = {}
    for i, fr in enumerate(frames):
        d = np.array([b for b, _ in fr]).reshape(-1, 5)
        f = np.array([e for _, e in fr]).reshape(len(fr), DIM)
        out[i + 1] = tr.update(d, f)
    return out


def ids(res, frames):
    return {int(r[4]) for f in frames for r in res[f]}


def test_single_person_one_id():
    frames = [[(det(100 + 3 * f), feat(0))] for f in range(60)]
    res = run_deep(frames)
    assert len(ids(res, range(1, 61))) == 1


def test_occlusion_keeps_id_and_reports_immediately_after():
    frames = []
    for f in range(70):
        frames.append([] if 25 <= f < 35 else [(det(100 + 3 * f), feat(0))])
    res = run_deep(frames, max_age=30)
    assert len(ids(res, range(1, 71))) == 1, "same ID after a 10-frame gap"
    assert all(len(res[f]) == 0 for f in range(27, 35))
    assert len(res[36]) == 1, "confirmed track is reported right after re-matching (frame 36 = first frame back)"


def test_long_gap_new_id():
    frames = []
    for f in range(120):
        frames.append([] if 20 <= f < 80 else [(det(100 + 2 * f), feat(0))])
    res = run_deep(frames, max_age=30)
    assert ids(res, range(1, 21)).isdisjoint(ids(res, range(85, 121)))


def _swap_scene():
    """Two people stand 20 px apart and swap places at frame 30 (the kind of ambiguity that
    appears when people cross or hide behind each other)."""
    frames = []
    for f in range(60):
        a_x, b_x = (300, 320) if f < 30 else (320, 300)
        frames.append([(det(a_x), feat(0)), (det(b_x), feat(1))])
    return frames


def _id_at(res, frame, x_near):
    rows = res[frame]
    return int(rows[np.argmin(np.abs(rows[:, 0] - x_near))][4])


def test_appearance_prevents_swap_while_sort_swaps():
    frames = _swap_scene()
    deep = run_deep(frames)
    a_before, a_after = _id_at(deep, 10, 300), _id_at(deep, 58, 320)   # person A: x=300 then x=320
    assert a_before == a_after, "DeepSORT should keep A's ID using appearance"

    sort = Sort()
    sres = {i + 1: sort.update(np.array([b for b, _ in fr])) for i, fr in enumerate(frames)}
    s_before, s_after = _id_at(sres, 10, 300), _id_at(sres, 58, 320)
    assert s_before != s_after, "IoU-only SORT is expected to swap here (that is the point of the test)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
