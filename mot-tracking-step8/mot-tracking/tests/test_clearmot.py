"""Check the evaluator against hand-computed answers.

Run with:  python tests/test_clearmot.py     (or: python -m pytest tests/ -q)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clearmot import TRACK_COLUMNS, evaluate_tracks  # noqa: E402

GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


class FakeSeq:
    def __init__(self, rows):
        self.gt = pd.DataFrame(rows, columns=GT_COLS)

    def load_gt(self, pedestrians_only=True):
        return self.gt


def gt_rows(n=100):
    rows = []
    for f in range(1, n + 1):
        rows.append((f, 1, 100 + f, 100, 50, 120, 1, 1, 1.0))    # person A (left, walks right)
        rows.append((f, 2, 400, 300, 50, 120, 1, 1, 1.0))        # person B (stands still)
    return rows


def hyp_from(gt_rows_, id_map, drop=lambda f, gid: False, extra=()):
    rows = []
    for f, gid, x, y, w, h, *_ in gt_rows_:
        if drop(f, gid):
            continue
        rows.append((f, id_map(f, gid), x, y, w, h, 0.9, -1, -1, -1))
    rows.extend(extra)
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def test_perfect_tracking():
    g = gt_rows()
    m, ev = evaluate_tracks(FakeSeq(g), hyp_from(g, lambda f, gid: gid))
    assert (m["FP"], m["FN"], m["IDSW"], m["Frag"]) == (0, 0, 0, 0)
    assert m["MOTA"] == 1.0 and m["IDF1"] == 1.0 and len(ev) == 0


def test_one_id_switch():
    g = gt_rows()
    # person A is tracked as ID 11 for frames 1-50 and as ID 13 for frames 51-100
    idm = lambda f, gid: (11 if f <= 50 else 13) if gid == 1 else 12
    m, ev = evaluate_tracks(FakeSeq(g), hyp_from(g, idm))
    assert m["IDSW"] == 1 and m["FP"] == 0 and m["FN"] == 0
    assert abs(m["MOTA"] - (1 - 1 / 200)) < 1e-9          # one error in 200 GT boxes
    assert abs(m["IDF1"] - 0.75) < 1e-9                   # IDTP = 150 of 200 GT and 200 hyp boxes
    assert list(ev.iloc[0][["frame", "gt_id", "prev_track", "new_track"]]) == [51, 1, 11, 13]
    assert ev.iloc[0]["last_matched_frame"] == 50 and ev.iloc[0]["new_track_prev_gt"] == -1


def test_gap_gives_misses_and_one_fragmentation_but_no_switch():
    g = gt_rows()
    # person A disappears for frames 40-49, then comes back with the SAME ID
    m, _ = evaluate_tracks(FakeSeq(g), hyp_from(g, lambda f, gid: gid, drop=lambda f, gid: gid == 1 and 40 <= f < 50))
    assert m["FN"] == 10 and m["IDSW"] == 0 and m["Frag"] == 1 and m["FP"] == 0


def test_new_id_after_gap_counts_as_switch():
    g = gt_rows()
    idm = lambda f, gid: (21 if f < 40 else 23) if gid == 1 else 22
    m, _ = evaluate_tracks(FakeSeq(g), hyp_from(g, idm, drop=lambda f, gid: gid == 1 and 40 <= f < 50))
    assert m["IDSW"] == 1 and m["FN"] == 10


def test_false_positive_counted_and_ignore_region_not():
    g = gt_rows(10) + [(f, 9, 500, 50, 40, 100, 1, 8, 1.0) for f in range(1, 11)]   # distractor GT
    extra = []
    for f in range(1, 11):
        extra.append((f, 90, 500, 50, 40, 100, 0.8, -1, -1, -1))                    # on the distractor
        if f <= 4:
            extra.append((f, 91, 20, 350, 40, 90, 0.8, -1, -1, -1))                 # genuine FP
    base = [r for r in g if r[1] in (1, 2)]
    m, _ = evaluate_tracks(FakeSeq(g), hyp_from(base, lambda f, gid: gid, extra=extra))
    assert m["FP"] == 4 and m["FN"] == 0 and m["IDSW"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
