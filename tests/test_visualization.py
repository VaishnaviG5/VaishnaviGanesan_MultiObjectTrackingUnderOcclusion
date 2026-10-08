"""Tests for the Step 8 visualization code on a tiny synthetic sequence.

Run with:  python tests/test_visualization.py     (or: python -m pytest tests/ -q)
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.render import TrailRenderer, busiest_windows, trajectories  # noqa: E402

N, W, H = 60, 320, 240


def make_dataset(root: Path):
    seq = root / "data" / "MOT17-93-TEST"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname=MOT17-93-TEST\nimDir=img1\nframeRate=30\nseqLength={N}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n")
    gt, good, bad = [], [], []
    for f in range(1, N + 1):
        cv2.imwrite(str(seq / "img1" / f"{f:06d}.jpg"), np.full((H, W, 3), 100, np.uint8))
        for pid in (1, 2, 3):
            x, y = 20 + 80 * (pid - 1) + 2 * f, 60 + 10 * pid
            gt.append((f, pid, x, y, 30, 80, 1, 1, 1.0))
            good.append((f, pid, x, y, 30, 80, 0.9, -1, -1, -1))
            if pid == 1 and 30 <= f < 35:                      # person 1 lost for 5 frames ...
                continue
            bad.append((f, pid if not (pid == 1 and f >= 35) else 11, x, y, 30, 80, 0.9, -1, -1, -1))  # ... new ID
    pd.DataFrame(gt).to_csv(seq / "gt" / "gt.txt", header=False, index=False)
    for name, rows in (("good", good), ("bad", bad)):
        d = root / "trk" / name
        d.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(d / "MOT17-93-TEST.txt", header=False, index=False)
    return seq, root / "trk" / "good", root / "trk" / "bad"


def test_trail_is_drawn_and_forgets_old_points():
    tr = TrailRenderer(trail=10)
    base = np.full((100, 100, 3), 100, np.uint8)
    for f in range(1, 31):
        tr.update(f, np.array([[10 + f, 20, 10, 30, 7, 0.9]]))
    assert min(p[0] for p in tr.hist[7]) >= 20, "only the last 10 frames are kept"
    out = tr.draw(base, 30)
    assert (out != base).any() and out.shape == base.shape
    tr.update(60, np.zeros((0, 6)))                            # nobody visible for a long time
    assert 7 not in tr.hist


def test_trajectories_filters_short_tracks():
    df = pd.DataFrame([dict(frame=f, id=1, x=f, y=0, w=10, h=20) for f in range(1, 31)]
                      + [dict(frame=f, id=2, x=f, y=0, w=10, h=20) for f in range(1, 5)])
    t = trajectories(df, min_len=15)
    assert list(t) == [1] and t[1].shape == (30, 2) and tuple(t[1][0]) == (6.0, 20.0)   # feet = bottom centre


def test_busiest_windows():
    assert busiest_windows({35: 1.0}, length=20, n_frames=60)[0] == (16, 1.0)
    wins = busiest_windows({10: 5.0, 50: 3.0}, length=10, top=2, n_frames=60)
    assert [w[0] for w in wins] == [1, 41] and [w[1] for w in wins] == [5.0, 3.0]


def test_script_writes_video_gif_and_trajectories():
    import scripts.visualize_tracks as vt
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        seq, good, bad = make_dataset(t)
        out = t / "out"
        old = sys.argv
        sys.argv = ["visualize_tracks.py", "--seq-dir", str(seq), "--track-dir", str(good), str(bad),
                    "--out-dir", str(out), "--start", "20", "--length", "30", "--scale", "0.5", "--trail", "10"]
        try:
            vt.main()
        finally:
            sys.argv = old
        mp4, gif = out / "MOT17-93-TEST_f20-49.mp4", out / "MOT17-93-TEST_f20-49.gif"
        assert mp4.exists() and gif.exists() and (out / "MOT17-93-TEST_trajectories.png").exists()
        cap = cv2.VideoCapture(str(mp4))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * int(W * 0.5), "two trackers side by side"
        cap.release()
        assert Image.open(gif).n_frames == 15                  # every 2nd frame

        sys.argv = ["visualize_tracks.py", "--seq-dir", str(seq), "--track-dir", str(bad),
                    "--length", "20", "--suggest-clips"]
        try:
            vt.main()                                         # must run and exit without writing anything
        finally:
            sys.argv = old


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
