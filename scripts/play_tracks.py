"""Interactive video player for visual tracking playback.

Opens an OpenCV window showing tracking boxes, IDs, and motion trails in real-time.

Controls:
    [Space] : Pause / Resume
    [q] / [Esc] : Quit
    [d] / [Right Arrow] : Step 1 frame forward when paused

Usage:
    python scripts/play_tracks.py --seq-dir data/MOT17/train/MOT17-02-FRCNN --track-dir results/trackers/sort
    python scripts/play_tracks.py --seq-dir data/MOT17/train/MOT17-04-FRCNN --track-dir results/trackers/deepsort_realtime
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clearmot import load_tracks
from src.data import MOTSequence
from src.render import TrailRenderer, render_frame


def rows_by_frame(df):
    if df.empty:
        return {}
    return {
        int(f): g[["x", "y", "w", "h", "id", "conf"]].to_numpy(dtype=float)
        for f, g in df.groupby("frame")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", default="data/MOT17/train/MOT17-02-FRCNN")
    ap.add_argument("--track-dir", default="results/trackers/sort")
    ap.add_argument("--trail", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    seq = MOTSequence(args.seq_dir)
    track_path = Path(args.track_dir) / f"{seq.info.name}.txt"
    if not track_path.exists():
        print(f"Error: Tracking results not found at {track_path}")
        print("Run the tracker first: python scripts/run_sort.py --seq-dir " + str(args.seq_dir))
        return

    tracks_df = load_tracks(track_path)
    track_rows = rows_by_frame(tracks_df)

    renderer = TrailRenderer(trail=args.trail, point="feet")
    delay = int(1000 / args.fps)

    window_name = f"MOT Tracking Playback - {seq.info.name} ({Path(args.track_dir).name})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 60)
    print(f"Playing tracking output for: {seq.info.name}")
    print(f"Tracker: {Path(args.track_dir).name}")
    print("Controls: [Space] Pause/Resume | [q]/[Esc] Quit")
    print("=" * 60 + "\n")

    paused = False
    f = 1

    while f <= seq.info.length:
        img = seq.read_frame(f)
        current_dets = track_rows.get(f, np.zeros((0, 6)))

        if not paused:
            renderer.update(f, current_dets)

        vis = render_frame(
            img,
            current_dets,
            renderer,
            f,
            label=f"{seq.info.name} | {Path(args.track_dir).name}",
            scale=args.scale,
        )

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in [ord("q"), 27]:  # 'q' or Esc
            break
        elif key == ord(" "):  # Space
            paused = not paused
        elif key in [ord("d"), 83]:  # 'd' or right arrow
            f += 1
            continue

        if not paused:
            f += 1

    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == "__main__":
    main()
