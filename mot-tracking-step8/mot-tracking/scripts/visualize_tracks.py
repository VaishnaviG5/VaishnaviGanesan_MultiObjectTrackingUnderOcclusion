"""Step 8: visualize tracking - a clip with boxes, IDs and fading trails, plus static trajectory plots.

Usage:
    # 1) which part of the sequence is interesting?
    python scripts/visualize_tracks.py --seq-dir data/MOT17/train/MOT17-04-FRCNN \
        --track-dir results/trackers/deepsort_resnet --suggest-clips

    # 2) render a clip (give several tracker folders for a side-by-side comparison)
    python scripts/visualize_tracks.py --seq-dir data/MOT17/train/MOT17-04-FRCNN \
        --track-dir results/trackers/sort results/trackers/deepsort_resnet \
        --start 300 --length 150 --trail 30 --scale 0.5

Outputs (results/step8/):
    <seq>_f<start>-<end>.mp4        the clip
    <seq>_f<start>-<end>.gif        a smaller looping version (good for the README)
    <seq>_trajectories.png          full trajectories over the background: ground truth next to each tracker

Tip: use a STATIC camera sequence (MOT17-02, -04, -09) for trajectories; with a moving camera the
image-plane paths include the camera motion.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clearmot import evaluate_tracks, load_tracks  # noqa: E402
from src.data import MOTSequence  # noqa: E402
from src.render import (TrailRenderer, busiest_windows, hstack_same_height,  # noqa: E402
                        plot_trajectories, render_frame, trajectories)


def rows_by_frame(df):
    return {int(f): g[["x", "y", "w", "h", "id", "conf"]].to_numpy(dtype=float) for f, g in df.groupby("frame")}


def suggest_clips(seq, track_dirs, length):
    print(f"\nSuggested clips of {length} frames in {seq.info.name}:")
    _, ev = evaluate_tracks(seq, load_tracks(Path(track_dirs[0]) / f"{seq.info.name}.txt"))
    sw = ev.groupby("frame").size().to_dict()
    crowd = seq.load_gt().groupby("frame").size().to_dict()
    for title, per_frame in ((f"most ID switches ({Path(track_dirs[0]).name})", sw), ("most people (crowded)", crowd)):
        wins = busiest_windows({int(k): float(v) for k, v in per_frame.items()}, length, n_frames=seq.info.length)
        print(f"  {title}:")
        for start, score in wins:
            print(f"    --start {start} --length {length}   (score {score:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--track-dir", nargs="+", required=True, help="1 folder, or several for side-by-side")
    ap.add_argument("--out-dir", default="results/step8")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--length", type=int, default=150, help="frames in the clip")
    ap.add_argument("--trail", type=int, default=30, help="how many past frames each trail shows")
    ap.add_argument("--point", choices=["feet", "center"], default="feet", help="which point of a box is trailed")
    ap.add_argument("--scale", type=float, default=0.5, help="resize frames (0.5 = half size)")
    ap.add_argument("--fps", type=float, default=None, help="default: the sequence's frame rate")
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--gif-step", type=int, default=2, help="keep every n-th frame in the gif")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--min-len", type=int, default=15, help="shortest trajectory (frames) drawn in the static plot")
    ap.add_argument("--suggest-clips", action="store_true", help="print interesting windows and exit")
    args = ap.parse_args()

    seq = MOTSequence(args.seq_dir)
    if args.suggest_clips:
        suggest_clips(seq, args.track_dir, args.length)
        return

    start = max(1, args.start)
    end = min(seq.info.length, start + args.length - 1)
    names = [Path(d).name for d in args.track_dir]
    tracks = [load_tracks(Path(d) / f"{seq.info.name}.txt") for d in args.track_dir]
    per_tracker = [rows_by_frame(t) for t in tracks]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{seq.info.name}_f{start}-{end}"
    fps = args.fps or float(seq.info.frame_rate)

    # ---- clip -------------------------------------------------------------
    if not (args.no_video and args.no_gif):
        renderers = [TrailRenderer(args.trail, args.point) for _ in names]
        warm = max(1, start - args.trail)                    # replay a few earlier frames so trails exist at the start
        writer, gif_frames = None, []
        for f in range(warm, end + 1):
            for r, rows in zip(renderers, per_tracker):
                r.update(f, rows.get(f, np.zeros((0, 6))))
            if f < start:
                continue
            img = seq.read_frame(f)
            panels = [render_frame(img, rows.get(f, np.zeros((0, 6))), r, f, name, args.scale)
                      for r, rows, name in zip(renderers, per_tracker, names)]
            frame_out = hstack_same_height(panels)
            if not args.no_video:
                if writer is None:
                    h, w = frame_out.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame_out)
            if not args.no_gif and (f - start) % args.gif_step == 0:
                gw = args.gif_width
                small = cv2.resize(frame_out, (gw, int(frame_out.shape[0] * gw / frame_out.shape[1])),
                                   interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if writer is not None:
            writer.release()
            print(f"saved {out_dir / f'{stem}.mp4'}  ({end - start + 1} frames)")
        if gif_frames:
            gif_frames[0].save(out_dir / f"{stem}.gif", save_all=True, append_images=gif_frames[1:],
                               duration=int(1000 / fps * args.gif_step), loop=0, optimize=True)
            print(f"saved {out_dir / f'{stem}.gif'}  ({len(gif_frames)} frames)")

    # ---- static trajectories ---------------------------------------------
    if not args.no_static:
        gt = seq.load_gt()
        bg = seq.read_frame(start)
        panels = [("Ground truth", gt)] + list(zip(names, tracks))
        fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.6))
        axes = np.atleast_1d(axes)
        for ax, (title, df) in zip(axes, panels):
            plot_trajectories(ax, bg, trajectories(df, args.min_len, args.point), title)
        fig.suptitle(f"{seq.info.name}: trajectories over the whole sequence "
                     f"(circle = start, X = end; more trajectories than ground truth = broken identities)", fontsize=10)
        fig.tight_layout()
        p = out_dir / f"{seq.info.name}_trajectories.png"
        fig.savefig(p, dpi=130)
        print(f"saved {p}")


if __name__ == "__main__":
    main()
