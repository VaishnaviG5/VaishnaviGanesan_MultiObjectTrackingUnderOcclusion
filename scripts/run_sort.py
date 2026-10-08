"""Step 3: run SORT on cached detections and write MOTChallenge-format results.

Usage:
    python scripts/run_sort.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN \
        --det-dir detections/yolov8m_1280 \
        --conf-thr 0.4 --max-age 30 --min-hits 3 --iou-thr 0.3 \
        --preview-frames 1 200 400

Output:
    results/trackers/sort/<sequence>.txt   frame,id,x,y,w,h,conf,-1,-1,-1
    results/trackers/sort/config.json
    results/step3_preview_<sequence>_<frame>.png   (if --preview-frames is given)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import MOTSequence  # noqa: E402
from src.detections import detections_by_frame, load_detections  # noqa: E402
from src.sort import Sort  # noqa: E402
from src.viz import draw_boxes  # noqa: E402


def run_sequence(seq, det_dir, conf_thr, max_age, min_hits, iou_thr):
    dets = detections_by_frame(load_detections(Path(det_dir) / f"{seq.info.name}.txt", min_conf=conf_thr))
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_thr=iou_thr)
    results = {}                                           # frame -> (M, 6) array
    t0 = time.time()
    for frame in range(1, seq.info.length + 1):
        d = dets.get(frame, np.zeros((0, 5)))              # call update on EVERY frame, even empty ones
        results[frame] = tracker.update(d)
    return results, time.time() - t0


def write_results(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame in sorted(results):
            for x, y, w, h, tid, conf in results[frame]:
                f.write(f"{frame},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--det-dir", required=True, help="folder created by detect.py")
    ap.add_argument("--out-dir", default="results/trackers/sort")
    ap.add_argument("--conf-thr", type=float, default=0.4, help="drop detections below this confidence")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--preview-frames", type=int, nargs="*", default=[], help="save annotated frames")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        results, elapsed = run_sequence(seq, args.det_dir, args.conf_thr,
                                        args.max_age, args.min_hits, args.iou_thr)
        write_results(results, out_dir / f"{seq.info.name}.txt")

        all_ids = {int(r[4]) for rows in results.values() for r in rows}
        n_boxes = sum(len(r) for r in results.values())
        gt = seq.load_gt()
        print(f"{seq.info.name}: {len(all_ids)} track IDs (GT has {gt['id'].nunique()} people), "
              f"{n_boxes / seq.info.length:.1f} tracked boxes/frame "
              f"(GT {len(gt) / seq.info.length:.1f}), tracker speed {seq.info.length / elapsed:.0f} FPS")

        for f in args.preview_frames:
            rows = results.get(f, np.zeros((0, 6)))
            vis = draw_boxes(seq.read_frame(f), rows[:, :4], ids=rows[:, 4])
            cv2.putText(vis, f"SORT  {seq.info.name}  frame {f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            p = Path(f"results/step3_preview_{seq.info.name}_{f}.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), vis)
            print(f"  saved {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
