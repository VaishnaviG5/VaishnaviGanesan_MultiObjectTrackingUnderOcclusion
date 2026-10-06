"""Step 1 check: load a MOT17 sequence and draw its ground truth on one frame.

Usage:
    python scripts/show_gt.py --seq-dir data/MOT17/train/MOT17-04-FRCNN --frame 1
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import MOTSequence  # noqa: E402
from src.viz import draw_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="path to one MOT17 sequence folder")
    ap.add_argument("--frame", type=int, default=1, help="1-indexed frame number")
    ap.add_argument("--out", default="results/step1_gt_frame.png")
    args = ap.parse_args()

    seq = MOTSequence(args.seq_dir)
    print(seq.summary())

    img = seq.read_frame(args.frame)
    gt = seq.gt_for_frame(args.frame)
    print(f"Frame {args.frame}: {len(gt)} pedestrians in ground truth")

    vis = draw_boxes(img, gt[["x", "y", "w", "h"]].to_numpy(), ids=gt["id"].to_numpy())
    cv2.putText(vis, f"{seq.info.name}  frame {args.frame}  GT={len(gt)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
