"""Step 5b: run the DeepSORT-style tracker on cached detections + embeddings.

Usage:
    python scripts/run_deepsort.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN ... \
        --emb-dir embeddings/resnet --conf-thr 0.4 \
        --max-age 30 --n-init 3 --max-cosine-dist 0.2 --max-iou-dist 0.7

Output:
    results/trackers/<out-name>/<sequence>.txt   (MOT format, same as SORT)
    results/trackers/<out-name>/config.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sort import write_results  # noqa: E402
from src.data import MOTSequence  # noqa: E402
from src.deepsort import DeepSort  # noqa: E402
from src.viz import draw_boxes  # noqa: E402


def load_embeddings(path: Path, min_conf: float):
    """-> {frame: (dets (n,5) x,y,w,h,conf; feats (n,D) unit length)}"""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/extract_embeddings.py first.")
    z = np.load(path)
    frame, box, conf, feat = z["frame"], z["box"], z["conf"], z["feat"].astype(np.float32)
    keep = conf >= min_conf
    frame, box, conf, feat = frame[keep], box[keep], conf[keep], feat[keep]
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    out = {}
    for f in np.unique(frame):
        m = frame == f
        out[int(f)] = (np.column_stack([box[m], conf[m]]).astype(float), feat[m].astype(float))
    return out


def run_sequence(seq, emb_dir, conf_thr, **tracker_kw):
    data = load_embeddings(Path(emb_dir) / f"{seq.info.name}.npz", conf_thr)
    tracker = DeepSort(**tracker_kw)
    results, t0 = {}, time.time()
    for frame in range(1, seq.info.length + 1):
        d, f = data.get(frame, (np.zeros((0, 5)), np.zeros((0, 1))))
        results[frame] = tracker.update(d, f)           # call on EVERY frame so lost tracks age
    return results, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--emb-dir", required=True, help="folder created by extract_embeddings.py")
    ap.add_argument("--out-dir", default=None, help="default: results/trackers/deepsort_<emb folder name>")
    ap.add_argument("--conf-thr", type=float, default=0.4)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--max-cosine-dist", type=float, default=0.2)
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument("--budget", type=int, default=100, help="appearance gallery size per track")
    ap.add_argument("--no-gating", action="store_true", help="disable Mahalanobis gating (ablation)")
    ap.add_argument("--preview-frames", type=int, nargs="*", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"results/trackers/deepsort_{Path(args.emb_dir).name}")
    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        results, elapsed = run_sequence(
            seq, args.emb_dir, args.conf_thr, max_age=args.max_age, n_init=args.n_init,
            max_cosine_distance=args.max_cosine_dist, max_iou_distance=args.max_iou_dist,
            nn_budget=args.budget, use_gating=not args.no_gating)
        write_results(results, out_dir / f"{seq.info.name}.txt")

        ids = {int(r[4]) for rows in results.values() for r in rows}
        n_boxes = sum(len(r) for r in results.values())
        gt = seq.load_gt()
        print(f"{seq.info.name}: {len(ids)} track IDs (GT has {gt['id'].nunique()} people), "
              f"{n_boxes / seq.info.length:.1f} tracked boxes/frame (GT {len(gt) / seq.info.length:.1f}), "
              f"tracker speed {seq.info.length / elapsed:.0f} FPS (embeddings precomputed)")

        for f in args.preview_frames:
            rows = results.get(f, np.zeros((0, 6)))
            vis = draw_boxes(seq.read_frame(f), rows[:, :4], ids=rows[:, 4])
            cv2.putText(vis, f"DeepSORT  {seq.info.name}  frame {f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            p = Path(f"results/step5_preview_{seq.info.name}_{f}.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), vis)
            print(f"  saved {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
