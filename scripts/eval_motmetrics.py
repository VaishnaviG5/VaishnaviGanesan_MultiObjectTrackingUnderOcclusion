"""Score tracker output against ground truth using py-motmetrics.

Computes standard MOTChallenge CLEAR-MOT metrics:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- IDF1 (Identification F1-score)
- IDSW (Identity Switches / num_switches)
- FP (False Positives)
- FN (False Negatives / Misses)
- Frag (Fragmentations)
- MT (Mostly Tracked, >80%)
- ML (Mostly Lost, <20%)
- Precision and Recall

Usage:
    python scripts/eval_motmetrics.py \
        --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
        --track-dir results/trackers/sort
"""
import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Circumvent Windows Application Control DLL block on scipy.io._mio_utils
sys.modules.setdefault("scipy.io", MagicMock())
import motmetrics as mm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.boxes import iou_matrix
from src.clearmot import IGNORE_CLASSES, TRACK_COLUMNS, XYWH, load_tracks
from src.data import MOTSequence


def eval_sequence(seq: MOTSequence, hyp: pd.DataFrame, iou_thr: float = 0.5):
    """Accumulates motmetrics events for one sequence."""
    gt_all = seq.load_gt(pedestrians_only=False)
    gt_groups = {int(f): g for f, g in gt_all.groupby("frame")}
    hyp_groups = {int(f): g for f, g in hyp.groupby("frame")} if len(hyp) else {}
    empty_gt, empty_h = gt_all.iloc[0:0], hyp.iloc[0:0]
    n_frames = int(max(max(gt_groups, default=0), max(hyp_groups, default=0)))

    acc = mm.MOTAccumulator(auto_id=True)

    for f in range(1, n_frames + 1):
        g = gt_groups.get(f, empty_gt)
        ped = g[(g["cls"] == 1) & (g["conf"] == 1)]
        h = hyp_groups.get(f, empty_h)
        h_boxes = h[XYWH].to_numpy(dtype=float)
        h_ids = h["id"].to_numpy(dtype=int)

        # Drop hypotheses overlapping ignore regions
        if len(h) and len(g):
            g_cls = g["cls"].to_numpy()
            g_boxes = g[XYWH].to_numpy(dtype=float)
            ious = iou_matrix(h_boxes, g_boxes)
            # Find best match for each hyp to ignore classes
            for h_idx in range(len(h_boxes)):
                matches_ign = [
                    ious[h_idx, j] >= iou_thr
                    for j in range(len(g_boxes))
                    if g_cls[j] in IGNORE_CLASSES
                ]
                if any(matches_ign):
                    h_boxes[h_idx] = np.nan  # mark to drop

            keep = [i for i, b in enumerate(h_boxes) if not np.isnan(b[0])]
            h_boxes = h_boxes[keep]
            h_ids = h_ids[keep]

        gt_ids = ped["id"].to_numpy(dtype=int)
        gt_boxes = ped[XYWH].to_numpy(dtype=float)

        if len(gt_boxes) == 0 and len(h_boxes) == 0:
            dists = np.empty((0, 0))
        elif len(gt_boxes) == 0:
            dists = np.empty((0, len(h_boxes)))
        elif len(h_boxes) == 0:
            dists = np.empty((len(gt_boxes), 0))
        else:
            ious = iou_matrix(gt_boxes, h_boxes)
            # Distance is 1 - IoU, thresholded at max (1 - iou_thr)
            dists = 1.0 - ious
            dists[ious < iou_thr] = np.nan

        acc.update(gt_ids.tolist(), h_ids.tolist(), dists)

    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--track-dir", required=True, help="folder with <seq>.txt tracking outputs")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    name = Path(args.track_dir).name
    accs = []
    names = []

    for s_path in args.seq_dir:
        seq = MOTSequence(s_path)
        track_path = Path(args.track_dir) / f"{seq.info.name}.txt"
        if not track_path.exists():
            raise FileNotFoundError(f"{track_path} not found. Run the tracker first.")
        hyp = load_tracks(track_path)
        acc = eval_sequence(seq, hyp, args.iou)
        accs.append(acc)
        names.append(seq.info.name)

    mh = mm.metrics.create()
    metric_keys = [
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_fragmentations",
        "mostly_tracked",
        "mostly_lost",
        "precision",
        "recall",
    ]

    summary = mh.compute_many(
        accs,
        names=names,
        metrics=metric_keys,
        generate_overall=True,
    )

    # Format into human-readable table
    display_df = summary.copy()
    display_df.rename(
        columns={
            "num_frames": "Frames",
            "mota": "MOTA(%)",
            "motp": "MOTP",
            "idf1": "IDF1(%)",
            "num_switches": "IDSW",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "num_fragmentations": "Frag",
            "mostly_tracked": "MT",
            "mostly_lost": "ML",
            "precision": "Prec(%)",
            "recall": "Rec(%)",
        },
        inplace=True,
    )

    display_df["MOTA(%)"] = (display_df["MOTA(%)"] * 100).round(1)
    display_df["IDF1(%)"] = (display_df["IDF1(%)"] * 100).round(1)
    display_df["Prec(%)"] = (display_df["Prec(%)"] * 100).round(1)
    display_df["Rec(%)"] = (display_df["Rec(%)"] * 100).round(1)
    display_df["MOTP"] = display_df["MOTP"].round(3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_motmetrics.csv"
    display_df.to_csv(csv_path)

    print("\n=== PY-MOTMETRICS EVALUATION RESULTS ===")
    print(display_df.to_string())
    print(f"\nSaved metrics to {csv_path}\n")


if __name__ == "__main__":
    main()
