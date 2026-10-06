"""Step 2 sanity check: how good is the detector ALONE, before any tracking?

For each frame, detections are matched one-to-one to ground-truth pedestrians
(Hungarian matching on IoU, threshold 0.5). We report precision / recall / F1
across several confidence thresholds, plus recall split by GT visibility
(low-visibility people are the occluded ones: this foreshadows tracking trouble).

Simplified version of the official MOT17 rule: detections that match an
"ignore" GT region (static person, person on vehicle, distractor, reflection)
are not counted as false positives.

Usage:
    python scripts/eval_detections.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN data/MOT17/train/MOT17-02-FRCNN \
        --det-dir detections/yolov8m_1280
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.boxes import iou_matrix  # noqa: E402
from src.data import MOTSequence  # noqa: E402
from src.detections import detections_by_frame, load_detections  # noqa: E402

IGNORE_CLASSES = {2, 7, 8, 12}  # person on vehicle, static person, distractor, reflection
VIS_BINS = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.01)]


def match_frame(det_boxes, gt_boxes, iou_thr):
    """One-to-one matching. Returns (matched_det_idx, matched_gt_idx) lists."""
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return [], []
    iou = iou_matrix(det_boxes, gt_boxes)
    cost = np.where(iou >= iou_thr, 1.0 - iou, 1e6)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < 1e6
    return list(r[keep]), list(c[keep])


def evaluate_sequence(seq, dets, thr, iou_thr):
    gt_all = seq.load_gt(pedestrians_only=False)
    gt_groups = {int(f): g for f, g in gt_all.groupby("frame")}
    empty = gt_all.iloc[0:0]

    tp = fp = fn = 0
    vis_total = np.zeros(len(VIS_BINS), dtype=int)
    vis_found = np.zeros(len(VIS_BINS), dtype=int)

    for frame in range(1, seq.info.length + 1):
        g = gt_groups.get(frame, empty)
        ped = g[(g["cls"] == 1) & (g["conf"] == 1)]
        ign = g[g["cls"].isin(IGNORE_CLASSES)]

        d = dets.get(frame, np.zeros((0, 5)))
        d = d[d[:, 4] >= thr]

        gt_boxes = ped[["x", "y", "w", "h"]].to_numpy()
        det_boxes = d[:, :4]
        mi, mj = match_frame(det_boxes, gt_boxes, iou_thr)

        tp += len(mi)
        fn += len(gt_boxes) - len(mj)

        unmatched = [i for i in range(len(det_boxes)) if i not in set(mi)]
        if unmatched and len(ign):
            ign_iou = iou_matrix(det_boxes[unmatched], ign[["x", "y", "w", "h"]].to_numpy())
            fp += int((ign_iou.max(axis=1) < iou_thr).sum())
        else:
            fp += len(unmatched)

        vis = ped["vis"].to_numpy()
        found = np.zeros(len(vis), dtype=bool)
        found[mj] = True
        for k, (lo, hi) in enumerate(VIS_BINS):
            in_bin = (vis >= lo) & (vis < hi)
            vis_total[k] += in_bin.sum()
            vis_found[k] += (in_bin & found).sum()

    return tp, fp, fn, vis_total, vis_found


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--det-dir", required=True, help="folder created by detect.py")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.4, 0.5, 0.6])
    ap.add_argument("--report-thr", type=float, default=0.25, help="threshold for the visibility breakdown")
    ap.add_argument("--out", default="results/step2_detector_eval.csv")
    args = ap.parse_args()

    records, vis_records = [], []
    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        dets = detections_by_frame(load_detections(Path(args.det_dir) / f"{seq.info.name}.txt"))
        for thr in args.thresholds:
            tp, fp, fn, vt, vf = evaluate_sequence(seq, dets, thr, args.iou)
            p, r, f1 = prf(tp, fp, fn)
            records.append(dict(seq=seq.info.name, conf_thr=thr, TP=tp, FP=fp, FN=fn,
                                precision=p, recall=r, F1=f1))
            if thr == args.report_thr:
                vis_records.append((seq.info.name, vt, vf))

    df = pd.DataFrame(records)

    # overall rows (sum counts over sequences, then recompute P/R/F1)
    overall = []
    for thr, g in df.groupby("conf_thr"):
        tp, fp, fn = g["TP"].sum(), g["FP"].sum(), g["FN"].sum()
        p, r, f1 = prf(tp, fp, fn)
        overall.append(dict(seq="OVERALL", conf_thr=thr, TP=tp, FP=fp, FN=fn,
                            precision=p, recall=r, F1=f1))
    df = pd.concat([df, pd.DataFrame(overall)], ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nSaved {out}")

    if vis_records:
        vt = sum(v[1] for v in vis_records)
        vf = sum(v[2] for v in vis_records)
        print(f"\nRecall by GT visibility (conf >= {args.report_thr}, IoU >= {args.iou}):")
        for k, (lo, hi) in enumerate(VIS_BINS):
            label = f"visibility {lo:.1f}-{min(hi, 1.0):.1f}"
            rec = vf[k] / max(vt[k], 1)
            print(f"  {label}: recall {rec:.3f}  ({vf[k]}/{vt[k]} GT boxes)")


if __name__ == "__main__":
    main()
