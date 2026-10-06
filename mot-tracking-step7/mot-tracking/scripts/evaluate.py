"""Score tracker output against ground truth (CLEAR-MOT + IDF1), with ID-switch events.

Usage:
    python scripts/evaluate.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN data/MOT17/train/MOT17-02-FRCNN \
        --track-dir results/trackers/sort

Outputs:
    results/eval/<name>_metrics.csv       one row per sequence + OVERALL
    results/eval/<name>_idsw_events.csv   every ID switch (frame, gt_id, prev_track, new_track)
where <name> is the tracker folder's name. The official TrackEval run in Step 6 cross-checks these.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clearmot import aggregate, evaluate_tracks, load_tracks  # noqa: E402
from src.data import MOTSequence  # noqa: E402

SHOW = ["seq", "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN", "MT", "ML", "num_gt_ids", "num_hyp_ids",
        "Precision", "Recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--track-dir", required=True, help="folder with <sequence>.txt tracker results")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    name = Path(args.track_dir).name
    rows, all_events = [], []
    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        path = Path(args.track_dir) / f"{seq.info.name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the tracker on this sequence first.")
        m, ev = evaluate_tracks(seq, load_tracks(path), args.iou)
        rows.append(m)
        ev.insert(0, "seq", seq.info.name)
        all_events.append(ev)

    rows.append(aggregate(rows))
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_metrics.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(out_dir / f"{name}_idsw_events.csv", index=False)

    show = df[SHOW].copy()
    for c in ("MOTA", "IDF1", "Precision", "Recall"):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))
    print(f"\nSaved {out_dir / f'{name}_metrics.csv'} and {out_dir / f'{name}_idsw_events.csv'}")


if __name__ == "__main__":
    main()
