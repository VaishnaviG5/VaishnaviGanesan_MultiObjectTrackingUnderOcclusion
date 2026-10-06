"""Step 4: how long should a lost track stay alive? Sweep SORT's max_age (and optionally
min_hits / iou_thr), evaluate each setting, and plot the trade-off.

Why it matters:
  max_age too SMALL -> a briefly occluded person's track dies; they come back as a NEW ID (ID switch)
  max_age too LARGE -> dead tracks linger and can "steal" a different person who walks by (wrong match),
                       and more ghost tracks are alive at any time

Usage:
    python scripts/ablate_max_age.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN data/MOT17/train/MOT17-02-FRCNN \
        --det-dir detections/yolov8m_1280 --conf-thr 0.4 \
        --max-ages 5 15 30 60

Tip: tune on some sequences and report on others (e.g. tune on 02 + 04, then run again on
the remaining sequences with the chosen value) and say so in the write-up.

Outputs:
    results/step4_ablation.csv   all settings x (each sequence + OVERALL)
    results/step4_ablation.png   IDSW, #IDs created, MOTA, IDF1 versus max_age
"""
import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sort import run_sequence  # noqa: E402
from src.clearmot import aggregate, evaluate_tracks, results_to_df  # noqa: E402
from src.data import MOTSequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--conf-thr", type=float, default=0.4)
    ap.add_argument("--max-ages", type=int, nargs="+", default=[5, 15, 30, 60])
    ap.add_argument("--min-hits", type=int, nargs="+", default=[3])
    ap.add_argument("--iou-thrs", type=float, nargs="+", default=[0.3])
    ap.add_argument("--out-csv", default="results/step4_ablation.csv")
    ap.add_argument("--out-plot", default="results/step4_ablation.png")
    args = ap.parse_args()

    seqs = [MOTSequence(d) for d in args.seq_dir]
    records = []
    for max_age, min_hits, iou_thr in itertools.product(args.max_ages, args.min_hits, args.iou_thrs):
        per_seq = []
        for seq in seqs:
            results, _ = run_sequence(seq, args.det_dir, args.conf_thr, max_age, min_hits, iou_thr)
            m, _ = evaluate_tracks(seq, results_to_df(results), seq_name=seq.info.name)
            per_seq.append(m)
        per_seq.append(aggregate(per_seq))
        for m in per_seq:
            records.append(dict(max_age=max_age, min_hits=min_hits, iou_thr=iou_thr, **m))
        o = per_seq[-1]
        print(f"max_age={max_age:>3} min_hits={min_hits} iou={iou_thr}: "
              f"MOTA {o['MOTA']*100:5.1f}  IDF1 {o['IDF1']*100:5.1f}  IDSW {o['IDSW']:>4}  "
              f"Frag {o['Frag']:>4}  FP {o['FP']:>5}  FN {o['FN']:>6}  IDs created {o['num_hyp_ids']}")

    df = pd.DataFrame(records)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    overall = df[df["seq"] == "OVERALL"]
    panels = [("IDSW", "ID switches (lower is better)"),
              ("num_hyp_ids", "Track IDs created (lower is better)"),
              ("MOTA", "MOTA (higher is better)"),
              ("IDF1", "IDF1 (higher is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for (mh, it), g in overall.groupby(["min_hits", "iou_thr"]):
            g = g.sort_values("max_age")
            ax.plot(g["max_age"], g[col], marker="o", label=f"min_hits={mh}, iou_thr={it}")
        ax.set_title(title)
        ax.set_xlabel("max_age (frames a lost track stays alive)")
        ax.set_xticks(sorted(overall["max_age"].unique()))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=130)
    print(f"\nSaved {args.out_csv} and {args.out_plot}")


if __name__ == "__main__":
    main()
