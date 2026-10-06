"""Step 5c: how good are the embeddings? (Use this BEFORE tuning --max-cosine-dist.)

Each detection is labelled with the ground-truth person it overlaps (IoU >= 0.5). We then compare
cosine distances for
   SAME person (different frames, up to --max-gap frames apart)   -> should be SMALL
   DIFFERENT people (same neighbourhood in time)                  -> should be LARGE
and report how well the two groups separate (AUC: 1.0 = perfect, 0.5 = useless), plus percentiles
to help choose max_cosine_distance (a value near the 90-95th percentile of the SAME-person
distances keeps most true matches while rejecting most wrong ones).

Usage:
    python scripts/embedding_stats.py --seq-dir data/MOT17/train/MOT17-04-FRCNN ... --emb-dir embeddings/resnet
Output: printed summary + results/step5_embedding_stats_<name>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.boxes import iou_matrix  # noqa: E402
from src.data import MOTSequence  # noqa: E402


def label_detections(seq, frame, box, iou_thr=0.5):
    """Give each detection the GT id it overlaps (or -1)."""
    gt = seq.load_gt()
    groups = {int(f): g for f, g in gt.groupby("frame")}
    labels = np.full(len(frame), -1, dtype=int)
    for f in np.unique(frame):
        g = groups.get(int(f))
        if g is None or g.empty:
            continue
        idx = np.where(frame == f)[0]
        iou = iou_matrix(box[idx], g[["x", "y", "w", "h"]].to_numpy())
        cost = np.where(iou >= iou_thr, 1 - iou, 1e6)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if cost[i, j] < 1e6:
                labels[idx[i]] = int(g["id"].to_numpy()[j])
    return labels


def sample_pairs(frame, labels, max_gap, n_pairs, rng):
    valid = np.where(labels >= 0)[0]
    pos, neg = [], []
    tries = 0
    while (len(pos) < n_pairs or len(neg) < n_pairs) and tries < n_pairs * 60 and len(valid) > 1:
        tries += 1
        i = rng.choice(valid)
        near = valid[(np.abs(frame[valid] - frame[i]) <= max_gap) & (frame[valid] != frame[i])]
        if len(near) == 0:
            continue
        j = rng.choice(near)
        if labels[i] == labels[j]:
            if len(pos) < n_pairs:
                pos.append((i, j))
        elif len(neg) < n_pairs:
            neg.append((i, j))
    return np.array(pos).reshape(-1, 2), np.array(neg).reshape(-1, 2)


def auc(same, diff):
    """P(random same-person distance < random different-person distance)."""
    if len(same) == 0 or len(diff) == 0:
        return float("nan")
    s = np.sort(diff)
    less = np.searchsorted(s, same, side="left")
    ties = np.searchsorted(s, same, side="right") - less
    return float(((len(s) - np.searchsorted(s, same, side="right")) + 0.5 * ties).sum() / (len(same) * len(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--max-gap", type=int, default=30, help="max frames between the two crops of a pair")
    ap.add_argument("--pairs", type=int, default=20000, help="pairs per sequence and per group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    same_all, diff_all = [], []
    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        z = np.load(Path(args.emb_dir) / f"{seq.info.name}.npz")
        frame, box, feat = z["frame"], z["box"], z["feat"].astype(np.float32)
        feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
        labels = label_detections(seq, frame, box)
        pos, neg = sample_pairs(frame, labels, args.max_gap, args.pairs, rng)
        d_same = 1 - np.sum(feat[pos[:, 0]] * feat[pos[:, 1]], axis=1) if len(pos) else np.array([])
        d_diff = 1 - np.sum(feat[neg[:, 0]] * feat[neg[:, 1]], axis=1) if len(neg) else np.array([])
        print(f"{seq.info.name}: {(labels >= 0).sum()}/{len(labels)} detections matched to GT, "
              f"{len(d_same)} same-person pairs, {len(d_diff)} different-person pairs, AUC {auc(d_same, d_diff):.3f}")
        same_all.append(d_same)
        diff_all.append(d_diff)

    same, diff = np.concatenate(same_all), np.concatenate(diff_all)
    pct = lambda x, q: float(np.percentile(x, q))
    print(f"\nOVERALL  AUC = {auc(same, diff):.3f}   (1.0 perfect, 0.5 useless)")
    print(f"  same person      : median {pct(same, 50):.3f}   90th pct {pct(same, 90):.3f}   95th pct {pct(same, 95):.3f}")
    print(f"  different people : median {pct(diff, 50):.3f}   5th pct {pct(diff, 5):.3f}   10th pct {pct(diff, 10):.3f}")
    print(f"  -> try --max-cosine-dist around {pct(same, 90):.2f} to {pct(same, 95):.2f}")

    name = Path(args.emb_dir).name
    out = Path(f"results/step5_embedding_stats_{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, max(float(np.max(diff)), float(np.max(same)), 0.5), 60)
    plt.hist(same, bins=bins, alpha=0.6, density=True, label="same person")
    plt.hist(diff, bins=bins, alpha=0.6, density=True, label="different people")
    plt.xlabel("cosine distance"); plt.ylabel("density"); plt.legend()
    plt.title(f"Embedding separation ({name}): AUC {auc(same, diff):.3f}")
    plt.tight_layout(); plt.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
