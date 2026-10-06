"""Step 7: failure analysis - where do ID switches happen, and why?

For every ID switch of each tracker this script determines
  * the MECHANISM   swap between two people / re-spawn after losing a person / ...
  * the CONTEXT     occlusion? fast motion? look-alike neighbour?   (see src/failure_analysis.py)
and compares those contexts with how common they are overall (base rates).

Usage:
    python scripts/analyze_failures.py \
        --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN ... \
        --track-dir results/trackers/sort results/trackers/deepsort_resnet \
        --emb-dir embeddings/resnet            # optional: enables the look-alike analysis

Outputs (results/step7/):
    failure_summary.md                 tables for the write-up
    failure_causes.png                 chart
    <tracker>/switch_events_annotated.csv   every switch with its features and cause
    <tracker>/examples/*.png           before/after pictures of example switches
    examples_sheet_<tracker>.png       all examples of a tracker on one page
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.embedding_stats import label_detections  # noqa: E402
from src.clearmot import evaluate_tracks, load_tracks  # noqa: E402
from src.data import MOTSequence  # noqa: E402
from src.failure_analysis import (CAUSES, OTHER, EmbeddingIndex, GTFeatures,  # noqa: E402
                                  annotate_events, summarize)
from src.trackeval_utils import to_markdown  # noqa: E402
from src.viz import draw_boxes  # noqa: E402

PRIMARY_ORDER = list(CAUSES) + [OTHER]


# ------------------------------------------------------------------ example pictures
def _label_bar(width, text, height=34, color=(30, 30, 30)):
    bar = np.full((height, width, 3), color, np.uint8)
    cv2.putText(bar, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def example_image(seq, hyp_by_frame, gt_peds, ev, tracker, panel_h=380, min_w=480):
    """Side-by-side crops: the last frame the person was tracked, and the frame of the switch."""
    f0, f1 = int(ev.last_matched_frame), int(ev.frame)

    def gt_box(gid, f):
        r = gt_peds[(gt_peds["id"] == gid) & (gt_peds["frame"] == f)]
        return r[["x", "y", "w", "h"]].to_numpy()[0] if len(r) else None

    boxes = [b for b in (gt_box(ev.gt_id, f0), gt_box(ev.gt_id, f1)) if b is not None]
    other = gt_box(ev.other_gt, f1) if ev.other_gt >= 0 else None
    if other is not None:
        boxes.append(other)
    if not boxes:
        return None
    arr = np.array(boxes)
    x1, y1 = arr[:, 0].min(), arr[:, 1].min()
    x2, y2 = (arr[:, 0] + arr[:, 2]).max(), (arr[:, 1] + arr[:, 3]).max()
    pad = 0.6 * max(arr[:, 2].max(), arr[:, 3].max() / 2)
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    W, H = seq.info.width, seq.info.height
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))

    panels = []
    for f, tid in ((f0, ev.prev_track), (f1, ev.new_track)):
        img = seq.read_frame(f)
        rows = hyp_by_frame.get(f, np.zeros((0, 6)))
        img = draw_boxes(img, rows[:, :4], ids=rows[:, 4]) if len(rows) else img
        g = gt_box(ev.gt_id, f)
        if g is not None:                                   # target person: thick white outline
            cv2.rectangle(img, (int(g[0]) - 3, int(g[1]) - 3), (int(g[0] + g[2]) + 3, int(g[1] + g[3]) + 3),
                          (255, 255, 255), 2)
        crop = img[y1:y2, x1:x2]
        scale = panel_h / max(crop.shape[0], 1)
        crop = cv2.resize(crop, (max(int(crop.shape[1] * scale), 50), panel_h))
        if crop.shape[1] < min_w:                          # pad narrow crops so the labels stay readable
            pad = np.full((panel_h, min_w, 3), 40, np.uint8)
            off = (min_w - crop.shape[1]) // 2
            pad[:, off:off + crop.shape[1]] = crop
            crop = pad
        panels.append(np.vstack([_label_bar(crop.shape[1], f"frame {f}: person {ev.gt_id} = ID {int(tid)}"),
                                 crop]))
    body = np.hstack(panels)
    head = _label_bar(body.shape[1], f"{tracker} | {seq.info.name} | cause: {ev.primary} | {ev.mechanism}, "
                                     f"gap {ev.gap} fr (white box = the person)", color=(90, 40, 20))
    return np.vstack([head, body])


def pick_examples(ann, per_category):
    chosen = []
    for cat in PRIMARY_ORDER:
        sub = ann[ann["primary"] == cat].sort_values(["seq", "frame"])
        if sub.empty:
            continue
        idx = np.unique(np.linspace(0, len(sub) - 1, num=min(per_category, len(sub))).round().astype(int))
        chosen += [sub.iloc[i] for i in idx]
    return chosen


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--track-dir", nargs="+", required=True)
    ap.add_argument("--emb-dir", default=None, help="enables the similar-appearance analysis")
    ap.add_argument("--out-dir", default="results/step7")
    ap.add_argument("--vis-thr", type=float, default=0.5, help="GT visibility below this = occluded")
    ap.add_argument("--occ-iou", type=float, default=0.3, help="IoU with another person above this = occluded")
    ap.add_argument("--fast-pct", type=float, default=90.0, help="'fast' = above this percentile of speeds")
    ap.add_argument("--sim-pct", type=float, default=10.0,
                    help="'similar' = closer than this percentile of neighbouring-pair distances")
    ap.add_argument("--pre", type=int, default=5, help="frames before the person was last tracked to include")
    ap.add_argument("--priority", nargs=3, default=list(CAUSES), choices=list(CAUSES))
    ap.add_argument("--examples", type=int, default=2, help="example pictures per cause and tracker")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs = [MOTSequence(d) for d in args.seq_dir]
    feats = {s.info.name: GTFeatures(s.load_gt(False), args.vis_thr, args.occ_iou, args.fast_pct) for s in seqs}
    for s in seqs:
        print(f"{s.info.name}: 'fast' = more than {feats[s.info.name].fast_thr:.4f} box-heights per frame")

    emb, sim_thr = {}, None
    if args.emb_dir:
        base = []
        for s in seqs:
            z = np.load(Path(args.emb_dir) / f"{s.info.name}.npz")
            frame, box = z["frame"], z["box"]
            emb[s.info.name] = EmbeddingIndex(frame, box, z["feat"].astype(np.float32),
                                              label_detections(s, frame, box))
            base.append(emb[s.info.name].baseline())
        base = np.concatenate(base)
        if len(base):
            sim_thr = float(np.percentile(base, args.sim_pct))
            print(f"look-alike threshold: cosine distance <= {sim_thr:.3f} "
                  f"(the closest {args.sim_pct:.0f}% of {len(base)} neighbouring pairs)")
    else:
        print("No --emb-dir: the similar-appearance analysis is skipped.")

    summaries, md = {}, ["# Failure analysis of ID switches\n"]
    md.append(f"Settings: occluded = GT visibility < {args.vis_thr} or IoU with another person >= {args.occ_iou}; "
              f"fast = above the {args.fast_pct:.0f}th percentile of per-sequence speed; "
              f"similar = closest {args.sim_pct:.0f}% of neighbouring pairs"
              + ("" if sim_thr is not None else " (not evaluated: no embeddings)")
              + f"; priority = {' > '.join(args.priority)} > other.\n")

    for track_dir in args.track_dir:
        name = Path(track_dir).name
        ann_parts, hyp_all = [], {}
        for s in seqs:
            hyp = load_tracks(Path(track_dir) / f"{s.info.name}.txt")
            hyp_all[s.info.name] = {int(f): g[["x", "y", "w", "h", "id", "conf"]].to_numpy()
                                    for f, g in hyp.groupby("frame")}
            _, ev = evaluate_tracks(s, hyp)
            a = annotate_events(ev, feats[s.info.name], pre=args.pre, emb=emb.get(s.info.name),
                                sim_thr=sim_thr, priority=tuple(args.priority))
            a.insert(0, "seq", s.info.name)
            a.insert(0, "tracker", name)
            ann_parts.append(a)
        ann = pd.concat(ann_parts, ignore_index=True)

        # base rates: share of GT person-frames that sit in the same context, using a window as long
        # as a typical switch window
        win = int(max(np.median(ann["gap"] + args.pre + 2), 2)) if len(ann) else args.pre + 2
        tot = occ = fast = 0
        for s in seqs:
            n, o, f = feats[s.info.name].base_counts(win)
            tot, occ, fast = tot + n, occ + o, fast + f
        summ = summarize(ann, occ / max(tot, 1), fast / max(tot, 1))
        summaries[name] = summ

        (out / name).mkdir(parents=True, exist_ok=True)
        ann.to_csv(out / name / "switch_events_annotated.csv", index=False)

        md.append(f"\n## {name}: {len(ann)} ID switches\n")
        md.append("**Mechanism**\n\n" + to_markdown(summ["mechanism"], float_cols=()))
        md.append("\n**Primary cause** (first applicable flag in priority order)\n\n"
                  + to_markdown(summ["primary"], float_cols=()))
        md.append("\n**Context flags** (not exclusive: one switch can be both occluded and fast)\n\n"
                  + to_markdown(summ["flags"], float_cols=()))
        md.append(f"\n**Over-representation** (window = {win} frames): share of switches in a context vs share of all "
                  "GT person-frames in that context; lift > 1 means the context is linked to switches\n\n"
                  + to_markdown(summ["lift"], float_cols=()))
        if len(ann):
            rg = ann[ann["mechanism"] == "respawn_gap"]["gap"]
            if len(rg):
                md.append(f"\nRe-spawns after a gap: median {rg.median():.0f} frames lost, "
                          f"longest {rg.max()} frames (compare with max_age).\n")

        print(f"\n=== {name}: {len(ann)} ID switches ===")
        print(summ["mechanism"].to_string(index=False))
        print(summ["primary"].to_string(index=False))
        print(summ["lift"].to_string(index=False))

        if not args.no_images and len(ann):
            ex_dir = out / name / "examples"
            ex_dir.mkdir(parents=True, exist_ok=True)
            by_name = {s.info.name: s for s in seqs}
            composites = []
            for ev in pick_examples(ann, args.examples):
                s = by_name[ev.seq]
                gt_peds = s.load_gt()
                img = example_image(s, hyp_all[ev.seq], gt_peds, ev, name)
                if img is None:
                    continue
                p = ex_dir / f"{ev.primary}_{ev.seq}_f{ev.frame}_gt{ev.gt_id}.png"
                cv2.imwrite(str(p), img)
                composites.append(img)
            if composites:
                w = 1100
                rs = [cv2.resize(c, (w, int(c.shape[0] * w / c.shape[1]))) for c in composites]
                cv2.imwrite(str(out / f"examples_sheet_{name}.png"), np.vstack(rs))
                print(f"saved {len(composites)} example pictures in {ex_dir} and examples_sheet_{name}.png")

    if len(summaries) > 1:
        cmp = pd.DataFrame({n: s["primary"].set_index("primary_cause")["count"] for n, s in summaries.items()})
        cmp.loc["TOTAL"] = cmp.sum()
        md.append("\n## Comparison of trackers (primary cause counts)\n\n"
                  + to_markdown(cmp.reset_index().rename(columns={"index": "primary_cause"}), float_cols=()))
    (out / "failure_summary.md").write_text("\n".join(md) + "\n")

    # chart
    names = list(summaries)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x, bw = np.arange(len(PRIMARY_ORDER)), 0.8 / len(names)
    for k, n in enumerate(names):
        counts = summaries[n]["primary"].set_index("primary_cause").reindex(PRIMARY_ORDER)["count"]
        ax[0].bar(x + k * bw, counts, bw, label=n)
    ax[0].set_xticks(x + bw * (len(names) - 1) / 2)
    ax[0].set_xticklabels([c.replace("_", "\n") for c in PRIMARY_ORDER])
    ax[0].set_ylabel("ID switches"); ax[0].set_title("Primary cause of ID switches"); ax[0].legend()
    ctx = ["occlusion", "fast_motion"]
    xs, bw2 = np.arange(len(ctx)), 0.8 / (len(names) + 1)
    first = summaries[names[0]]["lift"].set_index("context")
    ax[1].bar(xs, [first.loc[c, "all_gt_boxes_pct"] for c in ctx], bw2, color="lightgray", label="all GT person-frames")
    for k, n in enumerate(names):
        l = summaries[n]["lift"].set_index("context")
        ax[1].bar(xs + (k + 1) * bw2, [l.loc[c, "switches_pct"] for c in ctx], bw2, label=f"{n}: switches")
    ax[1].set_xticks(xs + bw2 * len(names) / 2); ax[1].set_xticklabels(ctx)
    ax[1].set_ylabel("% in this context"); ax[1].set_title("Switches vs. base rate"); ax[1].legend(fontsize=8)
    ax[0].margins(y=0.15); ax[1].margins(y=0.3)          # headroom for the legends
    fig.tight_layout(); fig.savefig(out / "failure_causes.png", dpi=130)
    print(f"\nSaved {out / 'failure_summary.md'} and {out / 'failure_causes.png'}")


if __name__ == "__main__":
    main()
