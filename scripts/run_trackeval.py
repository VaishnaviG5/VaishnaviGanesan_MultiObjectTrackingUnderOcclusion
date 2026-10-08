"""Step 6: official evaluation with TrackEval (HOTA, MOTA, IDF1, ID switches, ...) + cross-check.

Usage:
    python scripts/setup_trackeval.py          # once
    python scripts/run_trackeval.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN data/MOT17/train/MOT17-02-FRCNN ... \
        --track-dir results/trackers/sort results/trackers/deepsort_resnet

Outputs (in results/, or results/trackeval/):
    metrics.csv, metrics.md          official numbers: one row per tracker x sequence (+ OVERALL)
    trackeval/crosscheck.csv         official numbers next to our in-repo evaluator (src/clearmot.py)
    trackeval/output/...             TrackEval's own summary files

--tag NAME writes metrics_NAME.csv/.md instead, e.g. --tag tune and --tag heldout when you tune
parameters on some sequences and report on the others.

Note: we score ONE detector variant per sequence (e.g. MOT17-04-FRCNN). The three variants share
identical images and ground truth, so this is a faithful local evaluation, but numbers are not
directly comparable to the leaderboard (which averages all three variants on the hidden test set).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clearmot import evaluate_tracks, load_tracks  # noqa: E402
from src.data import MOTSequence  # noqa: E402
from src.trackeval_utils import (SUMMARY_COLS, crosscheck_table, extract_rows,  # noqa: E402
                                 prepare_layout, to_markdown)


def import_trackeval(path: str):
    try:
        import trackeval
        return trackeval
    except ImportError:
        p = Path(path).resolve()
        if not (p / "trackeval").exists():
            raise SystemExit(f"TrackEval not found at {p}. Run:  python scripts/setup_trackeval.py")
        sys.path.insert(0, str(p))
        import trackeval
        return trackeval


def run_trackeval(trackeval, seqs, gt_folder, trackers_folder, tracker_names, out_folder, verbose=False):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(USE_PARALLEL=False, PRINT_RESULTS=verbose, PRINT_CONFIG=False,
                       TIME_PROGRESS=verbose, PLOT_CURVES=False)

    ds_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_config.update(
        GT_FOLDER=str(gt_folder), TRACKERS_FOLDER=str(trackers_folder), OUTPUT_FOLDER=str(out_folder),
        TRACKERS_TO_EVAL=tracker_names, BENCHMARK="MOT17", SPLIT_TO_EVAL="train",
        SKIP_SPLIT_FOL=True,                                  # our folders have no 'MOT17-train' level
        SEQ_INFO={s.info.name: s.info.length for s in seqs},  # which sequences, and their lengths
        PRINT_CONFIG=False,
    )
    evaluator = trackeval.Evaluator(eval_config)
    datasets = [trackeval.datasets.MotChallenge2DBox(ds_config)]
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
               trackeval.metrics.Identity(), trackeval.metrics.Count()]
    output_res, _ = evaluator.evaluate(datasets, metrics)
    return output_res


def our_rows(seqs, track_dirs):
    """Same table, computed by src/clearmot.py (for the cross-check)."""
    from src.clearmot import aggregate
    rows = []
    for d in track_dirs:
        per = []
        for seq in seqs:
            m, _ = evaluate_tracks(seq, load_tracks(Path(d) / f"{seq.info.name}.txt"))
            per.append(m)
        per.append(aggregate(per))
        for m in per:
            rows.append(dict(tracker=Path(d).name, seq=m["seq"], MOTA=100 * m["MOTA"], IDF1=100 * m["IDF1"],
                             IDSW=m["IDSW"], FP=m["FP"], FN=m["FN"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--track-dir", nargs="+", required=True, help="one or more tracker result folders")
    ap.add_argument("--trackeval-dir", default="third_party/TrackEval")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true", help="show TrackEval's own tables")
    ap.add_argument("--no-crosscheck", action="store_true")
    args = ap.parse_args()

    seqs = [MOTSequence(d) for d in args.seq_dir]
    parents = {str(Path(d).resolve().parent) for d in args.seq_dir}
    if len(parents) != 1:
        raise SystemExit("All --seq-dir folders must live in the same parent folder (e.g. data/MOT17/train).")
    gt_folder = parents.pop()
    seq_names = [s.info.name for s in seqs]

    out_dir = Path(args.out_dir)
    te_dir = out_dir / "trackeval"
    tracker_names = prepare_layout(args.track_dir, seq_names, te_dir)

    trackeval = import_trackeval(args.trackeval_dir)
    output_res = run_trackeval(trackeval, seqs, gt_folder, te_dir / "trackers", tracker_names,
                               te_dir / "output", verbose=args.verbose)

    rows = []
    for name in tracker_names:
        rows += extract_rows(output_res, name, seq_names)
    df = pd.DataFrame(rows)[["tracker", "seq"] + SUMMARY_COLS]

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"metrics{suffix}.csv", index=False)
    overall = df[df["seq"] == "OVERALL"]
    md = ["## Overall (official TrackEval)\n", to_markdown(overall)]
    for name in tracker_names:
        md += [f"\n\n## Per sequence: {name}\n", to_markdown(df[(df.tracker == name) & (df.seq != "OVERALL")]
                                                                .drop(columns="tracker"))]
    (out_dir / f"metrics{suffix}.md").write_text("\n".join(md) + "\n")

    print("\nOFFICIAL RESULTS (TrackEval)")
    print(overall.drop(columns="seq").round(1).to_string(index=False))
    print(f"\nSaved {out_dir / f'metrics{suffix}.csv'} and {out_dir / f'metrics{suffix}.md'}")

    if not args.no_crosscheck:
        cc = crosscheck_table(df, our_rows(seqs, args.track_dir))
        cc.to_csv(te_dir / f"crosscheck{suffix}.csv", index=False)
        show = cc[cc.seq == "OVERALL"][["tracker", "MOTA_TE", "MOTA_ours", "IDF1_TE", "IDF1_ours",
                                        "IDSW_TE", "IDSW_ours", "FP_TE", "FP_ours", "FN_TE", "FN_ours"]]
        print("\nCROSS-CHECK: TrackEval vs our evaluator (OVERALL)")
        print(show.round(1).to_string(index=False))
        print("Small differences are expected (e.g. tie-breaking in matching, ignore-region details); "
              "large ones mean a bug or a data/format problem.")


if __name__ == "__main__":
    main()
