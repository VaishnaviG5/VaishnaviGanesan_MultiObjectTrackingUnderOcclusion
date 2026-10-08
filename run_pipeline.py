"""End-to-end pipeline driver for Multi-Object Tracking Under Occlusion.

Executes all pipeline stages:
  Step 3: Detection with YOLOv8
  Step 4: SORT tracking
  Step 5: DeepSORT tracking (deep-sort-realtime)
  Step 6: Evaluation with py-motmetrics & CLEAR-MOT
  Step 7: Failure analysis (ID switch categorization)
  Step 8: Trajectory visualization (MP4/GIF/PNG)

Usage:
    python run_pipeline.py
    python run_pipeline.py --sequences MOT17-02-FRCNN MOT17-04-FRCNN
    python run_pipeline.py --steps 4 5 6 7 8
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run([sys.executable] + cmd, check=True)
    return res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=["MOT17-02-FRCNN", "MOT17-04-FRCNN"],
        help="Sequence subfolders inside data/MOT17/train",
    )
    ap.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 7, 8],
        help="Pipeline steps to execute (default: 3 4 5 6 7 8)",
    )
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    seq_dirs = [f"data/MOT17/train/{s}" for s in args.sequences]
    det_dir = f"detections/{Path(args.model).stem}_{args.imgsz}"

    # Step 3: Detect
    if 3 in args.steps:
        print("\n" + "=" * 60 + "\nSTEP 3: YOLO Object Detection\n" + "=" * 60)
        run_cmd([
            "scripts/detect.py",
            "--seq-dir", *seq_dirs,
            "--model", args.model,
            "--imgsz", str(args.imgsz),
            "--conf", "0.1",
            "--out-dir", det_dir,
        ])

    # Step 4: SORT
    if 4 in args.steps:
        print("\n" + "=" * 60 + "\nSTEP 4: SORT Tracker\n" + "=" * 60)
        run_cmd([
            "scripts/run_sort.py",
            "--seq-dir", *seq_dirs,
            "--det-dir", det_dir,
            "--conf-thr", "0.4",
            "--max-age", "30",
            "--min-hits", "3",
            "--iou-thr", "0.3",
        ])

    # Step 5: DeepSORT (deep-sort-realtime)
    if 5 in args.steps:
        print("\n" + "=" * 60 + "\nSTEP 5: DeepSORT Tracker (deep-sort-realtime)\n" + "=" * 60)
        run_cmd([
            "scripts/run_deepsort_realtime.py",
            "--seq-dir", *seq_dirs,
            "--det-dir", det_dir,
            "--conf-thr", "0.4",
            "--max-age", "30",
            "--n-init", "3",
        ])

    # Step 6: Evaluation
    if 6 in args.steps:
        print("\n" + "=" * 60 + "\nSTEP 6: Evaluation (py-motmetrics & CLEAR-MOT)\n" + "=" * 60)
        run_cmd([
            "scripts/eval_motmetrics.py",
            "--seq-dir", *seq_dirs,
            "--track-dir", "results/trackers/sort",
        ])
        run_cmd([
            "scripts/eval_motmetrics.py",
            "--seq-dir", *seq_dirs,
            "--track-dir", "results/trackers/deepsort_realtime",
        ])
        run_cmd([
            "scripts/evaluate.py",
            "--seq-dir", *seq_dirs,
            "--track-dir", "results/trackers/sort",
        ])
        run_cmd([
            "scripts/evaluate.py",
            "--seq-dir", *seq_dirs,
            "--track-dir", "results/trackers/deepsort_realtime",
        ])
        run_cmd(["scripts/compare_trackers.py"])

    # Step 7: Failure Analysis
    if 7 in args.steps:
        print("\n" + "=" * 60 + "\nSTEP 7: Failure Analysis & ID-Switch Profiling\n" + "=" * 60)
        run_cmd([
            "scripts/analyze_failures.py",
            "--seq-dir", *seq_dirs,
            "--track-dir", "results/trackers/sort", "results/trackers/deepsort_realtime",
        ])

    # Step 8: Visualization
    if 8 in args.steps:
        print("\n" + "=" * 60 + "\nSTEP 8: Trajectory Visualization\n" + "=" * 60)
        run_cmd([
            "scripts/visualize_tracks.py",
            "--seq-dir", seq_dirs[0],
            "--track-dir", "results/trackers/sort", "results/trackers/deepsort_realtime",
            "--start", "50",
            "--length", "100",
            "--trail", "30",
            "--scale", "0.5",
        ])

    print("\n[SUCCESS] Pipeline executed successfully!")


if __name__ == "__main__":
    main()
