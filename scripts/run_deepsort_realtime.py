"""Run deep-sort-realtime tracker on cached YOLO detections.

Usage:
    python scripts/run_deepsort_realtime.py \
        --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
        --det-dir detections/yolov8n_640 \
        --conf-thr 0.4 --max-age 30 --n-init 3

Output:
    results/trackers/deepsort_realtime/<sequence>.txt
    results/trackers/deepsort_realtime/config.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sort import write_results
from src.data import MOTSequence
from src.detections import detections_by_frame, load_detections


def run_sequence(seq: MOTSequence, det_path: Path, conf_thr: float, max_age: int, n_init: int):
    dets = detections_by_frame(load_detections(det_path, min_conf=conf_thr))
    tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=100,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=False,
    )

    results = {}
    t0 = time.time()

    for frame in range(1, seq.info.length + 1):
        d = dets.get(frame, np.zeros((0, 5)))
        raw_dets = []
        for x, y, w, h, conf in d:
            raw_dets.append(([float(x), float(y), float(w), float(h)], float(conf), "person"))

        img = seq.read_frame(frame)
        tracks = tracker.update_tracks(raw_dets, frame=img)

        frame_rows = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            box = t.to_ltwh()
            conf = t.get_det_conf() if t.get_det_conf() is not None else 1.0
            frame_rows.append([box[0], box[1], box[2], box[3], int(t.track_id), conf])

        results[frame] = np.array(frame_rows).reshape(-1, 6)

    elapsed = time.time() - t0
    return results, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--out-dir", default="results/trackers/deepsort_realtime")
    ap.add_argument("--conf-thr", type=float, default=0.4)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s_path in args.seq_dir:
        seq = MOTSequence(s_path)
        det_path = Path(args.det_dir) / f"{seq.info.name}.txt"
        if not det_path.exists():
            raise FileNotFoundError(f"Detections not found at {det_path}")

        results, elapsed = run_sequence(seq, det_path, args.conf_thr, args.max_age, args.n_init)
        out_file = out_dir / f"{seq.info.name}.txt"
        write_results(results, out_file)

        all_ids = {int(r[4]) for rows in results.values() for r in rows}
        n_boxes = sum(len(r) for r in results.values())
        gt = seq.load_gt()
        fps = seq.info.length / max(elapsed, 1e-4)

        print(
            f"{seq.info.name} (deep-sort-realtime): {len(all_ids)} track IDs "
            f"(GT has {gt['id'].nunique()} people), "
            f"{n_boxes / seq.info.length:.1f} boxes/frame, speed {fps:.1f} FPS -> {out_file}"
        )

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
