"""Step 2: run a pretrained YOLO person detector on MOT17 sequences and cache results.

We cache detections at a LOW confidence threshold so that later steps (tracker,
threshold sweeps) can filter them without ever re-running the detector.

Usage (one or more sequences):
    python scripts/detect.py --seq-dir data/MOT17/train/MOT17-04-FRCNN \
        --model yolov8m.pt --imgsz 1280 --conf 0.1

Output:
    detections/<model>_<imgsz>/<sequence-name>.txt     (MOT format)
    detections/<model>_<imgsz>/config.json             (settings used, for reproducibility)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import MOTSequence  # noqa: E402
from src.detections import save_detections  # noqa: E402


def results_to_rows(frame, boxes_xyxy, confs, img_w, img_h):
    """Convert one frame of detector output to MOT rows (frame, x, y, w, h, conf).

    Boxes are clipped to the image; degenerate boxes are dropped.
    """
    rows = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(img_w), float(x2)), min(float(img_h), float(y2))
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        rows.append((frame, x1, y1, w, h, float(c)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True, help="one or more MOT17 sequence folders")
    ap.add_argument("--model", default="yolov8m.pt", help="Ultralytics weights (auto-downloaded)")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size; larger helps small/far people")
    ap.add_argument("--conf", type=float, default=0.1, help="cache threshold (keep it low)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--device", default=None, help="e.g. 0 for first GPU, 'cpu'; default = auto")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="default: detections/<model>_<imgsz>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from tqdm import tqdm
    from ultralytics import YOLO
    import ultralytics

    tag = f"{Path(args.model).stem}_{args.imgsz}"
    out_dir = Path(args.out_dir or f"detections/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        out_path = out_dir / f"{seq.info.name}.txt"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite to redo)")
            continue

        import torch

        paths = [str(seq.frame_path(f)) for f in range(1, seq.info.length + 1)]
        chunk_size = 200
        rows, t0 = [], time.time()
        pbar = tqdm(total=len(paths), desc=seq.info.name)
        with torch.no_grad():
            for c_start in range(0, len(paths), chunk_size):
                chunk = paths[c_start : c_start + chunk_size]
                stream = model.predict(
                    source=chunk, classes=[0],
                    conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                    device=args.device, batch=args.batch, stream=True, verbose=False,
                )
                for res in stream:
                    frame = int(Path(res.path).stem)  # 000123.jpg -> 123
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    rows.extend(results_to_rows(frame, xyxy, confs, seq.info.width, seq.info.height))
                    pbar.update(1)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        pbar.close()
        elapsed = time.time() - t0

        save_detections(rows, out_path)
        print(f"{seq.info.name}: {len(rows)} detections, {len(paths) / elapsed:.1f} FPS -> {out_path}")

    cfg = {**vars(args), "ultralytics_version": ultralytics.__version__}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
