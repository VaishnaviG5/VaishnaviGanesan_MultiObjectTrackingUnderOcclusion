"""Step 5a: compute an appearance embedding for every cached detection (once), and save it.

Reads detections from Step 2 (confidence >= --min-conf), crops each box from its frame, runs an
embedding backend, and writes one file per sequence:
    embeddings/<name>/<sequence>.npz  with  frame, box (N,4 xywh), conf, feat (N,D)

Usage:
    python scripts/extract_embeddings.py \
        --seq-dir data/MOT17/train/MOT17-04-FRCNN ... \
        --det-dir detections/yolov8m_1280 --min-conf 0.3 --backend resnet

Backends: hist (no downloads), resnet (torchvision), osnet (torchreid). Output folder is
embeddings/<backend>, so several backends can live side by side and be compared.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:          # progress bar is optional
    def tqdm(it, **_):
        return it

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import MOTSequence  # noqa: E402
from src.detections import load_detections  # noqa: E402
from src.embedder import crop_person, get_embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", nargs="+", required=True)
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--backend", default="resnet", choices=["hist", "resnet", "osnet"])
    ap.add_argument("--weights", default=None, help="osnet re-ID checkpoint (optional)")
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="only embed detections at least this confident (tracker --conf-thr must be >= this)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=None, help="default: embeddings/<backend>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"embeddings/{args.backend}")
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder(args.backend, device=args.device, weights=args.weights)

    for seq_dir in args.seq_dir:
        seq = MOTSequence(seq_dir)
        out_path = out_dir / f"{seq.info.name}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite to redo)")
            continue

        df = load_detections(Path(args.det_dir) / f"{seq.info.name}.txt", min_conf=args.min_conf)
        df = df.sort_values("frame", kind="stable").reset_index(drop=True)
        boxes = df[["x", "y", "w", "h"]].to_numpy(dtype=np.float32)
        frames = df["frame"].to_numpy(dtype=np.int32)

        feats, pending = [], []
        def flush():
            if pending:
                feats.append(embedder(pending))
                pending.clear()

        for frame, idx in tqdm(df.groupby("frame").indices.items(), desc=seq.info.name,
                               total=df["frame"].nunique()):
            img = seq.read_frame(int(frame))
            for i in idx:
                pending.append(crop_person(img, boxes[i]))
            if len(pending) >= args.batch_size:
                flush()
        flush()

        feat = np.concatenate(feats) if feats else np.zeros((0, 1), dtype=np.float32)
        assert len(feat) == len(df), "embedding count must equal detection count"
        np.savez_compressed(out_path, frame=frames, box=boxes, conf=df["conf"].to_numpy(dtype=np.float32),
                            feat=feat.astype(np.float16))
        print(f"{seq.info.name}: {len(df)} detections -> {out_path}  (dim {feat.shape[1]})")


if __name__ == "__main__":
    main()
