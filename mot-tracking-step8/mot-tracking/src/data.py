"""MOT17 sequence loader.

MOT17 layout (one folder per sequence):
    MOT17-04-FRCNN/
        seqinfo.ini      frame rate, length, image size
        img1/000001.jpg  frames, numbered from 1
        gt/gt.txt        ground truth boxes
        det/det.txt      public detections (we run our own detector instead)

gt.txt columns (comma separated):
    frame, id, x, y, w, h, conf, class, visibility
  - (x, y) is the top-left corner, in pixels
  - conf is 0/1: whether the box is "considered" in evaluation
  - class: 1 = pedestrian (the only class we track); others are distractors,
    people on vehicles, static persons, reflections, etc.
  - visibility: 0..1, fraction of the person that is visible (useful in Step 7)
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
PEDESTRIAN_CLASS = 1


@dataclass
class SeqInfo:
    name: str
    frame_rate: int
    length: int
    width: int
    height: int
    im_dir: str
    im_ext: str


def read_seqinfo(seq_dir: Path) -> SeqInfo:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        raise FileNotFoundError(f"{ini} not found. Is {seq_dir} a MOT17 sequence folder?")
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    s = cfg["Sequence"]
    return SeqInfo(
        name=s["name"],
        frame_rate=int(s["frameRate"]),
        length=int(s["seqLength"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        im_dir=s.get("imDir", "img1"),
        im_ext=s.get("imExt", ".jpg"),
    )


class MOTSequence:
    def __init__(self, seq_dir: str | Path):
        self.dir = Path(seq_dir)
        self.info = read_seqinfo(self.dir)
        self._gt: pd.DataFrame | None = None

    # ---- frames -------------------------------------------------------
    def frame_path(self, frame: int) -> Path:
        """Frames are 1-indexed: frame 1 -> img1/000001.jpg."""
        return self.dir / self.info.im_dir / f"{frame:06d}{self.info.im_ext}"

    def read_frame(self, frame: int) -> np.ndarray:
        """Return the frame as a BGR image (OpenCV convention)."""
        path = self.frame_path(frame)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read frame {frame} at {path}")
        return img

    def frames(self):
        """Iterate (frame_number, image) over the whole sequence."""
        for f in range(1, self.info.length + 1):
            yield f, self.read_frame(f)

    # ---- ground truth -------------------------------------------------
    def load_gt(self, pedestrians_only: bool = True) -> pd.DataFrame:
        """Ground truth as a DataFrame with GT_COLUMNS.

        With pedestrians_only=True (default) keeps only class 1 and conf == 1,
        which matches what the official evaluation scores.
        """
        if self._gt is None:
            path = self.dir / "gt" / "gt.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Ground truth exists only for the train split."
                )
            self._gt = pd.read_csv(path, header=None, names=GT_COLUMNS)
        gt = self._gt
        if pedestrians_only:
            gt = gt[(gt["cls"] == PEDESTRIAN_CLASS) & (gt["conf"] == 1)]
        return gt

    def gt_for_frame(self, frame: int, pedestrians_only: bool = True) -> pd.DataFrame:
        gt = self.load_gt(pedestrians_only)
        return gt[gt["frame"] == frame]

    def summary(self) -> str:
        gt = self.load_gt()
        per_frame = gt.groupby("frame").size()
        return (
            f"{self.info.name}: {self.info.length} frames @ {self.info.frame_rate} fps, "
            f"{self.info.width}x{self.info.height}\n"
            f"  GT pedestrian boxes: {len(gt)}, unique IDs: {gt['id'].nunique()}, "
            f"avg {per_frame.mean():.1f} people/frame (max {per_frame.max()})"
        )
