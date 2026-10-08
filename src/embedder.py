"""Appearance embeddings for person crops.

Every backend maps a list of BGR crops (OpenCV images) -> array (N, D) of L2-normalised vectors,
so cosine distance = 1 - dot product.

Backends
  hist    HSV colour histograms over head/torso/legs stripes. No downloads, runs on CPU, a decent
          weak baseline (clothing colour). Good for testing the pipeline.
  resnet  torchvision ResNet-50 with ImageNet weights (global-average-pooled, 2048-d). Easy to
          install, but NOT trained for person re-identification.
  osnet   OSNet from the `torchreid` package. Best choice if you can install it, especially with
          re-ID weights (e.g. osnet_x1_0 trained on MSMT17/Market1501 from the torchreid model zoo,
          passed with --weights). Without --weights it falls back to ImageNet weights.
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_person(img: np.ndarray, box) -> np.ndarray:
    """Crop an (x, y, w, h) box from a BGR image, clipped to the image (never returns an empty crop)."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
    x2, y2 = int(min(W, np.ceil(x + w))), int(min(H, np.ceil(y + h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros((8, 4, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class HistEmbedder:
    name = "hist"

    def __init__(self, **_):
        self.h_bins, self.s_bins = 16, 8

    def _one(self, crop: np.ndarray) -> np.ndarray:
        crop = cv2.resize(crop, (32, 64))
        # keep the central part horizontally: less background at the sides of the box
        crop = crop[:, 4:28]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        stripes = [hsv[0:16], hsv[16:40], hsv[40:64]]        # head / torso / legs
        parts = []
        for s in stripes:
            hist = cv2.calcHist([s], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256]).ravel()
            hist = hist / max(hist.sum(), 1e-9)
            parts.append(np.sqrt(hist))                      # Hellinger: cosine of these = Bhattacharyya
        return np.concatenate(parts)

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, 3 * self.h_bins * self.s_bins), dtype=np.float32)
        return _l2(np.stack([self._one(c) for c in crops])).astype(np.float32)


class ResNetEmbedder:
    name = "resnet"

    def __init__(self, device=None, **_):
        import torch
        import torchvision

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, crops):
        torch = self.torch
        if not crops:
            return np.zeros((0, 2048), dtype=np.float32)
        batch = []
        for c in crops:
            rgb = cv2.cvtColor(cv2.resize(c, (128, 256)), cv2.COLOR_BGR2RGB)     # (w=128, h=256)
            batch.append(torch.from_numpy(rgb).permute(2, 0, 1))
        x = torch.stack(batch).float().to(self.device) / 255.0
        x = (x - self.mean) / self.std
        with torch.no_grad():
            f = self.net(x).cpu().numpy()
        return _l2(f).astype(np.float32)


class OSNetEmbedder:
    name = "osnet"

    def __init__(self, device=None, weights=None, model_name="osnet_x1_0", **_):
        from torchreid.utils import FeatureExtractor      # pip install torchreid (see its README)

        self.ext = FeatureExtractor(model_name=model_name, model_path=weights or "",
                                    device=device or ("cuda" if _cuda() else "cpu"))

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        f = self.ext(rgb).cpu().numpy()                   # torchreid resizes to 256x128 itself
        return _l2(f).astype(np.float32)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_embedder(name: str, device=None, weights=None):
    name = name.lower()
    if name == "hist":
        return HistEmbedder()
    if name == "resnet":
        return ResNetEmbedder(device=device)
    if name == "osnet":
        return OSNetEmbedder(device=device, weights=weights)
    raise ValueError(f"unknown embedder '{name}' (choose hist, resnet or osnet)")
