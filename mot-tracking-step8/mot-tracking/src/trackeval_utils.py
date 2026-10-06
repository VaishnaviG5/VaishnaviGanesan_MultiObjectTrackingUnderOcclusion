"""Helpers for running the official TrackEval (https://github.com/JonathonLuiten/TrackEval)
on our tracker outputs, and for turning its results into tidy tables.

Kept separate from the TrackEval call itself so everything here can be unit-tested without it.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "MotChallenge2DBox"
SUMMARY_COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag", "MT", "ML", "GT_IDs", "IDs"]


# ---------------------------------------------------------------- layout
def prepare_layout(track_dirs, seq_names, root) -> list[str]:
    """Copy tracker results into the folder layout TrackEval expects:

        <root>/trackers/<tracker name>/data/<sequence>.txt

    Returns the tracker names (= the tracker folders' names).
    """
    root = Path(root)
    names = [Path(d).name for d in track_dirs]
    if len(set(names)) != len(names):
        raise ValueError(f"tracker folders must have distinct names, got {names}")
    for d, name in zip(track_dirs, names):
        dest = root / "trackers" / name / "data"
        if dest.parent.exists():
            shutil.rmtree(dest.parent)                 # always start from a clean copy
        dest.mkdir(parents=True)
        for seq in seq_names:
            src = Path(d) / f"{seq}.txt"
            if not src.exists():
                raise FileNotFoundError(f"{src} not found. Run tracker '{name}' on {seq} first.")
            shutil.copy(src, dest / f"{seq}.txt")
    return names


# ---------------------------------------------------------------- results -> table
def _num(x) -> float:
    return float(np.mean(x))


def extract_rows(output_res: dict, tracker: str, seq_names, dataset_name: str = DATASET_NAME) -> list[dict]:
    """TrackEval result dict -> one row per sequence plus an OVERALL row.

    TrackEval stores rates (MOTA, IDF1, HOTA, ...) as fractions and HOTA-family fields as arrays
    over 19 IoU thresholds; we average those and report everything in percent.
    """
    res = output_res[dataset_name][tracker]
    rows = []
    for seq in list(seq_names) + ["COMBINED_SEQ"]:
        d = res[seq]["pedestrian"]
        h, c, i, n = d["HOTA"], d["CLEAR"], d["Identity"], d["Count"]
        rows.append(dict(
            tracker=tracker, seq="OVERALL" if seq == "COMBINED_SEQ" else seq,
            HOTA=100 * _num(h["HOTA"]), DetA=100 * _num(h["DetA"]), AssA=100 * _num(h["AssA"]),
            MOTA=100 * _num(c["MOTA"]), IDF1=100 * _num(i["IDF1"]),
            IDSW=int(_num(c["IDSW"])), FP=int(_num(c["CLR_FP"])), FN=int(_num(c["CLR_FN"])),
            Frag=int(_num(c["Frag"])), MT=int(_num(c["MT"])), ML=int(_num(c["ML"])),
            GT_IDs=int(_num(n["GT_IDs"])), IDs=int(_num(n["IDs"])),
        ))
    return rows


def to_markdown(df: pd.DataFrame, float_cols=("HOTA", "DetA", "AssA", "MOTA", "IDF1"), digits: int = 1) -> str:
    """Small markdown-table writer (avoids needing the 'tabulate' package)."""
    d = df.copy()
    for c in float_cols:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.{digits}f}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def crosscheck_table(trackeval_df: pd.DataFrame, ours_df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side TrackEval vs. our in-repo evaluator (both with columns tracker, seq, MOTA, IDF1 in
    percent and IDSW, FP, FN as counts)."""
    m = trackeval_df.merge(ours_df, on=["tracker", "seq"], suffixes=("_TE", "_ours"))
    out = m[["tracker", "seq"]].copy()
    for col in ["MOTA", "IDF1"]:
        out[f"{col}_TE"], out[f"{col}_ours"] = m[f"{col}_TE"], m[f"{col}_ours"]
        out[f"d{col}"] = m[f"{col}_ours"] - m[f"{col}_TE"]
    for col in ["IDSW", "FP", "FN"]:
        out[f"{col}_TE"], out[f"{col}_ours"] = m[f"{col}_TE"], m[f"{col}_ours"]
        out[f"d{col}"] = m[f"{col}_ours"] - m[f"{col}_TE"]
    return out


# ---------------------------------------------------------------- numpy-alias patch
_ALIAS = re.compile(r"\bnp\.(float|int|bool|object|str)\b")


def patch_numpy_aliases(trackeval_pkg_dir) -> int:
    """TrackEval was written for old NumPy and uses aliases (np.float, np.int, np.bool, ...) that were
    removed in NumPy 1.24. Replace them by the builtins. np.float32, np.int64, np.bool_, np.floating
    etc. are left untouched. Returns the number of files changed."""
    changed = 0
    for path in Path(trackeval_pkg_dir).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = _ALIAS.sub(lambda m: m.group(1), text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed
