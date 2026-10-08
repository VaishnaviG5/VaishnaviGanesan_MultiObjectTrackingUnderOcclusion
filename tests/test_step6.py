"""Tests for the Step 6 helpers (no real TrackEval needed).

The real TrackEval is exercised by scripts/run_trackeval.py on your machine; here we check our own
code: layout preparation, result parsing, tables, the NumPy-alias patch, and the wrapper's plumbing
using a tiny stand-in for the TrackEval API.

Run with:  python tests/test_step6.py     (or: python -m pytest tests/ -q)
"""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trackeval_utils import (crosscheck_table, extract_rows, patch_numpy_aliases,  # noqa: E402
                                 prepare_layout, to_markdown)


def fake_result(trackers, seqs):
    def one(k):
        return {"pedestrian": {
            "HOTA": {"HOTA": np.full(19, 0.50 + k), "DetA": np.full(19, 0.6), "AssA": np.full(19, 0.4)},
            "CLEAR": {"MOTA": 0.70 + k, "IDSW": 12 + k * 100, "CLR_FP": 100, "CLR_FN": 200, "Frag": 30,
                      "MT": 5, "ML": 2},
            "Identity": {"IDF1": 0.65},
            "Count": {"GT_IDs": 20, "IDs": 30}}}
    return {"MotChallenge2DBox": {t: {**{s: one(0.0) for s in seqs}, "COMBINED_SEQ": one(0.0)}
                                  for t in trackers}}


def test_extract_rows_scales_and_averages():
    rows = extract_rows(fake_result(["sort"], ["A", "B"]), "sort", ["A", "B"])
    assert [r["seq"] for r in rows] == ["A", "B", "OVERALL"]
    r = rows[-1]
    assert abs(r["HOTA"] - 50.0) < 1e-9 and abs(r["MOTA"] - 70.0) < 1e-9 and abs(r["IDF1"] - 65.0) < 1e-9
    assert (r["IDSW"], r["FP"], r["FN"], r["IDs"]) == (12, 100, 200, 30)


def test_markdown_table():
    df = pd.DataFrame([dict(tracker="sort", HOTA=50.04, IDSW=12)])
    md = to_markdown(df).splitlines()
    assert md[0] == "| tracker | HOTA | IDSW |" and md[2] == "| sort | 50.0 | 12 |"


def test_prepare_layout_copies_and_validates():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for name in ("sort", "deep"):
            (t / name).mkdir()
            (t / name / "S1.txt").write_text("1,1,0,0,10,10,1,-1,-1,-1\n")
        names = prepare_layout([t / "sort", t / "deep"], ["S1"], t / "te")
        assert names == ["sort", "deep"] and (t / "te/trackers/deep/data/S1.txt").exists()
        try:
            prepare_layout([t / "sort"], ["S2"], t / "te2")
            assert False, "missing sequence file must raise"
        except FileNotFoundError:
            pass
        try:
            prepare_layout([t / "sort", t / "sort"], ["S1"], t / "te3")
            assert False, "duplicate names must raise"
        except ValueError:
            pass


def test_numpy_alias_patch():
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "m.py"
        f.write_text("a = np.zeros(3, dtype=np.float)\nb = x.astype(np.int)\nc = np.bool(1)\n"
                     "d = np.float32(1) + np.int64(2)\ne = np.bool_(0)\nf = np.floating\ng = np.str('x')\n")
        assert patch_numpy_aliases(t) == 1
        s = f.read_text()
        assert "dtype=float)" in s and "astype(int)" in s and "c = bool(1)" in s and "str('x')" in s
        assert "np.float32" in s and "np.int64" in s and "np.bool_" in s and "np.floating" in s
        assert patch_numpy_aliases(t) == 0, "patching must be idempotent"


def test_crosscheck_table():
    te = pd.DataFrame([dict(tracker="a", seq="OVERALL", MOTA=70.0, IDF1=65.0, IDSW=12, FP=100, FN=200)])
    ours = pd.DataFrame([dict(tracker="a", seq="OVERALL", MOTA=69.5, IDF1=65.2, IDSW=13, FP=100, FN=201)])
    cc = crosscheck_table(te, ours).iloc[0]
    assert abs(cc["dMOTA"] + 0.5) < 1e-9 and cc["dIDSW"] == 1 and cc["dFN"] == 1


def test_wrapper_plumbing_with_standin_trackeval():
    """Run scripts/run_trackeval.py end to end against a stand-in `trackeval` module that checks
    the folder layout and returns TrackEval-shaped results."""
    seen = {}

    class Evaluator:
        @staticmethod
        def get_default_eval_config():
            return {}

        def __init__(self, cfg):
            seen["eval_cfg"] = cfg

        def evaluate(self, datasets, metrics):
            cfg = datasets[0].cfg
            seen["ds_cfg"] = cfg
            for tr in cfg["TRACKERS_TO_EVAL"]:
                for seq in cfg["SEQ_INFO"]:
                    assert (Path(cfg["TRACKERS_FOLDER"]) / tr / "data" / f"{seq}.txt").exists()
                    assert (Path(cfg["GT_FOLDER"]) / seq / "gt" / "gt.txt").exists()
            return fake_result(cfg["TRACKERS_TO_EVAL"], list(cfg["SEQ_INFO"])), ""

    class DS:
        @staticmethod
        def get_default_dataset_config():
            return {}

        def __init__(self, cfg):
            self.cfg = cfg

    stub = types.ModuleType("trackeval")
    stub.Evaluator = Evaluator
    stub.datasets = types.SimpleNamespace(MotChallenge2DBox=DS)
    stub.metrics = types.SimpleNamespace(HOTA=lambda: 0, CLEAR=lambda: 0, Identity=lambda: 0, Count=lambda: 0)

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        seq_dir = t / "data" / "MOT17" / "train" / "MOT17-95-TEST"
        (seq_dir / "gt").mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text(
            "[Sequence]\nname=MOT17-95-TEST\nimDir=img1\nframeRate=30\nseqLength=5\nimWidth=640\nimHeight=480\nimExt=.jpg\n")
        gt = [f"{f},1,{10 + f},20,40,100,1,1,1.0" for f in range(1, 6)]
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
        trk = t / "res" / "mytracker"
        trk.mkdir(parents=True)
        (trk / "MOT17-95-TEST.txt").write_text(
            "\n".join(f"{f},7,{10 + f},20,40,100,0.9,-1,-1,-1" for f in range(1, 6)) + "\n")

        sys.modules["trackeval"] = stub
        import scripts.run_trackeval as rt
        old_argv = sys.argv
        sys.argv = ["run_trackeval.py", "--seq-dir", str(seq_dir), "--track-dir", str(trk),
                    "--out-dir", str(t / "out"), "--tag", "unit"]
        try:
            rt.main()
        finally:
            sys.argv = old_argv
            del sys.modules["trackeval"]

        df = pd.read_csv(t / "out" / "metrics_unit.csv")
        assert list(df["seq"]) == ["MOT17-95-TEST", "OVERALL"]
        assert (t / "out" / "metrics_unit.md").exists()
        cc = pd.read_csv(t / "out" / "trackeval" / "crosscheck_unit.csv")
        assert len(cc) == 2, "cross-check against our evaluator should have both rows"
        assert seen["ds_cfg"]["SKIP_SPLIT_FOL"] is True and seen["ds_cfg"]["SEQ_INFO"] == {"MOT17-95-TEST": 5}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
