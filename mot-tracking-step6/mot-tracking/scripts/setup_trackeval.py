"""Step 6 setup: download the official TrackEval and make it work with a modern NumPy.

Usage:
    python scripts/setup_trackeval.py

Needs git and an internet connection. Result: third_party/TrackEval/
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trackeval_utils import patch_numpy_aliases  # noqa: E402

DEST = Path("third_party/TrackEval")
URL = "https://github.com/JonathonLuiten/TrackEval.git"


def main():
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, str(DEST)], check=True)
    else:
        print(f"{DEST} already exists, skipping clone")
    n = patch_numpy_aliases(DEST / "trackeval")
    print(f"Patched deprecated NumPy aliases (np.float, np.int, ...) in {n} files.")
    req = DEST / "minimum_requirements.txt"
    print(f"\nNext: pip install -r {req}   (if that file is missing: pip install numpy scipy)")
    print("Then: python scripts/run_trackeval.py --seq-dir ... --track-dir results/trackers/sort")


if __name__ == "__main__":
    main()
