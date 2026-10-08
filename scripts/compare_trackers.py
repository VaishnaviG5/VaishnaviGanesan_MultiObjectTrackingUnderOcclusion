"""Summarize and compare tracking benchmarks across SORT and DeepSORT variants.

Usage:
    python scripts/compare_trackers.py
"""
import glob
from pathlib import Path
import pandas as pd


def main():
    eval_files = {
        "SORT (Baseline)": "results/eval/sort_motmetrics.csv",
        "DeepSORT (deep-sort-realtime)": "results/eval/deepsort_realtime_motmetrics.csv",
        "DeepSORT (Custom Hist)": "results/eval/deepsort_hist_motmetrics.csv",
    }

    rows = []
    for tracker_name, path in eval_files.items():
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        overall = df.loc["OVERALL"]
        rows.append({
            "Tracker": tracker_name,
            "MOTA (%)": overall["MOTA(%)"],
            "IDF1 (%)": overall["IDF1(%)"],
            "IDSW": int(overall["IDSW"]),
            "FP": int(overall["FP"]),
            "FN": int(overall["FN"]),
            "Frag": int(overall["Frag"]),
            "Prec (%)": overall["Prec(%)"],
            "Rec (%)": overall["Rec(%)"],
        })

    if not rows:
        print("No evaluation CSVs found in results/eval/.")
        return

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("MULTI-OBJECT TRACKING BENCHMARK COMPARISON (MOT17-02 & MOT17-04)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    summary_path = Path("results/benchmark_comparison.md")
    summary_path.write_text(
        "# Benchmark Comparison\n\n" + summary_df.to_markdown(index=False) + "\n"
    )
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
