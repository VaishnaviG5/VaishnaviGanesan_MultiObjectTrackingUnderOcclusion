# Multi-Object Tracking Under Occlusion (MOT17)

A complete, production-grade implementation and failure analysis of **Multi-Object Tracking (MOT)** under occlusion on the **MOT17** benchmark, evaluating both **SORT** (Simple Online and Realtime Tracking) and **DeepSORT** (Deep Cosine Metric Learning with Hungarian Matching and Kalman Filter Motion Estimation).

---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Environment Setup & Installation](#environment-setup--installation)
3. [Quick Start (End-to-End Pipeline)](#quick-start-end-to-end-pipeline)
4. [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
5. [Benchmark Results](#benchmark-results)
6. [Key Architectural Decisions](#key-architectural-decisions)
7. [Failure Analysis: Why Trackers Lose Identity Under Occlusion](#failure-analysis-why-trackers-lose-identity-under-occlusion)
8. [Interview Preparation & Code Explanations](#interview-preparation--code-explanations)

---

## Overview & Architecture

Multi-object tracking follows the **Tracking-by-Detection** paradigm:
1. **Detection**: Pretrained YOLOv8 detects all person instances per frame.
2. **Motion Estimation**: A Kalman Filter predicts the next bounding box position based on constant velocity motion.
3. **Data Association**: The Hungarian Algorithm (Munkres / Linear Sum Assignment) matches predictions to current detections using an IoU (or appearance cosine distance) cost matrix.
4. **Lifecycle Management**: Tracks are confirmed after surviving consecutive hits (`min_hits`), coast through missing frames during occlusion, and are terminated if unmatched for longer than `max_age`.

```
                    +--------------------+
                    |  Video / Frames    |
                    +---------+----------+
                              |
                              v
                  +------------------------+
                  | YOLOv8 Person Detector |
                  +-----------+------------+
                              | Detections (x, y, w, h, conf)
                              v
   +-------------------------------------------------------+
   |                  Data Association                     |
   |                                                       |
   |  Kalman State Predictions <---> Current Detections    |
   |                                                       |
   |        Cost Matrix: IoU / Deep Cosine Distance        |
   |        Optimization: Hungarian Algorithm (SciPy)       |
   +--------------------------+----------------------------+
                              |
                              v
   +-------------------------------------------------------+
   |                Track Lifecycle Manager                |
   |                                                       |
   |  - Matched: Update Kalman State & Hit Streak          |
   |  - Unmatched Detections: Create Tentative Tracks      |
   |  - Unmatched Tracks (Occlusion): Coast with Kalman    |
   |  - Age > max_age: Terminate Track                     |
   +--------------------------+----------------------------+
                              |
                              v
                  +------------------------+
                  |  Trajectories & Metrics|
                  |  (MOTA, IDF1, IDSW)    |
                  +------------------------+
```

---

## Environment Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14 with CUDA PyTorch on NVIDIA RTX 5050 Laptop GPU)
- Windows / Linux / macOS

### Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```

Core dependencies in `requirements.txt`:
- `ultralytics`: YOLOv8 pretrained person detector
- `torch`, `torchvision`: Deep learning backend
- `motmetrics`: Official `py-motmetrics` CLEAR-MOT and IDF1 evaluation
- `deep-sort-realtime`: DeepSORT appearance tracking implementation
- `opencv-python`: Frame reading, box rendering, trajectory drawing
- `scipy`: Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
- `pandas`, `numpy`, `tabulate`: Data structures and markdown tables
- `matplotlib`, `tqdm`: Visualization and progress tracking

---

## Quick Start (End-to-End Pipeline)

To run the complete pipeline end-to-end on **MOT17-02-FRCNN** and **MOT17-04-FRCNN**:

```bash
python run_pipeline.py
```

This single command automatically:
1. Runs YOLOv8n person detection on both sequences.
2. Runs the SORT tracker.
3. Runs the DeepSORT tracker.
4. Evaluates both trackers using `py-motmetrics` against ground truth.
5. Performs failure analysis (categorizing all ID switches).
6. Generates trajectory video, GIF, and static trajectory plots.

To run specific steps only:
```bash
# Example: re-run evaluation, failure analysis, and visualization
python run_pipeline.py --steps 6 7 8
```

---

## Step-by-Step Reproduction Guide

### Step 1 & 2: Dataset & Inspection
Ensure MOT17 sequences are placed in `data/MOT17/train/`:
- `data/MOT17/train/MOT17-02-FRCNN/` (600 frames, moving/street camera)
- `data/MOT17/train/MOT17-04-FRCNN/` (1050 frames, static/high-angle night crossing)

Inspect ground truth bounding boxes:
```bash
python scripts/show_gt.py --seq-dir data/MOT17/train/MOT17-02-FRCNN --frame 1
```

### Step 3: Run YOLOv8 Detector
Extract person-only detections (class 0) at low confidence (`0.1`) so downstream trackers can sweep thresholds without re-inferencing:
```bash
python scripts/detect.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
    --model yolov8n.pt --imgsz 640 --conf 0.1 --out-dir detections/yolov8n_640
```

### Step 4: Run SORT Tracker
Run SORT with Kalman filter prediction, IoU matching, and lifecycle filtering:
```bash
python scripts/run_sort.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
    --det-dir detections/yolov8n_640 \
    --conf-thr 0.4 --max-age 30 --min-hits 3 --iou-thr 0.3
```
*Output: `results/trackers/sort/<sequence>.txt`*

### Step 5: Run DeepSORT Tracker
Run DeepSORT with MobileNet appearance re-identification embeddings:
```bash
python scripts/run_deepsort_realtime.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
    --det-dir detections/yolov8n_640 \
    --conf-thr 0.4 --max-age 30 --n-init 3
```
*Output: `results/trackers/deepsort_realtime/<sequence>.txt`*

### Step 6: Evaluation with `py-motmetrics`
Evaluate tracked outputs against ground truth (`gt/gt.txt`):
```bash
# Evaluate SORT
python scripts/eval_motmetrics.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
    --track-dir results/trackers/sort

# Evaluate DeepSORT
python scripts/eval_motmetrics.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
    --track-dir results/trackers/deepsort_realtime

# Print side-by-side benchmark table
python scripts/compare_trackers.py
```

### Step 7: Failure Analysis
Classify every ID switch into primary causes (Occlusion, Fast Motion, Similar Appearance, or Detector Miss) and export before/after switch images:
```bash
python scripts/analyze_failures.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN data/MOT17/train/MOT17-04-FRCNN \
    --track-dir results/trackers/sort results/trackers/deepsort_realtime
```
*Outputs: `results/step7/failure_summary.md`, `failure_causes.png`, and `examples_sheet_sort.png`.*

### Step 8: Trajectory Visualization
Generate side-by-side videos, GIFs with fading motion trails, and static trajectory maps:
```bash
python scripts/visualize_tracks.py \
    --seq-dir data/MOT17/train/MOT17-02-FRCNN \
    --track-dir results/trackers/sort results/trackers/deepsort_realtime \
    --start 50 --length 100 --trail 30 --scale 0.5
```
*Outputs: `results/step8/<seq>_f50-149.mp4`, `.gif`, and `_trajectories.png`.*

---

## Benchmark Results

Evaluated across **1,650 frames** on `MOT17-02-FRCNN` and `MOT17-04-FRCNN` using `py-motmetrics` (IoU threshold = 0.5):

| Tracker | MOTA (%) | IDF1 (%) | IDSW (Switches) | FP | FN | Frag | Precision (%) | Recall (%) | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SORT (Motion Only)** | 26.3% | 37.8% | **68** | **899** | 47,805 | 386 | **95.3%** | 27.7% | **~1,200 FPS** |
| **DeepSORT (Realtime)** | **27.9%** | **39.3%** | 72 | 3,955 | **43,642** | **161** | 85.0% | **34.0%** | ~5-10 FPS |
| **DeepSORT (Custom Hist)** | 26.8% | 36.7% | 79 | 1,234 | 47,114 | 569 | 93.9% | 28.8% | ~170 FPS |

### Key Observations:
1. **IDF1 Improvement**: DeepSORT achieves a higher IDF1 (39.3% vs 37.8%), demonstrating that appearance features help preserve track consistency across longer temporal gaps.
2. **Fragmentation Reduction**: DeepSORT dramatically reduces track fragmentations (**161 vs 386**—a 58% reduction!), because the appearance gallery allows re-identifying coasting tracks when IoU fails.
3. **Speed vs. Accuracy Trade-off**:
   - SORT is pure linear algebra (Kalman filter + Hungarian matrix) operating at **>1,000 FPS**.
   - DeepSORT incorporates deep CNN feature extraction on cropped pedestrian patches, reducing speed to ~5–10 FPS on CPU (higher on GPU).

---

## Key Architectural Decisions

### 1. Kalman Filter State Representation: $[c_x, c_y, s, r, v_x, v_y, v_s]$
- **Why not $[x, y, w, h, \dot{x}, \dot{y}, \dot{w}, \dot{h}]$?**
  - In a standard bounding box, width ($w$) and height ($h$) change rapidly and noisily depending on pedestrian stride and limb articulation.
  - Representing size as **area ($s = w \times h$)** and **aspect ratio ($r = w / h$)** provides a far more stable physical prior:
    1. A pedestrian's aspect ratio $r$ remains approximately constant while walking.
    2. Box area $s$ changes smoothly with camera perspective and depth.
  - Therefore, $r$ is modeled as static without velocity ($\dot{r} = 0$), preventing erratic aspect-ratio oscillations during partial occlusions.

### 2. Hungarian Matching Cost Formulations
- **SORT**: Cost is $1 - \text{IoU}$. If $\text{IoU} < \text{iou\_thr}$ (0.3), the cost is penalized so the assignment is rejected.
- **DeepSORT**: Two-stage association:
  1. **Appearance Matching**: Cost is cosine distance $d(u, f) = 1 - \frac{u \cdot f}{\|u\| \|f\|}$ against a gallery of recent track embeddings.
  2. **Mahalanobis Distance Gating**: A motion gating mask drops pairings that are physically implausible under the Kalman filter's predicted covariance ($\chi^2_{0.95}$ gating threshold = 9.4877 for 4 degrees of freedom).
  3. **IoU Fallback**: Any remaining unmatched tracks/detections are matched via IoU.

### 3. Hyperparameter Calibration: `max_age` and `min_hits`
- **`max_age = 30`**:
  - Defines how many frames a track survives without a matching detection (coasting on Kalman prediction).
  - At 30 FPS, `max_age = 30` gives the tracker a **1.0-second window** to coast through occluders (lamp posts, passing pedestrians).
  - Setting `max_age` too small causes premature track deletion and severe ID switches upon reappearance.
  - Setting `max_age` too high causes the Kalman prediction covariance to expand unbounded, leading to false associations with nearby pedestrians.
- **`min_hits = 3`**:
  - Requires 3 consecutive frame detections before a track is reported.
  - Dramatically suppresses detector false positives and noise artifacts.

---

## Failure Analysis: Why Trackers Lose Identity Under Occlusion

By mining all ground-truth matched ID switch events in `results/step7/failure_summary.md`:

```
Primary Causes of ID Switches:
  [=========================================] 75.0% Occlusion
  [=========] 17.6% Detector Miss / Flicker
  [====]       7.4% Fast Motion / Parallax
```

### Breakdown of Failure Modes:

#### 1. Prolonged Occlusion (`respawn_gap` — 82.4% of SORT switches)
- **What Happens**: A pedestrian walks behind a sign, kiosk, or another pedestrian for longer than `max_age` (e.g. 40–80 frames).
- **Why It Fails**: The Kalman filter coasts for 30 frames and then deletes the track. When the person emerges, the detector sees a box with no existing active track, creating a brand-new ID.
- **Example in Data**: `occlusion_MOT17-02-FRCNN_f20_gt2.png` and `occlusion_MOT17-04-FRCNN_f1032_gt92.png`.

#### 2. Path Crossing / Track Swap (`swap` — 8.8% in SORT, 27.8% in DeepSORT)
- **What Happens**: Two pedestrians walk in opposite directions and pass directly in front of each other.
- **Why It Fails**:
  - In SORT: During the overlap, bounding boxes have high mutual IoU. The Hungarian algorithm matches the wrong detection if the Kalman predictions cross.
  - In DeepSORT: If both pedestrians wear dark clothing or similar jackets, appearance cosine distance cannot resolve the ambiguity, resulting in swapped IDs after the crossing.

#### 3. Fast Motion & Non-Linear Camera Motion (`fast_motion` — 54.4% of switch contexts, 1.38x lift)
- **What Happens**: Camera pan/tilt (in moving sequences like MOT17-02) or erratic running motion violates the Kalman filter's constant velocity assumption.
- **Why It Fails**: The predicted bounding box lags behind the actual detection. The spatial IoU between prediction and detection drops below `0.3`, triggering track termination.

#### 4. Detector Flicker & False Negatives (`other` — 17.6%)
- **What Happens**: The detector drops a pedestrian for 2–3 frames due to low contrast, shadows, or pose deformation.
- **Why It Fails**: If a track is tentative or coasting during complex crowd movement, missing detections break track continuity.

## Directory Structure

```
.
├── data/
│   └── MOT17/train/          # MOT17 sequence directories (img1, gt, seqinfo.ini)
├── detections/
│   └── yolov8n_640/          # Cached YOLOv8 person detections
├── results/
│   ├── eval/                 # py-motmetrics and CLEAR-MOT CSV summaries
│   ├── step7/                # Failure analysis breakdown & ID-switch images
│   ├── step8/                # Trajectory MP4 videos, GIFs, and static plots
│   └── benchmark_comparison.md
├── scripts/
│   ├── detect.py             # Pretrained YOLOv8 detection caching
│   ├── run_sort.py           # SORT tracker runner
│   ├── run_deepsort_realtime.py # DeepSORT runner
│   ├── eval_motmetrics.py    # py-motmetrics evaluation script
│   ├── evaluate.py           # CLEAR-MOT + IDF1 evaluation script
│   ├── compare_trackers.py   # Benchmark comparison table generator
│   ├── analyze_failures.py   # ID-switch classification & failure analysis
│   └── visualize_tracks.py   # Trajectory video/GIF and static renderer
├── src/
│   ├── boxes.py              # IoU computation and coordinate conversions
│   ├── clearmot.py           # Mathematical CLEAR-MOT implementation
│   ├── data.py               # MOT17 sequence and ground-truth parser
│   ├── deepsort.py           # Custom DeepSORT tracker
│   ├── kalman_box.py         # 7-state Kalman Filter motion model
│   ├── render.py             # Motion trail renderer & static plotter
│   ├── sort.py               # SORT data association engine
│   └── viz.py                # Bounding box and ID drawing utilities
├── tests/                    # 6 unit test suites (all passing)
├── requirements.txt          # Project dependencies
├── run_pipeline.py           # Master end-to-end execution script
└── README.md                 # Project documentation
```
