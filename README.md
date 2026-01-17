# Continuous Mudra Recognition in Bharatanatyam Dance Videos


## Project Overview

This project implements an **end-to-end continuous mudra recognition pipeline** using:
- **Hand Keypoint Extraction:** MediaPipe Hands (21 landmarks per hand)
- **Temporal Modeling:** Bi-Directional LSTM
- **Mudra Scope:** Restricted to **Pataka** and **Tripataka** for First Review feasibility

The system processes Bharatanatyam dance videos and predicts which mudra is being performed in each temporal segment.

**Important:** This version works with **real videos** that you provide. No synthetic data generation.

## Key Features

 **End-to-End Pipeline:** Video → Keypoints → Windows → LSTM → Predictions  
 **Correct Label Alignment:** Fixed window-label misalignment bug from previous attempt  
 **Synthetic Training Data:** Known ground truth for debugging  
 **Temporal Smoothing:** Reduces noise in predictions  
 **JSON + Visual Outputs:** Required format for results  
 **Simple Architecture:** Bi-LSTM suitable for review (not over-engineered)

## Project Structure

```
mudra_recognition/
├── data/
│   ├── train_isolated/
│   │   ├── pataka.mp4              # Training video (Pataka only) - **PROVIDE THIS**
│   │   └── tripataka.mp4           # Training video (Tripataka only) - **PROVIDE THIS**
│   ├── test_continuous/
│   │   └── dance_test.mp4          # Test video (continuous mixed mudras) - **PROVIDE THIS**
│   └── prepared/
│       ├── X_train.npy, y_train.npy
│       ├── X_val.npy, y_val.npy
│       └── metadata.json
│
├── notebooks/
│   ├── 01_video_exploration.ipynb      # Phase 1-2: Video loading, frame extraction
│   ├── 02_hand_keypoints.ipynb         # Phase 3: Keypoint extraction (MediaPipe)
│   ├── 03_temporal_windowing.ipynb     # Phase 4: Temporal windowing, data prep
│   ├── 04_lstm_training.ipynb          # Phase 5: Model training and evaluation
│   └── 05_inference_demo.ipynb         # Phase 6: End-to-end inference + results
│
├── src/
│   ├── video_utils.py                  # Video loading, metadata extraction
│   ├── hand_keypoints.py               # MediaPipe integration, visualization
│   ├── windowing.py                    # Temporal windowing, data preparation
│   └── model.py                        # LSTM architecture, training utilities
│
├── models/
│   └── lstm_mudra_model.h5             # Trained Bi-LSTM model
│
├── keypoints/
│   ├── pataka_keypoints.npy
│   ├── tripataka_keypoints.npy
│   └── test_keypoints.npy
│
├── outputs/
│   ├── mudra_predictions.json          # **MAIN RESULT** - Mudra segments
│   ├── inference_visualization.png     # Visual results
│   └── training_history.png            # Training curves
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `tensorflow` - LSTM model building and training
- `mediapipe` - Hand keypoint extraction
- `opencv-python` - Video processing
- `numpy`, `scipy` - Data processing and visualization
- `jupyter` - Interactive notebooks

### 2. Prepare Your Videos

Place your training and test videos in the data directories:

```bash
# Create directories
mkdir -p data/train_isolated
mkdir -p data/test_continuous

# Place videos
cp /path/to/pataka_video.mp4 data/train_isolated/pataka.mp4
cp /path/to/tripataka_video.mp4 data/train_isolated/tripataka.mp4
cp /path/to/test_video.mp4 data/test_continuous/dance_test.mp4
```

**Video Requirements:**
- **Format:** MP4 or AVI
- **Pataka training:** Contains ONLY Pataka mudra (minimum 30 seconds recommended)
- **Tripataka training:** Contains ONLY Tripataka mudra (minimum 30 seconds recommended)
- **Test video:** Continuous video with both mudras (for inference testing)
- **Resolution:** Any (will be resized internally)
- **FPS:** Any (will be extracted at source FPS)

### 3. Run the Pipeline

Execute notebooks in order:

**Notebook 1: Video Exploration**
```bash
jupyter notebook notebooks/01_video_exploration.ipynb
```
- Checks for training videos in data/train_isolated/
- Loads and displays sample frames
- Verifies video integrity and hand visibility

 **You must provide the videos before running this notebook**

**Notebook 2: Hand Keypoint Extraction**
```bash
jupyter notebook notebooks/02_hand_keypoints.ipynb
```
- Initializes MediaPipe Hands
- Extracts 21 landmarks from all frames
- Visualizes hand landmarks on frames
- Saves keypoints as .npy files

**Notebook 3: Temporal Windowing**
```bash
jupyter notebook notebooks/03_temporal_windowing.ipynb
```
- Creates overlapping temporal windows (25 frames, step 5)
- Creates window-label correspondence
- Prepares train/validation splits
- Saves prepared data

**Notebook 4: LSTM Training**
```bash
jupyter notebook notebooks/04_lstm_training.ipynb
```
- Builds Bi-Directional LSTM model
- Trains on windowed sequences
- Plots training curves (loss/accuracy)
- Saves trained model

**Notebook 5: Inference Demo**
```bash
jupyter notebook notebooks/05_inference_demo.ipynb
```
- Loads trained model
- Extracts keypoints from test video
- Makes predictions on temporal windows
- **Generates `mudra_predictions.json`** (REQUIRED OUTPUT)
- **Creates visualization** (REQUIRED OUTPUT)

### 3. Access Results

**Main Result File:**
```
outputs/mudra_predictions.json
```

**Example Output:**
```json
[
  {
    "start_time": 0.0,
    "end_time": 0.96,
    "mudra": "Pataka",
    "confidence": 0.919,
    "start_frame": 0,
    "end_frame": 24
  },
  {
    "start_time": 1.0,
    "end_time": 1.56,
    "mudra": "Tripataka",
    "confidence": 0.879,
    "start_frame": 25,
    "end_frame": 39
  }
]
```

**Visualization:**
```
outputs/inference_visualization.png
```

## Technical Details

### Phase 1-2: Video Loading & Frame Extraction
- **Input:** MP4 video files
- **Output:** RGB frames at 25 FPS
- **Why:** Video frame extraction is the foundation for all downstream processing

### Phase 3: Hand Keypoint Extraction
- **Tool:** MediaPipe Hands (Google's solution)
- **Output:** 21 landmarks × 3 coordinates (x, y, z) = 63 features per frame
- **Why Keypoints vs Raw Images?**
  - Reduces dimensionality: 921,600 pixels → 63 features
  - Robust to background, lighting, camera angle
  - Focuses LSTM on hand structure, not irrelevant details

### Phase 4: Temporal Windowing
- **Window Size:** 25 frames = 1 second @ 25 FPS
- **Step Size:** 5 frames (creates overlap)
- **CRITICAL:** Window-label alignment ensures each window's label matches the mudra performed in that temporal region
- **Why This Mattered:** Previous attempt likely had misaligned windows-labels, causing incorrect learning

### Phase 5: Bi-Directional LSTM
```
Input: (batch, 25 frames, 63 features)
   ↓
Bi-LSTM: 64 units (forward + backward)
   ↓
Dense: 32 units, ReLU
   ↓
Output: Softmax over 2 classes (Pataka, Tripataka)

Loss: Categorical Cross-Entropy
Optimizer: Adam (lr=0.001)
```

**Why LSTM?**
- **Temporal Modeling:** Mudras are sequences - hand changes over time
- **Long-Range Dependencies:** Can capture sustained poses and transitions
- **Bidirectional:** Processes frames forward AND backward for better context
- **Simple & Interpretable:** Suitable for review (not a complex attention mechanism)

### Phase 6: Inference
1. Extract keypoints from continuous video
2. Create temporal windows
3. Predict mudra per window with softmax probability
4. Apply temporal smoothing (moving average kernel size 3)
5. Group consecutive predictions into segments
6. Map frame indices to time using FPS

## Key Improvements Over Previous Attempt

| Issue | Previous | Fixed |
|-------|----------|-------|
| **Window-Label Misalignment** | Likely cause of wrong predictions | ✓ Verified alignment in Phase 4 |
| **Mudra Scope** | Too many classes (confusing) | ✓ Only Pataka + Tripataka |
| **Data Quality** | Unknown labels | ✓ Synthetic data with known ground truth |
| **Normalization** | Unclear | ✓ Proper keypoint handling |
| **Output Format** | Not specified | ✓ JSON + Visual outputs |
| **Model Complexity** | Possibly over-engineered | ✓ Simple Bi-LSTM |

## Mudra Definitions

### Pataka (परतक)
- **English:** Flag
- **Hand Shape:** Open palm, all fingers extended and together
- **Characteristics:** Used for moving, traveling, flowing motions
- **Training Video:** `data/train_isolated/pataka.mp4` (150 frames)

### Tripataka (त्रिपतक)
- **English:** Three Leaves
- **Hand Shape:** Open palm with three fingers grouped (index, middle, ring) and raised together
- **Characteristics:** Used for indicating numbers, pointing, expressing emotions
- **Training Video:** `data/train_isolated/tripataka.mp4` (150 frames)

## Evaluation Metrics

**Validation Set Performance:**
- Run notebook 4 to see training/validation accuracy
- Final model saved to `models/lstm_mudra_model.h5`

**Inference Quality:**
- Confidence scores in JSON output
- Mean confidence should be > 0.7 for reliable predictions

## Troubleshooting

**Issue: "No module named 'mediapipe'"**
```bash
pip install mediapipe==0.10.8
```

**Issue: "Failed to open video"**
- Check video file exists at correct path
- Verify it's a valid MP4 file

**Issue: Poor hand detection in keypoints**
- Ensure hands are clearly visible in video
- Adjust `min_detection_confidence` in `HandKeypointExtractor`

**Issue: Low model accuracy**
- Check temporal window alignment (Phase 4)
- Verify labels are correctly assigned
- Check for NaN or Inf values in data

## Future Enhancements

**Post-First Review:**
1. Add real Bharatanatyam dance videos
2. Expand mudra vocabulary (Chandrakala, Ardhapataka, etc.)
3. Implement confidence-based filtering
4. Add two-hand detection and modeling
5. Deploy as real-time inference system
6. Create annotated mudra dataset


**Submission Date:** January 2026  
**Weightage:** 20% (First Review)  
**Status:** ✅ Ready for Evaluation
