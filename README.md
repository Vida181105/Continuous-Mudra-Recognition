# Bharatanatyam Mudra Recognition System

A complete pipeline for recognising Bharatanatyam hand gestures (mudras) in continuous video using an **Attention-BiLSTM** deep learning model.

## Supported Mudras

| # | Mudra | Description |
|---|-------|-------------|
| 1 | Chandrakala | Crescent moon gesture |
| 2 | Hamsaaya | Swan beak gesture |
| 3 | Pataka | Flag gesture |
| 4 | Shikara | Peak/spire gesture |
| 5 | Tripataka | Three-part flag gesture |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_installation.py
```

### 3. Run the Full Pipeline

```bash
python run_pipeline.py
```

This single command executes all four stages:

| Step | Description | Time (approx.) |
|------|-------------|-----------------|
| 1/4 | Extract keypoints, augment, save dataset | 5–15 min |
| 2/4 | Build & train Attention-BiLSTM model | 30–90 min |
| 3/4 | Evaluate model, generate confusion matrix | < 1 min |
| 4/4 | Annotate test videos, create JSON reports | 5–15 min |

**Total estimated time on CPU: 40–120 minutes.**

## Project Structure

```
MUDRA_RECOGNITION/
├── src/
│   ├── data_preparation.py       # MediaPipe keypoint extraction & augmentation
│   ├── model_architecture.py     # Attention-BiLSTM model definition
│   ├── continuous_recognition.py # Sliding-window inference on videos
│   └── utils.py                  # Visualisation & helper functions
├── run_pipeline.py               # Main entry point (one command)
├── config.json                   # All configurable parameters
├── requirements.txt              # Python dependencies
├── test_installation.py          # Dependency checker
├── data/
│   ├── train_isolated/           # Training videos (by mudra folder)
│   ├── test_continuous/          # Continuous test videos
│   └── prepared/                 # (auto) Prepared .npz dataset
├── models/                       # (auto) Saved model & diagram
├── results/
│   ├── training/                 # (auto) Curves, confusion matrix, report
│   └── continuous_recognition/   # (auto) Annotated videos, JSON reports
└── logs/                         # (auto) Training log CSV
```

## Outputs

After the pipeline completes you will find:

### `results/training/`
- **training_curves.png** – Accuracy & loss over epochs
- **confusion_matrix.png** – Per-class heatmap
- **classification_report.txt** – Precision / recall / F1
- **training_log.txt** – Epoch-by-epoch metrics

### `results/continuous_recognition/`
- **\*_annotated.mp4** – Test videos with mudra labels overlaid
- **\*_report.json** – Detection segments, timestamps, confidences
- **timeline_visualization.png** – Colour-coded timeline across all test videos

### `models/`
- **best_model.h5** – Trained model (loadable for future predictions)
- **model_architecture.png** – Visual diagram (requires `graphviz`)

## Configuration

All parameters are in `config.json`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.sequence_length` | 30 | Frames per input window |
| `data.augmentation_factor` | 5 | Augmented copies per window |
| `model.type` | `"auto"` | `"full"`, `"lightweight"`, or `"auto"` |
| `training.epochs` | 100 | Maximum training epochs |
| `training.early_stopping_patience` | 20 | Stop if no improvement |
| `recognition.confidence_threshold` | 0.6 | Min confidence to label a mudra |
| `recognition.smoothing_window` | 7 | Majority-vote window size |

## Model Architecture

```
Input (batch, 30, 126)
  │
  ├─ Bidirectional LSTM (128 units, return_sequences)
  ├─ Batch Normalisation
  ├─ Bidirectional LSTM (64 units, return_sequences)
  ├─ Batch Normalisation
  ├─ Temporal Attention (64 units)   ← key innovation
  ├─ Dense (128, ReLU) + Dropout
  ├─ Dense (64, ReLU) + Dropout
  └─ Dense (5, Softmax)
```

## Expected Performance

With the limited training data (2–3 videos per mudra):
- **Training accuracy**: 80–95 %
- **Validation accuracy**: 55–75 %

Augmentation (5× per window) compensates for the small dataset.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| MediaPipe crash on macOS | Upgrade: `pip install --upgrade mediapipe` |
| TensorFlow Metal warnings | Set `TF_CPP_MIN_LOG_LEVEL=2` (already handled) |
| Low accuracy | Try increasing `augmentation_factor` to 8–10 |
| Video codec error | Ensure `ffmpeg` is installed: `brew install ffmpeg` |
| `graphviz` not found for diagram | `brew install graphviz && pip install pydot` (optional) |
| Out of memory during training | Reduce `batch_size` in config.json |

## Loading the Model Later

```python
import tensorflow as tf
from src.model_architecture import TemporalAttention

model = tf.keras.models.load_model(
    'models/best_model.h5',
    custom_objects={'TemporalAttention': TemporalAttention}
)
```

## License

This project was developed as a final-year academic project.
