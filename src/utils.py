"""
Utility Functions for Bharatanatyam Mudra Recognition System.

Contains helper functions for:
- Configuration loading
- Directory management
- Visualization (training curves, confusion matrix, timeline)
- Logging and reporting
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = 'config.json') -> dict:
    """
    Load pipeline configuration from a JSON file.

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        Configuration dictionary.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_directories(config: dict) -> None:
    """
    Create all required output directories.

    Args:
        config: Pipeline configuration dictionary.
    """
    dirs = [
        'models',
        'logs',
        os.path.join(config['paths']['results_dir'], 'training'),
        os.path.join(config['paths']['results_dir'], 'continuous_recognition'),
        os.path.dirname(config['paths']['prepared_data']),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Training Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_path: str) -> None:
    """
    Plot training and validation accuracy / loss curves.

    Args:
        history: Keras training History object.
        output_path: File path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    ax1.plot(epochs, history.history['accuracy'],
             'b-o', markersize=3, label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history.history['val_accuracy'],
             'r-o', markersize=3, label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history.history['loss'],
             'b-o', markersize=3, label='Train Loss', linewidth=2)
    ax2.plot(epochs, history.history['val_loss'],
             'r-o', markersize=3, label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str], output_path: str) -> None:
    """
    Plot and save a confusion-matrix heat-map.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        output_path: File path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str],
                               output_path: str) -> str:
    """
    Generate and save a classification report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        output_path: File path to save the report.

    Returns:
        The classification report string.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0
    )

    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n")

    return report


# ---------------------------------------------------------------------------
# Timeline Visualization
# ---------------------------------------------------------------------------

# Distinct colours for up to 10 mudras
MUDRA_COLORS = [
    '#2ecc71',  # green
    '#e67e22',  # orange
    '#3498db',  # blue
    '#9b59b6',  # purple
    '#f1c40f',  # yellow
    '#e74c3c',  # red
    '#1abc9c',  # teal
    '#e84393',  # pink
    '#00cec9',  # cyan
    '#6c5ce7',  # indigo
]


def plot_timeline(reports: List[Dict], mudra_names: List[str],
                  output_path: str) -> None:
    """
    Create a colour-coded timeline visualisation of mudra detections
    across all test videos.

    Args:
        reports: List of report dicts (one per test video).
        mudra_names: Ordered list of mudra class names.
        output_path: File path to save the figure.
    """
    n_videos = len(reports)
    if n_videos == 0:
        return

    color_map = {name: MUDRA_COLORS[i % len(MUDRA_COLORS)]
                 for i, name in enumerate(mudra_names)}
    color_map['Unknown'] = '#95a5a6'

    fig, axes = plt.subplots(n_videos, 1,
                             figsize=(14, max(2.5 * n_videos, 4)),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (report, ax) in enumerate(zip(reports, axes)):
        filename = report['video_info']['filename']
        duration = float(str(report['video_info']['duration']).replace('s', ''))

        for segment in report.get('segments', []):
            start = float(str(segment['start_time']).replace('s', ''))
            end = float(str(segment['end_time']).replace('s', ''))
            mudra = segment['mudra']
            color = color_map.get(mudra, '#95a5a6')

            ax.barh(0, end - start, left=start, height=0.6,
                    color=color, edgecolor='white', linewidth=0.5)

            # Label if segment is wide enough
            if (end - start) > duration * 0.06:
                ax.text((start + end) / 2, 0, mudra,
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='black')

        ax.set_xlim(0, duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(filename, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=10)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=color_map.get(name, '#95a5a6'), label=name)
        for name in mudra_names
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               ncol=min(len(mudra_names), 5),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------------------------

def save_training_log(history, output_path: str) -> None:
    """
    Save epoch-by-epoch training log as a readable text file.

    Args:
        history: Keras training History object.
        output_path: File path to save the log.
    """
    with open(output_path, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
                f"{'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}\n")
        f.write("-" * 80 + "\n")

        epochs = len(history.history['loss'])
        lr_history = history.history.get('lr', [None] * epochs)

        for i in range(epochs):
            lr_str = f"{lr_history[i]:.2e}" if lr_history[i] is not None else "N/A"
            f.write(
                f"{i+1:>6}  "
                f"{history.history['loss'][i]:>11.4f}  "
                f"{history.history['accuracy'][i]:>10.4f}  "
                f"{history.history['val_loss'][i]:>10.4f}  "
                f"{history.history['val_accuracy'][i]:>9.4f}  "
                f"{lr_str:>10}\n"
            )

        f.write("-" * 80 + "\n")
        best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
        best_val_acc = max(history.history['val_accuracy'])
        f.write(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n")


def print_section(title: str, char: str = '-', width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")
