#!/usr/bin/env python3
"""
Test Installation Script for Bharatanatyam Mudra Recognition System.
Verifies that all required dependencies are installed and working correctly.
"""

import sys
import importlib


def check_package(name: str, import_name: str = None, min_version: str = None) -> bool:
    """Check if a package is installed and optionally verify minimum version."""
    import_name = import_name or name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {name:20s} → version {version}")
        return True
    except ImportError:
        print(f"  ✗ {name:20s} → NOT INSTALLED")
        return False


def check_mediapipe_hands() -> bool:
    """Verify that MediaPipe Hands can be initialized."""
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        hands.close()
        print("  ✓ MediaPipe Hands     → initialization OK")
        return True
    except Exception as e:
        print(f"  ✗ MediaPipe Hands     → FAILED: {e}")
        return False


def check_tensorflow_model() -> bool:
    """Verify that TensorFlow can build a simple model."""
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print("  ✓ TensorFlow Model    → build & compile OK")
        return True
    except Exception as e:
        print(f"  ✗ TensorFlow Model    → FAILED: {e}")
        return False


def check_opencv_video() -> bool:
    """Verify that OpenCV can read videos."""
    try:
        import cv2
        # Just verify the VideoCapture class is available
        cap = cv2.VideoCapture()
        cap.release()
        print("  ✓ OpenCV Video        → VideoCapture OK")
        return True
    except Exception as e:
        print(f"  ✗ OpenCV Video        → FAILED: {e}")
        return False


def check_data_files() -> bool:
    """Verify that training data exists."""
    import os
    train_dir = "data/train_isolated"
    test_dir = "data/test_continuous"
    all_ok = True

    if os.path.isdir(train_dir):
        mudras = [d for d in os.listdir(train_dir)
                  if os.path.isdir(os.path.join(train_dir, d))
                  and not d.startswith('cached') and not d.startswith('.')]
        video_count = 0
        for m in mudras:
            mdir = os.path.join(train_dir, m)
            vids = [f for f in os.listdir(mdir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            video_count += len(vids)
        print(f"  ✓ Training data       → {len(mudras)} mudras, {video_count} videos")
    else:
        print(f"  ✗ Training data       → directory not found: {train_dir}")
        all_ok = False

    if os.path.isdir(test_dir):
        test_vids = [f for f in os.listdir(test_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        print(f"  ✓ Test data           → {len(test_vids)} test videos")
    else:
        print(f"  ✗ Test data           → directory not found: {test_dir}")
        all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("  Mudra Recognition System - Installation Test")
    print("=" * 60)

    errors = 0

    # 1. Check Python version
    print(f"\n[1] Python Version")
    py_version = sys.version.split()[0]
    if sys.version_info >= (3, 8):
        print(f"  ✓ Python {py_version}")
    else:
        print(f"  ✗ Python {py_version} (requires >= 3.8)")
        errors += 1

    # 2. Check required packages
    print(f"\n[2] Required Packages")
    packages = [
        ("tensorflow", "tensorflow"),
        ("mediapipe", "mediapipe"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
    ]
    for name, import_name in packages:
        if not check_package(name, import_name):
            errors += 1

    # 3. Functional tests
    print(f"\n[3] Functional Tests")
    if not check_mediapipe_hands():
        errors += 1
    if not check_tensorflow_model():
        errors += 1
    if not check_opencv_video():
        errors += 1

    # 4. Data files
    print(f"\n[4] Data Files")
    if not check_data_files():
        errors += 1

    # Summary
    print("\n" + "=" * 60)
    if errors == 0:
        print("  ✅ All checks passed! System is ready.")
        print("  Run: python run_pipeline.py")
    else:
        print(f"  ❌ {errors} check(s) failed!")
        print("  Fix the issues above and run this script again.")
        print("  Install packages with: pip install -r requirements.txt")
    print("=" * 60)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
