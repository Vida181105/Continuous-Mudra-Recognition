#!/usr/bin/env python3
"""
Bharatanatyam Mudra Recognition Pipeline – Main Entry Point.

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --config custom_config.json

Steps executed:
    1. Data Preparation   – extract keypoints, augment, save dataset
    2. Model Training     – build & train Attention-BiLSTM model
    3. Model Evaluation   – confusion matrix, classification report
    4. Continuous Recognition – annotate test videos, generate reports
"""

import os
import sys
import time
import json
import argparse
import warnings

# Suppress noisy TensorFlow / protobuf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np

# Ensure imports resolve from project root regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ===========================================================================
# Pipeline
# ===========================================================================

class MudraRecognitionPipeline:
    """Orchestrates the complete mudra recognition workflow."""

    def __init__(self, config_path: str = 'config.json'):
        from src.utils import load_config, ensure_directories
        self.config = load_config(config_path)
        ensure_directories(self.config)

    # ------------------------------------------------------------------
    # STEP 1 – Data Preparation
    # ------------------------------------------------------------------

    def step1_data_preparation(self) -> dict:
        """
        Scan training videos, extract two-hand keypoints via MediaPipe,
        create sliding windows, augment 5×, and save a compressed .npz
        dataset.
        """
        from src.data_preparation import DataPreparation

        train_dir   = self.config['paths']['train_dir']
        output_file = self.config['paths']['prepared_data']

        print(f"Scanning: {train_dir}/")

        with DataPreparation(self.config) as prep:
            stats = prep.prepare_dataset(train_dir, output_file)

        return stats

    # ------------------------------------------------------------------
    # STEP 2 – Model Training
    # ------------------------------------------------------------------

    def step2_train_model(self) -> None:
        """
        Load the prepared dataset, split train/val, build an
        Attention-BiLSTM model, train with callbacks, and save.
        """
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from src.model_architecture import (
            build_attention_lstm_model, ModelTrainer, save_model_diagram,
        )
        from src.utils import (
            plot_training_curves, plot_confusion_matrix,
            save_classification_report, save_training_log,
        )

        # Load dataset
        data_path = self.config['paths']['prepared_data']
        data = np.load(data_path, allow_pickle=True)
        sequences = data['sequences']
        labels    = data['labels']
        mudra_names = list(data['mudra_names'])

        n_classes = len(mudra_names)
        test_size = self.config['data']['train_test_split']

        print(f"Building Attention-LSTM model...")

        # Stratified train / validation split
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels,
            test_size=test_size,
            stratify=labels,
            random_state=42,
        )

        # Build model (auto-selects full / lightweight)
        model = build_attention_lstm_model(
            sequence_length=self.config['data']['sequence_length'],
            n_features=sequences.shape[2],
            n_classes=n_classes,
            model_type=self.config['model']['type'],
            n_sequences=len(X_train),
            config=self.config['model'],
        )

        model.summary(print_fn=lambda line: None)  # suppress duplicate

        # Save model architecture diagram (best effort)
        save_model_diagram(
            model, os.path.join('models', 'model_architecture.png'))

        # Train
        trainer = ModelTrainer(model, self.config)
        model_save_path = self.config['paths']['model_save']
        log_dir = 'logs'

        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            model_save_path=model_save_path,
            log_dir=log_dir,
        )

        # Also save model explicitly after training (best-weights restored)
        try:
            model.save(model_save_path, save_format='h5')
        except Exception:
            model.save(model_save_path)

        # ---- Visualisations --------------------------------------------
        results_dir = os.path.join(
            self.config['paths']['results_dir'], 'training')

        plot_training_curves(
            history,
            os.path.join(results_dir, 'training_curves.png'))

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

        plot_confusion_matrix(
            y_val, y_pred, mudra_names,
            os.path.join(results_dir, 'confusion_matrix.png'))

        report_txt = save_classification_report(
            y_val, y_pred, mudra_names,
            os.path.join(results_dir, 'classification_report.txt'))

        save_training_log(
            history,
            os.path.join(results_dir, 'training_log.txt'))

        best_val = max(history.history['val_accuracy'])
        print(f"\n✅ Training complete! Best val_accuracy: {best_val:.3f}")
        print(f"✅ Model saved: {model_save_path}")
        print(f"✅ Training curves: {results_dir}/training_curves.png")
        print(f"✅ Confusion matrix: {results_dir}/confusion_matrix.png")

        # Store for Step 3
        self._mudra_names = mudra_names
        self._X_val = X_val
        self._y_val = y_val
        self._y_pred = y_pred
        self._report_txt = report_txt

    # ------------------------------------------------------------------
    # STEP 3 – Evaluation Report
    # ------------------------------------------------------------------

    def step3_evaluate(self) -> None:
        """Print the classification report to the console."""
        print("Classification Report:")
        print(self._report_txt)

        acc = np.mean(self._y_val == self._y_pred)
        print(f"Overall Accuracy: {acc:.1%}")

    # ------------------------------------------------------------------
    # STEP 4 – Continuous Recognition on Test Videos
    # ------------------------------------------------------------------

    def step4_continuous_recognition(self) -> None:
        """
        For each video in ``data/test_continuous/``:
        - Run sliding-window inference
        - Generate an annotated video
        - Save a JSON report
        - Create a combined timeline visualisation
        """
        from src.continuous_recognition import ContinuousRecognizer
        from src.utils import plot_timeline

        test_dir   = self.config['paths']['test_dir']
        model_path = self.config['paths']['model_save']
        output_dir = os.path.join(
            self.config['paths']['results_dir'], 'continuous_recognition')

        # Discover mudra names from the saved dataset
        if not hasattr(self, '_mudra_names'):
            data = np.load(
                self.config['paths']['prepared_data'], allow_pickle=True)
            self._mudra_names = list(data['mudra_names'])

        print(f"Processing test videos from: {test_dir}/")

        # Find test videos
        test_videos = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ])

        if not test_videos:
            print("  ⚠ No test videos found – skipping.")
            return

        reports: list = []

        with ContinuousRecognizer(
                model_path, self._mudra_names, self.config) as recognizer:
            for vpath in test_videos:
                try:
                    report = recognizer.process_video(vpath, output_dir)
                    reports.append(report)
                except Exception as exc:
                    print(f"  ✗ {os.path.basename(vpath)} → Error: {exc}")

        # Timeline visualisation
        if reports and self.config['visualization'].get(
                'generate_timeline', True):
            timeline_path = os.path.join(
                output_dir, 'timeline_visualization.png')
            plot_timeline(reports, self._mudra_names, timeline_path)
            print(f"\n✅ Timeline: {timeline_path}")

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_complete_pipeline(self) -> None:
        """Execute all four pipeline steps in sequence."""

        t_start = time.time()

        print("=" * 60)
        print("  MUDRA RECOGNITION PIPELINE")
        print("=" * 60)

        # ---- Step 1 ---------------------------------------------------
        print("\n[1/4] Data Preparation...")
        print("-" * 60)
        try:
            self.step1_data_preparation()
        except Exception as exc:
            print(f"\n❌ Data Preparation failed: {exc}")
            raise

        # ---- Step 2 ---------------------------------------------------
        print("\n[2/4] Model Training...")
        print("-" * 60)
        try:
            self.step2_train_model()
        except Exception as exc:
            print(f"\n❌ Model Training failed: {exc}")
            raise

        # ---- Step 3 ---------------------------------------------------
        print("\n[3/4] Model Evaluation...")
        print("-" * 60)
        try:
            self.step3_evaluate()
        except Exception as exc:
            print(f"\n❌ Evaluation failed: {exc}")
            raise

        # ---- Step 4 ---------------------------------------------------
        print("\n[4/4] Continuous Recognition...")
        print("-" * 60)
        try:
            self.step4_continuous_recognition()
        except Exception as exc:
            print(f"\n❌ Continuous Recognition failed: {exc}")
            raise

        # ---- Summary ---------------------------------------------------
        elapsed = time.time() - t_start
        mins, secs = divmod(int(elapsed), 60)

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE!")
        print(f"  Total time: {mins}m {secs}s")
        print("=" * 60)

        self._print_output_summary()

    # ------------------------------------------------------------------

    def _print_output_summary(self) -> None:
        """List all generated files."""
        print("\nGenerated Files:")

        dirs_to_scan = [
            'models',
            os.path.join(self.config['paths']['results_dir'], 'training'),
            os.path.join(self.config['paths']['results_dir'],
                         'continuous_recognition'),
        ]
        for d in dirs_to_scan:
            if not os.path.isdir(d):
                continue
            files = sorted(os.listdir(d))
            if not files:
                continue
            print(f"\n  {d}/")
            for f in files:
                if f.startswith('.'):
                    continue
                fpath = os.path.join(d, f)
                size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                if size > 1_048_576:
                    size_str = f"{size / 1_048_576:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"    {f:40s}  {size_str}")

        print("\nNext Steps:")
        print("  1. Review training curves in results/training/")
        print("  2. Watch annotated videos in results/continuous_recognition/")
        print("  3. Check JSON reports for detailed metrics")
        print("  4. Use models/best_model.h5 for new predictions")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bharatanatyam Mudra Recognition Pipeline')
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='Path to configuration JSON file (default: config.json)')
    args = parser.parse_args()

    pipeline = MudraRecognitionPipeline(config_path=args.config)
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()
