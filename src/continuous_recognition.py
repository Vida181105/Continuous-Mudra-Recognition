"""
Continuous Recognition Module for Bharatanatyam Mudra Recognition System.

Performs real-time-style mudra recognition on continuous video by:
1. Sliding-window keypoint extraction
2. Per-window model inference
3. Temporal smoothing (majority voting)
4. Segment extraction with timestamps
5. Annotated video generation
6. JSON reporting
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict, Optional


class ContinuousRecognizer:
    """
    Processes a continuous video stream and detects mudra gestures
    using a trained Attention-LSTM model with sliding-window inference.
    """

    def __init__(self,
                 model_path: str,
                 mudra_names: List[str],
                 config: dict):
        """
        Initialise the recogniser.

        Args:
            model_path: Path to the saved ``.h5`` model.
            mudra_names: Ordered list of mudra class names.
            config: Full pipeline configuration dictionary.
        """
        self.mudra_names = list(mudra_names)
        self.config = config

        rec = config['recognition']
        self.sequence_length: int = config['data']['sequence_length']
        self.stride: int = rec['stride']
        self.confidence_threshold: float = rec['confidence_threshold']
        self.smoothing_window: int = rec['smoothing_window']
        self.min_segment_duration: float = rec['min_segment_duration']

        # Load the trained model with custom attention layer
        from src.model_architecture import TemporalAttention
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'TemporalAttention': TemporalAttention},
            compile=False,
        )
        # Re-compile so predict() works without warnings
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Keypoint extraction & normalisation (reuse data-preparation code)
        from src.data_preparation import KeypointExtractor, KeypointNormalizer
        self.extractor = KeypointExtractor(static_image_mode=False)
        self.normalizer = KeypointNormalizer()

        # Distinct colours per mudra (BGR for OpenCV)
        self._bgr_colors = self._make_color_map()

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def _make_color_map(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate distinct BGR colours for each mudra + 'Unknown'."""
        palette = [
            (0, 200, 0),      # green
            (0, 165, 255),    # orange
            (255, 191, 0),    # deep-sky-blue
            (255, 0, 255),    # magenta
            (0, 255, 255),    # yellow
            (60, 60, 255),    # red
            (200, 200, 0),    # teal
            (180, 100, 230),  # pink
        ]
        cmap: Dict[str, Tuple[int, int, int]] = {}
        for i, name in enumerate(self.mudra_names):
            cmap[name] = palette[i % len(palette)]
        cmap['Unknown'] = (128, 128, 128)
        return cmap

    @staticmethod
    def _confidence_color(confidence: float) -> Tuple[int, int, int]:
        """Return a BGR colour based on confidence level."""
        if confidence >= 0.7:
            return (0, 230, 0)       # green
        elif confidence >= 0.5:
            return (0, 230, 230)     # yellow
        else:
            return (0, 0, 230)       # red

    # ------------------------------------------------------------------
    # Per-window prediction
    # ------------------------------------------------------------------

    def predict_sequence(self,
                         sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict the mudra for a single normalised window.

        Args:
            sequence: np.ndarray of shape (sequence_length, 126).

        Returns:
            (mudra_name, confidence).  Returns ``('Unknown', conf)`` if
            the confidence is below the threshold.
        """
        inp = np.expand_dims(sequence, axis=0).astype(np.float32)
        probs = self.model.predict(inp, verbose=0)[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        if confidence < self.confidence_threshold:
            return 'Unknown', confidence
        return self.mudra_names[pred_class], confidence

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_predictions(
            buffer: deque) -> Tuple[str, float]:
        """
        Majority-vote smoothing over a prediction buffer.

        Args:
            buffer: deque of ``(mudra_name, confidence)`` tuples.

        Returns:
            (smoothed_mudra_name, average_confidence_of_winner).
        """
        if not buffer:
            return 'Unknown', 0.0

        votes: Dict[str, Dict] = {}
        for name, conf in buffer:
            if name not in votes:
                votes[name] = {'count': 0, 'total_conf': 0.0}
            votes[name]['count'] += 1
            votes[name]['total_conf'] += conf

        best = max(votes, key=lambda k: votes[k]['count'])
        avg_conf = votes[best]['total_conf'] / votes[best]['count']
        return best, avg_conf

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def draw_prediction(self, frame: np.ndarray,
                        mudra_name: str,
                        confidence: float) -> np.ndarray:
        """
        Overlay the current prediction on a video frame.

        Draws a semi-transparent header bar with:
        - mudra name
        - colour-coded confidence bar & percentage
        """
        h, w = frame.shape[:2]

        # Semi-transparent dark bar at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        colour = self._confidence_color(confidence)

        # Mudra name
        cv2.putText(frame, f"Mudra: {mudra_name}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, colour, 2, cv2.LINE_AA)

        # Confidence bar background
        bar_x, bar_y, bar_w, bar_h = 12, 48, 200, 16
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        # Filled portion
        fill_w = int(bar_w * min(confidence, 1.0))
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h),
                      colour, -1)
        # Border
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (200, 200, 200), 1)
        # Percentage text
        cv2.putText(frame, f"{confidence:.0%}",
                    (bar_x + bar_w + 8, bar_y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Video-level processing
    # ------------------------------------------------------------------

    def process_video(self, video_path: str,
                      output_dir: str) -> Dict:
        """
        Run continuous mudra recognition on a single video.

        Generates:
        - An annotated ``.mp4`` video with overlay labels.
        - A JSON report with segments and statistics.

        Args:
            video_path: Path to the input video.
            output_dir: Directory where outputs will be saved.

        Returns:
            Report dictionary (same content as the saved JSON).
        """
        video_name = Path(video_path).stem
        os.makedirs(output_dir, exist_ok=True)

        # Open source video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        print(f"\nProcessing: {os.path.basename(video_path)} "
              f"({total_frames} frames, {fps:.0f} FPS)")

        # Output annotated video
        out_video_path = os.path.join(output_dir,
                                      f"{video_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # Sliding-window state
        keypoint_buffer: List[np.ndarray] = []
        prediction_buffer: deque = deque(maxlen=self.smoothing_window)
        frame_predictions: List[Tuple[str, float]] = []

        current_pred: Tuple[str, float] = ('Unknown', 0.0)
        predict_counter = 0
        frame_idx = 0
        progress_step = max(total_frames // 20, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract keypoints for this frame
            kp = self.extractor.extract(frame)
            keypoint_buffer.append(kp)

            # Once we have enough frames, start predicting
            if len(keypoint_buffer) >= self.sequence_length:
                if predict_counter % self.stride == 0:
                    window = np.array(
                        keypoint_buffer[-self.sequence_length:])
                    window = self.normalizer.normalize(window)
                    pred_name, pred_conf = self.predict_sequence(window)
                    prediction_buffer.append((pred_name, pred_conf))
                    current_pred = self.smooth_predictions(
                        prediction_buffer)
                predict_counter += 1

                # Trim buffer to limit memory usage
                if len(keypoint_buffer) > self.sequence_length * 3:
                    keypoint_buffer = keypoint_buffer[
                        -self.sequence_length:]

            # Record frame-level prediction
            frame_predictions.append(current_pred)

            # Annotate & write
            annotated = self.draw_prediction(
                frame, current_pred[0], current_pred[1])
            out.write(annotated)

            frame_idx += 1
            if frame_idx % progress_step == 0:
                pct = frame_idx / max(total_frames, 1)
                bar = '=' * int(40 * pct)
                print(f"\r  [{bar:<40s}] {pct:.0%}", end='', flush=True)

        # Final progress line
        print(f"\r  [{'=' * 40}] 100%")

        cap.release()
        out.release()

        # ----- Post-processing -------------------------------------------
        segments = self._extract_segments(frame_predictions, fps)

        report = self._build_report(
            video_path, segments, frame_predictions,
            total_frames, fps, duration)

        report_path = os.path.join(output_dir, f"{video_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Detected {len(segments)} mudra segments")
        print(f"  ✓ Annotated video: {out_video_path}")
        print(f"  ✓ Report: {report_path}")

        return report

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _extract_segments(self,
                          frame_preds: List[Tuple[str, float]],
                          fps: float) -> List[Dict]:
        """
        Group consecutive same-label frames into segments.

        Filters out:
        - 'Unknown' segments
        - Segments shorter than ``min_segment_duration``
        """
        if not frame_preds:
            return []

        segments: List[Dict] = []
        cur_name = frame_preds[0][0]
        start_frame = 0
        confs: List[float] = [frame_preds[0][1]]

        for i in range(1, len(frame_preds)):
            name, conf = frame_preds[i]
            if name != cur_name:
                self._maybe_add_segment(
                    segments, cur_name, start_frame, i - 1, confs, fps)
                cur_name = name
                start_frame = i
                confs = [conf]
            else:
                confs.append(conf)

        # Last segment
        self._maybe_add_segment(
            segments, cur_name, start_frame,
            len(frame_preds) - 1, confs, fps)

        return segments

    def _maybe_add_segment(self,
                           segments: List[Dict],
                           mudra: str,
                           start_frame: int,
                           end_frame: int,
                           confs: List[float],
                           fps: float) -> None:
        """Append a segment if it passes duration & name filters."""
        dur = (end_frame - start_frame + 1) / fps
        if dur < self.min_segment_duration:
            return
        if mudra == 'Unknown':
            return
        segments.append({
            'segment_id': len(segments) + 1,
            'mudra': mudra,
            'start_time': f"{start_frame / fps:.2f}s",
            'end_time': f"{(end_frame + 1) / fps:.2f}s",
            'duration': f"{dur:.2f}s",
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'average_confidence': round(float(np.mean(confs)), 3),
        })

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report(video_path: str,
                      segments: List[Dict],
                      frame_preds: List[Tuple[str, float]],
                      total_frames: int,
                      fps: float,
                      duration: float) -> Dict:
        """Compile a detailed JSON-serialisable report."""
        mudra_stats: Dict[str, Dict] = {}
        for seg in segments:
            name = seg['mudra']
            dur = float(seg['duration'].replace('s', ''))
            if name not in mudra_stats:
                mudra_stats[name] = {'count': 0, 'total_duration': 0.0}
            mudra_stats[name]['count'] += 1
            mudra_stats[name]['total_duration'] += dur
        for name in mudra_stats:
            mudra_stats[name]['total_duration'] = (
                f"{mudra_stats[name]['total_duration']:.1f}s")

        avg_conf = (float(np.mean([s['average_confidence']
                                   for s in segments]))
                    if segments else 0.0)

        return {
            'video_info': {
                'filename': os.path.basename(video_path),
                'duration': f"{duration:.1f}s",
                'fps': fps,
                'total_frames': total_frames,
            },
            'detection_summary': {
                'total_segments': len(segments),
                'unique_mudras': len(mudra_stats),
                'average_confidence': round(avg_conf, 3),
            },
            'segments': segments,
            'mudra_statistics': mudra_stats,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release held resources."""
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
