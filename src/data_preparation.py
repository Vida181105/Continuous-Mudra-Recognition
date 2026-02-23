"""
Data Preparation Module for Bharatanatyam Mudra Recognition System.

Handles:
- Two-hand keypoint extraction using MediaPipe Hands (126 features)
- Wrist-relative normalization with scale invariance
- Realistic data augmentation (rotation, scaling, time-warp, jitter, dropout)
- Sliding-window sequence extraction
- Compressed dataset saving (.npz)
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict


# ===========================================================================
# Keypoint Extraction
# ===========================================================================

class KeypointExtractor:
    """
    Extracts two-hand keypoints from video frames using MediaPipe Hands.

    Each frame produces a 126-dimensional feature vector:
        [right_hand (63 values) | left_hand (63 values)]
    where each hand has 21 landmarks × 3 coordinates (x, y, z).
    """

    N_LANDMARKS: int = 21
    N_COORDS: int = 3
    N_HANDS: int = 2
    N_FEATURES: int = N_HANDS * N_LANDMARKS * N_COORDS  # 126

    def __init__(self,
                 static_image_mode: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialise MediaPipe Hands.

        Args:
            static_image_mode: True treats every frame independently (slower
                but more accurate for unrelated images).  False enables
                inter-frame tracking for video.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract two-hand keypoints from a single BGR frame.

        Right-hand landmarks go to indices [0:63],
        left-hand landmarks go to indices [63:126].
        If a hand is absent, its section is filled with zeros.

        Args:
            frame: BGR image (H, W, 3) from OpenCV.

        Returns:
            np.ndarray of shape (126,).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        keypoints = np.zeros(self.N_FEATURES, dtype=np.float64)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness):
                label = handedness.classification[0].label
                # Right hand → first 63, Left hand → last 63
                offset = 0 if label == 'Right' else 63

                for i, lm in enumerate(hand_landmarks.landmark):
                    keypoints[offset + i * 3]     = lm.x
                    keypoints[offset + i * 3 + 1] = lm.y
                    keypoints[offset + i * 3 + 2] = lm.z

        return keypoints

    # -- Context-manager support ------------------------------------------
    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ===========================================================================
# Keypoint Normalization
# ===========================================================================

class KeypointNormalizer:
    """
    Normalises hand keypoints for translation and scale invariance.

    For each hand independently:
    1. Subtract the wrist position (landmark 0) → translation invariance
    2. Divide by the maximum distance from wrist → scale invariance
    """

    @staticmethod
    def normalize(sequence: np.ndarray) -> np.ndarray:
        """
        Normalise a sequence of keypoints in-place style (returns copy).

        Args:
            sequence: np.ndarray of shape (T, 126).

        Returns:
            Normalised sequence of the same shape.
        """
        normalised = sequence.copy()

        for t in range(len(normalised)):
            for hand_offset in (0, 63):
                hand_kp = normalised[t, hand_offset:hand_offset + 63]

                # Skip hands that were not detected (all zeros)
                if not np.any(hand_kp):
                    continue

                # Wrist position (landmark 0)
                wrist = hand_kp[:3].copy()

                # Translate so wrist is at origin
                for i in range(21):
                    hand_kp[i * 3:i * 3 + 3] -= wrist

                # Scale so maximum landmark distance from wrist == 1.0
                coords = hand_kp.reshape(21, 3)
                max_dist = np.max(np.linalg.norm(coords, axis=1))
                if max_dist > 1e-6:
                    hand_kp /= max_dist

                normalised[t, hand_offset:hand_offset + 63] = hand_kp

        return normalised


# ===========================================================================
# Sequence Augmentation
# ===========================================================================

class SequenceAugmenter:
    """
    Generates augmented copies of a keypoint sequence by combining:
    - Random rotation (±15° in the image plane)
    - Random scaling   (0.9× – 1.1×)
    - Time warping     (0.8× – 1.2× speed)
    - Gaussian jitter  (σ = 0.02)
    - Frame dropout    (10 % probability, replaced by neighbour average)
    """

    def __init__(self, num_augmentations: int = 5):
        """
        Args:
            num_augmentations: How many augmented copies to generate per
                original sequence.
        """
        self.num_augmentations = num_augmentations

    def augment(self, sequence: np.ndarray) -> List[np.ndarray]:
        """
        Produce *num_augmentations* augmented versions of *sequence*.

        Each augmented version has ALL five transformations applied
        sequentially, each with random parameters.

        Args:
            sequence: np.ndarray of shape (T, 126).

        Returns:
            List of augmented np.ndarrays, each with shape (T, 126).
        """
        results: List[np.ndarray] = []
        for _ in range(self.num_augmentations):
            aug = sequence.copy()
            aug = self._random_rotation(aug)
            aug = self._random_scaling(aug)
            aug = self._time_warp(aug)
            aug = self._gaussian_jitter(aug)
            aug = self._frame_dropout(aug)
            results.append(aug)
        return results

    # ----- Individual augmentation transforms ----------------------------

    @staticmethod
    def _random_rotation(seq: np.ndarray,
                         max_angle_deg: float = 15.0) -> np.ndarray:
        """Rotate (x, y) coordinates in the image plane by a random angle."""
        angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        result = seq.copy()

        for t in range(len(result)):
            for hand_offset in (0, 63):
                hand_kp = result[t, hand_offset:hand_offset + 63]
                if not np.any(hand_kp):
                    continue
                for i in range(21):
                    x = hand_kp[i * 3]
                    y = hand_kp[i * 3 + 1]
                    hand_kp[i * 3]     = x * cos_a - y * sin_a
                    hand_kp[i * 3 + 1] = x * sin_a + y * cos_a
        return result

    @staticmethod
    def _random_scaling(seq: np.ndarray,
                        lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
        """Scale all non-zero keypoint values by a random factor."""
        scale = np.random.uniform(lo, hi)
        result = seq.copy()
        mask = result != 0
        result[mask] *= scale
        return result

    @staticmethod
    def _time_warp(seq: np.ndarray,
                   lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
        """
        Temporal warping: resample the sequence at a randomly altered speed.

        The output always has the same length as the input.
        """
        n_frames, n_features = seq.shape
        warp_factor = np.random.uniform(lo, hi)
        new_n = max(int(n_frames * warp_factor), 2)

        # Step 1 – resample to *new_n* frames
        orig_idx = np.arange(n_frames)
        warp_idx = np.linspace(0, n_frames - 1, new_n)
        warped = np.zeros((new_n, n_features))
        for feat in range(n_features):
            warped[:, feat] = np.interp(warp_idx, orig_idx, seq[:, feat])

        # Step 2 – resample back to the original length
        result_idx = np.linspace(0, new_n - 1, n_frames)
        result = np.zeros_like(seq)
        for feat in range(n_features):
            result[:, feat] = np.interp(result_idx, np.arange(new_n),
                                        warped[:, feat])
        return result

    @staticmethod
    def _gaussian_jitter(seq: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        """Add Gaussian noise only to non-zero keypoint values."""
        noise = np.random.normal(0, sigma, seq.shape)
        result = seq.copy()
        mask = result != 0
        result[mask] += noise[mask]
        return result

    @staticmethod
    def _frame_dropout(seq: np.ndarray,
                       drop_prob: float = 0.1) -> np.ndarray:
        """
        Randomly drop frames (replace with average of neighbours).

        First and last frames are never dropped.
        """
        result = seq.copy()
        for t in range(1, len(result) - 1):
            if np.random.random() < drop_prob:
                result[t] = (result[t - 1] + result[t + 1]) / 2.0
        return result


# ===========================================================================
# Main Data Preparation Class
# ===========================================================================

class DataPreparation:
    """
    Orchestrates full dataset preparation:
    scan videos → extract keypoints → normalise → window → augment → save.
    """

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, config: dict):
        """
        Args:
            config: Full pipeline configuration dictionary.
        """
        self.config = config
        self.sequence_length: int = config['data']['sequence_length']
        self.stride: int = config['data'].get('train_stride',
                                              self.sequence_length // 2)
        self.augmentation_factor: int = config['data']['augmentation_factor']

        self.extractor = KeypointExtractor(static_image_mode=False)
        self.normalizer = KeypointNormalizer()
        self.augmenter = SequenceAugmenter(self.augmentation_factor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_videos(self, folder: str) -> List[str]:
        """Return sorted list of video file paths in *folder*."""
        videos: List[str] = []
        for fname in sorted(os.listdir(folder)):
            _, ext = os.path.splitext(fname)
            if ext.lower() in self.VIDEO_EXTENSIONS:
                videos.append(os.path.join(folder, fname))
        return videos

    def _find_mudra_folders(self, train_dir: str) -> List[str]:
        """Return sorted mudra folder names, excluding cache/hidden dirs."""
        return sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
            and not d.startswith('cached')
            and not d.startswith('.')
        ])

    # ------------------------------------------------------------------
    # Video Processing
    # ------------------------------------------------------------------

    def extract_keypoints_from_video(self, video_path: str) -> np.ndarray:
        """
        Read all frames of a video and return normalised keypoints.

        Args:
            video_path: Path to the video file.

        Returns:
            np.ndarray of shape (T, 126) where T = number of frames.

        Raises:
            IOError: If the video cannot be opened.
            ValueError: If no frames are read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        keypoints_list: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            keypoints_list.append(self.extractor.extract(frame))

        cap.release()

        if len(keypoints_list) == 0:
            raise ValueError(f"No frames read from: {video_path}")

        sequence = np.array(keypoints_list)
        sequence = self.normalizer.normalize(sequence)
        return sequence

    # ------------------------------------------------------------------
    # Sliding Window
    # ------------------------------------------------------------------

    def create_sliding_windows(self, sequence: np.ndarray) -> List[np.ndarray]:
        """
        Create overlapping sliding windows of length *sequence_length*.

        If the video is shorter than one window it is zero-padded.

        Args:
            sequence: np.ndarray of shape (T, 126).

        Returns:
            List of np.ndarrays, each of shape (sequence_length, 126).
        """
        n_frames = len(sequence)
        windows: List[np.ndarray] = []

        if n_frames < self.sequence_length:
            padded = np.zeros((self.sequence_length, sequence.shape[1]))
            padded[:n_frames] = sequence
            windows.append(padded)
        else:
            for start in range(0, n_frames - self.sequence_length + 1,
                               self.stride):
                windows.append(
                    sequence[start:start + self.sequence_length].copy())

            # Make sure we cover the very last frames
            last_start = n_frames - self.sequence_length
            if not windows or last_start > (len(windows) - 1) * self.stride:
                windows.append(sequence[last_start:].copy())

        return windows

    # ------------------------------------------------------------------
    # Full Dataset Preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, train_dir: str,
                        output_file: str) -> Dict:
        """
        Process all training videos and create the augmented dataset.

        Steps for each video:
        1. Extract keypoints from every frame
        2. Normalise keypoints (wrist-relative + scale)
        3. Create sliding windows
        4. Augment each window 5× (plus keep original)
        5. Accumulate into arrays
        6. Shuffle and save as compressed .npz

        Args:
            train_dir: Path to ``data/train_isolated/``.
            output_file: Path for the output ``.npz`` file.

        Returns:
            Dictionary with dataset statistics.
        """
        mudra_folders = self._find_mudra_folders(train_dir)
        print(f"Found {len(mudra_folders)} mudras: "
              f"{', '.join(mudra_folders)}")

        all_sequences: List[np.ndarray] = []
        all_labels: List[int] = []
        stats: Dict[str, int] = {}

        for label_idx, mudra_name in enumerate(mudra_folders):
            mudra_dir = os.path.join(train_dir, mudra_name)
            videos = self._find_videos(mudra_dir)
            print(f"\nProcessing {mudra_name} ({len(videos)} videos)...")

            mudra_total = 0

            for video_path in videos:
                video_name = os.path.basename(video_path)
                try:
                    # Extract & normalise
                    sequence = self.extract_keypoints_from_video(video_path)

                    # Sliding windows
                    windows = self.create_sliding_windows(sequence)
                    base_count = len(windows)

                    # Augment: original + augmented copies
                    augmented_windows: List[np.ndarray] = []
                    for window in windows:
                        augmented_windows.append(window)          # original
                        augmented_windows.extend(
                            self.augmenter.augment(window))       # augmented

                    total_count = len(augmented_windows)

                    for w in augmented_windows:
                        all_sequences.append(w)
                        all_labels.append(label_idx)

                    mudra_total += total_count
                    print(f"  ✓ {video_name} → "
                          f"{base_count} base sequences → "
                          f"{total_count} augmented")

                except Exception as exc:
                    print(f"  ✗ {video_name} → Error: {exc}")

            stats[mudra_name] = mudra_total

        # Convert to arrays
        sequences_array = np.array(all_sequences, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)

        # Shuffle
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(sequences_array))
        sequences_array = sequences_array[indices]
        labels_array = labels_array[indices]

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez_compressed(
            output_file,
            sequences=sequences_array,
            labels=labels_array,
            mudra_names=np.array(mudra_folders),
        )

        # Summary
        print(f"\n✅ Dataset saved: {output_file}")
        print(f"   Total sequences: {len(sequences_array):,}")
        print(f"   Shape: {sequences_array.shape}")
        class_counts = [int(np.sum(labels_array == i))
                        for i in range(len(mudra_folders))]
        print(f"   Class distribution: {class_counts}")

        return {
            'total_sequences': len(sequences_array),
            'shape': tuple(sequences_array.shape),
            'mudra_names': mudra_folders,
            'class_distribution': stats,
            'class_counts': class_counts,
        }

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release held resources."""
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
