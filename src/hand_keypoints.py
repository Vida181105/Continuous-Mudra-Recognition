"""
Hand Keypoint Extraction using MediaPipe Hands

Handles:
- Initialize MediaPipe Hands model
- Extract 21 landmarks per hand
- Handle missing detections (zero-padding)
- Normalize keypoints
- Visualize landmarks on frames
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


class HandKeypointExtractor:
    """Extract hand keypoints using MediaPipe Hands."""
    
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        """
        Initialize MediaPipe Hands.
        
        Args:
            static_image_mode (bool): If False, optimized for video
            max_num_hands (int): Maximum number of hands to detect (use 1 for single hand)
            min_detection_confidence (float): Minimum confidence for detection
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        
        # 21 landmarks per hand
        self.num_landmarks = 21
        
    def extract_keypoints(self, frame):
        """
        Extract hand keypoints from a single frame.
        
        Args:
            frame (np.ndarray): Frame in RGB format (H, W, 3)
            
        Returns:
            keypoints (np.ndarray): Shape (21, 3) - x, y, z coordinates
                                    Returns zeros if no hand detected
            confidence (float): Detection confidence (0 if not detected)
            handedness (str): "Right" or "Left" (or None)
        """
        h, w, _ = frame.shape
        
        results = self.hands.process(frame)
        
        # Initialize with zeros (no hand detected)
        keypoints = np.zeros((self.num_landmarks, 3), dtype=np.float32)
        confidence = 0.0
        handedness = None
        
        if results.multi_hand_landmarks:
            # Use first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract x, y, z
            for i, landmark in enumerate(hand_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y, landmark.z]
            
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
                confidence = results.multi_handedness[0].classification[0].score
        
        return keypoints, confidence, handedness
    
    def extract_keypoints_batch(self, frames):
        """
        Extract keypoints from multiple frames.
        
        Args:
            frames (list): List of frames (RGB format)
            
        Returns:
            keypoints_sequence (np.ndarray): Shape (num_frames, 21, 3)
            confidences (list): Detection confidence per frame
            handedness_list (list): Handedness per frame
        """
        num_frames = len(frames)
        keypoints_sequence = np.zeros((num_frames, self.num_landmarks, 3), dtype=np.float32)
        confidences = []
        handedness_list = []
        
        for i, frame in enumerate(frames):
            keypoints, conf, handedness = self.extract_keypoints(frame)
            keypoints_sequence[i] = keypoints
            confidences.append(conf)
            handedness_list.append(handedness)
        
        return keypoints_sequence, confidences, handedness_list
    
    def normalize_keypoints(self, keypoints_sequence):
        """
        Normalize keypoints to zero-mean and unit variance.
        
        Args:
            keypoints_sequence (np.ndarray): Shape (num_frames, 21, 3)
            
        Returns:
            normalized (np.ndarray): Normalized keypoints (same shape)
            mean (np.ndarray): Mean values for inverse normalization
            std (np.ndarray): Std values for inverse normalization
        """
        # Flatten to (num_frames * 21, 3)
        num_frames = keypoints_sequence.shape[0]
        flat = keypoints_sequence.reshape(-1, 3)
        
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized = (flat - mean) / std
        normalized = normalized.reshape(num_frames, 21, 3)
        
        return normalized, mean, std
    
    def draw_landmarks(self, frame, keypoints, confidence=None, handedness=None):
        """
        Draw hand landmarks on frame.
        
        Args:
            frame (np.ndarray): Frame in RGB format
            keypoints (np.ndarray): Shape (21, 3) - hand landmarks
            confidence (float): Optional detection confidence
            handedness (str): Optional handedness label
            
        Returns:
            frame_with_landmarks (np.ndarray): Frame with drawn landmarks
        """
        frame_copy = frame.copy()
        h, w, _ = frame_copy.shape
        
        # Draw circles at each landmark
        for i, (x, y, z) in enumerate(keypoints):
            # Check if landmark is valid (not zero-padded)
            if not (x == 0 and y == 0 and z == 0):
                px = int(x * w)
                py = int(y * h)
                cv2.circle(frame_copy, (px, py), 3, (0, 255, 0), -1)
        
        # Draw connections between landmarks (hand skeleton)
        # MediaPipe hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        for start, end in connections:
            x1, y1, _ = keypoints[start]
            x2, y2, _ = keypoints[end]
            
            if not (x1 == 0 and y1 == 0) and not (x2 == 0 and y2 == 0):
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                cv2.line(frame_copy, (px1, py1), (px2, py2), (255, 0, 0), 2)
        
        # Add text annotations
        if handedness:
            cv2.putText(frame_copy, f"Hand: {handedness}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if confidence:
            cv2.putText(frame_copy, f"Conf: {confidence:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def draw_landmarks_batch(self, frames, keypoints_sequence, confidences=None):
        """
        Draw landmarks on multiple frames.
        
        Args:
            frames (list): List of frames
            keypoints_sequence (np.ndarray): Shape (num_frames, 21, 3)
            confidences (list): Optional confidence values
            
        Returns:
            frames_with_landmarks (list): Frames with drawn landmarks
        """
        frames_with_landmarks = []
        
        for i, frame in enumerate(frames):
            conf = confidences[i] if confidences else None
            frame_viz = self.draw_landmarks(frame, keypoints_sequence[i], confidence=conf)
            frames_with_landmarks.append(frame_viz)
        
        return frames_with_landmarks
    
    def save_keypoints(self, keypoints_sequence, output_path):
        """
        Save keypoints to .npy file.
        
        Args:
            keypoints_sequence (np.ndarray): Shape (num_frames, 21, 3)
            output_path (str): Path to save .npy file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(str(output_path), keypoints_sequence)
        print(f"Saved keypoints to {output_path}")
    
    def load_keypoints(self, keypoints_path):
        """
        Load keypoints from .npy file.
        
        Args:
            keypoints_path (str): Path to .npy file
            
        Returns:
            keypoints_sequence (np.ndarray): Shape (num_frames, 21, 3)
        """
        keypoints = np.load(str(keypoints_path))
        return keypoints
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
