"""
Video Loading and Frame Extraction Utilities

Handles:
- Loading video files with OpenCV
- Extracting frames at fixed FPS
- Frame validation
- Video metadata extraction
"""

import cv2
import numpy as np
from pathlib import Path


def load_video(video_path, target_fps=25):
    """
    Load video and extract frames at target FPS.
    
    Args:
        video_path (str): Path to video file
        target_fps (int): Target FPS for frame extraction
        
    Returns:
        frames (list): List of numpy arrays (H, W, 3)
        actual_fps (float): Actual FPS of source video
        num_frames (int): Total number of frames
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {actual_fps}")
    print(f"  Total Frames: {total_frames}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    print(f"  Extracted: {len(frames)} frames\n")
    
    return frames, actual_fps, len(frames)


def display_frame(frame, title="Frame"):
    """
    Display a single frame (for debugging).
    
    Args:
        frame (np.ndarray): Frame to display (RGB)
        title (str): Window title
    """
    cv2.imshow(title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_video_metadata(video_path):
    """
    Get basic metadata about video without loading all frames.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        metadata (dict): Video properties
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return metadata


def frame_to_time(frame_index, fps):
    """
    Convert frame index to time in seconds.
    
    Args:
        frame_index (int): Frame index (0-based)
        fps (float): Frames per second
        
    Returns:
        time_seconds (float): Time in seconds
    """
    return frame_index / fps


def time_to_frame(time_seconds, fps):
    """
    Convert time in seconds to frame index.
    
    Args:
        time_seconds (float): Time in seconds
        fps (float): Frames per second
        
    Returns:
        frame_index (int): Frame index (0-based)
    """
    return int(time_seconds * fps)
