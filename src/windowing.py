"""
Temporal Windowing for Sequence Data

Handles:
- Create overlapping temporal windows from keypoint sequences
- Window validation
- Padding for edge cases
- Window-label alignment
"""

import numpy as np
from typing import Tuple, List


def create_temporal_windows(keypoints_sequence, window_size=25, step_size=5):
    """
    Create overlapping temporal windows from keypoint sequence.
    
    Args:
        keypoints_sequence (np.ndarray): Shape (num_frames, 21, 3)
        window_size (int): Size of each window in frames
        step_size (int): Step size for sliding window
        
    Returns:
        windows (np.ndarray): Shape (num_windows, window_size, 21, 3)
        window_indices (list): List of (start_frame, end_frame) for each window
    """
    num_frames = keypoints_sequence.shape[0]
    
    if num_frames < window_size:
        raise ValueError(f"Sequence length ({num_frames}) < window_size ({window_size})")
    
    windows = []
    window_indices = []
    
    # Create sliding windows
    for start_idx in range(0, num_frames - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        window = keypoints_sequence[start_idx:end_idx]
        windows.append(window)
        window_indices.append((start_idx, end_idx - 1))  # end_idx-1 is inclusive
    
    windows = np.array(windows)
    
    print(f"Created temporal windows:")
    print(f"  Sequence length: {num_frames}")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    print(f"  Number of windows: {len(windows)}")
    print(f"  Output shape: {windows.shape}\n")
    
    return windows, window_indices


def create_labeled_windows(keypoints_sequences_dict, window_size=25, step_size=5):
    """
    Create windows and labels from multiple labeled sequences.
    
    Args:
        keypoints_sequences_dict (dict): {label: list_of_keypoint_arrays}
                                        e.g., {'Pataka': [seq1, seq2, ...], 'Tripataka': [...]}
        window_size (int): Window size in frames
        step_size (int): Step size for sliding window
        
    Returns:
        windows (np.ndarray): Shape (total_windows, window_size, 21, 3)
        labels (np.ndarray): Shape (total_windows,) - label indices
        label_to_idx (dict): Mapping from label name to index
        idx_to_label (dict): Mapping from index to label name
        window_metadata (list): List of dicts with metadata for each window
    """
    windows_list = []
    labels_list = []
    label_to_idx = {}
    idx_to_label = {}
    window_metadata = []
    
    # Create label mappings
    for idx, label in enumerate(sorted(keypoints_sequences_dict.keys())):
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    
    print(f"Label mapping: {label_to_idx}\n")
    
    # Process each label
    for label, sequences in keypoints_sequences_dict.items():
        label_idx = label_to_idx[label]
        
        for seq_idx, keypoints_seq in enumerate(sequences):
            try:
                windows, window_indices = create_temporal_windows(
                    keypoints_seq, window_size=window_size, step_size=step_size
                )
                
                windows_list.append(windows)
                
                # Create labels for all windows from this sequence
                for win_idx, (start_frame, end_frame) in enumerate(window_indices):
                    labels_list.append(label_idx)
                    window_metadata.append({
                        'label': label,
                        'label_idx': label_idx,
                        'sequence_idx': seq_idx,
                        'window_idx': win_idx,
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })
                
            except ValueError as e:
                print(f"Warning: Could not process sequence {seq_idx} for label {label}: {e}")
                continue
    
    # Concatenate all windows
    if windows_list:
        windows = np.concatenate(windows_list, axis=0)
        labels = np.array(labels_list)
    else:
        raise ValueError("No valid windows created from sequences")
    
    print(f"Total windows created: {len(windows)}")
    print(f"Label distribution: {np.bincount(labels)}\n")
    
    return windows, labels, label_to_idx, idx_to_label, window_metadata


def reshape_for_lstm(windows):
    """
    Reshape windows for LSTM input.
    
    LSTM expects input of shape (batch_size, time_steps, features)
    
    Args:
        windows (np.ndarray): Shape (num_windows, window_size, 21, 3)
        
    Returns:
        reshaped (np.ndarray): Shape (num_windows, window_size, 21*3=63)
    """
    num_windows, window_size, num_landmarks, coords = windows.shape
    
    # Flatten landmarks and coordinates
    reshaped = windows.reshape(num_windows, window_size, num_landmarks * coords)
    
    print(f"Reshaped for LSTM:")
    print(f"  Input shape: {windows.shape}")
    print(f"  Output shape: {reshaped.shape}\n")
    
    return reshaped


def verify_window_integrity(windows, labels, window_metadata):
    """
    Verify that windows and labels are properly aligned.
    
    Args:
        windows (np.ndarray): Shape (num_windows, window_size, features)
        labels (np.ndarray): Shape (num_windows,)
        window_metadata (list): Metadata for each window
    """
    print("Window integrity check:")
    print(f"  Number of windows: {len(windows)}")
    print(f"  Number of labels: {len(labels)}")
    print(f"  Number of metadata: {len(window_metadata)}")
    
    assert len(windows) == len(labels), "Windows and labels count mismatch!"
    assert len(windows) == len(window_metadata), "Windows and metadata count mismatch!"
    
    # Check for NaN or Inf
    nan_count = np.isnan(windows).sum()
    inf_count = np.isinf(windows).sum()
    
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("  WARNING: Invalid values detected in windows!")
    else:
        print("  ✓ All values valid")
    
    print()
