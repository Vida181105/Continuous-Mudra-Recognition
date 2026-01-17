"""
LSTM Model for Mudra Classification

Handles:
- Build Bi-Directional LSTM model
- Training utilities
- Model evaluation
- Model persistence
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


def build_lstm_model(input_shape, num_classes, lstm_units=64):
    """
    Build Bi-Directional LSTM model for mudra classification.
    
    Args:
        input_shape (tuple): (window_size, num_features) - e.g., (25, 63)
        num_classes (int): Number of mudra classes (2 for Pataka and Tripataka)
        lstm_units (int): Number of LSTM units
        
    Returns:
        model (keras.Model): Compiled LSTM model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Bi-directional LSTM layer
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False)),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    print()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    """
    Train the LSTM model.
    
    Args:
        model (keras.Model): Compiled model
        X_train (np.ndarray): Training data (num_samples, window_size, features)
        y_train (np.ndarray): Training labels (num_samples, num_classes) - one-hot encoded
        X_val (np.ndarray): Validation data
        y_val (np.ndarray): Validation labels (one-hot encoded)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        history: Training history object
    """
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}\n")
    
    # Define early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, idx_to_label=None):
    """
    Evaluate model on test data.
    
    Args:
        model (keras.Model): Trained model
        X_test (np.ndarray): Test data
        y_test (np.ndarray): Test labels (one-hot encoded)
        idx_to_label (dict): Mapping from class index to label name
        
    Returns:
        metrics (dict): Evaluation metrics
    """
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test_idx, y_pred)
    
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    
    # Classification report
    if idx_to_label:
        label_names = [idx_to_label[i] for i in range(len(idx_to_label))]
        print("Classification Report:")
        print(classification_report(y_test_idx, y_pred, target_names=label_names))
    else:
        print("Classification Report:")
        print(classification_report(y_test_idx, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_idx, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'y_true': y_test_idx
    }
    
    return metrics


def plot_training_history(history):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def save_model(model, model_path):
    """
    Save model to disk.
    
    Args:
        model (keras.Model): Model to save
        model_path (str): Path to save model (.h5 or .keras)
    """
    model.save(str(model_path))
    print(f"Model saved to {model_path}\n")


def load_model(model_path):
    """
    Load model from disk.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        model (keras.Model): Loaded model
    """
    model = keras.models.load_model(str(model_path))
    print(f"Model loaded from {model_path}\n")
    return model


def predict_on_window(model, window, idx_to_label):
    """
    Make prediction on a single window.
    
    Args:
        model (keras.Model): Trained model
        window (np.ndarray): Shape (window_size, features)
        idx_to_label (dict): Label mapping
        
    Returns:
        prediction (dict): {class_idx, class_name, confidence}
    """
    # Add batch dimension
    window_batch = np.expand_dims(window, axis=0)
    
    # Predict
    prob = model.predict(window_batch, verbose=0)[0]
    class_idx = np.argmax(prob)
    confidence = prob[class_idx]
    class_name = idx_to_label[class_idx]
    
    return {
        'class_idx': class_idx,
        'class_name': class_name,
        'confidence': float(confidence),
        'probabilities': prob
    }


def predict_on_sequence(model, windows, idx_to_label):
    """
    Make predictions on multiple windows from a sequence.
    
    Args:
        model (keras.Model): Trained model
        windows (np.ndarray): Shape (num_windows, window_size, features)
        idx_to_label (dict): Label mapping
        
    Returns:
        predictions (list): List of prediction dicts
    """
    predictions = []
    
    for window in windows:
        pred = predict_on_window(model, window, idx_to_label)
        predictions.append(pred)
    
    return predictions
