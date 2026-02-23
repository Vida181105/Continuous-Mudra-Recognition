"""
Model Architecture Module for Bharatanatyam Mudra Recognition System.

Implements an Attention-LSTM architecture:
    Input → BiLSTM → BN → BiLSTM → BN → TemporalAttention → Dense → Softmax

Key features:
- Bidirectional LSTMs capture past and future temporal context
- Custom TemporalAttention layer learns to focus on discriminative frames
- Two variants: "full" (>500 seqs) and "lightweight" (≤500 seqs)
- Proper regularisation for limited training data
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Tuple


# ===========================================================================
# Custom Temporal Attention Layer
# ===========================================================================

class TemporalAttention(tf.keras.layers.Layer):
    """
    Temporal Attention Layer for sequence classification.

    Given an input tensor of shape ``(batch, timesteps, features)``
    this layer computes:

    1. ``score = tanh(input @ W + b)``          → (batch, timesteps, units)
    2. ``alpha = softmax(score @ u, axis=time)`` → (batch, timesteps)
    3. ``context = sum(alpha * input, axis=time)``→ (batch, features)

    The learned parameters ``W``, ``b``, and ``u`` allow the network to
    attend selectively to the most informative frames.

    Attributes:
        units: Dimensionality of the internal attention space.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])

        self.W = self.add_weight(
            name='attention_weight',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )
        self.u = self.add_weight(
            name='attention_context_vector',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: Tensor of shape (batch, timesteps, features).
            mask: Optional boolean mask of shape (batch, timesteps).

        Returns:
            context_vector: Tensor of shape (batch, features).
        """
        # (batch, timesteps, units)
        score = tf.nn.tanh(
            tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b
        )

        # (batch, timesteps)
        attention_logits = tf.tensordot(score, self.u, axes=[[2], [0]])

        # Mask out padded timesteps if a mask is provided
        if mask is not None:
            attention_logits += (
                (1.0 - tf.cast(mask, dtype=tf.float32)) * -1e9
            )

        attention_weights = tf.nn.softmax(attention_logits, axis=1)

        # Weighted sum → (batch, features)
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, axis=-1),
            axis=1,
        )
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


# ===========================================================================
# Model Builder
# ===========================================================================

def build_attention_lstm_model(
    sequence_length: int = 30,
    n_features: int = 126,
    n_classes: int = 5,
    model_type: str = 'auto',
    n_sequences: int = 0,
    config: Optional[Dict] = None,
) -> tf.keras.Model:
    """
    Build and return an (uncompiled) Attention-BiLSTM Keras model.

    Architecture
    ------------
    Input (batch, T, 126)
      → Bidirectional LSTM 1 (return_sequences=True)
      → BatchNormalization
      → Bidirectional LSTM 2 (return_sequences=True)
      → BatchNormalization
      → TemporalAttention
      → Dense (ReLU) → Dropout
      → Dense (ReLU) → Dropout
      → Dense (softmax, n_classes)

    Args:
        sequence_length: Number of timesteps per input window.
        n_features: Feature dimension per timestep (126 for two hands).
        n_classes: Number of output mudra classes.
        model_type: ``'full'``, ``'lightweight'``, or ``'auto'``.
        n_sequences: Number of training sequences (used when ``'auto'``).
        config: Optional model config dict from config.json.

    Returns:
        A ``tf.keras.Model`` (not yet compiled).
    """

    # Auto-select variant based on data size
    if model_type == 'auto':
        model_type = 'full' if n_sequences > 500 else 'lightweight'

    # Hyperparameters per variant
    if model_type == 'full':
        lstm1 = config.get('lstm_units', 128) if config else 128
        lstm2 = lstm1 // 2
        att   = config.get('attention_units', 64) if config else 64
        d1, d2 = 128, 64
        drop  = config.get('dropout_rate', 0.5) if config else 0.5
    else:  # lightweight
        lstm1, lstm2 = 64, 32
        att   = 32
        d1, d2 = 64, 32
        drop  = 0.4

    print(f"   Model type: {model_type} "
          f"(data {'>' if model_type == 'full' else '≤'} 500 sequences)")

    # ---- Build graph ----------------------------------------------------
    inputs = tf.keras.Input(
        shape=(sequence_length, n_features), name='input_sequence')

    # BiLSTM 1
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm1,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.0,
            name='lstm_1',
        ),
        name='bilstm_1',
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)

    # BiLSTM 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm2,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.0,
            name='lstm_2',
        ),
        name='bilstm_2',
    )(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)

    # Temporal Attention
    x = TemporalAttention(units=att, name='temporal_attention')(x)

    # Classification head
    x = tf.keras.layers.Dense(d1, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(drop, name='dropout_1')(x)

    x = tf.keras.layers.Dense(d2, activation='relu', name='dense_2')(x)
    x = tf.keras.layers.Dropout(drop, name='dropout_2')(x)

    outputs = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='attention_lstm_mudra')

    # Print summary
    total_params = model.count_params()
    print(f"   Total parameters: {total_params:,}")

    return model


# ===========================================================================
# Model Trainer
# ===========================================================================

class ModelTrainer:
    """
    Handles model compilation, training callbacks, and training execution.
    """

    def __init__(self, model: tf.keras.Model, config: dict):
        """
        Args:
            model: A Keras model (compiled or not).
            config: Full pipeline configuration dictionary.
        """
        self.model = model
        self.config = config
        self.history = None

    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile with Adam optimiser and sparse categorical cross-entropy."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

    def get_callbacks(self, model_save_path: str,
                      log_dir: Optional[str] = None) -> list:
        """
        Create standard training callbacks.

        Includes:
        - ModelCheckpoint (best val_accuracy)
        - EarlyStopping
        - ReduceLROnPlateau
        - CSVLogger (if log_dir provided)
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['training']['early_stopping_patience'],
                mode='max',
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.CSVLogger(
                    os.path.join(log_dir, 'training_log.csv'))
            )

        return callbacks

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_save_path: str,
              log_dir: Optional[str] = None):
        """
        Train the model end-to-end.

        Args:
            X_train: Training sequences, shape (N_train, T, 126).
            y_train: Training labels, shape (N_train,).
            X_val: Validation sequences.
            y_val: Validation labels.
            model_save_path: Where to save the best checkpoint.
            log_dir: Optional directory for CSV training log.

        Returns:
            Keras History object.
        """
        lr = self.config['model']['learning_rate']
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']

        self.compile_model(learning_rate=lr)
        callbacks = self.get_callbacks(model_save_path, log_dir)

        # Balanced class weights for potentially imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight(
            'balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes.astype(int), weights))

        print(f"\n   Training/Validation split: "
              f"{len(X_train)} / {len(X_val)}")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}, "
              f"LR: {lr}\n")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        return self.history


# ===========================================================================
# Utility: Save Model Architecture Diagram
# ===========================================================================

def save_model_diagram(model: tf.keras.Model, output_path: str) -> None:
    """
    Attempt to save a visual diagram of the model architecture.

    Requires ``pydot`` and the ``graphviz`` system package.
    Fails silently if they are not installed.
    """
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            dpi=100,
        )
        print(f"✅ Model diagram saved: {output_path}")
    except Exception:
        print("   Note: Model diagram not generated "
              "(install graphviz + pydot for this feature)")
