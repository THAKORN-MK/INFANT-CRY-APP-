"""Corrected Stage 1 CNN + MFCC + BiLSTM + temporal attention model."""

from __future__ import annotations

from typing import Any, Sequence

from .attention import attention_layer_class


def _conv_block(layers: Any, x: Any, filters: int, pool: tuple[int, int], index: int):
    for conv_index in range(1, 3):
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            name=f"cnn_block_{index}_conv_{conv_index}",
        )(x)
        x = layers.BatchNormalization(name=f"cnn_block_{index}_bn_{conv_index}")(x)
        x = layers.Activation("relu", name=f"cnn_block_{index}_relu_{conv_index}")(x)
    x = layers.MaxPooling2D(pool, name=f"cnn_block_{index}_pool")(x)
    return layers.Dropout(0.25 if index < 3 else 0.30)(x)


def build_stage1_model(tf: Any, input_shape: Sequence[int], num_classes: int = 2):
    """Build Stage 1 with an explicit feature/time transpose before BiLSTM."""

    layers = tf.keras.layers
    AttentionLayer = attention_layer_class(tf)
    inputs = layers.Input(shape=tuple(input_shape), name="audio_features")
    x = _conv_block(layers, inputs, 32, (2, 2), 1)
    x = _conv_block(layers, x, 64, (2, 2), 2)
    x = _conv_block(layers, x, 128, (2, 1), 3)

    # CNN tensors are [feature, time, channel]. BiLSTM must consume time.
    x = layers.Permute((2, 1, 3), name="time_major")(x)
    shape = x.shape
    x = layers.Reshape(
        (int(shape[1]), int(shape[2]) * int(shape[3])), name="time_sequence"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bilstm_2"
    )(x)
    x = layers.Dropout(0.30)(x)
    x = AttentionLayer(name="temporal_attention")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classifier",
    )(x)
    return tf.keras.Model(inputs, outputs, name="Stage1_TimeMajor_CNN_BiLSTM_Attention")
