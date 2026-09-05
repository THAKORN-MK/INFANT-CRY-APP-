"""Corrected Stage 2 single- and multi-branch temporal models."""

from __future__ import annotations

from typing import Any, Sequence

from .attention import attention_layer_class


FEATURE_BLOCKS: dict[str, tuple[int, int]] = {
    "mfcc_derivatives": (0, 120),
    "log_mel": (120, 184),
    "chroma": (184, 196),
}


def _conv_block(
    layers: Any,
    x: Any,
    filters: int,
    pool: tuple[int, int],
    index: int,
    *,
    prefix: str = "cnn",
    two_convs: bool = True,
):
    count = 2 if two_convs else 1
    for conv_index in range(1, count + 1):
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            name=f"{prefix}_block_{index}_conv_{conv_index}",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_block_{index}_bn_{conv_index}")(x)
        x = layers.Activation(
            "relu", name=f"{prefix}_block_{index}_relu_{conv_index}"
        )(x)
    x = layers.MaxPooling2D(pool, name=f"{prefix}_block_{index}_pool")(x)
    return layers.Dropout(0.25 if index < 3 else 0.30)(x)


def _as_time_sequence(layers: Any, x: Any, name: str):
    x = layers.Permute((2, 1, 3), name=f"{name}_time_major")(x)
    shape = x.shape
    return layers.Reshape(
        (int(shape[1]), int(shape[2]) * int(shape[3])), name=name
    )(x)


def _single_branch(layers: Any, inputs: Any):
    x = _conv_block(layers, inputs, 32, (2, 2), 1)
    x = _conv_block(layers, x, 64, (2, 2), 2)
    x = _conv_block(layers, x, 128, (2, 1), 3)
    x = _conv_block(layers, x, 256, (2, 1), 4, two_convs=False)
    return _as_time_sequence(layers, x, "time_sequence")


def _crop_feature_block(layers: Any, inputs: Any, start: int, end: int, name: str):
    total = int(inputs.shape[1])
    return layers.Cropping2D(
        cropping=((start, total - end), (0, 0)), name=f"{name}_features"
    )(inputs)


def _multi_branch(layers: Any, inputs: Any):
    sequences = []
    definitions = (
        ("mfcc", FEATURE_BLOCKS["mfcc_derivatives"], ((2, 2), (2, 2), (2, 1))),
        ("mel", FEATURE_BLOCKS["log_mel"], ((2, 2), (2, 2), (2, 1))),
        ("chroma", FEATURE_BLOCKS["chroma"], ((2, 2), (2, 2))),
    )
    for prefix, (start, end), pools in definitions:
        x = _crop_feature_block(layers, inputs, start, end, prefix)
        for index, pool in enumerate(pools, start=1):
            x = _conv_block(
                layers,
                x,
                32 * min(index, 3),
                pool,
                index,
                prefix=prefix,
                two_convs=index < 3,
            )
        sequences.append(_as_time_sequence(layers, x, f"{prefix}_time_sequence"))
    return layers.Concatenate(axis=-1, name="time_sequence")(sequences)


def build_stage2_model(
    tf: Any,
    input_shape: Sequence[int],
    num_classes: int = 5,
    *,
    architecture: str = "corrected_single_branch",
):
    """Build a Stage 2 model whose recurrent sequence axis is always time."""

    if architecture not in {"corrected_single_branch", "corrected_multi_branch"}:
        raise ValueError(f"Unsupported Stage 2 architecture: {architecture}")
    layers = tf.keras.layers
    AttentionLayer = attention_layer_class(tf)
    inputs = layers.Input(shape=tuple(input_shape), name="audio_features")
    if architecture == "corrected_single_branch":
        x = _single_branch(layers, inputs)
    else:
        x = _multi_branch(layers, inputs)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bilstm_2"
    )(x)
    x = layers.Dropout(0.30)(x)
    x = AttentionLayer(name="temporal_attention")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classifier",
    )(x)
    return tf.keras.Model(
        inputs,
        outputs,
        name=f"Stage2_{architecture}_TimeMajor_BiLSTM_Attention",
    )
