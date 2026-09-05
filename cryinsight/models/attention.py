"""Serializable temporal attention shared by Stage 1 and Stage 2."""

from __future__ import annotations

from typing import Any


_CLASSES: dict[int, Any] = {}


def attention_layer_class(tf: Any):
    """Return a Keras-serializable AttentionLayer without eager TF import."""

    key = id(tf)
    if key in _CLASSES:
        return _CLASSES[key]

    @tf.keras.utils.register_keras_serializable(package="CryInsight")
    class AttentionLayer(tf.keras.layers.Layer):
        def build(self, input_shape):
            width = int(input_shape[-1])
            self.W = self.add_weight(
                shape=(width, width),
                initializer="glorot_uniform",
                trainable=True,
                name="attn_W",
            )
            self.b = self.add_weight(
                shape=(width,), initializer="zeros", trainable=True, name="attn_b"
            )
            self.u = self.add_weight(
                shape=(width,),
                initializer="glorot_uniform",
                trainable=True,
                name="attn_u",
            )
            super().build(input_shape)

        def call(self, inputs):
            score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            score = tf.tensordot(score, self.u, axes=1)
            weights = tf.nn.softmax(score, axis=1)
            return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

        def get_config(self):
            return super().get_config()

    _CLASSES[key] = AttentionLayer
    return AttentionLayer
