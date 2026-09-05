"""Runtime configuration helpers that do not import TensorFlow eagerly."""

from .device import RuntimeDeviceError, configure_tensorflow_runtime

__all__ = ["RuntimeDeviceError", "configure_tensorflow_runtime"]
