"""Auditable TensorFlow device and precision configuration."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


class RuntimeDeviceError(RuntimeError):
    """Raised when a requested compute device cannot be honored."""


def _device_name(device: Any) -> str:
    return str(getattr(device, "name", device))


def _compute_capability(tf: Any, device: Any) -> str | None:
    try:
        details = tf.config.experimental.get_device_details(device)
    except Exception:
        return None
    value = details.get("compute_capability") if isinstance(details, dict) else None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return f"{int(value[0])}.{int(value[1])}"
    return str(value) if value is not None else None


def configure_tensorflow_runtime(
    tf: Any,
    *,
    device: str = "auto",
    require_gpu: bool = False,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    """Select CPU/GPU before model creation and return a JSON-safe manifest."""

    requested = str(device).lower()
    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("device must be one of: auto, gpu, cpu")
    if require_gpu and requested == "cpu":
        raise RuntimeDeviceError("GPU is required but --device cpu was selected")

    physical_gpus = list(tf.config.list_physical_devices("GPU"))
    if requested == "cpu":
        tf.config.set_visible_devices([], "GPU")
        selected = "cpu"
    elif physical_gpus:
        selected = "gpu"
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                # TensorFlow rejects late changes after device initialization.
                pass
    else:
        if require_gpu or requested == "gpu":
            raise RuntimeDeviceError(
                "GPU is required/requested but TensorFlow found no physical GPU"
            )
        selected = "cpu"

    policy_name = "mixed_float16" if mixed_precision else "float32"
    tf.keras.mixed_precision.set_global_policy(policy_name)
    build_info: dict[str, Any] = {}
    try:
        raw_build_info = tf.sysconfig.get_build_info()
        if isinstance(raw_build_info, dict):
            build_info = {str(key): str(value) for key, value in raw_build_info.items()}
    except Exception:
        pass

    return {
        "schema_version": "1.0",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
        "tensorflow_version": str(getattr(tf, "__version__", "unknown")),
        "keras_version": str(getattr(tf.keras, "__version__", "unknown")),
        "requested_device": requested,
        "require_gpu": bool(require_gpu),
        "selected_device": selected,
        "physical_gpu_count": len(physical_gpus),
        "gpus": [
            {
                "name": _device_name(gpu),
                "compute_capability": _compute_capability(tf, gpu),
            }
            for gpu in physical_gpus
        ],
        "memory_growth_requested": selected == "gpu",
        "mixed_precision_requested": bool(mixed_precision),
        "precision_policy": str(tf.keras.mixed_precision.global_policy().name),
        "tensorflow_build": build_info,
    }
