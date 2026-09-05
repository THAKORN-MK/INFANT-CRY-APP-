"""Manual tiny GPU smoke test; this never reads Dataset files or starts full training."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryinsight.models.stage1_model import build_stage1_model
from cryinsight.models.stage2_model import build_stage2_model
from cryinsight.runtime.device import configure_tensorflow_runtime


def _exercise(model, shape: tuple[int, int, int], classes: int, name: str) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )
    features = tf.random.normal((1, *shape), dtype=tf.float32)
    labels = tf.one_hot([0], classes)
    loss = float(model.train_on_batch(features, labels))
    if not np.isfinite(loss):
        raise RuntimeError(f"{name} mini-batch loss is non-finite")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{name}.keras"
        model.save(path)
        loaded = tf.keras.models.load_model(path)
        output = loaded(features, training=False)
    if output.dtype != tf.float32 or not bool(tf.reduce_all(tf.math.is_finite(output))):
        raise RuntimeError(f"{name} save/load output contract failed")
    print(name, "loss", loss, "output_dtype", output.dtype.name, "device", output.device)


def main() -> int:
    environment = configure_tensorflow_runtime(
        tf, device="gpu", require_gpu=True, mixed_precision=True
    )
    print("selected_device", environment["selected_device"])
    print("gpus", environment["gpus"])
    _exercise(build_stage1_model(tf, (120, 128, 1), 2), (120, 128, 1), 2, "stage1")
    tf.keras.backend.clear_session()
    _exercise(
        build_stage2_model(
            tf, (196, 128, 1), 5, architecture="corrected_single_branch"
        ),
        (196, 128, 1),
        5,
        "stage2_single",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
