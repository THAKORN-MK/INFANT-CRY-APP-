"""Small real-Keras save/load smoke test for WSL checkpoint staging."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryinsight.training.checkpoint_staging import CheckpointStaging


def main() -> int:
    import tensorflow as tf

    destination_parent = PROJECT_ROOT / "Models_dbl" / "Main"
    with tempfile.TemporaryDirectory(
        prefix="checkpoint-smoke-",
        dir=destination_parent,
    ) as directory:
        destination = Path(directory) / "published.keras"
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(4,)),
                tf.keras.layers.Dense(2, activation="softmax", dtype="float32"),
            ]
        )
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
        x = tf.zeros((8, 4), dtype=tf.float32)
        y = tf.zeros((8,), dtype=tf.int32)
        with CheckpointStaging(
            "smoke_run",
            "fold_1",
            "selected.keras",
        ) as staging:
            local_path = staging.local_path
            history = model.fit(
                x,
                y,
                validation_data=(x, y),
                epochs=3,
                batch_size=4,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=str(local_path),
                        save_best_only=False,
                        verbose=0,
                    )
                ],
                verbose=0,
            )
            publication = staging.publish(destination)
            staging_is_native_linux = not str(staging.directory).startswith("/mnt/")
        loaded = tf.keras.models.load_model(str(destination))
        output_shape = tuple(loaded(tf.zeros((1, 4))).shape)
        if output_shape != (1, 2):
            raise RuntimeError(f"Unexpected loaded model output shape: {output_shape}")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "staging_path": str(local_path),
                    "staging_is_native_linux": staging_is_native_linux,
                    "published_to_wsl_mount": str(destination).startswith("/mnt/"),
                    "published_model_loaded": True,
                    "repeated_checkpoint_epochs": len(history.history["loss"]),
                    "output_shape": list(output_shape),
                    **publication,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
