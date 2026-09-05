from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cryinsight.training.artefacts import (
    apply_normalizer,
    fit_normalizer,
    load_normalizer,
    save_normalizer,
)


class PerFeatureNormalizerTests(unittest.TestCase):
    def test_fit_uses_training_tensor_only_and_broadcasts_by_feature_bin(self) -> None:
        training = np.asarray(
            [
                [[[1.0], [3.0]], [[10.0], [14.0]]],
                [[[5.0], [7.0]], [[18.0], [22.0]]],
            ],
            dtype=np.float32,
        )
        validation = np.full((1, 2, 2, 1), 1_000_000.0, dtype=np.float32)
        stats = fit_normalizer(
            training,
            epsilon=1e-8,
            run_id="run",
            fold=1,
            axis="per_feature_bin",
            feature_order=("mfcc", "log_mel"),
            feature_blocks=(("mfcc", 0, 1), ("log_mel", 1, 2)),
        )

        np.testing.assert_allclose(stats.mean[:, 0, 0], [4.0, 16.0])
        self.assertEqual(stats.mean.shape, (2, 1, 1))
        normalized_validation = apply_normalizer(validation, stats)
        self.assertGreater(float(normalized_validation.min()), 1_000.0)

    def test_per_feature_bundle_round_trip_records_shape_and_blocks(self) -> None:
        features = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5, 1)
        stats = fit_normalizer(
            features,
            epsilon=1e-8,
            run_id="run",
            fold="final_refit",
            axis="per_feature_bin",
            feature_order=("mfcc_delta_delta2",),
            feature_blocks=(("mfcc_delta_delta2", 0, 4),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalizer.npy"
            save_normalizer(path, stats)
            loaded = load_normalizer(
                path, expected_run_id="run", expected_fold="final_refit"
            )
            metadata = json.loads(
                path.with_suffix(".npy.metadata.json").read_text(encoding="utf-8")
            )

        np.testing.assert_allclose(loaded.mean, stats.mean)
        np.testing.assert_allclose(loaded.std, stats.std)
        self.assertEqual(metadata["axis"], "per_feature_bin")
        self.assertEqual(metadata["mean_shape"], [4, 1, 1])
        self.assertEqual(metadata["feature_blocks"][0][0], "mfcc_delta_delta2")

    def test_apply_rejects_feature_shape_mismatch(self) -> None:
        features = np.ones((2, 3, 4, 1), dtype=np.float32)
        stats = fit_normalizer(
            features, epsilon=1e-8, run_id="run", fold=1
        )
        with self.assertRaisesRegex(ValueError, "feature shape"):
            apply_normalizer(np.ones((2, 4, 4, 1), dtype=np.float32), stats)


if __name__ == "__main__":
    unittest.main()
