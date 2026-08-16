from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from cryinsight.audio.features import (
    AudioContractError,
    PreprocessingConfig,
    apply_augmentation,
    assert_validation_originals_only,
    extract_features,
    extract_fold_tensors,
    mixup_batch,
)
from cryinsight.training.protocol import OriginalRecord


class PreprocessingContractTests(unittest.TestCase):
    def test_stage1_uses_mfcc_derivatives_without_mel_or_chroma(self) -> None:
        config = PreprocessingConfig.stage1_binary()

        self.assertEqual(config.version, "cryinsight_stage1_features_v3")
        self.assertEqual(config.feature_order, ("mfcc", "delta", "delta2"))
        self.assertEqual(config.feature_shape, (120, 128, 1))

    def test_stage2_keeps_mfcc_mel_and_chroma_contract(self) -> None:
        config = PreprocessingConfig.stage2_main()

        self.assertEqual(config.version, "cryinsight_stage2_features_v3")
        self.assertEqual(
            config.feature_order,
            ("mfcc", "delta", "delta2", "logmel", "chroma"),
        )
        self.assertEqual(config.feature_shape, (196, 128, 1))

    def test_default_feature_contract_matches_deployed_tensor(self) -> None:
        config = PreprocessingConfig()

        self.assertEqual(config.version, "cryinsight_features_v2")
        self.assertEqual(config.sample_rate, 22050)
        self.assertEqual(config.delta_width, 9)
        self.assertEqual(config.minimum_waveform_samples, 4096)
        self.assertEqual(config.short_audio_policy, "right_zero_pad_to_delta_width")
        self.assertEqual(config.feature_order, ("mfcc", "delta", "delta2", "logmel", "chroma"))
        self.assertEqual(config.feature_shape, (196, 128, 1))
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.to_dict()["feature_shape"], [196, 128, 1])
        self.assertEqual(config.to_dict()["minimum_waveform_samples"], 4096)

    def test_non_finite_waveform_is_rejected(self) -> None:
        with self.assertRaisesRegex(AudioContractError, "finite"):
            apply_augmentation(
                np.array([0.0, np.nan], dtype=np.float32),
                sample_rate=22050,
                augmentation_type="amplitude_scale",
                augmentation_params_json=json.dumps({"factor": 0.9}),
                seed=42,
            )


class AugmentationContractTests(unittest.TestCase):
    def test_seeded_noise_is_deterministic_and_finite(self) -> None:
        waveform = np.sin(np.linspace(0, 20, 4096, dtype=np.float32))
        kwargs = {
            "sample_rate": 22050,
            "augmentation_type": "gaussian_noise",
            "augmentation_params_json": json.dumps({"noise_factor": 0.005}),
            "seed": 123,
        }

        first = apply_augmentation(waveform, **kwargs)
        second = apply_augmentation(waveform, **kwargs)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        self.assertLessEqual(float(np.max(np.abs(first))), 1.0 + 1e-6)

    def test_validation_contract_rejects_augmented_rows(self) -> None:
        assert_validation_originals_only(["original", "original"])

        with self.assertRaisesRegex(AudioContractError, "original-only"):
            assert_validation_originals_only(["original", "augmented"])


class FoldTensorErrorContextTests(unittest.TestCase):
    def test_final_refit_can_extract_training_without_validation_records(self) -> None:
        training = OriginalRecord(
            record_id="train-record",
            filepath=Path("D:/fixture/train.wav"),
            relative_path="baby/train.wav",
            label="baby",
            source_label="baby",
            source_dataset="fixture",
            group_id="train-group",
            group_rule="fixture",
            sha256="a" * 64,
        )
        config = PreprocessingConfig.stage1_binary()
        feature = np.ones(config.feature_shape, dtype=np.float32)

        with (
            patch(
                "cryinsight.audio.features.load_preprocessed_waveform",
                return_value=(np.ones(4096, dtype=np.float32), config.sample_rate),
            ),
            patch("cryinsight.audio.features.extract_features", return_value=feature),
        ):
            tensors = extract_fold_tensors(
                [training],
                [],
                [],
                config=config,
                allow_empty_validation=True,
            )

        self.assertEqual(tensors.train_features.shape, (1, *config.feature_shape))
        self.assertEqual(tensors.validation_features.shape, (0, *config.feature_shape))

    def test_original_audio_error_identifies_partition_record_and_path(self) -> None:
        missing_path = Path("D:/missing/train.wav")
        training = OriginalRecord(
            record_id="train-record",
            filepath=missing_path,
            relative_path="baby/train.wav",
            label="baby",
            source_label="baby",
            source_dataset="fixture",
            group_id="train-group",
            group_rule="fixture",
            sha256="a" * 64,
        )
        validation = OriginalRecord(
            record_id="validation-record",
            filepath=Path("D:/missing/validation.wav"),
            relative_path="baby/validation.wav",
            label="baby",
            source_label="baby",
            source_dataset="fixture",
            group_id="validation-group",
            group_rule="fixture",
            sha256="b" * 64,
        )

        with self.assertRaises(AudioContractError) as caught:
            extract_fold_tensors([training], [validation], [])

        message = str(caught.exception)
        self.assertIn("Training", message)
        self.assertIn("train-record", message)
        self.assertIn(str(missing_path), message)


class MixupContractTests(unittest.TestCase):
    def test_mixup_is_deterministic_and_preserves_shapes(self) -> None:
        features = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        labels = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]

        first_x, first_y = mixup_batch(features, labels, n_samples=5, alpha=0.2, seed=7)
        second_x, second_y = mixup_batch(features, labels, n_samples=5, alpha=0.2, seed=7)

        self.assertEqual(first_x.shape, (5, 3))
        self.assertEqual(first_y.shape, (5, 2))
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)
        np.testing.assert_allclose(first_y.sum(axis=1), np.ones(5), atol=1e-6)


@unittest.skipUnless(importlib.util.find_spec("librosa"), "librosa is unavailable")
class LibrosaFeatureTests(unittest.TestCase):
    def test_stage1_extraction_omits_mel_and_chroma_bins(self) -> None:
        config = PreprocessingConfig.stage1_binary()
        samples = int(config.sample_rate * 2.0)
        time = np.arange(samples, dtype=np.float32) / config.sample_rate
        waveform = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)

        features = extract_features(waveform, sample_rate=config.sample_rate, config=config)

        self.assertEqual(features.shape, (120, 128, 1))
        self.assertTrue(np.isfinite(features).all())

    def test_short_valid_waveform_is_right_zero_padded_for_delta_contract(self) -> None:
        config = PreprocessingConfig()
        samples = 3072
        time = np.arange(samples, dtype=np.float32) / config.sample_rate
        waveform = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)

        features = extract_features(waveform, sample_rate=config.sample_rate, config=config)

        self.assertEqual(features.shape, config.feature_shape)
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_feature_shape_dtype_and_finite_values(self) -> None:
        config = PreprocessingConfig()
        seconds = 2.0
        samples = int(config.sample_rate * seconds)
        time = np.arange(samples, dtype=np.float32) / config.sample_rate
        waveform = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)

        features = extract_features(waveform, sample_rate=config.sample_rate, config=config)

        self.assertEqual(features.shape, config.feature_shape)
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
