from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class ExperimentFeatureViewTests(unittest.TestCase):
    def test_cache_hit_does_not_load_or_extract_audio_again(self):
        from cryinsight.experiments.feature_views import build_feature_view
        from cryinsight.training.feature_cache import FeatureCache
        from cryinsight.audio.features import PreprocessingConfig
        from tests.test_experiment_selection import _record
        with tempfile.TemporaryDirectory() as directory:
            cache = FeatureCache(directory)
            config = PreprocessingConfig.stage2_main()
            with patch('cryinsight.experiments.feature_views.load_preprocessed_waveform', return_value=(np.zeros(4096), 22050)), patch('cryinsight.experiments.feature_views.extract_features', return_value=np.ones(config.feature_shape, dtype=np.float32)):
                first = build_feature_view(_record(1), 'mfcc_summary', config, cache)
            with patch('cryinsight.experiments.feature_views.load_preprocessed_waveform', side_effect=AssertionError('cache hit must not read audio')):
                second = build_feature_view(_record(1), 'mfcc_summary', config, cache)
            np.testing.assert_array_equal(first, second)

    def test_stage1_logmel_baseline_uses_a_contract_that_contains_logmel(self) -> None:
        from cryinsight.experiments.feature_views import (
            preprocessing_config_for_candidate,
        )
        from cryinsight.experiments.registry import experiment_registry

        registry = experiment_registry()
        logmel = preprocessing_config_for_candidate(
            registry["stage1_logmel_small_cnn"]
        )
        proposed = preprocessing_config_for_candidate(registry["stage1_mfcc_svm"])
        self.assertIn("logmel", logmel.feature_order)
        self.assertIn("logmel", {name for name, _start, _end in logmel.feature_blocks})
        self.assertEqual(logmel.feature_shape[0], 196)
        self.assertNotIn("logmel", proposed.feature_order)
        self.assertEqual(proposed.feature_shape[0], 120)

    def test_feature_block_boundaries_match_stage2_contract(self) -> None:
        from cryinsight.experiments.feature_views import select_feature_blocks

        features = np.arange(196 * 128, dtype=np.float32).reshape(196, 128, 1)
        selected = select_feature_blocks(
            features,
            ("mfcc", "delta", "delta2", "chroma"),
        )

        self.assertEqual(selected.shape, (132, 128, 1))
        np.testing.assert_array_equal(selected[:40], features[:40])
        np.testing.assert_array_equal(selected[-12:], features[184:196])

    def test_mfcc_summary_contains_literal_mean_then_standard_deviation(self) -> None:
        from cryinsight.experiments.feature_views import mfcc_summary

        features = np.zeros((196, 8, 1), dtype=np.float32)
        for coefficient in range(40):
            features[coefficient, :, 0] = float(coefficient)

        summary = mfcc_summary(features)

        self.assertEqual(summary.shape, (80,))
        np.testing.assert_array_equal(summary[:40], np.arange(40, dtype=np.float32))
        np.testing.assert_array_equal(summary[40:], np.zeros(40, dtype=np.float32))

    def test_log_mel_view_selects_exact_64_bins(self) -> None:
        from cryinsight.experiments.feature_views import log_mel_view

        features = np.arange(196 * 4, dtype=np.float32).reshape(196, 4, 1)

        selected = log_mel_view(features)

        self.assertEqual(selected.shape, (64, 4, 1))
        np.testing.assert_array_equal(selected, features[120:184])

    def test_cache_key_changes_when_feature_view_changes(self) -> None:
        from cryinsight.training.feature_cache import build_feature_cache_key

        common = {
            "source_sha256": "a" * 64,
            "preprocessing": {},
            "augmentation": None,
            "dtype": "float32",
            "shape": (196, 128, 1),
        }

        self.assertNotEqual(
            build_feature_cache_key(feature_view="all_blocks", **common),
            build_feature_cache_key(feature_view="log_mel", **common),
        )

if __name__ == "__main__":
    unittest.main()
