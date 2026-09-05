from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class FeatureCacheTests(unittest.TestCase):
    def test_key_changes_with_source_config_augmentation_dtype_or_shape(self) -> None:
        from cryinsight.training.feature_cache import build_feature_cache_key

        base = {
            "source_sha256": "a" * 64,
            "preprocessing": {"version": "v1", "sample_rate": 22050},
            "augmentation": None,
            "dtype": "float32",
            "shape": (120, 128, 1),
        }
        first = build_feature_cache_key(**base)
        variants = [
            {**base, "source_sha256": "b" * 64},
            {**base, "preprocessing": {"version": "v2"}},
            {**base, "augmentation": {"type": "noise", "seed": 42}},
            {**base, "dtype": "float64"},
            {**base, "shape": (196, 128, 1)},
        ]
        self.assertTrue(all(build_feature_cache_key(**row) != first for row in variants))

    def test_verified_round_trip_and_immutable_write(self) -> None:
        from cryinsight.training.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as directory:
            cache = FeatureCache(directory)
            payload = np.ones((3, 4, 1), dtype=np.float32)
            key = cache.key(
                source_sha256="a" * 64,
                preprocessing={"version": "v1"},
                augmentation=None,
                dtype="float32",
                shape=payload.shape,
            )
            cache.put(key, payload)
            np.testing.assert_array_equal(cache.get(key), payload)
            cache.put(key, payload)
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                cache.put(key, payload + 1.0)

    def test_corrupted_array_is_rejected(self) -> None:
        from cryinsight.training.feature_cache import FeatureCache, FeatureCacheError

        with tempfile.TemporaryDirectory() as directory:
            cache = FeatureCache(directory)
            payload = np.ones((3, 4, 1), dtype=np.float32)
            key = cache.key(
                source_sha256="a" * 64,
                preprocessing={"version": "v1"},
                augmentation=None,
                dtype="float32",
                shape=payload.shape,
            )
            cache.put(key, payload)
            array_path, _ = cache.paths(key)
            array_path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(FeatureCacheError, "SHA-256"):
                cache.get(key)

    def test_metadata_key_mismatch_is_rejected(self) -> None:
        from cryinsight.training.feature_cache import FeatureCache, FeatureCacheError

        with tempfile.TemporaryDirectory() as directory:
            cache = FeatureCache(directory)
            payload = np.ones((3, 4, 1), dtype=np.float32)
            key = cache.key(
                source_sha256="a" * 64,
                preprocessing={"version": "v1"},
                augmentation=None,
                dtype="float32",
                shape=payload.shape,
            )
            cache.put(key, payload)
            _, metadata_path = cache.paths(key)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["key"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(FeatureCacheError, "key mismatch"):
                cache.get(key)


if __name__ == "__main__":
    unittest.main()
