"""Content-addressed immutable feature cache with integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .artefacts import sha256_file
from .protocol import write_json_atomic


class FeatureCacheError(ValueError):
    """Raised when a cache entry is stale, corrupt, or incompatible."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_feature_cache_key(
    *,
    source_sha256: str,
    preprocessing: Mapping[str, Any],
    augmentation: Mapping[str, Any] | None,
    dtype: str,
    shape: Sequence[int],
    feature_view: str = "all_blocks",
) -> str:
    payload = {
        "schema_version": "1.0",
        "source_sha256": str(source_sha256).lower(),
        "preprocessing": _canonical(preprocessing),
        "augmentation": _canonical(augmentation),
        "dtype": str(dtype),
        "shape": [int(value) for value in shape],
        "feature_view": str(feature_view),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FeatureCache:
    """Store verified float tensors under SHA-256-derived immutable paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def key(self, **kwargs: Any) -> str:
        return build_feature_cache_key(**kwargs)

    def paths(self, key: str) -> tuple[Path, Path]:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Feature cache key must be a lowercase SHA-256 digest")
        directory = self.root / key[:2]
        return directory / f"{key}.npy", directory / f"{key}.json"

    def get(self, key: str) -> np.ndarray | None:
        array_path, metadata_path = self.paths(key)
        if not array_path.exists() and not metadata_path.exists():
            return None
        if not array_path.is_file() or not metadata_path.is_file():
            raise FeatureCacheError(f"Incomplete feature cache entry: {key}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FeatureCacheError(f"Unreadable feature cache metadata: {exc}") from exc
        if metadata.get("key") != key:
            raise FeatureCacheError(f"Feature cache key mismatch for {key}")
        actual_hash = sha256_file(array_path)
        if actual_hash != metadata.get("array_sha256"):
            raise FeatureCacheError(
                f"Feature cache SHA-256 mismatch for {array_path}"
            )
        try:
            values = np.load(array_path, allow_pickle=False)
        except Exception as exc:
            raise FeatureCacheError(f"Unreadable feature cache array: {exc}") from exc
        expected_shape = tuple(int(value) for value in metadata.get("shape", ()))
        expected_dtype = str(metadata.get("dtype", ""))
        if values.shape != expected_shape or str(values.dtype) != expected_dtype:
            raise FeatureCacheError(
                f"Feature cache tensor contract mismatch: {values.shape}/{values.dtype}"
            )
        if not np.isfinite(values).all():
            raise FeatureCacheError("Feature cache tensor contains non-finite values")
        return np.asarray(values)

    def put(self, key: str, values: np.ndarray) -> Path:
        array = np.asarray(values)
        if not np.isfinite(array).all():
            raise FeatureCacheError("Cannot cache non-finite features")
        array_path, metadata_path = self.paths(key)
        existing = self.get(key)
        if existing is not None:
            if existing.shape == array.shape and existing.dtype == array.dtype and np.array_equal(existing, array):
                return array_path
            raise FileExistsError(f"Refusing to replace immutable feature cache: {key}")

        array_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{array_path.name}.", suffix=".tmp", dir=array_path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            if array_path.exists() or metadata_path.exists():
                raise FileExistsError(
                    f"Refusing to replace immutable feature cache: {key}"
                )
            os.replace(temporary_name, array_path)
            write_json_atomic(
                metadata_path,
                {
                    "schema_version": "1.0",
                    "key": key,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "array_sha256": sha256_file(array_path),
                },
            )
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return array_path
