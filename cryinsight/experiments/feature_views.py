"""Reusable, cache-safe feature views for classical and neural candidates."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cryinsight.audio.features import (
    PreprocessingConfig,
    extract_features,
    load_preprocessed_waveform,
)
from cryinsight.training.artefacts import sha256_file
from cryinsight.training.feature_cache import FeatureCache, build_feature_cache_key
from cryinsight.training.protocol import OriginalRecord, write_json_atomic

from .contracts import CandidateSpec


class FeatureViewError(ValueError):
    """Raised when a feature view violates its declared input/output contract."""


FEATURE_BLOCK_SLICES: dict[str, slice] = {
    "mfcc": slice(0, 40),
    "delta": slice(40, 80),
    "delta2": slice(80, 120),
    "log_mel": slice(120, 184),
    "chroma": slice(184, 196),
}


def preprocessing_config_for_candidate(
    candidate: CandidateSpec,
) -> PreprocessingConfig:
    """Select an extraction contract that contains the candidate's feature view."""

    needs_extended_blocks = candidate.feature_view in {
        "log_mel",
        "all_blocks",
        "multi_branch_blocks",
        "feature_block_subset",
    }
    if candidate.stage == "stage1" and not needs_extended_blocks:
        return PreprocessingConfig.stage1_binary()
    return PreprocessingConfig.stage2_main()


def _feature_tensor(features: Any) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise FeatureViewError(
            f"Expected [feature_bins, time_frames, 1], received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise FeatureViewError("Feature view contains non-finite values")
    return np.asarray(values, dtype=np.float32)


def select_feature_blocks(
    features: Any,
    blocks: Sequence[str],
) -> np.ndarray:
    values = _feature_tensor(features)
    selected = tuple(str(block) for block in blocks)
    if not selected or len(selected) != len(set(selected)):
        raise FeatureViewError("Feature blocks must be unique and non-empty")
    unknown = sorted(set(selected) - set(FEATURE_BLOCK_SLICES))
    if unknown:
        raise FeatureViewError(f"Unknown feature blocks: {unknown}")
    arrays: list[np.ndarray] = []
    for name in selected:
        block_slice = FEATURE_BLOCK_SLICES[name]
        if int(block_slice.stop or 0) > values.shape[0]:
            raise FeatureViewError(
                f"Feature tensor does not contain the declared {name} block"
            )
        arrays.append(values[block_slice, :, :])
    result = np.concatenate(arrays, axis=0)
    return np.asarray(result, dtype=np.float32)


def mfcc_summary(features: Any) -> np.ndarray:
    values = _feature_tensor(features)
    if values.shape[0] < 40:
        raise FeatureViewError("MFCC summary requires at least 40 feature bins")
    mfcc = np.asarray(values[:40, :, 0], dtype=np.float64)
    result = np.concatenate(
        (mfcc.mean(axis=1), mfcc.std(axis=1, ddof=0)),
        axis=0,
    )
    return np.asarray(result, dtype=np.float32)


def log_mel_view(features: Any) -> np.ndarray:
    return select_feature_blocks(features, ("log_mel",))


def extract_yamnet_embedding(waveform: Any, model: Any) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise FeatureViewError("YAMNet requires a finite non-empty mono waveform")
    outputs = model(audio)
    patches: Any = None
    if isinstance(outputs, Mapping):
        if "embeddings" in outputs:
            patches = outputs["embeddings"]
        elif "embedding" in outputs:
            patches = outputs["embedding"]
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        patches = outputs[1]
    if patches is None:
        raise FeatureViewError("YAMNet output does not contain patch embeddings")
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise FeatureViewError(
            f"YAMNet embeddings must be finite [patches, dimensions], received {values.shape}"
        )
    return values.mean(axis=0, dtype=np.float64).astype(np.float32)


def _safe_archive_member(member: tarfile.TarInfo) -> bool:
    name = PurePosixPath(member.name)
    return (
        not name.is_absolute()
        and ".." not in name.parts
        and not member.issym()
        and not member.islnk()
        and (member.isfile() or member.isdir())
    )


def _verified_yamnet_directory(path: Path, archive_sha256: str) -> bool:
    manifest_path = path / "archive_manifest.json"
    required = (path / "saved_model.pb", path / "variables" / "variables.index")
    if not manifest_path.is_file() or not all(item.is_file() for item in required):
        return False
    if not tuple((path / "variables").glob("variables.data-*-of-*")):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("archive_sha256") == archive_sha256


def prepare_yamnet_model(
    archive: str | Path,
    destination: str | Path,
) -> Path:
    """Extract the local SavedModel archive into an immutable hash directory."""

    source = Path(archive).resolve()
    if not source.is_file():
        raise FeatureViewError(f"Local YAMNet archive does not exist: {source}")
    archive_sha256 = sha256_file(source)
    root = Path(destination).resolve()
    target = root / archive_sha256
    if _verified_yamnet_directory(target, archive_sha256):
        return target
    if target.exists():
        raise FeatureViewError(f"Incomplete or corrupt YAMNet cache directory: {target}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".yamnet-", dir=root))
    try:
        with tarfile.open(source, "r:gz") as handle:
            members = handle.getmembers()
            unsafe = [member.name for member in members if not _safe_archive_member(member)]
            if unsafe:
                raise FeatureViewError(f"YAMNet archive contains unsafe paths: {unsafe[:3]}")
            handle.extractall(temporary, members=members)
        required = (
            temporary / "saved_model.pb",
            temporary / "variables" / "variables.index",
        )
        if not all(path.is_file() for path in required):
            raise FeatureViewError("YAMNet archive is missing SavedModel files")
        if not tuple((temporary / "variables").glob("variables.data-*-of-*")):
            raise FeatureViewError("YAMNet archive is missing variable data")
        write_json_atomic(
            temporary / "archive_manifest.json",
            {
                "schema_version": "1.0",
                "archive_path": str(source),
                "archive_sha256": archive_sha256,
            },
        )
        if target.exists():
            raise FileExistsError(f"Refusing to replace YAMNet cache: {target}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _apply_view(
    features: np.ndarray,
    view_id: str,
    *,
    blocks: Sequence[str] | None,
) -> np.ndarray:
    if view_id in {"all_blocks", "multi_branch_blocks"}:
        return _feature_tensor(features)
    if view_id == "mfcc_summary":
        return mfcc_summary(features)
    if view_id == "log_mel":
        return log_mel_view(features)
    if view_id == "feature_block_subset":
        return select_feature_blocks(features, tuple(blocks or ()))
    raise FeatureViewError(f"Unsupported feature view: {view_id}")


def build_feature_view(
    record: OriginalRecord,
    view_id: str,
    config: PreprocessingConfig,
    cache: FeatureCache | None = None,
    *,
    blocks: Sequence[str] | None = None,
    yamnet_model: Any | None = None,
    yamnet_archive_sha256: str | None = None,
) -> np.ndarray:
    """Build one original-record feature view under an auditable cache identity."""

    active_view = str(view_id)
    active_config = replace(config, sample_rate=16000) if active_view == "yamnet_embedding" else config
    if active_view == 'labels_only':
        shape = (1,)
    elif active_view == 'yamnet_embedding':
        shape = (1024,)
    elif active_view == 'mfcc_summary':
        shape = (80,)
    else:
        shape = _apply_view(np.zeros(config.feature_shape, dtype=np.float32), active_view, blocks=blocks).shape
    preprocessing = active_config.to_dict()
    if blocks is not None:
        preprocessing = {**preprocessing, "selected_blocks": list(blocks)}
    if active_view == 'yamnet_embedding':
        if cache is not None and (not isinstance(yamnet_archive_sha256, str) or len(yamnet_archive_sha256) != 64):
            raise FeatureViewError('YAMNet cache requires its archive SHA-256')
        preprocessing = {**preprocessing, "yamnet_archive_sha256": yamnet_archive_sha256}
    key = build_feature_cache_key(source_sha256=record.sha256, preprocessing=preprocessing,
                                  augmentation=None, dtype='float32', shape=shape, feature_view=active_view)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            if cached.shape != shape or cached.dtype != np.float32:
                raise FeatureViewError('Cached feature shape/dtype differs from requested contract')
            return cached
    if active_view == "labels_only":
        result = np.zeros((1,), dtype=np.float32)
    elif active_view == "yamnet_embedding":
        if yamnet_model is None:
            raise FeatureViewError("YAMNet feature view requires a loaded local model")
        waveform, _ = load_preprocessed_waveform(record.filepath, config=active_config)
        result = extract_yamnet_embedding(waveform, yamnet_model)
    else:
        waveform, sample_rate = load_preprocessed_waveform(record.filepath, config=config)
        features = extract_features(waveform, sample_rate=sample_rate, config=config)
        result = _apply_view(features, active_view, blocks=blocks)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FeatureViewError("Feature view contains non-finite values")
    if result.shape != shape:
        raise FeatureViewError(f'Feature shape differs from declared view: {result.shape} != {shape}')
    if cache is not None:
        cache.put(key, result)
    return result
