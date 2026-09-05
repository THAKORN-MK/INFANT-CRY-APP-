"""Versioned audio and feature contract shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cryinsight.training.protocol import AugmentationPlanRow, OriginalRecord
from cryinsight.training.feature_cache import FeatureCache


class AudioContractError(ValueError):
    """Raised when audio or a feature tensor violates the frozen contract."""


class AudioDependencyUnavailable(RuntimeError):
    """Raised when librosa is unavailable for an audio operation."""


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str = "cryinsight_features_v2"
    sample_rate: int = 22050
    mono: bool = True
    trim_top_db: int = 20
    normalize_waveform: bool = True
    n_mfcc: int = 40
    n_mels: int = 64
    n_chroma: int = 12
    n_fft: int = 2048
    hop_length: int = 512
    delta_width: int = 9
    short_audio_policy: str = "right_zero_pad_to_delta_width"
    max_frames: int = 128
    feature_order: tuple[str, ...] = (
        "mfcc",
        "delta",
        "delta2",
        "logmel",
        "chroma",
    )
    dtype: str = "float32"
    normalization_epsilon: float = 1e-8

    @classmethod
    def stage1_binary(cls) -> "PreprocessingConfig":
        """Feature contract for the binary baby gate."""

        return cls(
            version="cryinsight_stage1_features_v3",
            feature_order=("mfcc", "delta", "delta2"),
        )

    @classmethod
    def stage2_main(cls) -> "PreprocessingConfig":
        """Feature contract for the five-class infant emotion model."""

        return cls(
            version="cryinsight_stage2_features_v3",
            feature_order=("mfcc", "delta", "delta2", "logmel", "chroma"),
        )

    @property
    def feature_bins(self) -> int:
        widths = {
            "mfcc": self.n_mfcc,
            "delta": self.n_mfcc,
            "delta2": self.n_mfcc,
            "logmel": self.n_mels,
            "chroma": self.n_chroma,
        }
        try:
            return sum(widths[name] for name in self.feature_order)
        except KeyError as exc:
            raise AudioContractError(
                f"Unsupported feature in feature_order: {exc.args[0]!r}"
            ) from exc

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return (self.feature_bins, self.max_frames, 1)

    @property
    def feature_blocks(self) -> tuple[tuple[str, int, int], ...]:
        widths = {
            "mfcc": self.n_mfcc,
            "delta": self.n_mfcc,
            "delta2": self.n_mfcc,
            "logmel": self.n_mels,
            "chroma": self.n_chroma,
        }
        rows: list[tuple[str, int, int]] = []
        start = 0
        for name in self.feature_order:
            end = start + widths[name]
            rows.append((name, start, end))
            start = end
        return tuple(rows)

    @property
    def minimum_waveform_samples(self) -> int:
        return (self.delta_width - 1) * self.hop_length

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_order"] = list(self.feature_order)
        payload["feature_shape"] = list(self.feature_shape)
        payload["feature_blocks"] = [list(row) for row in self.feature_blocks]
        payload["minimum_waveform_samples"] = self.minimum_waveform_samples
        return payload


@dataclass(frozen=True)
class FoldTensors:
    train_features: np.ndarray
    train_labels: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    train_sample_kinds: tuple[str, ...]
    validation_features: np.ndarray
    validation_labels: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    validation_sample_kinds: tuple[str, ...]


@dataclass(frozen=True)
class OriginalTensors:
    features: np.ndarray
    labels: tuple[str, ...]
    record_ids: tuple[str, ...]
    sample_kinds: tuple[str, ...]


def _require_librosa():
    try:
        import librosa  # type: ignore[import-not-found]
    except Exception as exc:
        raise AudioDependencyUnavailable(
            "librosa is required for waveform loading and feature extraction; "
            "activate the CryInsight ML environment"
        ) from exc
    return librosa


def _validated_waveform(waveform: np.ndarray | Sequence[float]) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise AudioContractError(
            f"Expected a mono one-dimensional waveform, received shape {audio.shape}"
        )
    if audio.size == 0:
        raise AudioContractError("Audio waveform is empty")
    if not np.isfinite(audio).all():
        raise AudioContractError("Audio waveform must contain only finite values")
    return audio


def _peak_normalize(waveform: np.ndarray, *, epsilon: float) -> np.ndarray:
    peak = float(np.max(np.abs(waveform)))
    if not np.isfinite(peak) or peak <= epsilon:
        raise AudioContractError("Audio waveform has zero or non-finite energy")
    normalized = waveform / peak
    normalized = np.asarray(normalized, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise AudioContractError("Waveform normalization produced non-finite values")
    return normalized


def load_preprocessed_waveform(
    filepath: str | Path,
    *,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Load, resample, trim, and normalize using the frozen train contract."""

    active = config or PreprocessingConfig()
    path = Path(filepath)
    if not path.is_file():
        raise AudioContractError(f"Audio file does not exist: {path}")
    librosa = _require_librosa()
    try:
        waveform, sample_rate = librosa.load(
            str(path), sr=active.sample_rate, mono=active.mono
        )
        waveform, _ = librosa.effects.trim(waveform, top_db=active.trim_top_db)
    except Exception as exc:
        raise AudioContractError(f"Could not decode audio file {path}: {exc}") from exc
    audio = _validated_waveform(waveform)
    if active.normalize_waveform:
        audio = _peak_normalize(audio, epsilon=active.normalization_epsilon)
    if int(sample_rate) != active.sample_rate:
        raise AudioContractError(
            f"Expected sample rate {active.sample_rate}, received {sample_rate}"
        )
    return audio, int(sample_rate)


def _parse_augmentation_params(
    augmentation_params_json: str | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(augmentation_params_json, str):
        try:
            raw = json.loads(augmentation_params_json)
        except json.JSONDecodeError as exc:
            raise AudioContractError(f"Invalid augmentation JSON: {exc}") from exc
    else:
        raw = dict(augmentation_params_json)
    if not isinstance(raw, dict):
        raise AudioContractError("Augmentation parameters must be a JSON object")
    try:
        return {str(key): float(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise AudioContractError("Augmentation parameters must be numeric") from exc


def apply_augmentation(
    waveform: np.ndarray | Sequence[float],
    *,
    sample_rate: int,
    augmentation_type: str,
    augmentation_params_json: str | Mapping[str, float],
    seed: int,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Apply one deterministic training-only waveform transformation."""

    active = config or PreprocessingConfig()
    if sample_rate != active.sample_rate:
        raise AudioContractError(
            f"Augmentation sample rate {sample_rate} does not match {active.sample_rate}"
        )
    audio = _validated_waveform(waveform)
    params = _parse_augmentation_params(augmentation_params_json)
    rng = np.random.default_rng(seed)

    if augmentation_type == "gaussian_noise":
        noise_factor = params.get("noise_factor")
        if noise_factor is None or not 0.0 < noise_factor <= 0.1:
            raise AudioContractError("noise_factor must be in (0, 0.1]")
        transformed = audio + rng.normal(0.0, noise_factor, audio.shape)
    elif augmentation_type == "pitch_shift":
        steps = params.get("steps")
        if steps is None or not -6.0 <= steps <= 6.0 or steps == 0.0:
            raise AudioContractError("pitch-shift steps must be nonzero and within [-6, 6]")
        librosa = _require_librosa()
        transformed = librosa.effects.pitch_shift(
            audio, sr=sample_rate, n_steps=steps
        )
    elif augmentation_type == "time_stretch":
        rate = params.get("rate")
        if rate is None or not 0.5 <= rate <= 2.0 or rate == 1.0:
            raise AudioContractError("time-stretch rate must be within [0.5, 2.0] and not 1")
        librosa = _require_librosa()
        transformed = librosa.effects.time_stretch(audio, rate=rate)
    elif augmentation_type == "time_shift":
        fraction = params.get("fraction")
        if fraction is not None:
            if not -0.5 <= fraction <= 0.5 or fraction == 0.0:
                raise AudioContractError(
                    "time-shift fraction must be nonzero and within [-0.5, 0.5]"
                )
            shift = int(round(len(audio) * fraction))
        else:
            max_fraction = params.get("max_fraction")
            if max_fraction is None or not 0.0 < max_fraction <= 0.5:
                raise AudioContractError("time-shift max_fraction must be in (0, 0.5]")
            max_samples = max(1, int(round(len(audio) * max_fraction)))
            shift = int(rng.integers(-max_samples, max_samples + 1))
        if shift == 0:
            shift = 1 if fraction is None or fraction > 0.0 else -1
        transformed = np.roll(audio, shift)
    elif augmentation_type == "amplitude_scale":
        factor = params.get("factor")
        if factor is None or not 0.1 <= factor <= 2.0 or factor == 1.0:
            raise AudioContractError("amplitude factor must be within [0.1, 2] and not 1")
        transformed = audio * factor
    else:
        raise AudioContractError(f"Unsupported augmentation type: {augmentation_type!r}")

    result = _validated_waveform(np.asarray(transformed, dtype=np.float32))
    if active.normalize_waveform:
        result = _peak_normalize(result, epsilon=active.normalization_epsilon)
    return result


def extract_features(
    waveform: np.ndarray | Sequence[float],
    *,
    sample_rate: int,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Extract only the feature blocks declared by the versioned contract."""

    active = config or PreprocessingConfig()
    if sample_rate != active.sample_rate:
        raise AudioContractError(
            f"Feature sample rate {sample_rate} does not match {active.sample_rate}"
        )
    audio = _validated_waveform(waveform)
    if active.delta_width < 3 or active.delta_width % 2 == 0:
        raise AudioContractError("delta_width must be an odd integer of at least 3")
    if active.short_audio_policy != "right_zero_pad_to_delta_width":
        raise AudioContractError(
            f"Unsupported short-audio policy: {active.short_audio_policy!r}"
        )
    if audio.size < active.minimum_waveform_samples:
        audio = np.pad(
            audio,
            (0, active.minimum_waveform_samples - audio.size),
            mode="constant",
        ).astype(np.float32, copy=False)
    librosa = _require_librosa()

    allowed_features = {"mfcc", "delta", "delta2", "logmel", "chroma"}
    if not active.feature_order or len(set(active.feature_order)) != len(
        active.feature_order
    ):
        raise AudioContractError("feature_order must contain unique feature names")
    unknown = sorted(set(active.feature_order) - allowed_features)
    if unknown:
        raise AudioContractError(f"Unsupported features in feature_order: {unknown!r}")

    blocks: dict[str, np.ndarray] = {}
    if {"mfcc", "delta", "delta2"} & set(active.feature_order):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=active.n_mfcc,
            n_fft=active.n_fft,
            hop_length=active.hop_length,
        )
        blocks["mfcc"] = mfcc
        if {"delta", "delta2"} & set(active.feature_order):
            if mfcc.shape[1] < active.delta_width:
                raise AudioContractError(
                    "Audio is too short for the configured delta feature contract "
                    f"(fewer than {active.delta_width} frames after padding)"
                )
            if "delta" in active.feature_order:
                blocks["delta"] = librosa.feature.delta(
                    mfcc, width=active.delta_width
                )
            if "delta2" in active.feature_order:
                blocks["delta2"] = librosa.feature.delta(
                    mfcc, order=2, width=active.delta_width
                )
    if "logmel" in active.feature_order:
        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=active.n_mels,
            n_fft=active.n_fft,
            hop_length=active.hop_length,
        )
        blocks["logmel"] = librosa.power_to_db(mel_power, ref=np.max)
    if "chroma" in active.feature_order:
        blocks["chroma"] = librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate,
            n_chroma=active.n_chroma,
            n_fft=active.n_fft,
            hop_length=active.hop_length,
        )
    combined = np.vstack(tuple(blocks[name] for name in active.feature_order))
    if combined.shape[0] != active.feature_bins:
        raise AudioContractError(
            f"Feature order produced {combined.shape[0]} bins, expected {active.feature_bins}"
        )
    if combined.shape[1] < active.max_frames:
        combined = np.pad(
            combined,
            ((0, 0), (0, active.max_frames - combined.shape[1])),
            mode="constant",
        )
    else:
        combined = combined[:, : active.max_frames]
    features = np.asarray(combined[..., np.newaxis], dtype=np.float32)
    if features.shape != active.feature_shape:
        raise AudioContractError(
            f"Feature tensor has shape {features.shape}, expected {active.feature_shape}"
        )
    if not np.isfinite(features).all():
        raise AudioContractError("Feature tensor contains non-finite values")
    return features


def _cache_key(
    cache: FeatureCache,
    record: OriginalRecord,
    config: PreprocessingConfig,
    augmentation: Mapping[str, Any] | None,
) -> str:
    return cache.key(
        source_sha256=record.sha256,
        preprocessing=config.to_dict(),
        augmentation=augmentation,
        dtype=config.dtype,
        shape=config.feature_shape,
    )


def _original_features(
    record: OriginalRecord,
    *,
    config: PreprocessingConfig,
    cache: FeatureCache | None,
) -> np.ndarray:
    key = _cache_key(cache, record, config, None) if cache is not None else None
    if cache is not None and key is not None:
        cached = cache.get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
    waveform, sample_rate = load_preprocessed_waveform(record.filepath, config=config)
    features = extract_features(waveform, sample_rate=sample_rate, config=config)
    if cache is not None and key is not None:
        cache.put(key, features)
    return features


def assert_validation_originals_only(sample_kinds: Iterable[str]) -> None:
    kinds = tuple(sample_kinds)
    invalid = sorted({kind for kind in kinds if kind != "original"})
    if invalid:
        raise AudioContractError(
            f"Validation is original-only; found sample kinds {invalid!r}"
        )


def extract_original_tensors(
    records: Iterable[OriginalRecord],
    *,
    config: PreprocessingConfig | None = None,
    partition_name: str = "Held-out test",
    feature_cache: FeatureCache | None = None,
) -> OriginalTensors:
    """Materialize original-only tensors for validation or held-out testing."""

    active = config or PreprocessingConfig()
    originals = tuple(records)
    if not originals:
        raise AudioContractError(f"{partition_name} records cannot be empty")
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    record_ids: list[str] = []
    sample_kinds: list[str] = []
    for record in originals:
        try:
            features = _original_features(
                record, config=active, cache=feature_cache
            )
        except AudioContractError as exc:
            raise AudioContractError(
                f"{partition_name} original audio failed "
                f"(record_id={record.record_id!r}, filepath={record.filepath}): {exc}"
            ) from exc
        feature_rows.append(features)
        labels.append(record.label)
        record_ids.append(record.record_id)
        sample_kinds.append("original")
    assert_validation_originals_only(sample_kinds)
    tensor = np.stack(feature_rows).astype(np.float32, copy=False)
    if not np.isfinite(tensor).all():
        raise AudioContractError(f"{partition_name} tensors contain non-finite values")
    return OriginalTensors(
        features=tensor,
        labels=tuple(labels),
        record_ids=tuple(record_ids),
        sample_kinds=tuple(sample_kinds),
    )


def extract_fold_tensors(
    train_records: Iterable[OriginalRecord],
    validation_records: Iterable[OriginalRecord],
    augmentation_rows: Iterable[AugmentationPlanRow],
    *,
    config: PreprocessingConfig | None = None,
    allow_empty_validation: bool = False,
    feature_cache: FeatureCache | None = None,
) -> FoldTensors:
    """Materialize one fold while enforcing training-only augmentation."""

    active = config or PreprocessingConfig()
    training = tuple(train_records)
    validation = tuple(validation_records)
    planned = tuple(augmentation_rows)
    train_by_id = {record.record_id: record for record in training}
    validation_ids = {record.record_id for record in validation}
    if set(train_by_id) & validation_ids:
        raise AudioContractError("Train and Validation share original record IDs")
    for row in planned:
        if row.partition != "train":
            raise AudioContractError(
                f"Augmentation {row.sample_id} is not assigned to Training"
            )
        if row.original_record_id not in train_by_id:
            raise AudioContractError(
                f"Augmentation source is outside Training: {row.original_record_id}"
            )
        if row.original_record_id in validation_ids:
            raise AudioContractError(
                f"Validation original used as augmentation source: {row.original_record_id}"
            )

    waveform_cache: dict[str, tuple[np.ndarray, int]] = {}

    def load(record: OriginalRecord) -> tuple[np.ndarray, int]:
        if record.record_id not in waveform_cache:
            waveform_cache[record.record_id] = load_preprocessed_waveform(
                record.filepath, config=active
            )
        return waveform_cache[record.record_id]

    train_features: list[np.ndarray] = []
    train_labels: list[str] = []
    train_sample_ids: list[str] = []
    train_sample_kinds: list[str] = []
    for record in training:
        try:
            features = _original_features(
                record, config=active, cache=feature_cache
            )
        except AudioContractError as exc:
            raise AudioContractError(
                "Training original audio failed "
                f"(record_id={record.record_id!r}, filepath={record.filepath}): {exc}"
            ) from exc
        train_features.append(features)
        train_labels.append(record.label)
        train_sample_ids.append(record.record_id)
        train_sample_kinds.append("original")
    for row in planned:
        source = train_by_id[row.original_record_id]
        try:
            augmentation_identity = {
                "type": row.augmentation_type,
                "parameters_json": row.augmentation_params_json,
                "seed": row.seed,
                "sample_id": row.sample_id,
            }
            key = (
                _cache_key(feature_cache, source, active, augmentation_identity)
                if feature_cache is not None
                else None
            )
            cached = feature_cache.get(key) if feature_cache is not None and key else None
            if cached is not None:
                features = np.asarray(cached, dtype=np.float32)
            else:
                waveform, sample_rate = load(source)
                derivative = apply_augmentation(
                    waveform,
                    sample_rate=sample_rate,
                    augmentation_type=row.augmentation_type,
                    augmentation_params_json=row.augmentation_params_json,
                    seed=row.seed,
                    config=active,
                )
                features = extract_features(
                    derivative, sample_rate=sample_rate, config=active
                )
                if feature_cache is not None and key is not None:
                    feature_cache.put(key, features)
        except AudioContractError as exc:
            raise AudioContractError(
                "Training augmented audio failed "
                f"(sample_id={row.sample_id!r}, source_record_id={source.record_id!r}, "
                f"filepath={source.filepath}): {exc}"
            ) from exc
        train_features.append(features)
        train_labels.append(row.label)
        train_sample_ids.append(row.sample_id)
        train_sample_kinds.append("augmented")

    validation_features: list[np.ndarray] = []
    validation_labels: list[str] = []
    validation_record_ids: list[str] = []
    validation_kinds: list[str] = []
    for record in validation:
        try:
            features = _original_features(
                record, config=active, cache=feature_cache
            )
        except AudioContractError as exc:
            raise AudioContractError(
                "Validation original audio failed "
                f"(record_id={record.record_id!r}, filepath={record.filepath}): {exc}"
            ) from exc
        validation_features.append(features)
        validation_labels.append(record.label)
        validation_record_ids.append(record.record_id)
        validation_kinds.append("original")
    assert_validation_originals_only(validation_kinds)

    if not train_features:
        raise AudioContractError("Training tensors cannot be empty")
    if not validation_features and not allow_empty_validation:
        raise AudioContractError("A fold must contain non-empty Validation tensors")
    train_tensor = np.stack(train_features).astype(np.float32, copy=False)
    validation_tensor = (
        np.stack(validation_features).astype(np.float32, copy=False)
        if validation_features
        else np.empty((0, *active.feature_shape), dtype=np.float32)
    )
    if not np.isfinite(train_tensor).all() or not np.isfinite(validation_tensor).all():
        raise AudioContractError("Fold tensors contain non-finite values")
    return FoldTensors(
        train_features=train_tensor,
        train_labels=tuple(train_labels),
        train_sample_ids=tuple(train_sample_ids),
        train_sample_kinds=tuple(train_sample_kinds),
        validation_features=validation_tensor,
        validation_labels=tuple(validation_labels),
        validation_record_ids=tuple(validation_record_ids),
        validation_sample_kinds=tuple(validation_kinds),
    )


def mixup_batch(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_samples: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, training-only Mixup rows after normalization."""

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if x.ndim < 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise AudioContractError("Mixup features and labels have incompatible shapes")
    if x.shape[0] < 2 and n_samples:
        raise AudioContractError("Mixup requires at least two Training samples")
    if n_samples < 0:
        raise AudioContractError("Mixup n_samples cannot be negative")
    if alpha <= 0.0:
        raise AudioContractError("Mixup alpha must be positive")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AudioContractError("Mixup inputs must contain only finite values")
    if n_samples == 0:
        return (
            np.empty((0, *x.shape[1:]), dtype=np.float32),
            np.empty((0, y.shape[1]), dtype=np.float32),
        )

    rng = np.random.default_rng(seed)
    left = rng.integers(0, x.shape[0], size=n_samples)
    offsets = rng.integers(1, x.shape[0], size=n_samples)
    right = (left + offsets) % x.shape[0]
    weights = rng.beta(alpha, alpha, size=n_samples).astype(np.float32)
    x_weights = weights.reshape((n_samples,) + (1,) * (x.ndim - 1))
    y_weights = weights[:, np.newaxis]
    mixed_x = x_weights * x[left] + (1.0 - x_weights) * x[right]
    mixed_y = y_weights * y[left] + (1.0 - y_weights) * y[right]
    return mixed_x.astype(np.float32), mixed_y.astype(np.float32)
