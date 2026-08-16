"""Immutable training artefacts, normalizer identity, and OOF evaluation."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cryinsight.training.protocol import (
    AugmentationPlanRow,
    DependencyUnavailable,
    OriginalRecord,
    ProtocolViolation,
    assert_exact_oof_coverage,
    write_json_atomic,
)


class ArtefactError(ValueError):
    """Raised when a saved artefact is incomplete, altered, or mismatched."""


@dataclass(frozen=True)
class NormalizerStats:
    mean: float
    std: float
    epsilon: float
    axis: str
    feature_shape: tuple[int, int, int]
    dtype: str
    fit_sample_count: int
    run_id: str
    fold: int | str


@dataclass(frozen=True)
class OofPrediction:
    record_id: str
    filepath: str
    label: str
    group_id: str
    fold: int | str
    predicted_label: str
    scores: tuple[float, ...]
    sample_kind: str
    model_path: str
    normalizer_path: str
    run_id: str


@dataclass(frozen=True)
class OofMetricsResult:
    pooled_metrics: dict[str, Any]
    classification_report: dict[str, Any]
    confusion_matrix: np.ndarray
    per_fold_metrics: tuple[dict[str, Any], ...]
    fold_summary: dict[str, dict[str, float]]
    bootstrap_intervals: dict[str, dict[str, float | int | str]]


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{secrets.token_hex(4)}"


def create_run_directory(stage_root: str | Path, *, run_id: str | None = None) -> Path:
    """Create a new immutable ``runs/<run_id>`` directory."""

    identifier = run_id or _default_run_id()
    if _RUN_ID_PATTERN.fullmatch(identifier) is None:
        raise ArtefactError(
            "run_id must contain only letters, numbers, '.', '_' or '-' and must "
            "not contain path separators"
        )
    runs_root = Path(stage_root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    output = runs_root / identifier
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to reuse immutable run directory: {output}"
        ) from exc
    return output


def select_final_refit_epoch(
    fold_metrics: Iterable[Mapping[str, Any]],
    *,
    expected_folds: int,
) -> dict[str, Any]:
    """Freeze final-refit duration from the median fold-selected epoch."""

    if expected_folds < 1:
        raise ArtefactError("expected_folds must be positive")
    rows: dict[int, int] = {}
    for raw in fold_metrics:
        try:
            fold = int(raw["fold"])
            best_epoch = int(raw["best_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtefactError(
                "Fold metrics require integer fold and best_epoch values"
            ) from exc
        if fold in rows or best_epoch < 1:
            raise ArtefactError(
                f"Final refit requires exactly folds 1..{expected_folds} once each"
            )
        rows[fold] = best_epoch

    expected = set(range(1, expected_folds + 1))
    if set(rows) != expected:
        raise ArtefactError(
            f"Final refit requires exactly folds 1..{expected_folds} once each"
        )
    ordered_epochs = [rows[fold] for fold in range(1, expected_folds + 1)]
    return {
        "selection_rule": "median_fold_best_epoch",
        "selection_source": "internal_fold_validation_only",
        "fold_best_epochs": ordered_epochs,
        "final_epoch": int(statistics.median(ordered_epochs)),
        "heldout_test_used_for_selection": False,
    }


def write_incomplete_run_verification(
    run_dir: str | Path,
    *,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    """Seal a failed immutable training run with machine-readable evidence."""

    run_path = Path(run_dir)
    completed_folds = sum(
        1
        for fold_dir in run_path.glob("fold_*")
        if fold_dir.is_dir() and (fold_dir / "fold_manifest.json").is_file()
    )
    payload = {
        "schema_version": "1.0",
        "status": "incomplete",
        "run_id": run_path.name,
        "stage": stage,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "training_started": True,
        "folds_completed": completed_folds,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "accuracy_status": "NOT_AVAILABLE_INCOMPLETE_RUN",
    }
    write_json_atomic(run_path / "verification.json", payload)
    return payload


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise ArtefactError(f"Artefact file does not exist: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_hash(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ArtefactError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected_sha256}, found {actual}"
        )


def _metadata_path(normalizer_path: Path) -> Path:
    return normalizer_path.with_suffix(normalizer_path.suffix + ".metadata.json")


def _validate_normalizer(stats: NormalizerStats) -> None:
    if not math.isfinite(stats.mean):
        raise ArtefactError("Normalizer mean must be finite")
    if not math.isfinite(stats.std) or stats.std <= stats.epsilon:
        raise ArtefactError(
            f"Normalizer std must be finite and greater than epsilon {stats.epsilon}"
        )
    if stats.epsilon <= 0.0 or not math.isfinite(stats.epsilon):
        raise ArtefactError("Normalizer epsilon must be finite and positive")
    if stats.axis != "global_scalar":
        raise ArtefactError("Only the audited global_scalar normalizer is supported")
    if len(stats.feature_shape) != 3 or any(dimension <= 0 for dimension in stats.feature_shape):
        raise ArtefactError(f"Invalid feature shape: {stats.feature_shape}")
    if stats.dtype != "float32":
        raise ArtefactError("Normalizer dtype must be float32")
    if stats.fit_sample_count <= 0:
        raise ArtefactError("Normalizer fit_sample_count must be positive")
    if not stats.run_id:
        raise ArtefactError("Normalizer run_id is required")
    if isinstance(stats.fold, int):
        if stats.fold < 1:
            raise ArtefactError("Normalizer fold must start at 1")
    elif stats.fold != "final_refit":
        raise ArtefactError("Normalizer identity must be a fold number or final_refit")


def _save_npy_once(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable artefact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to replace immutable artefact: {path}")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise


def save_normalizer(path: str | Path, stats: NormalizerStats) -> None:
    """Save compatibility numeric stats plus mandatory identity metadata."""

    _validate_normalizer(stats)
    output = Path(path)
    metadata = _metadata_path(output)
    if output.exists() or metadata.exists():
        collision = output if output.exists() else metadata
        raise FileExistsError(f"Refusing to replace immutable artefact: {collision}")
    _save_npy_once(output, np.asarray([stats.mean, stats.std], dtype=np.float64))
    payload = asdict(stats)
    payload["feature_shape"] = list(stats.feature_shape)
    payload["normalizer_path"] = str(output)
    payload["normalizer_sha256"] = sha256_file(output)
    try:
        write_json_atomic(metadata, payload)
    except BaseException:
        # A partial bundle is intentionally left detectable rather than silently
        # replacing an immutable file.  The enclosing run must then be discarded.
        raise


def load_normalizer(
    path: str | Path,
    *,
    expected_run_id: str,
    expected_fold: int | str,
) -> NormalizerStats:
    output = Path(path)
    metadata_path = _metadata_path(output)
    if not metadata_path.is_file():
        raise ArtefactError(f"Normalizer metadata is missing: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtefactError(f"Could not read normalizer metadata: {exc}") from exc
    if payload.get("run_id") != expected_run_id:
        raise ArtefactError(
            f"Normalizer run_id mismatch: expected {expected_run_id!r}, "
            f"found {payload.get('run_id')!r}"
        )
    saved_fold = payload.get("fold", -1)
    if isinstance(expected_fold, int):
        try:
            saved_fold = int(saved_fold)
        except (TypeError, ValueError):
            saved_fold = -1
    if saved_fold != expected_fold:
        raise ArtefactError(
            f"Normalizer fold mismatch: expected {expected_fold}, "
            f"found {payload.get('fold')!r}"
        )
    verify_file_hash(output, str(payload.get("normalizer_sha256", "")))
    try:
        values = np.load(output, allow_pickle=False)
    except Exception as exc:
        raise ArtefactError(f"Could not load normalizer array {output}: {exc}") from exc
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ArtefactError(
            f"Normalizer array must contain two finite scalars, received {values.shape}"
        )
    stats = NormalizerStats(
        mean=float(payload["mean"]),
        std=float(payload["std"]),
        epsilon=float(payload["epsilon"]),
        axis=str(payload["axis"]),
        feature_shape=tuple(int(value) for value in payload["feature_shape"]),
        dtype=str(payload["dtype"]),
        fit_sample_count=int(payload["fit_sample_count"]),
        run_id=str(payload["run_id"]),
        fold=(
            int(payload["fold"])
            if isinstance(payload["fold"], int)
            else str(payload["fold"])
        ),
    )
    _validate_normalizer(stats)
    if not np.allclose(values, [stats.mean, stats.std], rtol=0.0, atol=1e-12):
        raise ArtefactError("Normalizer numeric values disagree with metadata")
    return stats


def _validate_oof_contract(
    expected_records: Sequence[OriginalRecord],
    predictions: Sequence[OofPrediction],
    label_order: Sequence[str],
) -> None:
    if not expected_records:
        raise ProtocolViolation("Expected OOF cohort is empty")
    if len(label_order) < 2 or len(set(label_order)) != len(label_order):
        raise ProtocolViolation("OOF label order must contain unique class names")
    non_original = sorted(
        row.record_id for row in predictions if row.sample_kind != "original"
    )
    if non_original:
        raise ProtocolViolation(
            f"OOF support must contain original records only: {non_original!r}"
        )
    assert_exact_oof_coverage(
        [record.record_id for record in expected_records],
        [row.record_id for row in predictions],
    )
    expected_by_id = {record.record_id: record for record in expected_records}
    allowed_labels = set(label_order)
    for row in predictions:
        source = expected_by_id[row.record_id]
        if row.label != source.label or row.group_id != source.group_id:
            raise ProtocolViolation(
                f"OOF provenance mismatch for record {row.record_id}"
            )
        if row.label not in allowed_labels or row.predicted_label not in allowed_labels:
            raise ProtocolViolation(f"Unknown OOF label for record {row.record_id}")
        if len(row.scores) != len(label_order):
            raise ProtocolViolation(
                f"OOF score width mismatch for record {row.record_id}"
            )
        scores = np.asarray(row.scores, dtype=np.float64)
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ProtocolViolation(f"Invalid OOF scores for record {row.record_id}")
        if not np.isclose(float(scores.sum()), 1.0, atol=1e-4):
            raise ProtocolViolation(
                f"OOF scores do not sum to one for record {row.record_id}"
            )
        argmax_label = label_order[int(np.argmax(scores))]
        if row.predicted_label != argmax_label:
            raise ProtocolViolation(
                f"OOF predicted label disagrees with scores for record {row.record_id}"
            )


def _require_sklearn_metrics():
    try:
        from sklearn import metrics  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyUnavailable(
            "scikit-learn is required to compute corrected OOF metrics"
        ) from exc
    return metrics


def _metric_values(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    scores: np.ndarray,
    *,
    label_indices: Sequence[int],
    metrics_module: Any,
) -> dict[str, float | None]:
    all_labels_observed = set(np.unique(true_indices).tolist()) == set(label_indices)
    one_hot = np.eye(len(label_indices), dtype=np.float64)[true_indices]
    confidence = scores.max(axis=1)
    correct = (predicted_indices == true_indices).astype(np.float64)
    calibration_gap = 0.0
    bin_edges = np.linspace(0.0, 1.0, 11)
    for bin_index in range(10):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == 0:
            members = (confidence >= lower) & (confidence <= upper)
        else:
            members = (confidence > lower) & (confidence <= upper)
        if members.any():
            calibration_gap += float(members.mean()) * abs(
                float(correct[members].mean()) - float(confidence[members].mean())
            )

    values: dict[str, float | None] = {
        "accuracy": float(metrics_module.accuracy_score(true_indices, predicted_indices)),
        "balanced_accuracy": (
            float(metrics_module.balanced_accuracy_score(true_indices, predicted_indices))
            if all_labels_observed
            else None
        ),
        "macro_precision": float(
            metrics_module.precision_score(
                true_indices,
                predicted_indices,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            metrics_module.recall_score(
                true_indices,
                predicted_indices,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            metrics_module.f1_score(
                true_indices,
                predicted_indices,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            metrics_module.f1_score(
                true_indices,
                predicted_indices,
                labels=label_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": float(
            metrics_module.log_loss(true_indices, scores, labels=list(label_indices))
        ),
        "brier_score": float(np.mean(np.sum(np.square(scores - one_hot), axis=1))),
        "expected_calibration_error": calibration_gap,
    }
    if len(label_indices) == 2:
        matrix = metrics_module.confusion_matrix(
            true_indices,
            predicted_indices,
            labels=list(label_indices),
        )
        true_negative, false_positive, false_negative, true_positive = (
            int(value) for value in matrix.ravel()
        )
        positive_total = true_positive + false_negative
        negative_total = true_negative + false_positive
        values["sensitivity"] = (
            float(true_positive / positive_total) if positive_total else None
        )
        values["specificity"] = (
            float(true_negative / negative_total) if negative_total else None
        )
    try:
        if not all_labels_observed:
            raise ValueError("ROC AUC requires every configured class in y_true")
        if len(label_indices) == 2:
            values["roc_auc"] = float(
                metrics_module.roc_auc_score(true_indices, scores[:, 1])
            )
        else:
            values["roc_auc_macro_ovr"] = float(
                metrics_module.roc_auc_score(
                    true_indices,
                    scores,
                    labels=label_indices,
                    multi_class="ovr",
                    average="macro",
                )
            )
            values["roc_auc_weighted_ovr"] = float(
                metrics_module.roc_auc_score(
                    true_indices,
                    scores,
                    labels=label_indices,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except ValueError:
        key = "roc_auc" if len(label_indices) == 2 else "roc_auc_macro_ovr"
        values[key] = None
        if len(label_indices) > 2:
            values["roc_auc_weighted_ovr"] = None
    return values


def group_bootstrap_intervals(
    predictions: Sequence[OofPrediction],
    *,
    label_order: Sequence[str],
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float | int | str]]:
    """Calculate percentile intervals by resampling complete provenance groups."""

    if iterations <= 0:
        return {}
    if not 0.0 < confidence_level < 1.0:
        raise ProtocolViolation("confidence_level must be between zero and one")
    metrics_module = _require_sklearn_metrics()
    label_to_index = {label: index for index, label in enumerate(label_order)}
    group_rows: dict[str, list[OofPrediction]] = defaultdict(list)
    for row in predictions:
        group_rows[row.group_id].append(row)
    groups = sorted(group_rows)
    if len(groups) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        rows = [row for group_id in chosen for row in group_rows[str(group_id)]]
        true_indices = np.asarray([label_to_index[row.label] for row in rows], dtype=int)
        predicted_indices = np.asarray(
            [label_to_index[row.predicted_label] for row in rows], dtype=int
        )
        scores = np.asarray([row.scores for row in rows], dtype=np.float64)
        values = _metric_values(
            true_indices,
            predicted_indices,
            scores,
            label_indices=tuple(range(len(label_order))),
            metrics_module=metrics_module,
        )
        for name, value in values.items():
            if value is not None and math.isfinite(value):
                samples[name].append(float(value))

    tail = (1.0 - confidence_level) / 2.0
    intervals: dict[str, dict[str, float | int | str]] = {}
    for name, values in sorted(samples.items()):
        if not values:
            continue
        intervals[name] = {
            "method": "group_percentile_bootstrap",
            "confidence_level": confidence_level,
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
            "valid_iterations": len(values),
            "requested_iterations": iterations,
            "seed": seed,
        }
    return intervals


def aggregate_oof_metrics(
    expected_records: Iterable[OriginalRecord],
    predictions: Iterable[OofPrediction],
    *,
    label_order: Sequence[str],
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
) -> OofMetricsResult:
    """Compute corrected pooled metrics from original OOF predictions only."""

    expected = tuple(expected_records)
    rows = tuple(predictions)
    labels = tuple(label_order)
    _validate_oof_contract(expected, rows, labels)
    metrics_module = _require_sklearn_metrics()
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_indices = np.asarray([label_to_index[row.label] for row in rows], dtype=int)
    predicted_indices = np.asarray(
        [label_to_index[row.predicted_label] for row in rows], dtype=int
    )
    scores = np.asarray([row.scores for row in rows], dtype=np.float64)
    label_indices = tuple(range(len(labels)))

    pooled = _metric_values(
        true_indices,
        predicted_indices,
        scores,
        label_indices=label_indices,
        metrics_module=metrics_module,
    )
    pooled_metrics: dict[str, Any] = {
        "evaluation_scope": "corrected_grouped_internal_validation",
        "independent_external_validation_performed": False,
        "support_original_records": len(rows),
        **pooled,
    }
    report = metrics_module.classification_report(
        true_indices,
        predicted_indices,
        labels=label_indices,
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    confusion = metrics_module.confusion_matrix(
        true_indices, predicted_indices, labels=label_indices
    ).astype(int)

    per_fold: list[dict[str, Any]] = []
    for fold in sorted({row.fold for row in rows}):
        positions = np.asarray(
            [index for index, row in enumerate(rows) if row.fold == fold], dtype=int
        )
        fold_values = _metric_values(
            true_indices[positions],
            predicted_indices[positions],
            scores[positions],
            label_indices=label_indices,
            metrics_module=metrics_module,
        )
        per_fold.append(
            {
                "fold": fold,
                "support_original_records": int(len(positions)),
                **fold_values,
            }
        )

    summary: dict[str, dict[str, float]] = {}
    metric_names = sorted(
        {
            key
            for row in per_fold
            for key, value in row.items()
            if key not in {"fold", "support_original_records"} and value is not None
        }
    )
    for name in metric_names:
        values = np.asarray(
            [float(row[name]) for row in per_fold if row.get(name) is not None],
            dtype=np.float64,
        )
        if values.size:
            summary[name] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "fold_count": float(values.size),
            }

    intervals = group_bootstrap_intervals(
        rows,
        label_order=labels,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    return OofMetricsResult(
        pooled_metrics=pooled_metrics,
        classification_report=_json_ready(report),
        confusion_matrix=confusion,
        per_fold_metrics=tuple(per_fold),
        fold_summary=summary,
        bootstrap_intervals=intervals,
    )


def aggregate_heldout_metrics(
    expected_records: Iterable[OriginalRecord],
    predictions: Iterable[OofPrediction],
    *,
    label_order: Sequence[str],
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
) -> OofMetricsResult:
    """Compute final-refit metrics on the locked original-only test cohort."""

    result = aggregate_oof_metrics(
        expected_records,
        predictions,
        label_order=label_order,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    pooled_metrics = dict(result.pooled_metrics)
    pooled_metrics["evaluation_scope"] = "locked_internal_heldout_test"
    pooled_metrics["independent_external_validation_performed"] = False
    return OofMetricsResult(
        pooled_metrics=pooled_metrics,
        classification_report=result.classification_report,
        confusion_matrix=result.confusion_matrix,
        per_fold_metrics=result.per_fold_metrics,
        fold_summary=result.fold_summary,
        bootstrap_intervals=result.bootstrap_intervals,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _write_csv_once(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    from cryinsight.training.protocol import _write_text_once

    _write_text_once(path, buffer.getvalue())


def write_augmentation_manifest_csv(
    path: str | Path,
    rows: Iterable[AugmentationPlanRow],
) -> None:
    fieldnames = [
        "sample_id",
        "original_record_id",
        "original_filepath",
        "original_relative_path",
        "label",
        "partition",
        "fold",
        "augmentation_type",
        "augmentation_params",
        "augmentation_index",
        "seed",
        "output_reference",
    ]
    payload = []
    for row in rows:
        payload.append(
            {
                "sample_id": row.sample_id,
                "original_record_id": row.original_record_id,
                "original_filepath": row.original_filepath,
                "original_relative_path": row.original_relative_path,
                "label": row.label,
                "partition": row.partition,
                "fold": row.fold,
                "augmentation_type": row.augmentation_type,
                "augmentation_params": row.augmentation_params_json,
                "augmentation_index": row.augmentation_index,
                "seed": row.seed,
                "output_reference": f"virtual:{row.sample_id}",
            }
        )
    _write_csv_once(Path(path), fieldnames, payload)


def write_oof_predictions_csv(
    path: str | Path,
    rows: Iterable[OofPrediction],
    *,
    label_order: Sequence[str],
) -> None:
    score_fields = [f"score_{label}" for label in label_order]
    fieldnames = [
        "record_id",
        "filepath",
        "label",
        "group_id",
        "fold",
        "predicted_label",
        *score_fields,
        "sample_kind",
        "model_path",
        "normalizer_path",
        "run_id",
    ]
    payload = []
    for row in rows:
        item = {
            "record_id": row.record_id,
            "filepath": row.filepath,
            "label": row.label,
            "group_id": row.group_id,
            "fold": row.fold,
            "predicted_label": row.predicted_label,
            "sample_kind": row.sample_kind,
            "model_path": row.model_path,
            "normalizer_path": row.normalizer_path,
            "run_id": row.run_id,
        }
        item.update(
            {field: score for field, score in zip(score_fields, row.scores, strict=True)}
        )
        payload.append(item)
    _write_csv_once(Path(path), fieldnames, payload)


def write_fold_metrics_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = tuple(rows)
    fieldnames: list[str] = []
    for row in payload:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    _write_csv_once(Path(path), fieldnames, payload)


def write_confusion_matrix_csv(
    path: str | Path,
    matrix: np.ndarray,
    *,
    label_order: Sequence[str],
) -> None:
    values = np.asarray(matrix, dtype=int)
    if values.shape != (len(label_order), len(label_order)):
        raise ArtefactError("Confusion-matrix shape does not match label order")
    fieldnames = ["actual_label", *[f"predicted_{label}" for label in label_order]]
    rows = []
    for index, label in enumerate(label_order):
        row: dict[str, Any] = {"actual_label": label}
        row.update(
            {
                f"predicted_{predicted_label}": int(values[index, predicted_index])
                for predicted_index, predicted_label in enumerate(label_order)
            }
        )
        rows.append(row)
    _write_csv_once(Path(path), fieldnames, rows)


def render_confusion_matrix_png(
    path: str | Path,
    matrix: np.ndarray,
    *,
    label_order: Sequence[str],
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyUnavailable(
            "matplotlib is required to render the OOF confusion matrix"
        ) from exc
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable artefact: {output}")
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(max(6, len(label_order) + 1), 5))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(label_order)),
        yticks=np.arange(len(label_order)),
        xticklabels=label_order,
        yticklabels=label_order,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title,
    )
    threshold = values.max() / 2.0 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def build_fold_manifest(
    *,
    run_id: str,
    fold: int,
    artefact_paths: Mapping[str, str | Path],
    splitter_name: str,
    split_seed: int,
    checkpoint_monitor: str,
    best_epoch: int,
    feature_shape: Sequence[int],
    label_order: Sequence[str],
    training_original_count: int,
    training_augmented_count: int,
    validation_original_count: int,
    source_snapshot_identifier: str,
) -> dict[str, Any]:
    files = {
        name: {"path": str(Path(path)), "sha256": sha256_file(path)}
        for name, path in artefact_paths.items()
    }
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "fold": fold,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "splitter_name": splitter_name,
        "split_seed": split_seed,
        "checkpoint_monitor": checkpoint_monitor,
        "best_epoch": best_epoch,
        "feature_shape": list(feature_shape),
        "label_order": list(label_order),
        "training_original_count": training_original_count,
        "training_augmented_count": training_augmented_count,
        "validation_original_count": validation_original_count,
        "source_snapshot_identifier": source_snapshot_identifier,
    }


def oof_metrics_payload(result: OofMetricsResult) -> dict[str, Any]:
    return {
        "pooled_metrics": _json_ready(result.pooled_metrics),
        "classification_report": _json_ready(result.classification_report),
        "fold_summary": _json_ready(result.fold_summary),
        "bootstrap_intervals": _json_ready(result.bootstrap_intervals),
        "limitations": [
            "This study reports model development and corrected internal validation.",
            "Independent/external validation has not yet been performed.",
            "Fold estimates are correlated and are not independent experiments.",
        ],
    }


def heldout_metrics_payload(result: OofMetricsResult) -> dict[str, Any]:
    return {
        "pooled_metrics": _json_ready(result.pooled_metrics),
        "classification_report": _json_ready(result.classification_report),
        "bootstrap_intervals": _json_ready(result.bootstrap_intervals),
        "limitations": [
            "The locked test partition is evaluated once after final refit.",
            "Test results do not select epochs, hyperparameters, or the deployment model.",
            "Train and test originate from the same corpus; this is not external validation.",
            "The underlying corpus was available during legacy model development.",
        ],
    }
