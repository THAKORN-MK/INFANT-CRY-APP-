"""Verified grouped-OOF aggregation and deterministic experiment selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cryinsight.training.artefacts import (
    OofPrediction,
    aggregate_oof_metrics,
    oof_metrics_payload,
    sha256_file,
    write_oof_predictions_csv,
)
from cryinsight.training.protocol import (
    OriginalRecord,
    ProtocolViolation,
    assert_exact_oof_coverage,
    write_json_atomic,
)

from .contracts import FoldResult


class ExperimentVerificationError(ValueError):
    """Raised when candidate evidence is unsafe to compare or publish."""


class ExperimentSelectionError(ValueError):
    """Raised when ranking evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class VerifiedProbabilities:
    probabilities: np.ndarray
    max_sum_deviation: float
    tolerance: float


def verify_probabilities(
    probabilities: Any,
    *,
    tolerance: float = 1e-5,
) -> VerifiedProbabilities:
    """Validate probability rows and normalize only bounded numeric drift."""

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Probability tolerance must be finite and positive")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ExperimentVerificationError(
            "Probabilities must be a non-empty two-dimensional class matrix"
        )
    if not np.isfinite(values).all():
        raise ExperimentVerificationError("Probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ExperimentVerificationError("Probabilities must be within [0, 1]")
    row_sums = values.sum(axis=1, dtype=np.float64)
    deviation = np.abs(row_sums - 1.0)
    max_deviation = float(deviation.max(initial=0.0))
    if np.any(row_sums <= 0.0) or max_deviation > float(tolerance):
        raise ExperimentVerificationError(
            "Probability rows do not sum to one within "
            f"tolerance {tolerance:g}; maximum deviation was {max_deviation:.12g}"
        )
    normalized = values / row_sums[:, np.newaxis]
    return VerifiedProbabilities(
        probabilities=np.asarray(normalized, dtype=np.float64),
        max_sum_deviation=max_deviation,
        tolerance=float(tolerance),
    )


def _validated_fold_results(
    results: Iterable[FoldResult],
    *,
    label_order: tuple[str, ...],
    probability_tolerance: float,
) -> tuple[tuple[FoldResult, VerifiedProbabilities], ...]:
    rows = tuple(results)
    if not rows:
        raise ExperimentVerificationError("Candidate seed has no fold results")
    folds = [int(row.fold) for row in rows]
    if sorted(folds) != [1, 2, 3, 4, 5]:
        raise ExperimentVerificationError(
            "Candidate seed requires exactly folds 1, 2, 3, 4, and 5 once each"
        )
    candidate_ids = {row.candidate_id for row in rows}
    seeds = {int(row.seed) for row in rows}
    if len(candidate_ids) != 1 or len(seeds) != 1:
        raise ExperimentVerificationError(
            "All fold results must belong to one candidate and one seed"
        )
    validated: list[tuple[FoldResult, VerifiedProbabilities]] = []
    allowed_labels = set(label_order)
    for row in sorted(rows, key=lambda item: int(item.fold)):
        if len(row.validation_record_ids) != len(row.true_labels):
            raise ExperimentVerificationError(
                f"Fold {row.fold} validation IDs and labels have different lengths"
            )
        verified = verify_probabilities(
            row.probabilities,
            tolerance=probability_tolerance,
        )
        if verified.probabilities.shape != (
            len(row.validation_record_ids),
            len(label_order),
        ):
            raise ExperimentVerificationError(
                f"Fold {row.fold} probability shape does not match its records/classes"
            )
        unknown = sorted(set(row.true_labels) - allowed_labels)
        if unknown:
            raise ExperimentVerificationError(
                f"Fold {row.fold} contains unknown labels: {unknown}"
            )
        if not Path(row.model_path).is_file() or not Path(row.manifest_path).is_file():
            raise ExperimentVerificationError(
                f"Fold {row.fold} is missing its model or manifest artefact"
            )
        validated.append((row, verified))
    return tuple(validated)


def aggregate_candidate_seed(
    expected_records: Iterable[OriginalRecord],
    fold_results: Iterable[FoldResult],
    *,
    label_order: Sequence[str],
    output_dir: str | Path,
    probability_tolerance: float = 1e-5,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Pool one candidate/seed's five folds and seal verified OOF artefacts."""

    expected = tuple(expected_records)
    labels = tuple(str(label) for label in label_order)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ExperimentVerificationError("Label order must contain unique classes")
    validated = _validated_fold_results(
        fold_results,
        label_order=labels,
        probability_tolerance=probability_tolerance,
    )
    expected_by_id = {record.record_id: record for record in expected}
    prediction_ids = tuple(
        record_id
        for result, _ in validated
        for record_id in result.validation_record_ids
    )
    try:
        assert_exact_oof_coverage(expected_by_id, prediction_ids)
    except ProtocolViolation as exc:
        raise ExperimentVerificationError(str(exc)) from exc

    candidate_id = validated[0][0].candidate_id
    seed = int(validated[0][0].seed)
    oof_rows: list[OofPrediction] = []
    max_deviation = 0.0
    fold_hashes: dict[str, dict[str, str]] = {}
    parameter_counts: dict[str, int] = {}
    adapters: set[str] = set()
    for result, verified in validated:
        max_deviation = max(max_deviation, verified.max_sum_deviation)
        fold_hashes[f"fold_{result.fold}"] = {
            "model": sha256_file(result.model_path),
            "manifest": sha256_file(result.manifest_path),
        }
        try:
            manifest_payload = json.loads(
                Path(result.manifest_path).read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentVerificationError(
                f"Could not read fold manifest {result.manifest_path}: {exc}"
            ) from exc
        if isinstance(manifest_payload, Mapping) and "parameter_count" in manifest_payload:
            count = manifest_payload["parameter_count"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ExperimentVerificationError("Invalid fold parameter count")
            parameter_counts[str(result.fold)] = count
            adapters.add(str(manifest_payload.get("adapter", "neural")))
        for position, record_id in enumerate(result.validation_record_ids):
            source = expected_by_id.get(record_id)
            if source is None:
                raise ExperimentVerificationError(
                    f"Unexpected OOF prediction: {record_id!r}"
                )
            true_label = result.true_labels[position]
            if true_label != source.label:
                raise ExperimentVerificationError(
                    f"OOF label provenance mismatch for {record_id}"
                )
            scores = tuple(float(value) for value in verified.probabilities[position])
            predicted_label = labels[int(np.argmax(verified.probabilities[position]))]
            oof_rows.append(
                OofPrediction(
                    record_id=record_id,
                    filepath=str(source.filepath),
                    label=source.label,
                    group_id=source.group_id,
                    fold=int(result.fold),
                    predicted_label=predicted_label,
                    scores=scores,
                    sample_kind="original",
                    model_path=str(Path(result.model_path)),
                    normalizer_path="recorded_in_fold_manifest",
                    run_id=f"{candidate_id}:seed_{seed}",
                )
            )
    oof_rows.sort(key=lambda row: (int(row.fold), row.record_id))
    try:
        metrics = aggregate_oof_metrics(
            expected,
            oof_rows,
            label_order=labels,
            bootstrap_iterations=int(bootstrap_iterations),
            bootstrap_seed=int(bootstrap_seed),
        )
    except (ProtocolViolation, ValueError) as exc:
        raise ExperimentVerificationError(str(exc)) from exc

    if len(adapters) > 1:
        raise ExperimentVerificationError("Adapter changed across folds")
    adapter = next(iter(adapters), "neural")
    counts = list(parameter_counts.values()) or [0]
    if adapter != "classical" and len(set(counts)) > 1:
        raise ExperimentVerificationError(
            "Model parameter count changed across folds for one candidate seed"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "oof_predictions.csv"
    metrics_path = output / "oof_metrics.json"
    summary_path = output / "seed_summary.json"
    verification_path = output / "verification.json"
    write_oof_predictions_csv(predictions_path, oof_rows, label_order=labels)
    write_json_atomic(metrics_path, oof_metrics_payload(metrics))

    class_recalls = {
        label: float(metrics.classification_report[label]["recall"])
        for label in labels
    }
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "candidate_id": candidate_id,
        "seed": seed,
        "folds": [1, 2, 3, 4, 5],
        "oof_support": len(oof_rows),
        "pooled_metrics": metrics.pooled_metrics,
        "class_recall": class_recalls,
        "minimum_class_recall": min(class_recalls.values()),
        "adapter": adapter,
        "parameter_count": statistics.fmean(counts),
        "parameter_count_mean": statistics.fmean(counts),
        "parameter_count_min": min(counts),
        "parameter_count_max": max(counts),
        "fold_parameter_counts": parameter_counts,
        "probability_max_sum_deviation_before_normalization": max_deviation,
        "probability_tolerance": float(probability_tolerance),
        "heldout_test_used_for_selection": False,
    }
    write_json_atomic(summary_path, summary)
    verification = {
        "schema_version": "1.0",
        "status": "complete",
        "candidate_id": candidate_id,
        "seed": seed,
        "folds": [1, 2, 3, 4, 5],
        "oof_support": len(oof_rows),
        "probability_validation": {
            "dtype": "float64",
            "tolerance": float(probability_tolerance),
            "max_sum_deviation_before_normalization": max_deviation,
        },
        "fold_artefact_sha256": fold_hashes,
        "artefact_sha256": {
            "oof_predictions": sha256_file(predictions_path),
            "oof_metrics": sha256_file(metrics_path),
            "seed_summary": sha256_file(summary_path),
        },
        "selection_scope": "corrected_grouped_internal_validation",
        "heldout_test_used_for_selection": False,
    }
    write_json_atomic(verification_path, verification)
    return summary


_SCREENING_TOLERANCE = 0.005
_CONFIRMATION_SEEDS = (42, 123, 2026)


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentSelectionError(f"Missing or invalid selection value: {key}") from exc
    if not math.isfinite(value):
        raise ExperimentSelectionError(f"Selection value must be finite: {key}")
    return value


def _parameter_count(row: Mapping[str, Any]) -> float:
    try:
        value = float(row["parameter_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentSelectionError("Missing or invalid parameter_count") from exc
    if not math.isfinite(value) or value < 0:
        raise ExperimentSelectionError("parameter_count cannot be negative")
    return value


def _verified_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload = [dict(row) for row in rows]
    if not payload:
        raise ExperimentSelectionError("No candidate rows are available for ranking")
    candidate_ids = [str(row.get("candidate_id", "")).strip() for row in payload]
    if any(not value for value in candidate_ids):
        raise ExperimentSelectionError("Every ranking row requires a candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ExperimentSelectionError("Screening candidate IDs must be unique")
    incomplete = sorted(
        candidate_ids[index]
        for index, row in enumerate(payload)
        if row.get("verification_status") != "complete"
    )
    if incomplete:
        raise ExperimentSelectionError(
            f"Only complete verified candidates may be ranked: {incomplete}"
        )
    return payload


def _tolerance_ranking(
    rows: list[dict[str, Any]],
    *,
    primary: str,
    secondary: str,
    final_key: Any,
    secondary_tolerance: float = _SCREENING_TOLERANCE,
) -> list[dict[str, Any]]:
    remaining = list(rows)
    ordered: list[dict[str, Any]] = []
    while remaining:
        primary_best = max(_finite_float(row, primary) for row in remaining)
        primary_group = [
            row
            for row in remaining
            if primary_best - _finite_float(row, primary) <= _SCREENING_TOLERANCE
        ]
        secondary_remaining = list(primary_group)
        while secondary_remaining:
            secondary_best = max(
                _finite_float(row, secondary) for row in secondary_remaining
            )
            tied = [
                row
                for row in secondary_remaining
                if secondary_best - _finite_float(row, secondary)
                <= secondary_tolerance
            ]
            tied.sort(key=final_key)
            ordered.extend(tied)
            tied_ids = {id(row) for row in tied}
            secondary_remaining = [
                row for row in secondary_remaining if id(row) not in tied_ids
            ]
        group_ids = {id(row) for row in primary_group}
        remaining = [row for row in remaining if id(row) not in group_ids]
    return [dict(row, rank=index) for index, row in enumerate(ordered, start=1)]


def rank_screening_candidates(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank Wave A/B rows by the frozen tolerance and tie-break rules."""

    payload = _verified_rows(rows)
    for row in payload:
        _finite_float(row, "oof_macro_f1")
        _finite_float(row, "minimum_class_recall")
        _parameter_count(row)
    return _tolerance_ranking(
        payload,
        primary="oof_macro_f1",
        secondary="minimum_class_recall",
        final_key=lambda row: (
            _parameter_count(row),
            str(row["candidate_id"]),
        ),
    )


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_repeated_seeds(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate exactly the three frozen Wave C seeds per candidate."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ExperimentSelectionError("Every seed row requires a candidate_id")
        grouped.setdefault(candidate_id, []).append(row)
    if not grouped:
        raise ExperimentSelectionError("No repeated-seed rows are available")
    summaries: list[dict[str, Any]] = []
    metrics = (
        "oof_macro_f1",
        "oof_balanced_accuracy",
        "oof_accuracy",
        "minimum_class_recall",
    )
    expected = set(_CONFIRMATION_SEEDS)
    for candidate_id, candidate_rows in sorted(grouped.items()):
        seeds = [int(row.get("seed", -1)) for row in candidate_rows]
        if set(seeds) != expected or len(seeds) != len(expected):
            raise ExperimentSelectionError(
                f"Wave C candidate {candidate_id} requires seeds 42, 123, 2026 exactly"
            )
        if any(row.get("verification_status") != "complete" for row in candidate_rows):
            raise ExperimentSelectionError(
                f"Wave C candidate {candidate_id} has incomplete seed verification"
            )
        parameter_counts = [_parameter_count(row) for row in candidate_rows]
        adapters = {row.get("adapter", "neural") for row in candidate_rows}
        if len(adapters) != 1:
            raise ExperimentSelectionError("Adapter changed across seeds")
        adapter = next(iter(adapters))
        if adapter != "classical" and len(set(parameter_counts)) != 1:
            raise ExperimentSelectionError(
                f"Wave C candidate {candidate_id} parameter count changed across seeds"
            )
        summary: dict[str, Any] = {
            "candidate_id": candidate_id,
            "seeds": list(_CONFIRMATION_SEEDS),
            "adapter": adapter,
            "parameter_count": statistics.fmean(parameter_counts),
            "parameter_count_mean": statistics.fmean(parameter_counts),
            "parameter_count_min": min(parameter_counts),
            "parameter_count_max": max(parameter_counts),
            "seed_parameter_counts": {str(row["seed"]): _parameter_count(row) for row in candidate_rows},
            "verification_status": "complete",
        }
        for metric in metrics:
            values = [_finite_float(row, metric) for row in candidate_rows]
            stats = _metric_stats(values)
            if metric == "minimum_class_recall":
                stem = "minimum_class_recall"
            else:
                stem = metric
            summary[f"mean_{stem}"] = stats["mean"]
            summary[f"std_{stem}"] = stats["std"]
            summary[f"min_{stem}"] = stats["min"]
            summary[f"max_{stem}"] = stats["max"]
        summaries.append(summary)
    return summaries


def rank_confirmation_candidates(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank verified Wave C summaries with stability-aware tie-breaks."""

    payload = _verified_rows(rows)
    for row in payload:
        if list(row.get("seeds", [])) != list(_CONFIRMATION_SEEDS):
            raise ExperimentSelectionError(
                "Confirmation ranking requires seeds 42, 123, 2026"
            )
        _finite_float(row, "mean_oof_macro_f1")
        _finite_float(row, "mean_minimum_class_recall")
        _finite_float(row, "std_oof_macro_f1")
        _parameter_count(row)
    return _tolerance_ranking(
        payload,
        primary="mean_oof_macro_f1",
        secondary="mean_minimum_class_recall",
        secondary_tolerance=0.0,
        final_key=lambda row: (
            _finite_float(row, "std_oof_macro_f1"),
            _parameter_count(row),
            str(row["candidate_id"]),
        ),
    )


def promotion_decision(
    winner: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Recommend Training 3 only when every predeclared OOF gate passes."""

    checks = {
        "three_seed_confirmation": list(winner.get("seeds", []))
        == list(_CONFIRMATION_SEEDS),
        "macro_f1_gain_at_least_0_01": _finite_float(
            winner, "mean_oof_macro_f1"
        )
        >= _finite_float(reference, "oof_macro_f1") + 0.01,
        "balanced_accuracy_not_lower": _finite_float(
            winner, "mean_oof_balanced_accuracy"
        )
        >= _finite_float(reference, "oof_balanced_accuracy"),
        "minimum_recall_drop_within_0_02": _finite_float(
            winner, "mean_minimum_class_recall"
        )
        >= _finite_float(reference, "minimum_class_recall") - 0.02,
        "verification_complete": winner.get("verification_status") == "complete",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "1.0",
        "candidate_id": str(winner.get("candidate_id", "")),
        "status": (
            "recommended_for_training_3"
            if all(checks.values())
            else "no_promotion_recommended"
        ),
        "checks": checks,
        "failed_checks": failed,
        "selection_scope": "corrected_grouped_internal_validation",
        "heldout_test_used_for_selection": False,
    }


__all__ = [
    "ExperimentSelectionError",
    "ExperimentVerificationError",
    "VerifiedProbabilities",
    "aggregate_candidate_seed",
    "aggregate_repeated_seeds",
    "promotion_decision",
    "rank_confirmation_candidates",
    "rank_screening_candidates",
    "verify_probabilities",
]
