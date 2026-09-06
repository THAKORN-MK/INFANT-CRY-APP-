"""Single-source candidate registry and experiment config validation."""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .contracts import CandidateSpec, ExperimentConfig


class ExperimentProtocolError(ValueError):
    """Raised when an experiment violates the frozen OOF-only protocol."""


def _candidate(
    candidate_id: str,
    stage: str,
    family: str,
    feature_view: str,
    adapter: str,
    model: str,
    augmentation: str,
    normalization: str,
    loss: str,
    *,
    requires_gpu: bool = False,
    parameters: Mapping[str, Any] | None = None,
) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=candidate_id,
        stage=stage,
        family=family,
        feature_view=feature_view,
        adapter=adapter,
        model=model,
        augmentation=augmentation,
        normalization=normalization,
        loss=loss,
        requires_gpu=requires_gpu,
        parameters=dict(parameters or {}),
    )


def experiment_registry() -> dict[str, CandidateSpec]:
    rows = (
        _candidate("stage1_majority", "stage1", "baseline", "labels_only", "classical", "dummy_most_frequent", "none", "none", "not_applicable"),
        _candidate("stage1_mfcc_svm", "stage1", "baseline", "mfcc_summary", "classical", "rbf_svm", "none", "standard_scaler_train_only", "not_applicable"),
        _candidate("stage1_logmel_small_cnn", "stage1", "baseline", "log_mel", "neural", "small_cnn", "waveform_only", "per_feature_bin", "categorical_crossentropy", requires_gpu=True),
        _candidate("stage2_majority", "stage2", "baseline", "labels_only", "classical", "dummy_most_frequent", "none", "none", "not_applicable"),
        _candidate("stage2_mfcc_svm", "stage2", "baseline", "mfcc_summary", "classical", "rbf_svm", "none", "standard_scaler_train_only", "not_applicable"),
        _candidate("stage2_logmel_small_cnn", "stage2", "baseline", "log_mel", "neural", "small_cnn", "waveform_only", "per_feature_bin", "categorical_crossentropy", requires_gpu=True),
        _candidate("stage2_cnn_only", "stage2", "ablation", "all_blocks", "neural", "cnn_only", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True),
        _candidate("stage2_cnn_bilstm", "stage2", "ablation", "all_blocks", "neural", "cnn_bilstm", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True),
        _candidate("stage2_corrected_attention", "stage2", "proposed", "all_blocks", "neural", "cnn_bilstm_attention", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True, parameters={"architecture": "corrected_single_branch"}),
        _candidate("stage2_multi_branch_attention", "stage2", "candidate", "multi_branch_blocks", "neural", "cnn_bilstm_attention", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True, parameters={"architecture": "corrected_multi_branch"}),
        _candidate("stage2_feature_blocks", "stage2", "ablation", "feature_block_subset", "neural", "cnn_bilstm_attention", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True, parameters={"architecture": "corrected_single_branch"}),
        _candidate("stage2_normalization", "stage2", "ablation", "all_blocks", "neural", "cnn_bilstm_attention", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True, parameters={"architecture": "corrected_single_branch"}),
        _candidate("stage2_augmentation_mixup", "stage2", "ablation", "all_blocks", "neural", "cnn_bilstm_attention", "waveform_plus_mixup", "per_feature_bin", "categorical_crossentropy", requires_gpu=True, parameters={"architecture": "corrected_single_branch"}),
    )
    return {row.candidate_id: row for row in rows}


def validate_selection_metric(metric: str) -> str:
    value = str(metric).strip().lower()
    if "test" in value or "heldout" in value or "held-out" in value:
        raise ValueError("Experiment selection cannot use a held-out Test metric")
    if not value.startswith("oof_"):
        raise ValueError("Experiment selection metric must be a grouped OOF metric")
    return value


def _forbidden_name(value: object) -> bool:
    normalized = str(value).strip().casefold().replace("-", "_")
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", normalized) if token)
    return (
        "heldout" in normalized
        or "held_out" in normalized
        or "final_test" in normalized
        or "test" in tokens
    )


def _reject_test_references(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _forbidden_name(key):
                raise ExperimentProtocolError(
                    f"Test/held-out fields are forbidden in experiment selection: {path}.{key}"
                )
            _reject_test_references(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_test_references(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _forbidden_name(value):
        raise ExperimentProtocolError(
            f"Test/held-out values are forbidden in experiment selection: {path}"
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentProtocolError(f"Could not read experiment config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentProtocolError("Experiment config must be a JSON object")
    _reject_test_references(payload)
    try:
        metric = validate_selection_metric(payload.get("selection_metric", ""))
        config = ExperimentConfig(
            schema_version=payload.get("schema_version", ""),
            wave=payload.get("wave", ""),
            seeds=tuple(payload.get("seeds", ())),
            selection_metric=metric,
            candidates=tuple(payload.get("candidates", ())),
            parameters=payload.get("parameters", {}),
            candidate_source=payload.get("candidate_source", "explicit"),
            continue_on_candidate_failure=bool(
                payload.get("continue_on_candidate_failure", True)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentProtocolError(str(exc)) from exc
    unknown = sorted(set(config.candidates) - set(experiment_registry()))
    if unknown:
        raise ExperimentProtocolError(f"Unknown experiment candidates: {unknown}")
    validate_experiment_policy(config)
    return config


def validate_experiment_policy(config: ExperimentConfig, *, resolved: bool = False) -> None:
    _reject_test_references(asdict(config))
    if config.schema_version != '1.0' or config.wave not in {'stage1_baselines', 'A', 'B_features', 'B_augmentation', 'B_loss', 'C'}:
        raise ExperimentProtocolError('Unsupported experiment schema or wave')
    expected_seeds = (42, 123, 2026) if config.wave == 'C' else (42,)
    if config.seeds != expected_seeds:
        raise ExperimentProtocolError(f'Wave {config.wave} requires seeds {expected_seeds}')
    expected_source = 'parent_top_2' if config.wave == 'C' else 'parent_rank_1' if config.wave.startswith('B_') else 'explicit'
    if not resolved and config.candidate_source != expected_source:
        raise ExperimentProtocolError(f'Wave {config.wave} requires candidate_source {expected_source}')
    if resolved and config.wave == 'C' and not 1 <= len(config.candidates) <= 2:
        raise ExperimentProtocolError('Wave C requires at most two eligible candidates')


_VARIANT_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def derive_candidate(
    anchor: CandidateSpec,
    variant_id: str,
    overrides: Mapping[str, Any],
) -> CandidateSpec:
    variant = str(variant_id).strip().lower()
    if not _VARIANT_ID.fullmatch(variant):
        raise ExperimentProtocolError(f"Invalid variant ID: {variant_id!r}")
    values = dict(overrides)
    if variant == "anchor":
        if values:
            raise ExperimentProtocolError("The anchor variant cannot contain overrides")
        return anchor
    allowed = {"feature_view", "augmentation", "normalization", "loss", "parameters"}
    forbidden = sorted(set(values) - allowed)
    if forbidden:
        raise ExperimentProtocolError(f"Candidate overrides are forbidden: {forbidden}")
    parameters = dict(anchor.parameters)
    raw_parameters = values.pop("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise ExperimentProtocolError("Candidate parameter overrides must be a mapping")
    parameters.update(raw_parameters)
    return replace(
        anchor,
        candidate_id=f"{anchor.candidate_id}__{variant}",
        parameters=parameters,
        **values,
    )


def load_fold_assignments(path: str | Path) -> tuple[dict[str, str], ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    required = {"record_id", "group_id", "label", "validation_fold"}
    if not rows or not required.issubset(rows[0]):
        raise ExperimentProtocolError(
            f"Fold assignments must contain {sorted(required)}"
        )
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ExperimentProtocolError("Fold assignments contain duplicate record IDs")
    group_folds: dict[str, set[str]] = {}
    for row in rows:
        group_folds.setdefault(row["group_id"], set()).add(row["validation_fold"])
    split_groups = sorted(group for group, folds in group_folds.items() if len(folds) > 1)
    if split_groups:
        raise ExperimentProtocolError(
            f"One or more groups are split across folds: {split_groups[:3]}"
        )
    folds = {int(row["validation_fold"]) for row in rows}
    if folds != {1, 2, 3, 4, 5}:
        raise ExperimentProtocolError(f"Expected folds 1-5, found {sorted(folds)}")
    return rows


def fold_assignment_sha256(rows: Iterable[Mapping[str, str]]) -> str:
    canonical = [
        {
            "record_id": str(row["record_id"]),
            "group_id": str(row["group_id"]),
            "label": str(row["label"]),
            "validation_fold": str(row["validation_fold"]),
        }
        for row in rows
    ]
    canonical.sort(key=lambda row: row["record_id"])
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def registry_payload(experiment_ids: Iterable[str] | None = None) -> dict[str, object]:
    registry = experiment_registry()
    selected = tuple(experiment_ids or registry.keys())
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ExperimentProtocolError(f"Unknown experiments: {unknown}")
    for candidate_id in selected:
        validate_selection_metric(registry[candidate_id].selection_metric)
    return {
        "schema_version": "2.0",
        "selection_scope": "grouped_oof_only",
        "heldout_test_available_for_ranking": False,
        "experiments": [asdict(registry[name]) for name in selected],
    }
