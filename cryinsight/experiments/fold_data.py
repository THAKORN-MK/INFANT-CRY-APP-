"""Frozen reference-run and grouped-fold data contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from cryinsight.training.artefacts import sha256_file
from cryinsight.training.protocol import OriginalRecord, assert_fold_integrity

from .registry import (
    ExperimentProtocolError,
    fold_assignment_sha256,
    load_fold_assignments,
)


@dataclass(frozen=True)
class ReferenceStage:
    stage: str
    run_id: str
    project_root: Path
    run_dir: Path
    verification_status: str
    fold_count: int
    fold_assignment_path: Path
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class ReferencePipeline:
    pipeline_run_id: str
    stage1: ReferenceStage
    stage2: ReferenceStage

    def stage(self, name: str) -> ReferenceStage:
        if name == "stage1":
            return self.stage1
        if name == "stage2":
            return self.stage2
        raise ValueError("stage must be 'stage1' or 'stage2'")


@dataclass(frozen=True)
class FrozenRecordSet:
    records: tuple[OriginalRecord, ...]
    validation_folds: Mapping[str, int]


@dataclass(frozen=True)
class FoldDataset:
    fold: int
    train_records: tuple[OriginalRecord, ...]
    validation_records: tuple[OriginalRecord, ...]


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentProtocolError(f"Could not read reference artefact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentProtocolError(f"Reference artefact must be a JSON object: {path}")
    return payload


def _load_reference_stage(
    *,
    stage: str,
    project_root: Path,
    run_dir: Path,
    pipeline_run_id: str,
) -> ReferenceStage:
    required = {
        "verification": run_dir / "verification.json",
        "fold_assignments_file": run_dir / "fold_assignments.csv",
        "oof_predictions": run_dir / "oof_predictions.csv",
        "oof_metrics": run_dir / "oof_metrics.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ExperimentProtocolError(
            f"Reference {stage} run is missing required artefacts: {missing}"
        )
    verification = _read_json_object(required["verification"])
    if verification.get("run_id") != pipeline_run_id:
        raise ExperimentProtocolError(
            f"Reference {stage} verification run ID does not match {pipeline_run_id}"
        )
    status = str(verification.get("status", ""))
    if status != "complete":
        raise ExperimentProtocolError(
            f"Reference {stage} verification must be complete, found {status!r}"
        )
    fold_count = int(verification.get("folds_completed", 0))
    if fold_count != 5:
        raise ExperimentProtocolError(
            f"Reference {stage} must contain five completed folds, found {fold_count}"
        )
    assignment_rows = load_fold_assignments(required["fold_assignments_file"])
    hashes = {name: sha256_file(path) for name, path in required.items()}
    evidence = verification.get("artefact_sha256")
    if not isinstance(evidence, dict):
        raise ExperimentProtocolError(f"Reference {stage} verification artefact hashes are missing")
    for name, hash_key in (("fold_assignments", "fold_assignments_file"), ("oof_predictions", "oof_predictions"), ("oof_metrics", "oof_metrics")):
        entry = evidence.get(name)
        if not isinstance(entry, dict) or entry.get("sha256") != hashes[hash_key]:
            raise ExperimentProtocolError(f"Reference {stage} original verification hash mismatch: {name}")
    hashes["fold_assignments_contract"] = fold_assignment_sha256(assignment_rows)
    return ReferenceStage(
        stage=stage,
        run_id=pipeline_run_id,
        project_root=project_root,
        run_dir=run_dir,
        verification_status=status,
        fold_count=fold_count,
        fold_assignment_path=required["fold_assignments_file"],
        hashes=hashes,
    )


def load_reference_pipeline(
    project_root: str | Path,
    pipeline_run_id: str,
) -> ReferencePipeline:
    """Load only complete Stage 1/2 OOF evidence for one paired pipeline run."""

    root = Path(project_root).resolve()
    run_id = str(pipeline_run_id).strip()
    if not run_id:
        raise ExperimentProtocolError("pipeline run ID cannot be empty")
    if Path(run_id).name != run_id or run_id in {'.', '..'} or '/' in run_id or '\\' in run_id:
        raise ExperimentProtocolError("pipeline run ID must be a directory name")
    stage1 = _load_reference_stage(
        stage="stage1",
        project_root=root,
        run_dir=root / "Models_dbl" / "binary" / "runs" / run_id,
        pipeline_run_id=run_id,
    )
    stage2 = _load_reference_stage(
        stage="stage2",
        project_root=root,
        run_dir=root / "Models_dbl" / "Main" / "runs" / run_id,
        pipeline_run_id=run_id,
    )
    return ReferencePipeline(run_id, stage1, stage2)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def reject_test_path(path: str | Path, project_root: str | Path) -> Path:
    """Reject a resolved path anywhere below the locked Test directory."""

    target = Path(path).resolve()
    test_root = (Path(project_root).resolve() / "data_set_dbl_split" / "test").resolve()
    if target == test_root or _is_relative_to(target, test_root):
        raise ExperimentProtocolError(
            f"Test dataset paths are forbidden in experiment development: {target}"
        )
    return target


def load_reference_records(
    reference: ReferenceStage,
    data_root: str | Path,
) -> FrozenRecordSet:
    """Reconstruct and re-hash original development records from frozen assignments."""

    root = reject_test_path(data_root, reference.project_root)
    if not root.is_dir():
        raise ExperimentProtocolError(f"Development data root does not exist: {root}")
    rows = load_fold_assignments(reference.fold_assignment_path)
    required = {
        "record_id",
        "filepath",
        "relative_path",
        "label",
        "validation_fold",
        "source_dataset",
        "group_id",
        "group_rule",
        "sha256",
    }
    if not required.issubset(rows[0]):
        raise ExperimentProtocolError(
            f"Frozen assignments are missing fields: {sorted(required - set(rows[0]))}"
        )
    records: list[OriginalRecord] = []
    validation_folds: dict[str, int] = {}
    for row in rows:
        path = reject_test_path(row["filepath"], reference.project_root)
        if not path.is_file() or not _is_relative_to(path, root):
            raise ExperimentProtocolError(
                f"Frozen development record is outside the approved Train root: {path}"
            )
        relative_path = path.relative_to(root).as_posix()
        if relative_path != str(row["relative_path"]).replace("\\", "/"):
            raise ExperimentProtocolError(
                f"Frozen relative path does not match resolved path: {row['record_id']}"
            )
        expected_sha256 = str(row["sha256"]).lower()
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ExperimentProtocolError(
                f"Frozen record SHA-256 mismatch for {row['record_id']}: {path}"
            )
        record = OriginalRecord(
            record_id=str(row["record_id"]),
            filepath=path,
            relative_path=relative_path,
            label=str(row["label"]),
            source_label=str(row["label"]),
            source_dataset=str(row["source_dataset"]),
            group_id=str(row["group_id"]),
            group_rule=str(row["group_rule"]),
            sha256=actual_sha256,
        )
        records.append(record)
        validation_folds[record.record_id] = int(row["validation_fold"])
    if set(validation_folds.values()) != {1, 2, 3, 4, 5}:
        raise ExperimentProtocolError("Frozen record set must contain validation folds 1-5")
    return FrozenRecordSet(tuple(records), validation_folds)


def build_fold_dataset(
    records: tuple[OriginalRecord, ...],
    validation_folds: Mapping[str, int],
    *,
    fold: int,
) -> FoldDataset:
    """Materialize one frozen train/validation view and assert group isolation."""

    if fold not in {1, 2, 3, 4, 5}:
        raise ExperimentProtocolError("fold must be one of 1, 2, 3, 4, 5")
    record_ids = {record.record_id for record in records}
    if set(validation_folds) != record_ids:
        raise ExperimentProtocolError(
            "Validation-fold mapping must contain every frozen record exactly once"
        )
    invalid = sorted(set(validation_folds.values()) - {1, 2, 3, 4, 5})
    if invalid:
        raise ExperimentProtocolError(f"Invalid validation folds: {invalid}")
    train = tuple(
        record for record in records if int(validation_folds[record.record_id]) != fold
    )
    validation = tuple(
        record for record in records if int(validation_folds[record.record_id]) == fold
    )
    if not train or not validation:
        raise ExperimentProtocolError(f"Fold {fold} has an empty partition")
    assert_fold_integrity(train, validation)
    return FoldDataset(fold, train, validation)
