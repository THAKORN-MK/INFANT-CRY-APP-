"""Experiment run state and immutable artefact storage.

The training lifecycle is added incrementally; this module first owns run
identity, state transitions, job identity, and resume safety.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import platform
import sys
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from cryinsight.training.artefacts import sha256_file
from cryinsight.training.protocol import OriginalRecord, ProtocolViolation, write_json_atomic

from .contracts import (
    CandidateAdapter,
    CandidateSpec,
    ExperimentConfig,
    FoldRequest,
    FoldResult,
)
from .fold_data import (
    FrozenRecordSet,
    ReferencePipeline,
    ReferenceStage,
    build_fold_dataset,
    load_reference_pipeline,
    load_reference_records,
)
from .registry import (
    ExperimentProtocolError,
    experiment_registry,
    fold_assignment_sha256,
    load_fold_assignments,
    validate_experiment_policy,
)
from .selection import aggregate_candidate_seed, ExperimentVerificationError, verify_probabilities


class ExperimentStateError(RuntimeError):
    """Raised when an experiment run state or resume hash is invalid."""


@dataclass(frozen=True, order=True)
class JobKey:
    candidate_id: str
    seed: int
    fold: int


@dataclass(frozen=True)
class ExperimentPreparation:
    """Inputs required to freeze a new OOF-only experiment run."""

    project_root: Path
    pipeline_run_id: str
    config: ExperimentConfig
    stage_data_roots: Mapping[str, Path]
    runs_root: Path
    run_id: str | None = None
    candidate_specs: Mapping[str, CandidateSpec] | None = None
    source_config: ExperimentConfig | None = None
    parent_provenance: Mapping[str, Any] | None = None


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "prepared": {"running", "failed"},
    "running": {"complete", "failed"},
    "failed": {"running"},
    "complete": set(),
}

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@contextmanager
def _run_lock(store: "ExperimentRunStore"):
    """OS ownership lock: released by the OS on process death, never by PID guesses."""
    path = store.run_dir / ".execution.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExperimentStateError("Another process owns this experiment run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _publish_file(source: Path, destination: Path) -> None:
    """Retry publication only for byte-identical evidence; never replace a conflict."""
    content = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != content:
            raise ExperimentProtocolError(f"Conflicting immutable artefact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # link publishes the durable complete file atomically and refuses replacement.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise ExperimentProtocolError(f'Conflicting immutable artefact: {destination}')
    finally:
        temporary.unlink(missing_ok=True)


def _json_once_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != json.loads(json.dumps(payload)):
            raise ExperimentProtocolError(f'Conflicting immutable JSON artefact: {path}')
    else:
        write_json_atomic(path, payload)


def _verify_hashes(root: Path, hashes: Mapping[str, Any]) -> None:
    if not hashes:
        raise ExperimentProtocolError(f"Missing artefact hashes: {root}")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ExperimentProtocolError(f"Artefact hash mismatch: {path}")


def _next_directory(root: Path, prefix: str) -> Path:
    numbers = [int(p.name[len(prefix):]) for p in root.glob(prefix + '*') if p.name[len(prefix):].isdigit()]
    return root / f"{prefix}{max(numbers, default=0) + 1}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentStateError(f"Could not read experiment state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentStateError(f"Experiment state must be a JSON object: {path}")
    return payload


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _generated_run_id(pipeline_run_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{pipeline_run_id}__exp_{timestamp}_{secrets.token_hex(4)}"


def _validate_component(value: object, *, name: str) -> str:
    text = str(value)
    if not _COMPONENT.fullmatch(text):
        raise ValueError(f"Invalid {name}: {text!r}")
    return text


class ExperimentRunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        pipeline_run_id: str,
        config: ExperimentConfig,
        reference: ReferencePipeline,
        *,
        run_id: str | None = None,
        candidate_specs: Mapping[str, CandidateSpec] | None = None,
    ) -> "ExperimentRunStore":
        validate_experiment_policy(config, resolved=True)
        pipeline_id = _validate_component(pipeline_run_id, name="pipeline run ID")
        if reference.pipeline_run_id != pipeline_id:
            raise ExperimentStateError("Reference pipeline run ID does not match")
        if config.candidate_source != "explicit":
            raise ExperimentStateError(
                "Parent-derived configs must be resolved before creating a run"
            )
        registry = dict(candidate_specs or experiment_registry())
        invalid_specs = sorted(
            key for key, value in registry.items() if key != value.candidate_id
        )
        if invalid_specs:
            raise ExperimentStateError(
                f"Candidate snapshot keys do not match their IDs: {invalid_specs}"
            )
        unknown = sorted(set(config.candidates) - set(registry))
        if unknown:
            raise ExperimentStateError(f"Unknown candidates: {unknown}")
        identifier = run_id or _generated_run_id(pipeline_id)
        _validate_component(identifier, name="experiment run ID")
        if not identifier.startswith(pipeline_id + "__exp_"):
            raise ExperimentStateError(
                "Experiment run ID must start with its pipeline run ID"
            )
        root = Path(runs_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / identifier
        run_dir.mkdir(parents=False, exist_ok=False)

        config_payload = asdict(config)
        config_hash = _payload_sha256(config_payload)
        assignment_hashes = {
            "stage1": reference.stage1.hashes["fold_assignments_contract"],
            "stage2": reference.stage2.hashes["fold_assignments_contract"],
        }
        jobs = [
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "fold": fold,
                "stage": registry[candidate_id].stage,
            }
            for candidate_id in config.candidates
            for seed in config.seeds
            for fold in range(1, 6)
        ]
        write_json_atomic(
            run_dir / "protocol.json",
            {
                "schema_version": "1.0",
                "pipeline_run_id": pipeline_id,
                "experiment_run_id": identifier,
                "selection_scope": "grouped_oof_only",
                "heldout_test_available_for_ranking": False,
                "config": config_payload,
                "config_sha256": config_hash,
                "assignment_hashes": assignment_hashes,
            },
        )
        write_json_atomic(
            run_dir / "reference_run.json",
            {
                "schema_version": "1.0",
                "pipeline_run_id": pipeline_id,
                "stage1": {
                    "run_dir": str(reference.stage1.run_dir),
                    "hashes": dict(reference.stage1.hashes),
                },
                "stage2": {
                    "run_dir": str(reference.stage2.run_dir),
                    "hashes": dict(reference.stage2.hashes),
                },
            },
        )
        cls._write_shared_assignments(run_dir, reference)
        write_json_atomic(
            run_dir / "candidate_matrix.json",
            {
                "schema_version": "1.0",
                "expected_job_count": len(jobs),
                "jobs": jobs,
            },
        )
        write_json_atomic(
            run_dir / "resolved_config.json",
            {
                "schema_version": "1.0",
                "config": config_payload,
                "candidate_specs": [
                    asdict(registry[candidate_id])
                    for candidate_id in config.candidates
                ],
            },
        )
        write_json_atomic(
            run_dir / "state.json",
            {
                "schema_version": "1.0",
                "pipeline_run_id": pipeline_id,
                "experiment_run_id": identifier,
                "status": "prepared",
                "config_sha256": config_hash,
                "assignment_hashes": assignment_hashes,
                "expected_job_count": len(jobs),
            },
        )
        write_json_atomic(run_dir / "integrity.json", {
            "schema_version": "1.0",
            "artefact_sha256": {name: sha256_file(run_dir / name) for name in (
                "protocol.json", "reference_run.json", "shared_fold_assignments.csv",
                "candidate_matrix.json", "resolved_config.json",
            )},
        })
        return cls(run_dir)

    @staticmethod
    def _write_shared_assignments(
        run_dir: Path,
        reference: ReferencePipeline,
    ) -> None:
        path = run_dir / "shared_fold_assignments.csv"
        fields = ("stage", "record_id", "group_id", "label", "validation_fold")
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for stage, reference_stage in (
                ("stage1", reference.stage1),
                ("stage2", reference.stage2),
            ):
                for row in load_fold_assignments(reference_stage.fold_assignment_path):
                    writer.writerow(
                        {
                            "stage": stage,
                            "record_id": row["record_id"],
                            "group_id": row["group_id"],
                            "label": row["label"],
                            "validation_fold": row["validation_fold"],
                        }
                    )

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        *,
        expected_config_hash: str | None = None,
        expected_assignment_hashes: Mapping[str, str] | None = None,
    ) -> "ExperimentRunStore":
        store = cls(Path(run_dir))
        state = store._state_payload()
        if expected_config_hash is not None and state.get("config_sha256") != expected_config_hash:
            raise ExperimentStateError("Experiment config hash does not match resume request")
        if expected_assignment_hashes is not None:
            actual = state.get("assignment_hashes")
            if not isinstance(actual, dict):
                raise ExperimentStateError("Experiment assignment hashes are missing")
            for stage, expected in expected_assignment_hashes.items():
                if actual.get(stage) != expected:
                    raise ExperimentStateError(
                        f"Experiment assignment hash does not match for {stage}"
                    )
        return store

    @property
    def state(self) -> str:
        return str(self._state_payload().get("status", ""))

    @property
    def config_hash(self) -> str:
        return str(self._state_payload().get("config_sha256", ""))

    def _state_payload(self) -> dict[str, Any]:
        return _read_json(self.run_dir / "state.json")

    def _transition(self, target: str) -> None:
        payload = self._state_payload()
        current = str(payload.get("status", ""))
        if target not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ExperimentStateError(
                f"Cannot transition experiment from {current!r} to {target!r}; "
                f"{current or 'unknown'} runs are immutable"
            )
        payload["status"] = target
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _replace_json(self.run_dir / "state.json", payload)

    def mark_running(self) -> None:
        self._transition("running")

    def _jobs(self) -> tuple[JobKey, ...]:
        payload = _read_json(self.run_dir / "candidate_matrix.json")
        rows = payload.get("jobs")
        if not isinstance(rows, list):
            raise ExperimentStateError("Candidate matrix jobs are invalid")
        return tuple(
            JobKey(str(row["candidate_id"]), int(row["seed"]), int(row["fold"]))
            for row in rows
        )

    def _require_job(self, candidate_id: str, seed: int, fold: int) -> JobKey:
        job = JobKey(
            _validate_component(candidate_id, name="candidate ID"),
            int(seed),
            int(fold),
        )
        if job not in self._jobs():
            raise ExperimentStateError(f"Job is not in the candidate matrix: {job}")
        return job

    def job_dir(self, candidate_id: str, seed: int, fold: int) -> Path:
        job = self._require_job(candidate_id, seed, fold)
        return (
            self.run_dir
            / "candidates"
            / job.candidate_id
            / f"seed_{job.seed}"
            / f"fold_{job.fold}"
        )

    def mark_job_complete(
        self,
        candidate_id: str,
        seed: int,
        fold: int,
        artefact_hashes: Mapping[str, str],
        *,
        fold_result_path: str | None = None,
    ) -> Path:
        if self.state != "running":
            raise ExperimentStateError("Jobs can complete only while the run is running")
        invalid = sorted(
            name for name, value in artefact_hashes.items() if not _SHA256.fullmatch(str(value))
        )
        if invalid or not artefact_hashes:
            raise ExperimentStateError(
                f"Completed jobs require valid artefact hashes: {invalid}"
            )
        path = self.job_dir(candidate_id, seed, fold) / "complete.json"
        write_json_atomic(
            path,
            {
                "schema_version": "1.0",
                "status": "complete",
                "candidate_id": candidate_id,
                "seed": int(seed),
                "fold": int(fold),
                "artefact_sha256": dict(artefact_hashes),
                "fold_result_path": fold_result_path,
            },
        )
        return path

    def mark_job_failed(
        self,
        candidate_id: str,
        seed: int,
        fold: int,
        *,
        error_type: str,
        message: str,
        fail_run: bool = True,
        attempt_dir: Path | None = None,
    ) -> Path:
        if self.state != "running":
            raise ExperimentStateError("Jobs can fail only while the run is running")
        directory = self.job_dir(candidate_id, seed, fold)
        attempt = Path(attempt_dir) if attempt_dir is not None else _next_directory(directory, 'attempt_')
        if attempt.parent.resolve() != directory.resolve() or not re.fullmatch(r'attempt_[1-9][0-9]*', attempt.name):
            raise ExperimentStateError('Failure attempt must belong to its frozen job')
        path = attempt / "failure.json"
        write_json_atomic(
            path,
            {
                "schema_version": "1.0",
                "status": "failed",
                "candidate_id": candidate_id,
                "seed": int(seed),
                "fold": int(fold),
                "error_type": str(error_type),
                "message": str(message),
            },
        )
        if fail_run:
            self._transition("failed")
        return path

    def mark_failed(self) -> None:
        if self.state in {"prepared", "running"}:
            self._transition("failed")

    def pending_jobs(self) -> tuple[JobKey, ...]:
        return tuple(
            job
            for job in self._jobs()
            if _completed_fold_result(self, job) is None
        )

    def verify_integrity(self) -> None:
        core = _read_json(self.run_dir / "integrity.json")
        _verify_hashes(self.run_dir, core.get("artefact_sha256", {}))
        if (self.run_dir / "inputs.json").exists():
            preparation = _read_json(self.run_dir / "preparation_integrity.json")
            _verify_hashes(self.run_dir, preparation.get("artefact_sha256", {}))
        protocol = _read_json(self.run_dir / "protocol.json")
        state = self._state_payload()
        for key in ("pipeline_run_id", "experiment_run_id", "config_sha256", "assignment_hashes"):
            if state.get(key) != protocol.get(key):
                raise ExperimentProtocolError(f"State/protocol identity mismatch: {key}")
        if protocol.get("experiment_run_id") != self.run_id:
            raise ExperimentProtocolError("Experiment directory identity mismatch")

    def finalize(self, *, status: str) -> Path:
        target = str(status)
        if target not in {"complete", "failed"}:
            raise ValueError("Final status must be complete or failed")
        if target == "complete" and self.pending_jobs():
            raise ExperimentStateError("Cannot complete an experiment with pending jobs")
        if target == "complete" and self.state != "running":
            raise ExperimentStateError("Only a running experiment can become complete")
        if target == "failed" and self.state == "running":
            self._transition("failed")
        verification_path = self.run_dir / "verification.json"
        jobs = self._jobs()
        pending = self.pending_jobs()
        payload = {
                "schema_version": "1.0",
                "experiment_run_id": self.run_id,
                "status": target,
                "expected_jobs": len(jobs),
                "completed_jobs": len(jobs) - len(pending),
                "pending_jobs": len(pending),
                "failed_jobs": len(pending),
                "artefact_sha256": {
                    str(path.relative_to(self.run_dir)).replace('\\', '/'): sha256_file(path)
                    for path in sorted(self.run_dir.rglob('*'))
                    if path.is_file() and path.name not in {'state.json', '.execution.lock'}
                    and path != verification_path
                    and not any(part.startswith(('attempt_', 'aggregation_attempt_', 'summary_attempt_', 'report_attempt_')) for part in path.relative_to(self.run_dir).parts)
                    and 'feature_cache' not in path.relative_to(self.run_dir).parts
                },
                "heldout_test_used_for_selection": False,
            }
        if verification_path.exists():
            if _read_json(verification_path) != payload:
                raise ExperimentProtocolError("Conflicting final verification")
        else:
            write_json_atomic(verification_path, payload)
        if target == "complete":
            self._transition("complete")
        return verification_path


ReferenceLoader = Callable[[str | Path, str], ReferencePipeline]
RecordLoader = Callable[[ReferenceStage, str | Path], FrozenRecordSet]
AdapterResolver = Callable[[Any], CandidateAdapter]


def _experiment_config(payload: Mapping[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig(
            schema_version=str(payload["schema_version"]),
            wave=str(payload["wave"]),
            seeds=tuple(payload["seeds"]),
            selection_metric=str(payload["selection_metric"]),
            candidates=tuple(payload["candidates"]),
            parameters=payload.get("parameters", {}),
            candidate_source=str(payload.get("candidate_source", "explicit")),
            continue_on_candidate_failure=bool(
                payload.get("continue_on_candidate_failure", True)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentProtocolError(f"Invalid frozen experiment config: {exc}") from exc


def _validate_stage_roots(roots: Mapping[str, Path]) -> dict[str, Path]:
    if set(roots) != {"stage1", "stage2"}:
        raise ExperimentProtocolError(
            "Experiment preparation requires exactly stage1 and stage2 data roots"
        )
    return {stage: Path(path).resolve() for stage, path in roots.items()}


def prepare_experiment(
    request: ExperimentPreparation,
    *,
    reference_loader: ReferenceLoader = load_reference_pipeline,
    record_loader: RecordLoader = load_reference_records,
) -> Path:
    """Audit inputs and create an immutable prepared experiment run."""

    project_root = Path(request.project_root).resolve()
    if request.config.wave.startswith('B_') or request.config.wave == 'C':
        _validate_parent_provenance(request.runs_root, request.pipeline_run_id, request.config.wave, request.parent_provenance or {}, verify_complete=True)
    stage_roots = _validate_stage_roots(request.stage_data_roots)
    reference = reference_loader(project_root, request.pipeline_run_id)
    label_orders: dict[str, list[str]] = {}
    record_counts: dict[str, int] = {}
    snapshots: dict[str, Any] = {}
    for stage in ("stage1", "stage2"):
        frozen = record_loader(reference.stage(stage), stage_roots[stage])
        labels = sorted({record.label for record in frozen.records})
        if len(labels) < 2:
            raise ExperimentProtocolError(
                f"Frozen {stage} development cohort must contain at least two classes"
            )
        label_orders[stage] = labels
        record_counts[stage] = len(frozen.records)
        snapshots[stage] = _record_snapshot(frozen)
    store = ExperimentRunStore.create(
        request.runs_root,
        request.pipeline_run_id,
        request.config,
        reference,
        run_id=request.run_id,
        candidate_specs=request.candidate_specs,
    )
    write_json_atomic(
        store.run_dir / "inputs.json",
        {
            "schema_version": "1.0",
            "project_root": str(project_root),
            "pipeline_run_id": request.pipeline_run_id,
            "stage_data_roots": {
                stage: str(path) for stage, path in stage_roots.items()
            },
            "label_orders": label_orders,
            "record_counts": record_counts,
            "heldout_test_used_for_selection": False,
        },
    )
    write_json_atomic(store.run_dir / "record_snapshot.json", snapshots)
    write_json_atomic(store.run_dir / "source_config.json", asdict(request.source_config or request.config))
    write_json_atomic(store.run_dir / "parent_provenance.json", dict(request.parent_provenance or {}))
    write_json_atomic(store.run_dir / "preparation_integrity.json", {
        "schema_version": "1.0",
        "artefact_sha256": {name: sha256_file(store.run_dir / name) for name in (
            "integrity.json", "inputs.json", "record_snapshot.json", "source_config.json", "parent_provenance.json",
        )},
    })
    return store.run_dir


def _record_snapshot(frozen: FrozenRecordSet) -> dict[str, Any]:
    return {"records": [{**asdict(record), "filepath": str(record.filepath)} for record in frozen.records],
            "validation_folds": dict(frozen.validation_folds)}


def _validate_parent_provenance(runs_root: Path, pipeline_id: str, wave: str, provenance: Mapping[str, Any], *, verify_complete: bool = False) -> None:
    predecessor = {'B_features': 'A', 'B_augmentation': 'B_features', 'B_loss': 'B_augmentation', 'C': 'B_loss'}
    parent_id = provenance.get('parent_experiment_run_id')
    if not isinstance(parent_id, str) or not _COMPONENT.fullmatch(parent_id):
        raise ExperimentProtocolError('Parent-derived wave requires frozen parent evidence')
    parent = Path(runs_root).resolve() / parent_id
    if verify_complete:
        verify_experiment(parent)
    _verify_hashes(parent, provenance.get('parent_artefact_sha256', {}))
    protocol = _read_json(parent / 'protocol.json')
    if protocol.get('pipeline_run_id') != pipeline_id or protocol.get('config', {}).get('wave') != predecessor.get(wave):
        raise ExperimentProtocolError('Parent pipeline or predecessor wave mismatch')


def _verify_reference_snapshot(store: ExperimentRunStore, reference: ReferencePipeline | None = None) -> None:
    snapshot = _read_json(store.run_dir / "reference_run.json")
    for stage in ("stage1", "stage2"):
        frozen = snapshot[stage]
        hashes = frozen["hashes"]
        if reference is not None and (dict(reference.stage(stage).hashes) != hashes or str(reference.stage(stage).run_dir) != frozen["run_dir"]):
            raise ExperimentProtocolError(f"Reference evidence changed: {stage}")
        paths = {"verification": "verification.json", "fold_assignments_file": "fold_assignments.csv", "oof_predictions": "oof_predictions.csv", "oof_metrics": "oof_metrics.json"}
        _verify_hashes(Path(frozen["run_dir"]), {filename: hashes[key] for key, filename in paths.items()})


def resolve_adapter(candidate: Any) -> CandidateAdapter:
    """Resolve adapters lazily so classical audit paths do not import TensorFlow."""

    if candidate.adapter == "classical":
        from .classical import ClassicalAdapter

        return ClassicalAdapter()
    if candidate.adapter == "neural":
        from .neural import NeuralAdapter

        return NeuralAdapter()
    raise ExperimentProtocolError(f"Unsupported adapter: {candidate.adapter}")


def _resolved_registry(run_dir: Path) -> dict[str, CandidateSpec]:
    path = run_dir / "resolved_config.json"
    if not path.is_file():
        return experiment_registry()
    payload = _read_json(path)
    rows = payload.get("candidate_specs")
    if not isinstance(rows, list) or not rows:
        raise ExperimentProtocolError("Resolved candidate snapshot is missing")
    result: dict[str, CandidateSpec] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ExperimentProtocolError("Resolved candidate snapshot is invalid")
        try:
            spec = CandidateSpec(**row)
        except (TypeError, ValueError) as exc:
            raise ExperimentProtocolError(
                f"Resolved candidate snapshot is invalid: {exc}"
            ) from exc
        if spec.candidate_id in result:
            raise ExperimentProtocolError(
                f"Duplicate resolved candidate: {spec.candidate_id}"
            )
        result[spec.candidate_id] = spec
    return result


def _load_lifecycle_inputs(
    store: ExperimentRunStore,
    *,
    reference_loader: ReferenceLoader,
    record_loader: RecordLoader,
) -> tuple[ExperimentConfig, ReferencePipeline, dict[str, FrozenRecordSet], dict[str, Any]]:
    store.verify_integrity()
    protocol = _read_json(store.run_dir / "protocol.json")
    inputs = _read_json(store.run_dir / "inputs.json")
    raw_config = protocol.get("config")
    if not isinstance(raw_config, dict):
        raise ExperimentProtocolError("Frozen experiment config is missing")
    config = _experiment_config(raw_config)
    validate_experiment_policy(config, resolved=True)
    if config.wave.startswith('B_') or config.wave == 'C':
        _validate_parent_provenance(store.run_dir.parent, str(inputs['pipeline_run_id']), config.wave, _read_json(store.run_dir / 'parent_provenance.json'))
    if _payload_sha256(asdict(config)) != protocol.get("config_sha256"):
        raise ExperimentProtocolError("Frozen experiment config hash does not match")
    project_root = Path(str(inputs.get("project_root", ""))).resolve()
    pipeline_run_id = str(inputs.get("pipeline_run_id", ""))
    reference = reference_loader(project_root, pipeline_run_id)
    _verify_reference_snapshot(store, reference)
    if reference.pipeline_run_id != pipeline_run_id:
        raise ExperimentProtocolError("Reference pipeline run ID changed")
    expected_hashes = protocol.get("assignment_hashes")
    if not isinstance(expected_hashes, dict):
        raise ExperimentProtocolError("Frozen assignment hashes are missing")
    frozen_sets: dict[str, FrozenRecordSet] = {}
    raw_roots = inputs.get("stage_data_roots")
    if not isinstance(raw_roots, dict):
        raise ExperimentProtocolError("Frozen stage data roots are missing")
    for stage in ("stage1", "stage2"):
        reference_stage = reference.stage(stage)
        try:
            actual_hash = fold_assignment_sha256(
                load_fold_assignments(reference_stage.fold_assignment_path)
            )
        except (OSError, ValueError) as exc:
            raise ExperimentProtocolError(
                f"Could not revalidate {stage} fold assignments: {exc}"
            ) from exc
        if actual_hash != expected_hashes.get(stage):
            raise ExperimentProtocolError(
                f"Frozen assignment hash changed for {stage}"
            )
        frozen_sets[stage] = record_loader(reference_stage, raw_roots[stage])
        if _record_snapshot(frozen_sets[stage]) != _read_json(store.run_dir / "record_snapshot.json")[stage]:
            raise ExperimentProtocolError(f"Frozen record identity/order/fold mapping changed: {stage}")
    return config, reference, frozen_sets, inputs


def _write_environment_once(
    store: ExperimentRunStore,
    config: ExperimentConfig,
) -> dict[str, Any]:
    path = store.run_dir / "environment.json"
    registry = _resolved_registry(store.run_dir)
    selected = [registry[candidate_id] for candidate_id in config.candidates]
    neural = any(candidate.adapter == "neural" for candidate in selected)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "requested_device": str(config.parameters.get("device", "auto")),
        "contains_neural_candidates": neural,
    }
    if neural:
        import tensorflow as tf

        from cryinsight.runtime.device import configure_tensorflow_runtime

        payload["tensorflow_runtime"] = configure_tensorflow_runtime(
            tf,
            device=str(config.parameters.get("device", "auto")),
            require_gpu=bool(config.parameters.get("require_gpu", False)) or any(candidate.requires_gpu for candidate in selected),
            mixed_precision=bool(config.parameters.get("mixed_precision", False)),
        )
    if path.is_file():
        if _read_json(path) != payload:
            raise ExperimentProtocolError("Runtime environment contract changed on resume")
    else:
        write_json_atomic(path, payload)
    return payload


def _next_attempt_dir(store: ExperimentRunStore, job: JobKey) -> Path:
    job_dir = store.job_dir(job.candidate_id, job.seed, job.fold)
    return _next_directory(job_dir, "attempt_")


def _validate_fold_result(request: FoldRequest, result: FoldResult) -> None:
    if (
        result.candidate_id != request.candidate.candidate_id
        or int(result.seed) != int(request.seed)
        or int(result.fold) != int(request.fold)
    ):
        raise ExperimentProtocolError("Adapter returned a mismatched job identity")
    expected_ids = tuple(record.record_id for record in request.validation_records)
    expected_labels = tuple(record.label for record in request.validation_records)
    if tuple(result.validation_record_ids) != expected_ids:
        raise ExperimentProtocolError("Adapter returned mismatched validation record IDs")
    if tuple(result.true_labels) != expected_labels:
        raise ExperimentProtocolError("Adapter returned mismatched validation labels")
    if not Path(result.model_path).is_file() or not Path(result.manifest_path).is_file():
        raise ExperimentProtocolError("Adapter did not publish its model and manifest")
    values = verify_probabilities(result.probabilities).probabilities
    if values.shape != (len(expected_ids), len(request.label_order)):
        raise ExperimentProtocolError("Adapter returned mismatched probability dimensions")
    manifest = _read_json(Path(result.manifest_path))
    required = {
        "experiment_run_id": request.experiment_run_id, "pipeline_run_id": request.pipeline_run_id,
        "candidate_id": request.candidate.candidate_id, "seed": request.seed, "fold": request.fold,
        "adapter": request.candidate.adapter, "label_order": list(request.label_order),
        "train_record_ids": [r.record_id for r in request.train_records],
        "validation_record_ids": list(expected_ids),
        "config_sha256": request.runtime["config_sha256"],
        "assignment_sha256": request.runtime["assignment_sha256"],
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ExperimentProtocolError(f"Fold manifest identity mismatch: {key}")
    _verify_fold_manifest_artifacts(result)


def _verify_fold_manifest_artifacts(result: FoldResult) -> None:
    manifest = _read_json(Path(result.manifest_path))
    root = Path(result.manifest_path).resolve().parent
    if not Path(result.model_path).resolve().is_relative_to(root):
        raise ExperimentProtocolError("Fold model is outside its selected attempt")
    paths = {"model_sha256": Path(result.model_path).name,
             "normalizer_sha256": "normalizer.npy", "history_sha256": "history.csv",
             "checkpoint_publication_sha256": "checkpoint_publication.json",
             "augmentation_manifest_sha256": "augmentation_manifest.json",
             "normalizer_metadata_sha256": "normalizer.npy.metadata.json",
             "normalizer_manifest_sha256": "normalizer_manifest.json"}
    _verify_hashes(root, {name: manifest[key] for key, name in paths.items() if key in manifest})


def _save_fold_result(request: FoldRequest, result: FoldResult) -> tuple[Path, dict[str, str]]:
    from cryinsight.training.artefacts import OofPrediction, aggregate_oof_metrics, oof_metrics_payload, write_oof_predictions_csv
    verified = verify_probabilities(result.probabilities)
    rows = [OofPrediction(record_id=record.record_id, filepath=str(record.filepath),
                          label=record.label, group_id=record.group_id, fold=request.fold,
                          predicted_label=request.label_order[int(np.argmax(scores))],
                          scores=tuple(float(value) for value in scores), sample_kind='original',
                          model_path=str(result.model_path), normalizer_path='recorded_in_fold_manifest',
                          run_id=request.experiment_run_id)
            for record, scores in zip(request.validation_records, verified.probabilities)]
    prediction_path = Path(request.output_dir) / 'predictions.csv'
    metrics_path = Path(request.output_dir) / 'metrics.json'
    write_oof_predictions_csv(prediction_path, rows, label_order=request.label_order)
    metrics = oof_metrics_payload(aggregate_oof_metrics(request.validation_records, rows, label_order=request.label_order, bootstrap_iterations=0, bootstrap_seed=request.seed))
    metrics['max_probability_sum_deviation'] = verified.max_sum_deviation
    write_json_atomic(metrics_path, metrics)
    payload_path = Path(request.output_dir) / "fold_result.json"
    write_json_atomic(
        payload_path,
        {
            "schema_version": "1.0",
            "candidate_id": result.candidate_id,
            "seed": int(result.seed),
            "fold": int(result.fold),
            "validation_record_ids": list(result.validation_record_ids),
            "true_labels": list(result.true_labels),
            "probabilities": np.asarray(result.probabilities, dtype=np.float64).tolist(),
            "model_path": str(Path(result.model_path).resolve()),
            "manifest_path": str(Path(result.manifest_path).resolve()),
        },
    )
    return payload_path, {
        "fold_result": sha256_file(payload_path),
        "model": sha256_file(result.model_path),
        "manifest": sha256_file(result.manifest_path),
        "predictions": sha256_file(prediction_path),
        "metrics": sha256_file(metrics_path),
    }


def _completed_fold_result(store: ExperimentRunStore, job: JobKey) -> FoldResult | None:
    job_dir = store.job_dir(job.candidate_id, job.seed, job.fold)
    marker_path = job_dir / "complete.json"
    if not marker_path.is_file():
        return None
    marker = _read_json(marker_path)
    hashes = marker.get("artefact_sha256")
    if not isinstance(hashes, dict):
        raise ExperimentProtocolError(f"Completed job marker is invalid: {marker_path}")
    if (marker.get("status"), marker.get("candidate_id"), marker.get("seed"), marker.get("fold")) != ("complete", job.candidate_id, job.seed, job.fold):
        raise ExperimentProtocolError("Completed job marker identity mismatch")
    relative = marker.get("fold_result_path")
    if not isinstance(relative, str) or not re.fullmatch(r"attempt_[1-9][0-9]*/fold_result.json", relative):
        raise ExperimentProtocolError("Completed job marker must select its fold result attempt")
    payload_path = job_dir / relative
    payload = _read_json(payload_path)
    model_path = Path(str(payload.get("model_path", "")))
    manifest_path = Path(str(payload.get("manifest_path", "")))
    actual = {
        "fold_result": sha256_file(payload_path),
        "model": sha256_file(model_path),
        "manifest": sha256_file(manifest_path),
    }
    for key, name in (('predictions', 'predictions.csv'), ('metrics', 'metrics.json')):
        if key in hashes:
            path = payload_path.parent / name
            if not path.is_file():
                raise ExperimentProtocolError(f'Missing completed fold artefact: {path}')
            actual[key] = sha256_file(path)
    if actual != hashes:
        raise ExperimentProtocolError(f"Completed job artefact hash mismatch: {job_dir}")
    result = FoldResult(
        candidate_id=str(payload["candidate_id"]),
        seed=int(payload["seed"]),
        fold=int(payload["fold"]),
        validation_record_ids=tuple(payload["validation_record_ids"]),
        true_labels=tuple(payload["true_labels"]),
        probabilities=np.asarray(payload["probabilities"], dtype=np.float64),
        model_path=model_path,
        manifest_path=manifest_path,
    )
    if (result.candidate_id, result.seed, result.fold) != (job.candidate_id, job.seed, job.fold):
        raise ExperimentProtocolError("Completed fold result identity mismatch")
    _verify_fold_manifest_artifacts(result)
    verify_probabilities(result.probabilities)
    return result


def _aggregate_ready_seeds(
    store: ExperimentRunStore,
    config: ExperimentConfig,
    frozen_sets: Mapping[str, FrozenRecordSet],
    inputs: Mapping[str, Any],
    *,
    bootstrap_iterations: int,
) -> None:
    registry = _resolved_registry(store.run_dir)
    label_orders = inputs.get("label_orders")
    if not isinstance(label_orders, dict):
        raise ExperimentProtocolError("Frozen label orders are missing")
    for candidate_id in config.candidates:
        candidate = registry[candidate_id]
        for seed in config.seeds:
            jobs = [JobKey(candidate_id, seed, fold) for fold in range(1, 6)]
            results = [_completed_fold_result(store, job) for job in jobs]
            if any(result is None for result in results):
                continue
            output = store.run_dir / "candidates" / candidate_id / f"seed_{seed}"
            verification_path = output / "verification.json"
            if verification_path.is_file():
                _verify_seed(store, candidate_id, int(seed))
                continue
            attempt = _next_directory(output, "aggregation_attempt_")
            aggregate_candidate_seed(
                frozen_sets[candidate.stage].records,
                [result for result in results if result is not None],
                label_order=tuple(label_orders[candidate.stage]),
                output_dir=attempt,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=int(seed),
            )
            for name in ("oof_predictions.csv", "oof_metrics.json", "seed_summary.json", "verification.json"):
                _publish_file(attempt / name, output / name)


def _verify_seed(store: ExperimentRunStore, candidate_id: str, seed: int) -> dict[str, Any] | None:
    output = store.run_dir / "candidates" / candidate_id / f"seed_{seed}"
    if not (output / "verification.json").exists():
        return None
    verification = _read_json(output / "verification.json")
    if (verification.get("status"), verification.get("candidate_id"), verification.get("seed"), verification.get("folds")) != ("complete", candidate_id, seed, [1, 2, 3, 4, 5]):
        raise ExperimentProtocolError(f"Candidate seed identity invalid: {output}")
    names = {"oof_predictions": "oof_predictions.csv", "oof_metrics": "oof_metrics.json", "seed_summary": "seed_summary.json"}
    hashes = verification.get("artefact_sha256", {})
    if set(hashes) != set(names):
        raise ExperimentProtocolError("Candidate seed hashes incomplete")
    _verify_hashes(output, {names[key]: value for key, value in hashes.items()})
    candidate = _resolved_registry(store.run_dir)[candidate_id]
    snapshot = _read_json(store.run_dir / 'record_snapshot.json')[candidate.stage]
    records = tuple(OriginalRecord(**{**row, 'filepath': Path(row['filepath'])}) for row in snapshot['records'])
    inputs = _read_json(store.run_dir / 'inputs.json')
    assignments = store._state_payload()['assignment_hashes']
    for fold in range(1, 6):
        result = _completed_fold_result(store, JobKey(candidate_id, seed, fold))
        if result is None or verification.get("fold_artefact_sha256", {}).get(f"fold_{fold}") != {"model": sha256_file(result.model_path), "manifest": sha256_file(result.manifest_path)}:
            raise ExperimentProtocolError("Candidate seed fold hashes mismatch")
        dataset = build_fold_dataset(records, snapshot['validation_folds'], fold=fold)
        request = FoldRequest(experiment_run_id=store.run_id, pipeline_run_id=inputs['pipeline_run_id'],
                              candidate=candidate, seed=seed, fold=fold,
                              train_records=dataset.train_records, validation_records=dataset.validation_records,
                              label_order=tuple(inputs['label_orders'][candidate.stage]), output_dir=Path(result.manifest_path).parent,
                              runtime={'config_sha256': store.config_hash, 'assignment_sha256': assignments[candidate.stage]})
        _validate_fold_result(request, result)
    return _read_json(output / "seed_summary.json")


def verify_experiment(run_dir: str | Path) -> "ExperimentRunStore":
    """Read-only validation for completed run, summary and parent-wave consumers."""
    store = ExperimentRunStore.open(run_dir)
    store.verify_integrity()
    _verify_reference_snapshot(store)
    if store.state != "complete":
        raise ExperimentProtocolError("Experiment must be verified complete")
    verification = _read_json(store.run_dir / "verification.json")
    if verification.get("status") != "complete" or verification.get("experiment_run_id") != store.run_id:
        raise ExperimentProtocolError("Experiment completion identity invalid")
    _verify_hashes(store.run_dir, verification.get("artefact_sha256", {}))
    if store.pending_jobs():
        raise ExperimentProtocolError("Completed experiment has pending jobs")
    for candidate_id, seed in {(j.candidate_id, j.seed) for j in store._jobs()}:
        if _verify_seed(store, candidate_id, seed) is None:
            raise ExperimentProtocolError("Completed experiment has unverified seed")
    inputs = _read_json(store.run_dir / "inputs.json")
    report = Path(inputs["project_root"]) / "Report" / "experiments" / f"{store.run_id}.md"
    if not report.is_file() or sha256_file(report) != sha256_file(store.run_dir / "comparison.md"):
        raise ExperimentProtocolError("Published comparison report hash mismatch")
    return store


def _summarize_experiment(run_dir: str | Path) -> Path:
    """Finalize a run only when every job and candidate/seed is verified."""

    store = ExperimentRunStore.open(run_dir)
    store.verify_integrity()
    _verify_reference_snapshot(store)
    if store.state == "complete":
        verify_experiment(run_dir)
        return store.run_dir / "verification.json"
    if store.pending_jobs():
        if store.state in {"running", "failed"}:
            jobs = store._jobs()
            pending = store.pending_jobs()
            output = _next_directory(store.run_dir, "summary_attempt_")
            path = output / "summary.json"
            _write_run_reports(store, output, final=False)
            write_json_atomic(
                path,
                {
                    "schema_version": "1.0",
                    "experiment_run_id": store.run_id,
                    "status": "incomplete",
                    "expected_jobs": len(jobs),
                    "completed_jobs": len(jobs) - len(pending),
                    "pending_jobs": len(pending),
                    "resumable": True,
                    "heldout_test_used_for_selection": False,
                },
            )
            store.mark_failed()
            return path
        raise ExperimentStateError("Cannot summarize a run with pending jobs")
    if store.state != "running":
        raise ExperimentStateError("Only a running experiment can be summarized")
    matrix = _read_json(store.run_dir / "candidate_matrix.json")
    rows = matrix.get("jobs", [])
    seed_dirs = {
        store.run_dir
        / "candidates"
        / str(row["candidate_id"])
        / f"seed_{int(row['seed'])}"
        for row in rows
    }
    incomplete = [
        str(path)
        for path in seed_dirs
        if not (path / "verification.json").is_file()
        or _read_json(path / "verification.json").get("status") != "complete"
    ]
    if incomplete:
        store.mark_failed()
        raise ExperimentProtocolError(
            f"Candidate seed verification is missing or incomplete: {incomplete}"
        )
    output = _next_directory(store.run_dir, "report_attempt_")
    _write_run_reports(store, output, final=True)
    for path in sorted(output.iterdir()):
        _publish_file(path, store.run_dir / path.name)
    inputs = _read_json(store.run_dir / "inputs.json")
    _publish_file(output / "comparison.md", Path(inputs["project_root"]) / "Report" / "experiments" / f"{store.run_id}.md")
    return store.finalize(status="complete")


def _reference_oof_metrics(store: ExperimentRunStore, stage: str = "stage2") -> dict[str, float]:
    reference = _read_json(store.run_dir / "reference_run.json")
    stage2 = reference.get(stage)
    if not isinstance(stage2, dict):
        raise ExperimentProtocolError("Stage 2 reference evidence is missing")
    reference_run_dir = Path(str(stage2.get("run_dir", "")))
    payload = _read_json(reference_run_dir / "oof_metrics.json")
    pooled = payload.get("pooled_metrics", payload)
    if not isinstance(pooled, dict):
        raise ExperimentProtocolError("Reference OOF metrics are invalid")
    result: dict[str, float] = {}
    for source, target in (
        ("macro_f1", "oof_macro_f1"),
        ("balanced_accuracy", "oof_balanced_accuracy"),
        ("accuracy", "oof_accuracy"),
    ):
        if source in pooled:
            result[target] = float(pooled[source])
    report = payload.get("classification_report")
    inputs = _read_json(store.run_dir / "inputs.json")
    labels = inputs.get("label_orders", {}).get(stage, [])
    if isinstance(report, dict) and labels:
        recalls = [
            float(report[label]["recall"])
            for label in labels
            if isinstance(report.get(label), dict) and "recall" in report[label]
        ]
        if len(recalls) == len(labels):
            result["minimum_class_recall"] = min(recalls)
    return result


def _write_run_reports(store: ExperimentRunStore, output: Path, *, final: bool) -> None:
    from .reporting import (
        write_experiment_report,
        write_leaderboard_csv,
        write_leaderboard_markdown,
        write_promotion_recommendation,
    )
    from .selection import (
        aggregate_repeated_seeds,
        promotion_decision,
        rank_confirmation_candidates,
        rank_screening_candidates,
    )

    protocol = _read_json(store.run_dir / "protocol.json")
    config_payload = protocol.get("config")
    if not isinstance(config_payload, dict):
        raise ExperimentProtocolError("Frozen config is missing during summary")
    config = _experiment_config(config_payload)
    registry = _resolved_registry(store.run_dir)
    seed_rows: list[dict[str, Any]] = []
    per_class: dict[str, dict[str, float]] = {}
    exclusions: list[dict[str, Any]] = []
    for candidate_id in config.candidates:
        summaries = {seed: _verify_seed(store, candidate_id, seed) for seed in config.seeds}
        if any(summary is None for summary in summaries.values()):
            failures = [_read_json(path) for path in sorted((store.run_dir / 'candidates' / candidate_id).glob('seed_*/fold_*/attempt_*/failure.json'))]
            exclusions.append({"candidate_id": candidate_id, "reason": "incomplete candidate/seed evidence", "failures": failures})
            continue
        for seed in config.seeds:
            seed_dir = (
                store.run_dir / "candidates" / candidate_id / f"seed_{seed}"
            )
            summary = summaries[seed]
            pooled = summary.get("pooled_metrics")
            if not isinstance(pooled, dict):
                raise ExperimentProtocolError(
                    f"Candidate seed metrics are invalid: {seed_dir}"
                )
            seed_rows.append(
                {
                    "candidate_id": candidate_id,
                    "wave": config.wave,
                    "seed": int(seed),
                    "oof_macro_f1": float(pooled["macro_f1"]),
                    "oof_balanced_accuracy": float(pooled["balanced_accuracy"]),
                    "oof_accuracy": float(pooled["accuracy"]),
                    "minimum_class_recall": float(
                        summary["minimum_class_recall"]
                    ),
                    "parameter_count": float(summary.get("parameter_count", 0)),
                    "adapter": registry[candidate_id].adapter,
                    "fold_parameter_counts": summary.get("fold_parameter_counts", {}),
                    "verification_status": "complete",
                }
            )
            if len(config.seeds) == 1:
                per_class[candidate_id] = {
                    str(label): float(value)
                    for label, value in summary.get("class_recall", {}).items()
                }
    if not seed_rows:
        ranked = []
    elif len(config.seeds) == 1:
        ranked = rank_screening_candidates(seed_rows)
    else:
        repeated = aggregate_repeated_seeds(seed_rows)
        for row in repeated:
            row["wave"] = config.wave
        ranked = rank_confirmation_candidates(repeated)
    stages = {registry[candidate_id].stage for candidate_id in config.candidates}
    if len(stages) != 1:
        raise ExperimentProtocolError("Ranking requires candidates from one stage")
    reference_stage = next(iter(stages))
    reference = _reference_oof_metrics(store, reference_stage)
    if final and config.wave == "C" and len(config.seeds) == 3 and ranked and {
        "oof_macro_f1",
        "oof_balanced_accuracy",
        "minimum_class_recall",
    }.issubset(reference):
        promotion = promotion_decision(ranked[0], reference)
    else:
        promotion = {
            "schema_version": "1.0",
            "status": "no_promotion_recommended",
            "checks": {"three_seed_confirmation": False},
            "failed_checks": ["three_seed_confirmation"],
            "selection_scope": "corrected_grouped_internal_validation",
            "heldout_test_used_for_selection": False,
        }
    leaderboard_path = output / "leaderboard.json"
    write_json_atomic(
        leaderboard_path,
        {
            "schema_version": "1.0",
            "experiment_run_id": store.run_id,
            "wave": config.wave,
            "selection_scope": "corrected_grouped_internal_validation",
            "rows": ranked,
            "exclusions": exclusions,
            "heldout_test_used_for_selection": False,
        },
    )
    write_leaderboard_csv(output / "leaderboard.csv", ranked)
    write_leaderboard_markdown(
        output / "leaderboard.md",
        ranked,
        exclusions,
    )
    write_json_atomic(output / "promotion_recommendation.json", promotion)
    write_promotion_recommendation(output / "promotion_recommendation.md", promotion)
    write_json_atomic(output / "selection.json", {
        "schema_version": "1.0", "experiment_run_id": store.run_id,
        "wave": config.wave, "status": "complete" if final else "incomplete",
        "reference_stage": reference_stage, "reference_oof_metrics": reference,
        "ranked_candidate_ids": [row["candidate_id"] for row in ranked],
        "exclusions": exclusions, "promotion": promotion,
        "parent_provenance": _read_json(store.run_dir / "parent_provenance.json"),
        "heldout_test_used_for_selection": False,
    })
    report_payload = {
        "experiment_run_id": store.run_id,
        "wave": config.wave,
        "reference": reference,
        "ranked_candidates": ranked,
        "per_class_oof": per_class,
        "exclusions": exclusions,
        "promotion": promotion,
        "limitations": [
            "Candidate ranking uses corrected grouped internal OOF validation.",
            "Fold estimates are correlated and are not independent experiments.",
            "Independent external validation has not yet been performed.",
        ],
    }
    write_experiment_report(output / "comparison.md", report_payload)


def _train_experiment(
    run_dir: str | Path,
    *,
    adapter_resolver: AdapterResolver = resolve_adapter,
    reference_loader: ReferenceLoader = load_reference_pipeline,
    record_loader: RecordLoader = load_reference_records,
    bootstrap_iterations: int = 2000,
) -> Path:
    """Execute pending Candidate/Seed/Fold jobs without reading locked Test data."""

    store = ExperimentRunStore.open(run_dir)
    try:
        config, _reference, frozen_sets, inputs = _load_lifecycle_inputs(
            store,
            reference_loader=reference_loader,
            record_loader=record_loader,
        )
    except (ExperimentProtocolError, ExperimentStateError, KeyError) as exc:
        store.mark_failed()
        if isinstance(exc, ExperimentProtocolError):
            raise
        raise ExperimentProtocolError(str(exc)) from exc
    if store.state in {"prepared", "failed"}:
        store.mark_running()
    elif store.state != "running":
        raise ExperimentStateError(
            f"train_experiment requires prepared/running state, found {store.state!r}"
        )
    _write_environment_once(store, config)
    registry = _resolved_registry(store.run_dir)
    label_orders = inputs["label_orders"]
    runtime_base = dict(config.parameters)
    from cryinsight.training.feature_cache import FeatureCache

    cache_root = Path(
        str(runtime_base.pop("feature_cache_dir", store.run_dir / "feature_cache"))
    )
    runtime_base["feature_cache"] = FeatureCache(cache_root)
    selected_candidates = [registry[candidate_id] for candidate_id in config.candidates]
    if any(candidate.feature_view == "yamnet_embedding" for candidate in selected_candidates):
        import tensorflow as tf

        from .feature_views import prepare_yamnet_model

        archive = Path(str(inputs["project_root"])) / "yamnet-tensorflow2-yamnet-v1.tar.gz"
        saved_model = prepare_yamnet_model(
            archive,
            store.run_dir / "yamnet_model_cache",
        )
        runtime_base["yamnet_model"] = tf.saved_model.load(str(saved_model))
        runtime_base["yamnet_archive_sha256"] = sha256_file(archive)
    runtime_base["config_sha256"] = store.config_hash
    assignment_hashes = store._state_payload()["assignment_hashes"]
    failed_candidates: set[str] = set()
    for job in store._jobs():
        if job.candidate_id in failed_candidates:
            continue
        store.verify_integrity()
        _verify_reference_snapshot(store)
        candidate = registry[job.candidate_id]
        seed_dir = store.run_dir / 'candidates' / candidate.candidate_id / f'seed_{job.seed}'
        _json_once_or_identical(seed_dir / 'config.json', {
            'candidate': asdict(candidate), 'seed': job.seed, 'config_sha256': store.config_hash,
            'assignment_sha256': assignment_hashes[candidate.stage],
        })
        _publish_file(store.run_dir / 'environment.json', seed_dir / 'environment.json')
        frozen = frozen_sets[candidate.stage]
        dataset = build_fold_dataset(
            frozen.records,
            frozen.validation_folds,
            fold=job.fold,
        )
        output_dir = _next_attempt_dir(store, job)
        request = FoldRequest(
            experiment_run_id=store.run_id,
            pipeline_run_id=str(inputs["pipeline_run_id"]),
            candidate=candidate,
            seed=job.seed,
            fold=job.fold,
            train_records=dataset.train_records,
            validation_records=dataset.validation_records,
            label_order=tuple(label_orders[candidate.stage]),
            output_dir=output_dir,
            runtime={
                **runtime_base,
                "assignment_sha256": assignment_hashes[candidate.stage],
            },
        )
        try:
            previous = _completed_fold_result(store, job)
            if previous is not None:
                _validate_fold_result(request, previous)
                continue
            result = adapter_resolver(candidate).fit_predict_fold(request)
            store.verify_integrity()
            _verify_reference_snapshot(store)
            _validate_fold_result(request, result)
            _payload_path, hashes = _save_fold_result(request, result)
            store.mark_job_complete(
                job.candidate_id,
                job.seed,
                job.fold,
                hashes,
                fold_result_path=str(_payload_path.relative_to(store.job_dir(job.candidate_id, job.seed, job.fold))).replace('\\', '/'),
            )
        except (ExperimentProtocolError, ExperimentStateError, ExperimentVerificationError, ProtocolViolation):
            store.mark_failed()
            raise
        except Exception as exc:
            store.mark_job_failed(
                job.candidate_id,
                job.seed,
                job.fold,
                error_type=type(exc).__name__,
                message=str(exc),
                fail_run=not config.continue_on_candidate_failure,
                attempt_dir=output_dir,
            )
            if config.continue_on_candidate_failure:
                failed_candidates.add(job.candidate_id)
                continue
            raise
    _aggregate_ready_seeds(
        store,
        config,
        frozen_sets,
        inputs,
        bootstrap_iterations=bootstrap_iterations,
    )
    return _summarize_experiment(store.run_dir)


def train_experiment(run_dir: str | Path, **kwargs: Any) -> Path:
    store = ExperimentRunStore.open(run_dir)
    if store.state not in {"prepared", "running"}:
        raise ExperimentStateError(f"Training requires prepared/running state, found {store.state!r}")
    with _run_lock(store):
        try:
            return _train_experiment(store.run_dir, **kwargs)
        except BaseException:
            store.mark_failed()
            raise


def summarize_experiment(run_dir: str | Path) -> Path:
    store = ExperimentRunStore.open(run_dir)
    if store.state == "complete":
        verify_experiment(run_dir)
        return store.run_dir / "verification.json"
    with _run_lock(store):
        try:
            return _summarize_experiment(run_dir)
        except BaseException:
            store.mark_failed()
            raise


def resume_experiment(
    run_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Resume only pending jobs while preserving prior attempt evidence."""

    store = ExperimentRunStore.open(run_dir)
    if store.state not in {"failed", "running"}:
        raise ExperimentStateError("Only a failed/interrupted running experiment can be resumed")
    with _run_lock(store):
        try:
            return _train_experiment(store.run_dir, **kwargs)
        except BaseException:
            store.mark_failed()
            raise


__all__ = [
    "ExperimentPreparation",
    "ExperimentRunStore",
    "ExperimentStateError",
    "JobKey",
    "prepare_experiment",
    "resolve_adapter",
    "resume_experiment",
    "summarize_experiment",
    "train_experiment",
]
