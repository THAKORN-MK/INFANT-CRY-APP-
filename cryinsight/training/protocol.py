"""Dependency-light protocol primitives for leakage-safe model training.

This module deliberately uses only the Python standard library.  Dataset
auditing, cohort resolution, augmentation planning, and split integrity can
therefore be checked before importing the optional ML/audio stack.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable, Sequence


class ProtocolViolation(ValueError):
    """Raised when data would violate the declared experimental protocol."""


class DependencyUnavailable(RuntimeError):
    """Raised when an optional dependency is required for the requested phase."""


@dataclass(frozen=True)
class Esc50Metadata:
    esc_fold: int
    source_file: str
    take: str
    target: int
    category_rule: str


@dataclass(frozen=True)
class CandidateRecord:
    filepath: Path
    relative_path: str
    source_label: str
    model_label: str
    source_dataset: str
    sha256: str


@dataclass(frozen=True)
class OriginalRecord:
    record_id: str
    filepath: Path
    relative_path: str
    label: str
    source_label: str
    source_dataset: str
    group_id: str
    group_rule: str
    sha256: str


@dataclass(frozen=True)
class AuditRow:
    record_id: str
    relative_path: str
    source_label: str
    model_label: str
    source_dataset: str
    sha256: str
    status: str
    exclusion_reason: str = ""
    canonical_record_id: str = ""
    group_id: str = ""
    group_rule: str = ""


@dataclass(frozen=True)
class CohortResolution:
    eligible: tuple[OriginalRecord, ...]
    audit: tuple[AuditRow, ...]


@dataclass(frozen=True)
class HeldoutReservationReport:
    train_original_count_before: int
    train_original_count_after: int
    removed_train_count: int
    heldout_original_count: int
    group_overlap_after: int
    sha256_overlap_after: int


@dataclass(frozen=True)
class AugmentationPlanRow:
    sample_id: str
    original_record_id: str
    original_filepath: str
    original_relative_path: str
    label: str
    partition: str
    fold: int | str
    augmentation_index: int
    augmentation_type: str
    augmentation_params_json: str
    seed: int


@dataclass(frozen=True)
class AugmentationPlan:
    target_samples_per_class: int
    original_by_label: dict[str, int]
    generated_by_label: dict[str, int]
    final_by_label: dict[str, int]
    rows: tuple[AugmentationPlanRow, ...]


@dataclass(frozen=True)
class FoldIntegrityReport:
    train_count: int
    validation_count: int
    train_group_count: int
    validation_group_count: int


@dataclass(frozen=True)
class OofCoverageReport:
    expected_count: int
    predicted_count: int
    duplicate_count: int
    missing_count: int
    unexpected_count: int


@dataclass(frozen=True)
class FoldAssignment:
    record: OriginalRecord
    validation_fold: int
    splitter_name: str
    split_seed: int


@dataclass(frozen=True)
class SplitResult:
    assignments: tuple[FoldAssignment, ...]
    splitter_name: str
    split_seed: int
    n_folds: int
    reliable_groups: bool
    splitter_reason: str
    limitation: str
    validation_class_counts: dict[int, dict[str, int]]


@dataclass(frozen=True)
class FoldAssignmentValidationReport:
    record_count: int
    group_count: int
    content_hash_count: int
    n_folds: int


_ESC50_FILENAME = re.compile(
    r"^(?P<fold>\d+)-(?P<source>\d+)-(?P<take>[A-Z]+)-(?P<target>\d+)\.wav$",
    flags=re.IGNORECASE,
)


def parse_esc50_filename(filename: str) -> Esc50Metadata:
    """Parse official ESC-50 naming metadata without relying on folder order."""

    match = _ESC50_FILENAME.fullmatch(Path(filename).name)
    if match is None:
        raise ProtocolViolation(f"Malformed ESC-50 filename: {filename!r}")
    target = int(match.group("target"))
    return Esc50Metadata(
        esc_fold=int(match.group("fold")),
        source_file=match.group("source"),
        take=match.group("take").upper(),
        target=target,
        category_rule="exclude_crying_baby" if target == 20 else "eligible_negative",
    )


def sha256_path(path: str | Path) -> str:
    """Hash one original file for identity and leakage checks."""

    target = Path(path)
    if not target.is_file():
        raise ProtocolViolation(f"Dataset file does not exist: {target}")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProtocolViolation(f"Could not hash dataset file {target}: {exc}") from exc
    return digest.hexdigest()


def _wav_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.casefold() == ".wav"
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    )


def discover_stage2_candidates(
    data_root: str | Path,
    *,
    label_order: Sequence[str],
) -> tuple[CandidateRecord, ...]:
    """Discover and hash the full original five-class infant corpus."""

    root = Path(data_root)
    if not root.is_dir():
        raise ProtocolViolation(f"Stage 2 dataset directory does not exist: {root}")
    if not label_order or len(set(label_order)) != len(label_order):
        raise ProtocolViolation("Stage 2 label order must contain unique labels")
    candidates: list[CandidateRecord] = []
    for label in label_order:
        label_directory = root / label
        if not label_directory.is_dir():
            raise ProtocolViolation(f"Stage 2 label directory is missing: {label_directory}")
        files = _wav_files(label_directory)
        if not files:
            raise ProtocolViolation(f"Stage 2 label has no WAV files: {label!r}")
        for filepath in files:
            candidates.append(
                CandidateRecord(
                    filepath=filepath.resolve(),
                    relative_path=filepath.relative_to(root).as_posix(),
                    source_label=label,
                    model_label=label,
                    source_dataset="infantcry_dbl",
                    sha256=sha256_path(filepath),
                )
            )
    return tuple(candidates)


def discover_stage1_candidates(
    data_root: str | Path,
    *,
    infant_labels: Sequence[str],
    negative_label: str = "not_baby",
) -> tuple[tuple[CandidateRecord, ...], tuple[CandidateRecord, ...]]:
    """Discover infant positives and ESC-50 negative candidates for Stage 1."""

    root = Path(data_root)
    if not root.is_dir():
        raise ProtocolViolation(f"Stage 1 dataset directory does not exist: {root}")
    infant = list(
        discover_stage2_candidates(root, label_order=infant_labels)
    )
    infant = [
        CandidateRecord(
            filepath=row.filepath,
            relative_path=row.relative_path,
            source_label=row.source_label,
            model_label="baby",
            source_dataset=row.source_dataset,
            sha256=row.sha256,
        )
        for row in infant
    ]
    negative_directory = root / negative_label
    if not negative_directory.is_dir():
        raise ProtocolViolation(
            f"Stage 1 negative directory is missing: {negative_directory}"
        )
    negative_files = _wav_files(negative_directory)
    if not negative_files:
        raise ProtocolViolation("Stage 1 negative class has no WAV files")
    negatives = tuple(
        CandidateRecord(
            filepath=filepath.resolve(),
            relative_path=filepath.relative_to(root).as_posix(),
            source_label=negative_label,
            model_label="not_baby",
            source_dataset="esc50",
            sha256=sha256_path(filepath),
        )
        for filepath in negative_files
    )
    return tuple(infant), negatives


def _record_id(candidate: CandidateRecord) -> str:
    identity = f"{candidate.source_dataset}|{candidate.relative_path}".encode("utf-8")
    return f"rec_{hashlib.sha256(identity).hexdigest()[:20]}"


def _audit_row(
    candidate: CandidateRecord,
    *,
    status: str,
    exclusion_reason: str = "",
    canonical_record_id: str = "",
    group_id: str = "",
    group_rule: str = "",
) -> AuditRow:
    return AuditRow(
        record_id=_record_id(candidate),
        relative_path=candidate.relative_path,
        source_label=candidate.source_label,
        model_label=candidate.model_label,
        source_dataset=candidate.source_dataset,
        sha256=candidate.sha256,
        status=status,
        exclusion_reason=exclusion_reason,
        canonical_record_id=canonical_record_id,
        group_id=group_id,
        group_rule=group_rule,
    )


def _original_record(
    candidate: CandidateRecord,
    *,
    label: str,
    group_id: str,
    group_rule: str,
) -> OriginalRecord:
    return OriginalRecord(
        record_id=_record_id(candidate),
        filepath=candidate.filepath,
        relative_path=candidate.relative_path,
        label=label,
        source_label=candidate.source_label,
        source_dataset=candidate.source_dataset,
        group_id=group_id,
        group_rule=group_rule,
        sha256=candidate.sha256,
    )


def _validate_candidates(records: Sequence[CandidateRecord]) -> None:
    seen_paths: set[tuple[str, str]] = set()
    for record in records:
        key = (record.source_dataset.casefold(), record.relative_path.casefold())
        if key in seen_paths:
            raise ProtocolViolation(
                "Duplicate candidate identity: "
                f"{record.source_dataset}/{record.relative_path}"
            )
        seen_paths.add(key)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", record.sha256):
            raise ProtocolViolation(
                f"Invalid SHA-256 for {record.relative_path!r}: {record.sha256!r}"
            )


def resolve_stage2_records(records: Iterable[CandidateRecord]) -> CohortResolution:
    """Resolve the five-class cohort under exact-content duplicate rules.

    A same-label duplicate group keeps only its lexicographically first path.
    A hash observed under more than one model label is excluded in full.
    """

    candidates = tuple(records)
    _validate_candidates(candidates)
    by_hash: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        by_hash[candidate.sha256.lower()].append(candidate)

    eligible: list[OriginalRecord] = []
    audit: list[AuditRow] = []
    for content_hash in sorted(by_hash):
        duplicates = sorted(
            by_hash[content_hash],
            key=lambda item: (item.relative_path.casefold(), item.source_dataset.casefold()),
        )
        labels = {item.model_label for item in duplicates}
        if len(labels) > 1:
            audit.extend(
                _audit_row(
                    item,
                    status="excluded",
                    exclusion_reason="cross_label_exact_duplicate",
                )
                for item in duplicates
            )
            continue

        canonical = duplicates[0]
        canonical_id = _record_id(canonical)
        group_id = f"sha256:{content_hash}"
        group_rule = "exact_content_sha256"
        eligible.append(
            _original_record(
                canonical,
                label=canonical.model_label,
                group_id=group_id,
                group_rule=group_rule,
            )
        )
        audit.append(
            _audit_row(
                canonical,
                status="eligible",
                canonical_record_id=canonical_id,
                group_id=group_id,
                group_rule=group_rule,
            )
        )
        audit.extend(
            _audit_row(
                duplicate,
                status="excluded",
                exclusion_reason="same_label_exact_duplicate",
                canonical_record_id=canonical_id,
            )
            for duplicate in duplicates[1:]
        )

    eligible.sort(key=lambda item: (item.label, item.relative_path.casefold()))
    audit.sort(key=lambda item: (item.relative_path.casefold(), item.source_dataset.casefold()))
    return CohortResolution(tuple(eligible), tuple(audit))


def resolve_stage1_records(
    infant_records: Iterable[CandidateRecord],
    esc50_records: Iterable[CandidateRecord],
) -> CohortResolution:
    """Resolve Stage 1 while excluding ESC-50's ``crying_baby`` category."""

    infants = tuple(infant_records)
    negatives = tuple(esc50_records)
    _validate_candidates(infants + negatives)

    audit: list[AuditRow] = []
    included: list[tuple[CandidateRecord, str, str, str]] = []
    for candidate in infants:
        included.append(
            (
                candidate,
                "baby",
                f"infant_sha256:{candidate.sha256.lower()}",
                "exact_content_sha256",
            )
        )

    for candidate in negatives:
        metadata = parse_esc50_filename(candidate.filepath.name)
        if metadata.target == 20:
            audit.append(
                _audit_row(
                    candidate,
                    status="excluded",
                    exclusion_reason="esc50_crying_baby",
                )
            )
            continue
        included.append(
            (
                candidate,
                "not_baby",
                f"esc50_source:{metadata.source_file}",
                "esc50_source_file",
            )
        )

    by_hash: dict[str, list[tuple[CandidateRecord, str, str, str]]] = defaultdict(list)
    for item in included:
        by_hash[item[0].sha256.lower()].append(item)

    eligible: list[OriginalRecord] = []
    for content_hash in sorted(by_hash):
        duplicates = sorted(
            by_hash[content_hash],
            key=lambda item: (
                item[0].relative_path.casefold(),
                item[0].source_dataset.casefold(),
            ),
        )
        if len({item[1] for item in duplicates}) > 1:
            audit.extend(
                _audit_row(
                    item[0],
                    status="excluded",
                    exclusion_reason="cross_label_exact_duplicate",
                )
                for item in duplicates
            )
            continue

        canonical, label, group_id, group_rule = duplicates[0]
        canonical_id = _record_id(canonical)
        eligible.append(
            _original_record(
                canonical,
                label=label,
                group_id=group_id,
                group_rule=group_rule,
            )
        )
        audit.append(
            _audit_row(
                canonical,
                status="eligible",
                canonical_record_id=canonical_id,
                group_id=group_id,
                group_rule=group_rule,
            )
        )
        audit.extend(
            _audit_row(
                item[0],
                status="excluded",
                exclusion_reason="same_label_exact_duplicate",
                canonical_record_id=canonical_id,
            )
            for item in duplicates[1:]
        )

    eligible.sort(key=lambda item: (item.label, item.relative_path.casefold()))
    audit.sort(key=lambda item: (item.relative_path.casefold(), item.source_dataset.casefold()))
    return CohortResolution(tuple(eligible), tuple(audit))


def reserve_heldout_groups(
    training: CohortResolution,
    heldout_records: Iterable[OriginalRecord],
) -> tuple[CohortResolution, HeldoutReservationReport]:
    """Give a locked held-out cohort priority over overlapping train records."""

    heldout = tuple(heldout_records)
    if not heldout:
        raise ProtocolViolation("Locked held-out test cohort cannot be empty")
    heldout_groups = {row.group_id for row in heldout}
    heldout_hashes = {row.sha256.lower() for row in heldout}
    removed_reasons: dict[str, str] = {}
    retained: list[OriginalRecord] = []
    for record in training.eligible:
        if record.group_id in heldout_groups:
            removed_reasons[record.record_id] = "reserved_heldout_test_group"
        elif record.sha256.lower() in heldout_hashes:
            removed_reasons[record.record_id] = "reserved_heldout_test_hash"
        else:
            retained.append(record)

    retained_groups = {row.group_id for row in retained}
    retained_hashes = {row.sha256.lower() for row in retained}
    group_overlap = retained_groups & heldout_groups
    hash_overlap = retained_hashes & heldout_hashes
    if group_overlap or hash_overlap:
        raise ProtocolViolation(
            "Could not isolate locked held-out test groups from training"
        )

    audit = tuple(
        replace(
            row,
            status="excluded",
            exclusion_reason=removed_reasons[row.record_id],
        )
        if row.record_id in removed_reasons and row.status == "eligible"
        else row
        for row in training.audit
    )
    filtered = CohortResolution(tuple(retained), audit)
    report = HeldoutReservationReport(
        train_original_count_before=len(training.eligible),
        train_original_count_after=len(retained),
        removed_train_count=len(removed_reasons),
        heldout_original_count=len(heldout),
        group_overlap_after=len(group_overlap),
        sha256_overlap_after=len(hash_overlap),
    )
    return filtered, report


_AUGMENTATIONS: tuple[tuple[str, dict[str, float]], ...] = (
    ("gaussian_noise", {"noise_factor": 0.005}),
    ("pitch_shift", {"steps": 1.0}),
    ("pitch_shift", {"steps": -1.0}),
    ("time_stretch", {"rate": 1.1}),
    ("time_stretch", {"rate": 0.9}),
    ("time_shift", {"max_fraction": 0.1}),
    ("gaussian_noise", {"noise_factor": 0.0025}),
    ("pitch_shift", {"steps": 2.0}),
    ("pitch_shift", {"steps": -2.0}),
    ("time_stretch", {"rate": 1.2}),
    ("time_stretch", {"rate": 0.8}),
)


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _materialize_augmentation_params(
    augmentation_type: str,
    base_params: dict[str, float],
    *,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    if augmentation_type == "gaussian_noise":
        base = base_params["noise_factor"]
        return {"noise_factor": round(base * rng.uniform(0.85, 1.15), 9)}
    if augmentation_type == "pitch_shift":
        base = base_params["steps"]
        return {"steps": round(base + rng.uniform(-0.2, 0.2), 6)}
    if augmentation_type == "time_stretch":
        base = base_params["rate"]
        candidate = base + rng.uniform(-0.025, 0.025)
        if abs(candidate - 1.0) < 0.01:
            candidate = 1.01 if base > 1.0 else 0.99
        return {"rate": round(candidate, 6)}
    if augmentation_type == "time_shift":
        maximum = base_params["max_fraction"]
        fraction = rng.uniform(-maximum, maximum)
        if abs(fraction) < 1e-4:
            fraction = 1e-4 if rng.random() >= 0.5 else -1e-4
        return {"fraction": round(fraction, 7)}
    return dict(base_params)


def build_target_augmentation_plan(
    records: Iterable[OriginalRecord],
    *,
    fold: int | str,
    seed: int,
) -> AugmentationPlan:
    """Plan deterministic training-only augmentation to the largest class."""

    originals = tuple(records)
    if not originals:
        raise ProtocolViolation("Cannot build an augmentation plan for an empty fold")
    if isinstance(fold, int):
        if fold < 1:
            raise ProtocolViolation("Fold numbers must start at 1")
    elif fold != "final_refit":
        raise ProtocolViolation(
            "Augmentation identity must be a fold number or final_refit"
        )

    by_label: dict[str, list[OriginalRecord]] = defaultdict(list)
    for record in originals:
        by_label[record.label].append(record)
    for label in by_label:
        by_label[label].sort(key=lambda item: item.record_id)

    original_counts = {label: len(by_label[label]) for label in sorted(by_label)}
    target = max(original_counts.values())
    generated_counts = {label: target - count for label, count in original_counts.items()}
    final_counts = {label: target for label in original_counts}

    rows: list[AugmentationPlanRow] = []
    for label in sorted(by_label):
        sources = list(by_label[label])
        random.Random(_derived_seed(seed, fold, label, "sources")).shuffle(sources)
        needed = generated_counts[label]
        for augmentation_index in range(needed):
            source = sources[augmentation_index % len(sources)]
            augmentation_type, base_params = _AUGMENTATIONS[
                augmentation_index % len(_AUGMENTATIONS)
            ]
            sample_seed = _derived_seed(
                seed,
                fold,
                label,
                source.record_id,
                augmentation_index,
                augmentation_type,
            )
            params = _materialize_augmentation_params(
                augmentation_type,
                base_params,
                seed=sample_seed,
            )
            sample_id = "aug_" + hashlib.sha256(
                (
                    f"{fold}|{label}|{source.record_id}|{augmentation_index}|"
                    f"{augmentation_type}|{sample_seed}"
                ).encode("utf-8")
            ).hexdigest()[:20]
            rows.append(
                AugmentationPlanRow(
                    sample_id=sample_id,
                    original_record_id=source.record_id,
                    original_filepath=str(source.filepath),
                    original_relative_path=source.relative_path,
                    label=label,
                    partition="train",
                    fold=fold,
                    augmentation_index=augmentation_index,
                    augmentation_type=augmentation_type,
                    augmentation_params_json=json.dumps(
                        params, sort_keys=True, separators=(",", ":")
                    ),
                    seed=sample_seed,
                )
            )

    return AugmentationPlan(
        target_samples_per_class=target,
        original_by_label=original_counts,
        generated_by_label=generated_counts,
        final_by_label=final_counts,
        rows=tuple(rows),
    )


def assert_fold_integrity(
    train_records: Iterable[OriginalRecord],
    validation_records: Iterable[OriginalRecord],
) -> FoldIntegrityReport:
    """Reject direct, group-level, or exact-content leakage across a fold."""

    train = tuple(train_records)
    validation = tuple(validation_records)
    record_overlap = {item.record_id for item in train} & {
        item.record_id for item in validation
    }
    if record_overlap:
        raise ProtocolViolation(
            f"record overlap between train and validation: {sorted(record_overlap)!r}"
        )

    group_overlap = {item.group_id for item in train} & {
        item.group_id for item in validation
    }
    if group_overlap:
        raise ProtocolViolation(
            f"group overlap between train and validation: {sorted(group_overlap)!r}"
        )

    hash_overlap = {item.sha256.lower() for item in train} & {
        item.sha256.lower() for item in validation
    }
    if hash_overlap:
        raise ProtocolViolation(
            f"hash overlap between train and validation: {sorted(hash_overlap)!r}"
        )

    return FoldIntegrityReport(
        train_count=len(train),
        validation_count=len(validation),
        train_group_count=len({item.group_id for item in train}),
        validation_group_count=len({item.group_id for item in validation}),
    )


def assert_exact_oof_coverage(
    expected_record_ids: Iterable[str],
    predicted_record_ids: Iterable[str],
) -> OofCoverageReport:
    """Require exactly one out-of-fold prediction for every expected original."""

    expected = tuple(expected_record_ids)
    predicted = tuple(predicted_record_ids)
    expected_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    repeated_expected = sorted(
        record_id for record_id, count in expected_counts.items() if count != 1
    )
    if repeated_expected:
        raise ProtocolViolation(
            "duplicate identifiers in expected OOF cohort: " f"{repeated_expected!r}"
        )

    duplicates = sorted(
        record_id for record_id, count in predicted_counts.items() if count > 1
    )
    if duplicates:
        raise ProtocolViolation(f"duplicate OOF predictions: {duplicates!r}")

    expected_set = set(expected_counts)
    predicted_set = set(predicted_counts)
    missing = sorted(expected_set - predicted_set)
    if missing:
        raise ProtocolViolation(f"missing OOF predictions: {missing!r}")
    unexpected = sorted(predicted_set - expected_set)
    if unexpected:
        raise ProtocolViolation(f"unexpected OOF predictions: {unexpected!r}")

    return OofCoverageReport(
        expected_count=len(expected),
        predicted_count=len(predicted),
        duplicate_count=0,
        missing_count=0,
        unexpected_count=0,
    )


def validate_fold_assignments(
    assignments: Iterable[FoldAssignment],
    *,
    n_folds: int,
) -> FoldAssignmentValidationReport:
    """Validate that every original, group, and content hash has one fold."""

    rows = tuple(assignments)
    if not rows:
        raise ProtocolViolation("Fold assignments are empty")
    if n_folds < 2:
        raise ProtocolViolation("n_folds must be at least 2")

    record_counts = Counter(row.record.record_id for row in rows)
    duplicate_records = sorted(
        record_id for record_id, count in record_counts.items() if count != 1
    )
    if duplicate_records:
        raise ProtocolViolation(
            f"duplicate records in fold assignments: {duplicate_records!r}"
        )

    expected_folds = set(range(1, n_folds + 1))
    observed_folds = {row.validation_fold for row in rows}
    invalid_folds = sorted(observed_folds - expected_folds)
    if invalid_folds:
        raise ProtocolViolation(f"invalid validation fold numbers: {invalid_folds!r}")
    missing_folds = sorted(expected_folds - observed_folds)
    if missing_folds:
        raise ProtocolViolation(f"validation folds contain no records: {missing_folds!r}")

    group_folds: dict[str, set[int]] = defaultdict(set)
    hash_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group_folds[row.record.group_id].add(row.validation_fold)
        hash_folds[row.record.sha256.lower()].add(row.validation_fold)

    split_groups = sorted(group for group, folds in group_folds.items() if len(folds) > 1)
    if split_groups:
        raise ProtocolViolation(
            "groups assigned to multiple validation folds: " f"{split_groups!r}"
        )
    split_hashes = sorted(content_hash for content_hash, folds in hash_folds.items() if len(folds) > 1)
    if split_hashes:
        raise ProtocolViolation(
            "content hashes assigned to multiple validation folds: " f"{split_hashes!r}"
        )

    return FoldAssignmentValidationReport(
        record_count=len(rows),
        group_count=len(group_folds),
        content_hash_count=len(hash_folds),
        n_folds=n_folds,
    )


def _assignments_from_splits(
    records: Sequence[OriginalRecord],
    splits: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    splitter_name: str,
    seed: int,
    n_folds: int,
) -> tuple[FoldAssignment, ...]:
    by_record_id: dict[str, FoldAssignment] = {}
    for fold, (_, validation_indices) in enumerate(splits, start=1):
        for raw_index in validation_indices:
            index = int(raw_index)
            record = records[index]
            if record.record_id in by_record_id:
                raise ProtocolViolation(
                    f"record assigned to validation more than once: {record.record_id}"
                )
            by_record_id[record.record_id] = FoldAssignment(
                record=record,
                validation_fold=fold,
                splitter_name=splitter_name,
                split_seed=seed,
            )
    missing = sorted({record.record_id for record in records} - set(by_record_id))
    if missing:
        raise ProtocolViolation(f"records missing a validation fold: {missing!r}")
    assignments = tuple(
        sorted(
            by_record_id.values(),
            key=lambda row: (
                row.validation_fold,
                row.record.label,
                row.record.relative_path.casefold(),
            ),
        )
    )
    validate_fold_assignments(assignments, n_folds=n_folds)
    return assignments


def _class_counts_by_fold(
    assignments: Sequence[FoldAssignment],
    *,
    n_folds: int,
) -> dict[int, dict[str, int]]:
    all_labels = sorted({row.record.label for row in assignments})
    result: dict[int, dict[str, int]] = {}
    for fold in range(1, n_folds + 1):
        counts = Counter(
            row.record.label for row in assignments if row.validation_fold == fold
        )
        missing = sorted(set(all_labels) - set(counts))
        if missing:
            raise ProtocolViolation(
                f"validation fold {fold} is missing labels: {missing!r}"
            )
        result[fold] = {label: counts[label] for label in all_labels}
    return result


def assign_grouped_folds(
    records: Iterable[OriginalRecord],
    *,
    n_folds: int = 5,
    seed: int = 42,
    reliable_groups: bool = True,
    no_group_evidence: str = "",
) -> SplitResult:
    """Assign deterministic validation folds using the prompt's splitter order.

    Scikit-learn is imported only when this operation is requested so audit and
    record-resolution commands can run in a dependency-light environment.
    """

    originals = tuple(
        sorted(records, key=lambda item: (item.label, item.relative_path.casefold()))
    )
    if len(originals) < n_folds:
        raise ProtocolViolation(
            f"Need at least {n_folds} originals, found {len(originals)}"
        )
    if n_folds < 2:
        raise ProtocolViolation("n_folds must be at least 2")
    if len({record.record_id for record in originals}) != len(originals):
        raise ProtocolViolation("Original record IDs must be unique before splitting")
    if len({record.sha256.lower() for record in originals}) != len(originals):
        raise ProtocolViolation("Exact-content duplicates must be resolved before splitting")

    try:
        from sklearn.model_selection import (  # type: ignore[import-not-found]
            GroupKFold,
            StratifiedGroupKFold,
            StratifiedKFold,
        )
    except ImportError as exc:
        raise DependencyUnavailable(
            "scikit-learn is required for fold assignment; activate the CryInsight "
            "ML environment before using --prepare-only or training"
        ) from exc

    indices = list(range(len(originals)))
    labels = [record.label for record in originals]
    groups = [record.group_id for record in originals]

    if not reliable_groups:
        if not no_group_evidence.strip():
            raise ProtocolViolation(
                "StratifiedKFold requires explicit evidence that no reliable group "
                "relationship is available"
            )
        label_counts = Counter(labels)
        too_small = {label: count for label, count in label_counts.items() if count < n_folds}
        if too_small:
            raise ProtocolViolation(
                f"StratifiedKFold is infeasible for class counts: {too_small!r}"
            )
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        assignments = _assignments_from_splits(
            originals,
            splitter.split(indices, labels),
            splitter_name="StratifiedKFold",
            seed=seed,
            n_folds=n_folds,
        )
        class_counts = _class_counts_by_fold(assignments, n_folds=n_folds)
        return SplitResult(
            assignments=assignments,
            splitter_name="StratifiedKFold",
            split_seed=seed,
            n_folds=n_folds,
            reliable_groups=False,
            splitter_reason=no_group_evidence.strip(),
            limitation=(
                "clip-level internal validation; no reliable subject/session/source "
                "group relationship was available"
            ),
            validation_class_counts=class_counts,
        )

    if any(not group_id for group_id in groups):
        raise ProtocolViolation("Reliable grouped splitting requires non-empty group IDs")
    group_count = len(set(groups))
    if group_count < n_folds:
        raise ProtocolViolation(
            f"Grouped {n_folds}-fold validation needs at least {n_folds} groups; "
            f"found {group_count}"
        )
    groups_by_label: dict[str, set[str]] = defaultdict(set)
    for label, group_id in zip(labels, groups, strict=True):
        groups_by_label[label].add(group_id)
    infeasible = {
        label: len(label_groups)
        for label, label_groups in groups_by_label.items()
        if len(label_groups) < n_folds
    }
    if infeasible:
        raise ProtocolViolation(
            "Every class needs validation support in every fold; distinct group "
            f"counts are infeasible: {infeasible!r}"
        )

    sgkf_error = ""
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        assignments = _assignments_from_splits(
            originals,
            splitter.split(indices, labels, groups),
            splitter_name="StratifiedGroupKFold",
            seed=seed,
            n_folds=n_folds,
        )
        class_counts = _class_counts_by_fold(assignments, n_folds=n_folds)
        return SplitResult(
            assignments=assignments,
            splitter_name="StratifiedGroupKFold",
            split_seed=seed,
            n_folds=n_folds,
            reliable_groups=True,
            splitter_reason=(
                "Reliable group IDs are available and stratified grouped splitting "
                "preserved every class in every validation fold"
            ),
            limitation="",
            validation_class_counts=class_counts,
        )
    except (ProtocolViolation, ValueError) as exc:
        sgkf_error = str(exc)

    try:
        splitter = GroupKFold(n_splits=n_folds)
        assignments = _assignments_from_splits(
            originals,
            splitter.split(indices, labels, groups),
            splitter_name="GroupKFold",
            seed=seed,
            n_folds=n_folds,
        )
        class_counts = _class_counts_by_fold(assignments, n_folds=n_folds)
    except (ProtocolViolation, ValueError) as exc:
        raise ProtocolViolation(
            "Neither StratifiedGroupKFold nor GroupKFold produced a valid split "
            f"with every class in every fold. SGKF: {sgkf_error}; GroupKFold: {exc}"
        ) from exc
    return SplitResult(
        assignments=assignments,
        splitter_name="GroupKFold",
        split_seed=seed,
        n_folds=n_folds,
        reliable_groups=True,
        splitter_reason=(
            "StratifiedGroupKFold was infeasible; GroupKFold preserved group "
            f"isolation and class support. SGKF detail: {sgkf_error}"
        ),
        limitation="GroupKFold does not explicitly optimize class stratification",
        validation_class_counts=class_counts,
    )


def _write_text_once(path: Path, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable artefact: {output}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"Refusing to replace immutable artefact: {output}")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write a UTF-8 JSON artefact once, using an atomic final rename."""

    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text_once(Path(path), content)


def write_fold_assignments_csv(
    path: Path,
    assignments: Iterable[FoldAssignment],
) -> None:
    """Write the auditable fold-assignment contract without array-order reliance."""

    rows = tuple(
        sorted(
            assignments,
            key=lambda row: (
                row.validation_fold,
                row.record.label,
                row.record.relative_path.casefold(),
            ),
        )
    )
    buffer = io.StringIO(newline="")
    fieldnames = [
        "record_id",
        "filepath",
        "relative_path",
        "label",
        "validation_fold",
        "source_dataset",
        "group_id",
        "group_rule",
        "sha256",
        "splitter_name",
        "split_seed",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for assignment in rows:
        record = assignment.record
        writer.writerow(
            {
                "record_id": record.record_id,
                "filepath": str(record.filepath),
                "relative_path": record.relative_path,
                "label": record.label,
                "validation_fold": assignment.validation_fold,
                "source_dataset": record.source_dataset,
                "group_id": record.group_id,
                "group_rule": record.group_rule,
                "sha256": record.sha256.lower(),
                "splitter_name": assignment.splitter_name,
                "split_seed": assignment.split_seed,
            }
        )
    _write_text_once(Path(path), buffer.getvalue())


def write_record_audit_csv(path: Path, rows: Iterable[AuditRow]) -> None:
    """Write every discovered candidate and its eligibility decision."""

    ordered = tuple(
        sorted(rows, key=lambda row: (row.relative_path.casefold(), row.source_dataset))
    )
    buffer = io.StringIO(newline="")
    fieldnames = [field.name for field in AuditRow.__dataclass_fields__.values()]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow({field: getattr(row, field) for field in fieldnames})
    _write_text_once(Path(path), buffer.getvalue())
