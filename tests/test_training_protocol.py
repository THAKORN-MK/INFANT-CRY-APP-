from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from cryinsight.training.protocol import (
    AuditRow,
    CandidateRecord,
    CohortResolution,
    FoldAssignment,
    OriginalRecord,
    ProtocolViolation,
    assign_grouped_folds,
    assert_exact_oof_coverage,
    assert_fold_integrity,
    build_target_augmentation_plan,
    parse_esc50_filename,
    resolve_stage1_records,
    resolve_stage2_records,
    reserve_heldout_groups,
    validate_fold_assignments,
    write_fold_assignments_csv,
    write_json_atomic,
)


def candidate(
    name: str,
    *,
    source_label: str,
    model_label: str,
    sha256: str,
    source_dataset: str = "infantcry_dbl",
) -> CandidateRecord:
    return CandidateRecord(
        filepath=Path("D:/INFANT CRY/data_set_dbl") / source_label / name,
        relative_path=f"{source_label}/{name}",
        source_label=source_label,
        model_label=model_label,
        source_dataset=source_dataset,
        sha256=sha256,
    )


def original(record_id: str, label: str, group_id: str, sha256: str) -> OriginalRecord:
    return OriginalRecord(
        record_id=record_id,
        filepath=Path(f"D:/data/{record_id}.wav"),
        relative_path=f"{label}/{record_id}.wav",
        label=label,
        source_label=label,
        source_dataset="fixture",
        group_id=group_id,
        group_rule="fixture_group",
        sha256=sha256,
    )


class Esc50ParsingTests(unittest.TestCase):
    def test_target_20_is_marked_for_exclusion_from_negative_class(self) -> None:
        metadata = parse_esc50_filename("1-187207-A-20.wav")

        self.assertEqual(metadata.esc_fold, 1)
        self.assertEqual(metadata.source_file, "187207")
        self.assertEqual(metadata.take, "A")
        self.assertEqual(metadata.target, 20)
        self.assertEqual(metadata.category_rule, "exclude_crying_baby")

    def test_non_crying_baby_target_is_an_eligible_negative(self) -> None:
        metadata = parse_esc50_filename("1-100032-A-0.wav")

        self.assertEqual(metadata.category_rule, "eligible_negative")

    def test_malformed_esc50_filename_is_rejected(self) -> None:
        with self.assertRaises(ProtocolViolation):
            parse_esc50_filename("not-an-esc50-name.wav")


class CohortResolutionTests(unittest.TestCase):
    def test_stage2_keeps_one_canonical_copy_for_same_label_hash(self) -> None:
        records = [
            candidate("z.wav", source_label="hungry", model_label="hungry", sha256="a" * 64),
            candidate("a.wav", source_label="hungry", model_label="hungry", sha256="a" * 64),
        ]

        result = resolve_stage2_records(records)

        self.assertEqual(len(result.eligible), 1)
        self.assertEqual(result.eligible[0].filepath.name, "a.wav")
        excluded = [row for row in result.audit if row.status == "excluded"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].exclusion_reason, "same_label_exact_duplicate")

    def test_stage2_excludes_entire_cross_label_hash_group(self) -> None:
        records = [
            candidate("a.wav", source_label="burping", model_label="burping", sha256="b" * 64),
            candidate(
                "b.wav",
                source_label="discomfort",
                model_label="discomfort",
                sha256="b" * 64,
            ),
        ]

        result = resolve_stage2_records(records)

        self.assertEqual(result.eligible, ())
        self.assertEqual({row.exclusion_reason for row in result.audit}, {"cross_label_exact_duplicate"})

    def test_stage1_excludes_crying_baby_and_groups_esc50_by_source(self) -> None:
        infant = [
            candidate("baby.wav", source_label="hungry", model_label="baby", sha256="c" * 64),
        ]
        esc50 = [
            candidate(
                "1-211527-A-20.wav",
                source_label="not_baby",
                model_label="not_baby",
                sha256="d" * 64,
                source_dataset="esc50",
            ),
            candidate(
                "1-100210-A-36.wav",
                source_label="not_baby",
                model_label="not_baby",
                sha256="e" * 64,
                source_dataset="esc50",
            ),
            candidate(
                "1-100210-B-36.wav",
                source_label="not_baby",
                model_label="not_baby",
                sha256="f" * 64,
                source_dataset="esc50",
            ),
        ]

        result = resolve_stage1_records(infant, esc50)

        self.assertEqual([record.label for record in result.eligible].count("baby"), 1)
        negative = [record for record in result.eligible if record.label == "not_baby"]
        self.assertEqual(len(negative), 2)
        self.assertEqual({record.group_id for record in negative}, {"esc50_source:100210"})
        excluded = [row for row in result.audit if row.exclusion_reason == "esc50_crying_baby"]
        self.assertEqual(len(excluded), 1)


class AugmentationPlanningTests(unittest.TestCase):
    def test_final_refit_plan_has_explicit_non_fold_identity(self) -> None:
        records = [
            original("a1", "a", "ga1", "1" * 64),
            original("b1", "b", "gb1", "2" * 64),
            original("b2", "b", "gb2", "3" * 64),
        ]

        plan = build_target_augmentation_plan(
            records,
            fold="final_refit",
            seed=42,
        )

        self.assertTrue(plan.rows)
        self.assertTrue(all(row.fold == "final_refit" for row in plan.rows))

    def test_target_is_computed_from_each_fold_training_original_count(self) -> None:
        records = [
            original("a1", "a", "ga1", "1" * 64),
            original("a2", "a", "ga2", "2" * 64),
            original("b1", "b", "gb1", "3" * 64),
            original("b2", "b", "gb2", "4" * 64),
            original("b3", "b", "gb3", "5" * 64),
            original("b4", "b", "gb4", "6" * 64),
        ]

        plan = build_target_augmentation_plan(records, fold=1, seed=42)

        self.assertEqual(plan.target_samples_per_class, 4)
        self.assertEqual(plan.original_by_label, {"a": 2, "b": 4})
        self.assertEqual(plan.generated_by_label, {"a": 2, "b": 0})
        self.assertEqual(plan.final_by_label, {"a": 4, "b": 4})
        self.assertTrue(all(row.partition == "train" for row in plan.rows))

    def test_augmentation_plan_is_deterministic_for_same_seed(self) -> None:
        records = [
            original("a1", "a", "ga1", "1" * 64),
            original("b1", "b", "gb1", "2" * 64),
            original("b2", "b", "gb2", "3" * 64),
            original("b3", "b", "gb3", "4" * 64),
        ]

        first = build_target_augmentation_plan(records, fold=3, seed=42)
        second = build_target_augmentation_plan(records, fold=3, seed=42)

        self.assertEqual(first.rows, second.rows)

    def test_manifest_row_keeps_absolute_source_path(self) -> None:
        records = [
            original("a1", "a", "ga1", "1" * 64),
            original("b1", "b", "gb1", "2" * 64),
            original("b2", "b", "gb2", "3" * 64),
        ]

        plan = build_target_augmentation_plan(records, fold=1, seed=42)

        self.assertEqual(plan.rows[0].original_filepath, str(records[0].filepath))

    def test_repeated_transform_on_one_source_gets_distinct_recorded_parameters(self) -> None:
        records = [original("a1", "a", "ga1", "1" * 64)]
        records.extend(
            original(
                f"b{index}",
                "b",
                f"gb{index}",
                f"{index + 2:064x}",
            )
            for index in range(30)
        )

        plan = build_target_augmentation_plan(records, fold=2, seed=42)
        gaussian_rows = [
            row
            for row in plan.rows
            if row.label == "a" and row.augmentation_type == "gaussian_noise"
        ]

        self.assertGreater(len(gaussian_rows), 2)
        self.assertEqual(
            len({row.augmentation_params_json for row in gaussian_rows}),
            len(gaussian_rows),
        )


class IntegrityTests(unittest.TestCase):
    def test_heldout_groups_have_priority_over_training_records(self) -> None:
        training_records = (
            original("shared_group_train", "a", "shared", "1" * 64),
            original("shared_hash_train", "a", "train-only-group", "2" * 64),
            original("kept", "a", "kept-group", "3" * 64),
        )
        heldout_records = (
            original("shared_group_test", "a", "shared", "4" * 64),
            original("shared_hash_test", "a", "test-only-group", "2" * 64),
        )
        training_audit = tuple(
            AuditRow(
                record_id=row.record_id,
                relative_path=row.relative_path,
                source_label=row.source_label,
                model_label=row.label,
                source_dataset=row.source_dataset,
                sha256=row.sha256,
                status="eligible",
                canonical_record_id=row.record_id,
                group_id=row.group_id,
                group_rule=row.group_rule,
            )
            for row in training_records
        )

        filtered, report = reserve_heldout_groups(
            CohortResolution(training_records, training_audit),
            heldout_records,
        )

        self.assertEqual([row.record_id for row in filtered.eligible], ["kept"])
        self.assertEqual(report.train_original_count_before, 3)
        self.assertEqual(report.train_original_count_after, 1)
        self.assertEqual(report.removed_train_count, 2)
        self.assertEqual(report.group_overlap_after, 0)
        self.assertEqual(report.sha256_overlap_after, 0)
        excluded = {
            row.record_id: row.exclusion_reason
            for row in filtered.audit
            if row.status == "excluded"
        }
        self.assertEqual(
            excluded,
            {
                "shared_group_train": "reserved_heldout_test_group",
                "shared_hash_train": "reserved_heldout_test_hash",
            },
        )

    def test_fold_integrity_rejects_group_overlap(self) -> None:
        train = [original("train", "a", "shared", "1" * 64)]
        validation = [original("validation", "a", "shared", "2" * 64)]

        with self.assertRaisesRegex(ProtocolViolation, "group overlap"):
            assert_fold_integrity(train, validation)

    def test_fold_integrity_rejects_hash_overlap(self) -> None:
        train = [original("train", "a", "g1", "1" * 64)]
        validation = [original("validation", "a", "g2", "1" * 64)]

        with self.assertRaisesRegex(ProtocolViolation, "hash overlap"):
            assert_fold_integrity(train, validation)

    def test_oof_coverage_requires_each_expected_record_exactly_once(self) -> None:
        with self.assertRaisesRegex(ProtocolViolation, "duplicate"):
            assert_exact_oof_coverage(["r1", "r2"], ["r1", "r1", "r2"])

        report = assert_exact_oof_coverage(["r1", "r2"], ["r2", "r1"])
        self.assertEqual(report.expected_count, 2)
        self.assertEqual(report.predicted_count, 2)
        self.assertEqual(report.duplicate_count, 0)
        self.assertEqual(report.missing_count, 0)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is unavailable")
class GroupedFoldTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_keeps_groups_isolated(self) -> None:
        records = []
        for label in ("a", "b"):
            for group_number in range(5):
                records.append(
                    original(
                        f"{label}{group_number}",
                        label,
                        f"{label}_group_{group_number}",
                        f"{group_number + (0 if label == 'a' else 5):064x}",
                    )
                )

        first = assign_grouped_folds(records, n_folds=5, seed=42)
        second = assign_grouped_folds(records, n_folds=5, seed=42)

        self.assertEqual(first.assignments, second.assignments)
        self.assertEqual(first.splitter_name, "StratifiedGroupKFold")
        self.assertEqual(
            {assignment.validation_fold for assignment in first.assignments},
            {1, 2, 3, 4, 5},
        )
        for fold in range(1, 6):
            validation = [
                assignment.record
                for assignment in first.assignments
                if assignment.validation_fold == fold
            ]
            train = [
                assignment.record
                for assignment in first.assignments
                if assignment.validation_fold != fold
            ]
            assert_fold_integrity(train, validation)
            self.assertEqual({record.label for record in validation}, {"a", "b"})

    def test_no_reliable_group_requires_explicit_opt_in(self) -> None:
        records = [
            original(f"a{i}", "a", f"a{i}", f"{i:064x}") for i in range(5)
        ] + [
            original(f"b{i}", "b", f"b{i}", f"{i + 5:064x}") for i in range(5)
        ]

        result = assign_grouped_folds(
            records,
            n_folds=5,
            seed=42,
            reliable_groups=False,
            no_group_evidence="fixture contains independently acquired clips only",
        )

        self.assertEqual(result.splitter_name, "StratifiedKFold")
        self.assertIn("clip-level", result.limitation)


class FoldManifestTests(unittest.TestCase):
    def test_validation_rejects_one_group_assigned_to_multiple_folds(self) -> None:
        assignments = [
            FoldAssignment(
                record=original("r1", "a", "shared", "1" * 64),
                validation_fold=1,
                splitter_name="fixture",
                split_seed=42,
            ),
            FoldAssignment(
                record=original("r2", "a", "shared", "2" * 64),
                validation_fold=2,
                splitter_name="fixture",
                split_seed=42,
            ),
        ]

        with self.assertRaisesRegex(ProtocolViolation, "multiple validation folds"):
            validate_fold_assignments(assignments, n_folds=2)

    def test_fold_manifest_writer_is_write_once_and_complete(self) -> None:
        assignment = FoldAssignment(
            record=original("r1", "a", "g1", "1" * 64),
            validation_fold=1,
            splitter_name="StratifiedGroupKFold",
            split_seed=42,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fold_assignments.csv"
            write_fold_assignments_csv(output, [assignment])

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["record_id"], "r1")
            self.assertEqual(rows[0]["validation_fold"], "1")
            self.assertEqual(rows[0]["splitter_name"], "StratifiedGroupKFold")

            with self.assertRaisesRegex(FileExistsError, "immutable artefact"):
                write_fold_assignments_csv(output, [assignment])

    def test_json_writer_refuses_to_replace_an_existing_artefact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.json"
            write_json_atomic(output, {"seed": 42})

            with self.assertRaisesRegex(FileExistsError, "immutable artefact"):
                write_json_atomic(output, {"seed": 7})


if __name__ == "__main__":
    unittest.main()
