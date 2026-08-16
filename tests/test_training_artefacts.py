from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from cryinsight.training import artefacts as artefact_module
from cryinsight.training.artefacts import (
    ArtefactError,
    NormalizerStats,
    OofPrediction,
    aggregate_oof_metrics,
    create_run_directory,
    load_normalizer,
    save_normalizer,
    sha256_file,
    verify_file_hash,
    write_incomplete_run_verification,
)
from cryinsight.training.protocol import OriginalRecord, ProtocolViolation


def record(record_id: str, label: str, group_id: str) -> OriginalRecord:
    return OriginalRecord(
        record_id=record_id,
        filepath=Path(f"D:/data/{record_id}.wav"),
        relative_path=f"{label}/{record_id}.wav",
        label=label,
        source_label=label,
        source_dataset="fixture",
        group_id=group_id,
        group_rule="fixture",
        sha256=(record_id.encode("utf-8").hex() + "0" * 64)[:64],
    )


def prediction(
    source: OriginalRecord,
    *,
    fold: int | str,
    predicted_label: str,
    scores: tuple[float, ...],
    sample_kind: str = "original",
) -> OofPrediction:
    return OofPrediction(
        record_id=source.record_id,
        filepath=str(source.filepath),
        label=source.label,
        group_id=source.group_id,
        fold=fold,
        predicted_label=predicted_label,
        scores=scores,
        sample_kind=sample_kind,
        model_path=f"fold_{fold}/model.keras",
        normalizer_path=f"fold_{fold}/norm.npy",
        run_id="run_fixture",
    )


class ImmutableRunTests(unittest.TestCase):
    def test_run_directory_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage_root = Path(directory)
            first = create_run_directory(stage_root, run_id="fixed_run")

            self.assertEqual(first.name, "fixed_run")
            with self.assertRaisesRegex(FileExistsError, "immutable run"):
                create_run_directory(stage_root, run_id="fixed_run")

    def test_incomplete_verification_is_write_once_and_counts_completed_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_run_directory(directory, run_id="failed_run")
            completed = run_dir / "fold_1"
            completed.mkdir()
            (completed / "fold_manifest.json").write_text("{}", encoding="utf-8")

            payload = write_incomplete_run_verification(
                run_dir,
                stage="stage1_binary_baby_gate",
                error=RuntimeError("fixture failure"),
            )

            saved = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, payload)
            self.assertEqual(saved["status"], "incomplete")
            self.assertTrue(saved["training_started"])
            self.assertEqual(saved["folds_completed"], 1)
            self.assertEqual(saved["error"]["type"], "RuntimeError")
            self.assertEqual(saved["error"]["message"], "fixture failure")
            with self.assertRaisesRegex(FileExistsError, "immutable artefact"):
                write_incomplete_run_verification(
                    run_dir,
                    stage="stage1_binary_baby_gate",
                    error=RuntimeError("second failure"),
                )


class FinalRefitSelectionTests(unittest.TestCase):
    def test_final_epoch_is_median_of_all_five_fold_best_epochs(self) -> None:
        self.assertTrue(hasattr(artefact_module, "select_final_refit_epoch"))

        selection = artefact_module.select_final_refit_epoch(
            [
                {"fold": 4, "best_epoch": 39},
                {"fold": 2, "best_epoch": 41},
                {"fold": 5, "best_epoch": 35},
                {"fold": 1, "best_epoch": 32},
                {"fold": 3, "best_epoch": 36},
            ],
            expected_folds=5,
        )

        self.assertEqual(selection["final_epoch"], 36)
        self.assertEqual(selection["fold_best_epochs"], [32, 41, 36, 39, 35])
        self.assertEqual(selection["selection_rule"], "median_fold_best_epoch")

    def test_final_epoch_rejects_missing_or_duplicate_folds(self) -> None:
        self.assertTrue(hasattr(artefact_module, "select_final_refit_epoch"))
        selector = artefact_module.select_final_refit_epoch

        with self.assertRaisesRegex(ArtefactError, "exactly folds"):
            selector(
                [
                    {"fold": 1, "best_epoch": 10},
                    {"fold": 2, "best_epoch": 20},
                    {"fold": 2, "best_epoch": 30},
                ],
                expected_folds=3,
            )


class NormalizerTests(unittest.TestCase):
    def test_normalizer_round_trip_requires_matching_run_and_fold(self) -> None:
        stats = NormalizerStats(
            mean=1.25,
            std=2.5,
            epsilon=1e-8,
            axis="global_scalar",
            feature_shape=(196, 128, 1),
            dtype="float32",
            fit_sample_count=20,
            run_id="run_fixture",
            fold=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "norm_stats_main_dbl.npy"
            save_normalizer(path, stats)

            loaded = load_normalizer(path, expected_run_id="run_fixture", expected_fold=3)
            self.assertEqual(loaded, stats)

            with self.assertRaisesRegex(ArtefactError, "run_id"):
                load_normalizer(path, expected_run_id="another_run", expected_fold=3)
            with self.assertRaisesRegex(ArtefactError, "fold"):
                load_normalizer(path, expected_run_id="run_fixture", expected_fold=2)

    def test_hash_verification_detects_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artefact.bin"
            path.write_bytes(b"first")
            expected = sha256_file(path)
            verify_file_hash(path, expected)

            path.write_bytes(b"second")
            with self.assertRaisesRegex(ArtefactError, "SHA-256 mismatch"):
                verify_file_hash(path, expected)

    def test_final_refit_normalizer_has_explicit_non_fold_identity(self) -> None:
        stats = NormalizerStats(
            mean=0.5,
            std=1.5,
            epsilon=1e-8,
            axis="global_scalar",
            feature_shape=(120, 128, 1),
            dtype="float32",
            fit_sample_count=100,
            run_id="run_fixture",
            fold="final_refit",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "norm_stats_binary_dbl.npy"
            save_normalizer(path, stats)

            loaded = load_normalizer(
                path,
                expected_run_id="run_fixture",
                expected_fold="final_refit",
            )

            self.assertEqual(loaded.fold, "final_refit")


class OofContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [record("r1", "a", "g1"), record("r2", "b", "g2")]

    def test_augmented_oof_row_is_rejected_before_metric_calculation(self) -> None:
        rows = [
            prediction(self.records[0], fold=1, predicted_label="a", scores=(0.9, 0.1)),
            prediction(
                self.records[1],
                fold=2,
                predicted_label="b",
                scores=(0.1, 0.9),
                sample_kind="augmented",
            ),
        ]

        with self.assertRaisesRegex(ProtocolViolation, "original"):
            aggregate_oof_metrics(self.records, rows, label_order=("a", "b"))

    def test_duplicate_oof_row_is_rejected_before_metric_calculation(self) -> None:
        rows = [
            prediction(self.records[0], fold=1, predicted_label="a", scores=(0.9, 0.1)),
            prediction(self.records[0], fold=2, predicted_label="a", scores=(0.8, 0.2)),
            prediction(self.records[1], fold=2, predicted_label="b", scores=(0.1, 0.9)),
        ]

        with self.assertRaisesRegex(ProtocolViolation, "duplicate"):
            aggregate_oof_metrics(self.records, rows, label_order=("a", "b"))


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is unavailable")
class MetricCalculationTests(unittest.TestCase):
    def test_single_class_bootstrap_samples_do_not_emit_metric_warnings(self) -> None:
        records = [
            record("a1", "a", "g1"),
            record("a2", "a", "g2"),
            record("b1", "b", "g3"),
            record("b2", "b", "g4"),
        ]
        rows = [
            prediction(records[0], fold=1, predicted_label="a", scores=(0.9, 0.1)),
            prediction(records[1], fold=2, predicted_label="a", scores=(0.8, 0.2)),
            prediction(records[2], fold=1, predicted_label="b", scores=(0.2, 0.8)),
            prediction(records[3], fold=2, predicted_label="b", scores=(0.1, 0.9)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            aggregate_oof_metrics(
                records,
                rows,
                label_order=("a", "b"),
                bootstrap_iterations=20,
                bootstrap_seed=42,
            )

    def test_heldout_metrics_are_not_reported_as_external_validation(self) -> None:
        self.assertTrue(hasattr(artefact_module, "aggregate_heldout_metrics"))
        records = [record("a1", "a", "g1"), record("b1", "b", "g2")]
        rows = [
            prediction(records[0], fold=1, predicted_label="a", scores=(0.9, 0.1)),
            prediction(records[1], fold=1, predicted_label="b", scores=(0.1, 0.9)),
        ]

        result = artefact_module.aggregate_heldout_metrics(
            records,
            rows,
            label_order=("a", "b"),
            bootstrap_iterations=0,
        )

        self.assertEqual(
            result.pooled_metrics["evaluation_scope"],
            "locked_internal_heldout_test",
        )
        self.assertFalse(
            result.pooled_metrics["independent_external_validation_performed"]
        )

    def test_perfect_binary_predictions_have_unit_accuracy(self) -> None:
        records = [
            record("a1", "a", "g1"),
            record("a2", "a", "g2"),
            record("b1", "b", "g3"),
            record("b2", "b", "g4"),
        ]
        rows = [
            prediction(records[0], fold=1, predicted_label="a", scores=(0.9, 0.1)),
            prediction(records[1], fold=2, predicted_label="a", scores=(0.8, 0.2)),
            prediction(records[2], fold=1, predicted_label="b", scores=(0.2, 0.8)),
            prediction(records[3], fold=2, predicted_label="b", scores=(0.1, 0.9)),
        ]

        result = aggregate_oof_metrics(
            records,
            rows,
            label_order=("a", "b"),
            bootstrap_iterations=20,
            bootstrap_seed=42,
        )

        self.assertEqual(result.pooled_metrics["accuracy"], 1.0)
        self.assertEqual(result.pooled_metrics["sensitivity"], 1.0)
        self.assertEqual(result.pooled_metrics["specificity"], 1.0)
        self.assertIn("log_loss", result.pooled_metrics)
        self.assertIn("brier_score", result.pooled_metrics)
        self.assertIn("expected_calibration_error", result.pooled_metrics)
        np.testing.assert_array_equal(result.confusion_matrix, np.eye(2, dtype=int) * 2)


if __name__ == "__main__":
    unittest.main()
