from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path


class ExperimentProtocolTests(unittest.TestCase):
    def test_registry_contains_required_stage_baselines_and_ablations(self) -> None:
        from cryinsight.training.experiments import experiment_registry

        registry = experiment_registry()
        required = {
            "stage1_majority",
            "stage1_mfcc_svm",
            "stage1_logmel_small_cnn",
            "stage2_majority",
            "stage2_mfcc_svm",
            "stage2_logmel_small_cnn",
            "stage2_yamnet_linear",
            "stage2_yamnet_mlp",
            "stage2_cnn_only",
            "stage2_cnn_bilstm",
            "stage2_corrected_attention",
            "stage2_feature_blocks",
            "stage2_normalization",
            "stage2_augmentation_mixup",
        }
        self.assertTrue(required.issubset(registry))

    def test_fold_hash_is_order_independent_but_assignment_sensitive(self) -> None:
        from cryinsight.training.experiments import fold_assignment_sha256

        rows = [
            {"record_id": "a", "group_id": "g1", "label": "x", "validation_fold": "1"},
            {"record_id": "b", "group_id": "g2", "label": "y", "validation_fold": "2"},
        ]
        self.assertEqual(fold_assignment_sha256(rows), fold_assignment_sha256(reversed(rows)))
        changed = [dict(row) for row in rows]
        changed[0]["validation_fold"] = "2"
        self.assertNotEqual(fold_assignment_sha256(rows), fold_assignment_sha256(changed))

    def test_load_fold_assignments_rejects_group_split_across_folds(self) -> None:
        from cryinsight.training.experiments import ExperimentProtocolError, load_fold_assignments

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("record_id", "group_id", "label", "validation_fold"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"record_id": "a", "group_id": "g", "label": "x", "validation_fold": 1},
                        {"record_id": "b", "group_id": "g", "label": "x", "validation_fold": 2},
                    ]
                )
            with self.assertRaisesRegex(ExperimentProtocolError, "group"):
                load_fold_assignments(path)

    def test_ranking_contract_forbids_heldout_test_metric(self) -> None:
        from cryinsight.training.experiments import validate_selection_metric

        self.assertEqual(validate_selection_metric("oof_macro_f1"), "oof_macro_f1")
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_selection_metric("final_test_accuracy")


if __name__ == "__main__":
    unittest.main()
