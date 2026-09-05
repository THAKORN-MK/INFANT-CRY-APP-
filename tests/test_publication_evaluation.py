from __future__ import annotations

import unittest

import numpy as np


class PublicationEvaluationTests(unittest.TestCase):
    def test_probability_rows_are_renormalized_within_tolerance(self) -> None:
        from cryinsight.training.artefacts import normalize_probability_rows

        values = np.asarray([[0.2, 0.79999994], [0.1, 0.2, 0.6999999]], dtype=object)
        first = normalize_probability_rows(np.asarray([values[0]], dtype=np.float64))
        second = normalize_probability_rows(np.asarray([values[1]], dtype=np.float64))
        np.testing.assert_allclose(first.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(second.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)

    def test_invalid_probability_rows_are_rejected(self) -> None:
        from cryinsight.training.artefacts import normalize_probability_rows

        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_probability_rows(np.asarray([[1.1, -0.1]]))

    def test_curve_tables_include_roc_and_pr_for_each_class(self) -> None:
        from cryinsight.evaluation.curves import compute_roc_pr_tables

        labels = ("a", "b")
        y_true = ("a", "a", "b", "b")
        scores = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        tables = compute_roc_pr_tables(y_true, scores, labels)
        self.assertEqual(set(tables), {"a", "b"})
        self.assertIn("roc_auc", tables["a"])
        self.assertIn("average_precision", tables["b"])

    def test_cascade_counts_stage1_false_reject_as_end_to_end_error(self) -> None:
        from cryinsight.evaluation.cascade import aggregate_cascade_rows

        stage1 = [
            {"record_id": "r1", "label": "baby", "predicted_label": "not_baby"},
            {"record_id": "r2", "label": "baby", "predicted_label": "baby"},
            {"record_id": "n1", "label": "not_baby", "predicted_label": "not_baby"},
        ]
        stage2 = [
            {"record_id": "r1", "label": "hungry", "predicted_label": "hungry"},
            {"record_id": "r2", "label": "tired", "predicted_label": "tired"},
        ]
        result = aggregate_cascade_rows(stage1, stage2)
        self.assertEqual(result["support"], 3)
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["stage1_false_rejects"], 1)


if __name__ == "__main__":
    unittest.main()
