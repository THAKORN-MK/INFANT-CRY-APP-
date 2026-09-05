from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cryinsight.experiments.reporting import (
    write_experiment_report,
    write_leaderboard_csv,
    write_leaderboard_markdown,
)


class ExperimentReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "comparison.md"
        self.valid_rows = [
            {
                "rank": 1,
                "candidate_id": "candidate_a",
                "wave": "A",
                "seeds": [42],
                "oof_macro_f1": 0.91,
                "oof_balanced_accuracy": 0.90,
                "oof_accuracy": 0.92,
                "minimum_class_recall": 0.81,
                "parameter_count": 100,
                "verification_status": "complete",
            }
        ]
        self.failures = [
            {
                "candidate_id": "broken_candidate",
                "reason": "ResourceError",
            }
        ]
        self.payload = {
            "experiment_run_id": "pipeline__exp_fixture",
            "wave": "A",
            "reference": {
                "oof_macro_f1": 0.8878,
                "oof_balanced_accuracy": 0.8872,
                "minimum_class_recall": 0.79,
                "final_test_accuracy": 0.92409,
                "final_test_percent": "92.409",
            },
            "ranked_candidates": self.valid_rows,
            "per_class_oof": {"candidate_a": {"hungry": 0.81}},
            "exclusions": self.failures,
            "promotion": {
                "status": "no_promotion_recommended",
                "checks": {"three_seed_confirmation": False},
            },
            "limitations": ["Internal grouped validation only."],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_report_excludes_final_test_metrics_and_states_oof_scope(self) -> None:
        write_experiment_report(self.output, self.payload)
        text = self.output.read_text(encoding="utf-8")
        self.assertIn("grouped OOF only", text)
        self.assertNotIn("final_test_accuracy", text)
        self.assertNotIn("92.409", text)

    def test_failed_candidate_is_listed_but_not_ranked(self) -> None:
        write_leaderboard_markdown(self.output, self.valid_rows, self.failures)
        text = self.output.read_text(encoding="utf-8")
        self.assertIn("Excluded/failed candidates", text)
        self.assertIn("broken_candidate", text)
        self.assertNotIn("| 1 | broken_candidate |", text)

    def test_report_write_refuses_overwrite(self) -> None:
        self.output.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_experiment_report(self.output, self.payload)

    def test_csv_uses_frozen_column_order(self) -> None:
        output = self.root / "leaderboard.csv"
        write_leaderboard_csv(output, self.valid_rows)
        header = output.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(
            header,
            "rank,candidate_id,wave,seeds,oof_macro_f1_mean,oof_macro_f1_std,"
            "oof_balanced_accuracy_mean,oof_accuracy_mean,"
            "minimum_class_recall_mean,parameter_count,verification_status",
        )


if __name__ == "__main__":
    unittest.main()
