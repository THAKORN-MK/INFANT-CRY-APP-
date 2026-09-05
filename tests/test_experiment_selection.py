from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cryinsight.experiments.contracts import FoldResult
from cryinsight.experiments.selection import (
    ExperimentSelectionError,
    ExperimentVerificationError,
    aggregate_candidate_seed,
    aggregate_repeated_seeds,
    promotion_decision,
    rank_confirmation_candidates,
    rank_screening_candidates,
    verify_probabilities,
)
from cryinsight.training.protocol import OriginalRecord


def _record(index: int) -> OriginalRecord:
    label = "a" if index % 2 else "b"
    return OriginalRecord(
        record_id=f"record-{index}",
        filepath=Path(f"/dev/audio-{index}.wav"),
        relative_path=f"audio-{index}.wav",
        label=label,
        source_label=label,
        source_dataset="fixture",
        group_id=f"group-{index}",
        group_rule="fixture",
        sha256=f"{index:064x}",
    )


class ProbabilityVerificationTests(unittest.TestCase):
    def test_small_sum_error_is_normalized_in_float64_and_reported(self) -> None:
        result = verify_probabilities(
            np.asarray([[0.4, 0.600009], [0.25, 0.75]], dtype=np.float32),
            tolerance=1e-5,
        )
        self.assertEqual(result.probabilities.dtype, np.float64)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
        self.assertGreater(result.max_sum_deviation, 0.0)
        self.assertLessEqual(result.max_sum_deviation, 1e-5)

    def test_large_sum_error_fails_closed(self) -> None:
        with self.assertRaisesRegex(ExperimentVerificationError, "sum to one"):
            verify_probabilities(
                np.asarray([[0.4, 0.60002]], dtype=np.float64),
                tolerance=1e-5,
            )


class CandidateSeedAggregationTests(unittest.TestCase):
    def test_classical_fold_counts_use_untruncated_mean(self):
        records = tuple(_record(index) for index in range(1, 6))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = self._fold_results(root, records)
            for result, count in zip(results, (10, 10, 10, 10, 13)):
                result.manifest_path.write_text(json.dumps({"adapter": "classical", "parameter_count": count}))
            summary = aggregate_candidate_seed(records, results, label_order=("a", "b"), output_dir=root / "summary", bootstrap_iterations=0)
            self.assertEqual(summary["parameter_count"], 10.6)
            self.assertEqual(summary["parameter_count_min"], 10)
            self.assertEqual(summary["parameter_count_max"], 13)
            self.assertEqual(summary["fold_parameter_counts"], {"1": 10, "2": 10, "3": 10, "4": 10, "5": 13})

    def test_neural_count_mismatch_fails_before_publication(self):
        records = tuple(_record(index) for index in range(1, 6))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = self._fold_results(root, records)
            for result in results:
                result.manifest_path.write_text(json.dumps({"adapter": "neural", "parameter_count": result.fold}))
            with self.assertRaises(ExperimentVerificationError):
                aggregate_candidate_seed(records, results, label_order=("a", "b"), output_dir=root / "summary", bootstrap_iterations=0)
            self.assertFalse((root / "summary/oof_predictions.csv").exists())

    def _fold_results(
        self,
        root: Path,
        records: tuple[OriginalRecord, ...],
    ) -> tuple[FoldResult, ...]:
        rows = []
        for fold, record in enumerate(records, start=1):
            model = root / f"fold_{fold}.keras"
            manifest = root / f"fold_{fold}_manifest.json"
            model.write_bytes(b"model")
            manifest.write_text("{}", encoding="utf-8")
            probabilities = (
                np.asarray([[0.8, 0.2]])
                if record.label == "a"
                else np.asarray([[0.2, 0.8]])
            )
            rows.append(
                FoldResult(
                    candidate_id="candidate",
                    seed=42,
                    fold=fold,
                    validation_record_ids=(record.record_id,),
                    true_labels=(record.label,),
                    probabilities=probabilities,
                    model_path=model,
                    manifest_path=manifest,
                )
            )
        return tuple(rows)

    def test_duplicate_oof_record_fails(self) -> None:
        records = tuple(_record(index) for index in range(1, 6))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = list(self._fold_results(root, records))
            duplicate = results[-1]
            results[-1] = FoldResult(
                candidate_id=duplicate.candidate_id,
                seed=duplicate.seed,
                fold=duplicate.fold,
                validation_record_ids=(records[0].record_id,),
                true_labels=(records[0].label,),
                probabilities=duplicate.probabilities,
                model_path=duplicate.model_path,
                manifest_path=duplicate.manifest_path,
            )
            with self.assertRaisesRegex(ExperimentVerificationError, "duplicate OOF"):
                aggregate_candidate_seed(
                    records,
                    results,
                    label_order=("a", "b"),
                    output_dir=root / "summary",
                    bootstrap_iterations=0,
                )

    def test_verified_five_fold_seed_writes_immutable_oof_outputs(self) -> None:
        records = tuple(_record(index) for index in range(1, 6))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = aggregate_candidate_seed(
                records,
                self._fold_results(root, records),
                label_order=("a", "b"),
                output_dir=root / "summary",
                bootstrap_iterations=0,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["folds"], [1, 2, 3, 4, 5])
            for name in (
                "oof_predictions.csv",
                "oof_metrics.json",
                "seed_summary.json",
                "verification.json",
            ):
                self.assertTrue((root / "summary" / name).is_file(), name)
            verification = json.loads(
                (root / "summary" / "verification.json").read_text(encoding="utf-8")
            )
            self.assertFalse(verification["heldout_test_used_for_selection"])
            self.assertEqual(verification["oof_support"], 5)


class DeterministicSelectionTests(unittest.TestCase):
    def test_confirmation_recall_difference_is_not_a_screening_tolerance_tie(self):
        rows = [{"candidate_id": name, "seeds": [42, 123, 2026],
                 "mean_oof_macro_f1": .9, "mean_minimum_class_recall": recall,
                 "std_oof_macro_f1": deviation, "parameter_count": 100,
                 "verification_status": "complete"}
                for name, recall, deviation in (("stable_lower_recall", .800, .0001), ("higher_recall", .801, .01))]
        self.assertEqual(rank_confirmation_candidates(rows)[0]['candidate_id'], 'higher_recall')
    def test_fractional_classical_parameter_mean_is_not_truncated_for_ties(self):
        rows = [self._screening_row(name, macro_f1=.9, min_recall=.8, params=count) for name, count in (("a", 10.9), ("z", 10.1))]
        self.assertEqual(rank_screening_candidates(rows)[0]["candidate_id"], "z")

    def test_classical_seed_counts_may_vary(self):
        rows = [{**self._screening_row("svm", macro_f1=.9, min_recall=.8, params=count), "seed": seed, "adapter": "classical"} for seed, count in zip((42, 123, 2026), (10.2, 10.4, 10.6))]
        summary = aggregate_repeated_seeds(rows)[0]
        self.assertAlmostEqual(summary["parameter_count"], 10.4)
        self.assertEqual(summary["parameter_count_min"], 10.2)
    @staticmethod
    def _screening_row(
        candidate_id: str,
        *,
        macro_f1: float,
        min_recall: float,
        params: int,
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "oof_macro_f1": macro_f1,
            "oof_balanced_accuracy": macro_f1,
            "oof_accuracy": macro_f1,
            "minimum_class_recall": min_recall,
            "parameter_count": params,
            "verification_status": "complete",
        }

    def test_screening_tie_uses_minimum_class_recall_then_parameters(self) -> None:
        ranked = rank_screening_candidates(
            [
                self._screening_row(
                    "large", macro_f1=0.900, min_recall=0.80, params=500_000
                ),
                self._screening_row(
                    "small", macro_f1=0.897, min_recall=0.85, params=100_000
                ),
            ]
        )
        self.assertEqual(ranked[0]["candidate_id"], "small")
        self.assertEqual([row["rank"] for row in ranked], [1, 2])

    def test_confirmation_requires_exactly_three_seeds(self) -> None:
        rows = [
            {
                **self._screening_row(
                    "candidate", macro_f1=0.9, min_recall=0.8, params=100
                ),
                "seed": seed,
            }
            for seed in (42, 123)
        ]
        with self.assertRaisesRegex(ExperimentSelectionError, "42, 123, 2026"):
            aggregate_repeated_seeds(rows)

    def test_confirmation_aggregates_and_ranks_three_seeds(self) -> None:
        rows = []
        for candidate_id, offset, params in (
            ("stable", 0.0, 100),
            ("variable", 0.002, 200),
        ):
            for seed, delta in zip((42, 123, 2026), (-0.001, 0.0, 0.001)):
                rows.append(
                    {
                        **self._screening_row(
                            candidate_id,
                            macro_f1=0.9 + offset + delta,
                            min_recall=0.84 if candidate_id == "stable" else 0.82,
                            params=params,
                        ),
                        "seed": seed,
                    }
                )
        summaries = aggregate_repeated_seeds(rows)
        ranked = rank_confirmation_candidates(summaries)
        self.assertEqual(ranked[0]["candidate_id"], "stable")
        self.assertEqual(ranked[0]["seeds"], [42, 123, 2026])

    def test_promotion_requires_one_point_macro_f1_gain(self) -> None:
        decision = promotion_decision(
            {
                "candidate_id": "winner",
                "seeds": [42, 123, 2026],
                "mean_oof_macro_f1": 0.8977,
                "mean_oof_balanced_accuracy": 0.89,
                "mean_minimum_class_recall": 0.80,
                "verification_status": "complete",
            },
            {
                "oof_macro_f1": 0.8878,
                "oof_balanced_accuracy": 0.8872,
                "minimum_class_recall": 0.79,
            },
        )
        self.assertEqual(decision["status"], "no_promotion_recommended")
        self.assertFalse(decision["checks"]["macro_f1_gain_at_least_0_01"])


if __name__ == "__main__":
    unittest.main()
