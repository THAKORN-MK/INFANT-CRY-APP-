from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cryinsight.training.protocol import OriginalRecord


LABELS = ("belly_pain", "burping", "discomfort", "hungry", "tired")


def fixture_feature(record: OriginalRecord, candidate, request) -> np.ndarray:
    del candidate, request
    label_index = LABELS.index(record.label)
    sequence = int(record.record_id.rsplit("_", 1)[-1])
    return np.array(
        [float(label_index * 10 + sequence), float(label_index), float(sequence)],
        dtype=np.float32,
    )


class ClassicalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        from cryinsight.experiments.classical import ClassicalAdapter

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train_records = tuple(
            self._record(label, sequence, "train")
            for label in LABELS
            for sequence in range(4 if label == "belly_pain" else 3)
        )
        self.validation_records = tuple(
            self._record(label, 9, "validation") for label in LABELS
        )
        self.adapter = ClassicalAdapter(feature_builder=fixture_feature)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _record(self, label: str, sequence: int, partition: str) -> OriginalRecord:
        return OriginalRecord(
            record_id=f"{partition}_{label}_{sequence}",
            filepath=self.root / f"{partition}_{label}_{sequence}.wav",
            relative_path=f"{label}/{partition}_{sequence}.wav",
            label=label,
            source_label=label,
            source_dataset="fixture",
            group_id=f"group_{partition}_{label}_{sequence}",
            group_rule="fixture",
            sha256=f"{sequence:x}" * 64,
        )

    def _request(self, candidate_id: str):
        from cryinsight.experiments.contracts import FoldRequest
        from cryinsight.experiments.registry import experiment_registry

        return FoldRequest(
            experiment_run_id="pipeline__exp_fixture",
            pipeline_run_id="pipeline",
            candidate=experiment_registry()[candidate_id],
            seed=42,
            fold=1,
            train_records=self.train_records,
            validation_records=self.validation_records,
            label_order=LABELS,
            output_dir=self.root / candidate_id,
            runtime={
                "config_sha256": "a" * 64,
                "assignment_sha256": "b" * 64,
            },
        )

    def test_majority_returns_probabilities_in_declared_label_order(self) -> None:
        result = self.adapter.fit_predict_fold(self._request("stage2_majority"))

        self.assertEqual(result.probabilities.shape, (len(self.validation_records), 5))
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
        self.assertEqual(result.validation_record_ids, tuple(row.record_id for row in self.validation_records))

    def test_svm_scaler_is_fitted_only_on_training_features(self) -> None:
        result = self.adapter.fit_predict_fold(self._request("stage2_mfcc_svm"))
        metadata = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            metadata["normalizer_fit_record_ids"],
            [row.record_id for row in self.train_records],
        )
        self.assertTrue(
            {row.record_id for row in self.validation_records}.isdisjoint(
                metadata["normalizer_fit_record_ids"]
            )
        )
        self.assertGreater(metadata["parameter_count"], 0)

    def test_serialized_estimator_round_trip_preserves_probabilities(self) -> None:
        request = self._request("stage2_mfcc_svm")
        result = self.adapter.fit_predict_fold(request)
        loaded = joblib.load(result.model_path)
        validation_x = np.stack(
            [fixture_feature(row, request.candidate, request) for row in self.validation_records]
        )

        np.testing.assert_allclose(
            loaded.predict_proba(validation_x),
            result.probabilities,
        )

    def test_classical_adapter_does_not_import_tensorflow(self) -> None:
        tensorflow_was_loaded = "tensorflow" in sys.modules
        self.adapter.fit_predict_fold(self._request("stage2_majority"))
        if not tensorflow_was_loaded:
            self.assertNotIn("tensorflow", sys.modules)


if __name__ == "__main__":
    unittest.main()
