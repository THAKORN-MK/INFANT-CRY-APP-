from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class ExperimentFoldDataTests(unittest.TestCase):
    run_id = "20260821T164332Z_490383ff"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train_root = self.root / "data_set_dbl_split" / "train"
        self.test_root = self.root / "data_set_dbl_split" / "test"
        self.train_root.mkdir(parents=True)
        self.test_root.mkdir(parents=True)
        self.stage_roots = {
            "stage1": self.root / "Models_dbl" / "binary" / "runs" / self.run_id,
            "stage2": self.root / "Models_dbl" / "Main" / "runs" / self.run_id,
        }
        self.rows = self._make_records()
        for stage, run_dir in self.stage_roots.items():
            self._write_reference_stage(stage, run_dir, status="complete")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_records(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for fold in range(1, 6):
            relative = Path("class_a") / f"clip_{fold}.wav"
            path = self.train_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"audio-{fold}".encode("utf-8")
            path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            rows.append(
                {
                    "record_id": f"rec_{fold}",
                    "filepath": str(path.resolve()),
                    "relative_path": relative.as_posix(),
                    "label": "class_a",
                    "validation_fold": str(fold),
                    "source_dataset": "fixture",
                    "group_id": f"group_{fold}",
                    "group_rule": "fixture_group",
                    "sha256": digest,
                    "splitter_name": "StratifiedGroupKFold",
                    "split_seed": "42",
                }
            )
        return rows

    def _write_reference_stage(self, stage: str, run_dir: Path, *, status: str) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = run_dir / "fold_assignments.csv"
        with assignment_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (run_dir / "oof_predictions.csv").write_text(
            "record_id,true_label,predicted_label\n",
            encoding="utf-8",
        )
        (run_dir / "oof_metrics.json").write_text(
            json.dumps({"macro_f1": 0.5}),
            encoding="utf-8",
        )
        (run_dir / "verification.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "run_id": self.run_id,
                    "stage": stage,
                    "status": status,
                    "folds_completed": 5,
                    "artefact_sha256": {name: {"path": str(run_dir / filename), "sha256": hashlib.sha256((run_dir / filename).read_bytes()).hexdigest()} for name, filename in (("fold_assignments", "fold_assignments.csv"), ("oof_predictions", "oof_predictions.csv"), ("oof_metrics", "oof_metrics.json"))},
                }
            ),
            encoding="utf-8",
        )

    def test_reference_requires_both_complete_stages_with_same_id(self) -> None:
        from cryinsight.experiments.fold_data import load_reference_pipeline

        reference = load_reference_pipeline(self.root, self.run_id)

        self.assertEqual(reference.pipeline_run_id, self.run_id)
        self.assertEqual(reference.stage1.fold_count, 5)
        self.assertEqual(reference.stage2.fold_count, 5)
        self.assertEqual(
            set(reference.stage2.hashes),
            {
                "verification",
                "fold_assignments_file",
                "fold_assignments_contract",
                "oof_predictions",
                "oof_metrics",
            },
        )

    def test_reference_rejects_evidence_changed_after_original_verification(self):
        from cryinsight.experiments.fold_data import load_reference_pipeline
        from cryinsight.experiments.registry import ExperimentProtocolError
        (self.stage_roots['stage2'] / 'oof_metrics.json').write_text('{"macro_f1": 1.0}')
        with self.assertRaisesRegex(ExperimentProtocolError, 'hash'):
            load_reference_pipeline(self.root, self.run_id)

    def test_reference_rejects_incomplete_verification(self) -> None:
        from cryinsight.experiments.fold_data import load_reference_pipeline
        from cryinsight.experiments.registry import ExperimentProtocolError

        self._write_reference_stage(
            "stage2",
            self.stage_roots["stage2"],
            status="incomplete",
        )

        with self.assertRaisesRegex(ExperimentProtocolError, "complete"):
            load_reference_pipeline(self.root, self.run_id)

    def test_reference_rejects_verification_run_id_mismatch(self) -> None:
        from cryinsight.experiments.fold_data import load_reference_pipeline
        from cryinsight.experiments.registry import ExperimentProtocolError

        verification = self.stage_roots["stage1"] / "verification.json"
        payload = json.loads(verification.read_text(encoding="utf-8"))
        payload["run_id"] = "different"
        verification.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ExperimentProtocolError, "run ID"):
            load_reference_pipeline(self.root, self.run_id)

    def test_test_directory_is_rejected_after_path_resolution(self) -> None:
        from cryinsight.experiments.fold_data import reject_test_path
        from cryinsight.experiments.registry import ExperimentProtocolError

        with self.assertRaisesRegex(ExperimentProtocolError, "Test dataset"):
            reject_test_path(self.test_root / "class_a" / "clip.wav", self.root)

    def test_records_are_reconstructed_and_each_fold_is_isolated(self) -> None:
        from cryinsight.experiments.fold_data import (
            build_fold_dataset,
            load_reference_pipeline,
            load_reference_records,
        )

        reference = load_reference_pipeline(self.root, self.run_id)
        frozen = load_reference_records(reference.stage2, self.train_root)
        dataset = build_fold_dataset(
            frozen.records,
            frozen.validation_folds,
            fold=1,
        )

        self.assertEqual(len(frozen.records), 5)
        self.assertEqual(
            tuple(row.record_id for row in dataset.validation_records),
            ("rec_1",),
        )
        self.assertEqual(len(dataset.train_records), 4)
        self.assertTrue(
            {row.group_id for row in dataset.train_records}.isdisjoint(
                {row.group_id for row in dataset.validation_records}
            )
        )

    def test_record_hash_mismatch_is_rejected(self) -> None:
        from cryinsight.experiments.fold_data import (
            load_reference_pipeline,
            load_reference_records,
        )
        from cryinsight.experiments.registry import ExperimentProtocolError

        reference = load_reference_pipeline(self.root, self.run_id)
        (self.train_root / "class_a" / "clip_1.wav").write_bytes(b"changed")

        with self.assertRaisesRegex(ExperimentProtocolError, "SHA-256"):
            load_reference_records(reference.stage2, self.train_root)


if __name__ == "__main__":
    unittest.main()
