from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path


class ExperimentRunStoreTests(unittest.TestCase):
    pipeline_id = "20260821T164332Z_490383ff"

    def setUp(self) -> None:
        from cryinsight.experiments.contracts import ExperimentConfig
        from cryinsight.experiments.fold_data import ReferencePipeline, ReferenceStage

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runs = self.root / "experiment_runs"
        stage1_dir = self.root / "stage1"
        stage2_dir = self.root / "stage2"
        stage1_assignment = self._write_assignments(stage1_dir, "baby")
        stage2_assignment = self._write_assignments(stage2_dir, "hungry")
        hashes = {
            "verification": "a" * 64,
            "fold_assignments_file": "b" * 64,
            "fold_assignments_contract": "c" * 64,
            "oof_predictions": "d" * 64,
            "oof_metrics": "e" * 64,
        }
        self.reference = ReferencePipeline(
            pipeline_run_id=self.pipeline_id,
            stage1=ReferenceStage(
                stage="stage1",
                run_id=self.pipeline_id,
                project_root=self.root,
                run_dir=stage1_dir,
                verification_status="complete",
                fold_count=5,
                fold_assignment_path=stage1_assignment,
                hashes=hashes,
            ),
            stage2=ReferenceStage(
                stage="stage2",
                run_id=self.pipeline_id,
                project_root=self.root,
                run_dir=stage2_dir,
                verification_status="complete",
                fold_count=5,
                fold_assignment_path=stage2_assignment,
                hashes=hashes,
            ),
        )
        self.config = ExperimentConfig(
            schema_version="1.0",
            wave="A",
            seeds=(42,),
            selection_metric="oof_macro_f1",
            candidates=("stage2_majority",),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_assignments(run_dir: Path, label: str) -> Path:
        run_dir.mkdir(parents=True)
        path = run_dir / "fold_assignments.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "record_id",
                    "group_id",
                    "label",
                    "validation_fold",
                ),
            )
            writer.writeheader()
            for fold in range(1, 6):
                writer.writerow(
                    {
                        "record_id": f"{label}_{fold}",
                        "group_id": f"{label}_group_{fold}",
                        "label": label,
                        "validation_fold": fold,
                    }
                )
        return path

    def _create(self):
        from cryinsight.experiments.runner import ExperimentRunStore

        return ExperimentRunStore.create(
            self.runs,
            self.pipeline_id,
            self.config,
            self.reference,
            run_id=self.pipeline_id + "__exp_20260822T120000Z_a1b2c3d4",
        )

    def _complete(self, store, fold):
        from cryinsight.training.artefacts import sha256_file
        attempt = store.job_dir('stage2_majority', 42, fold) / 'attempt_1'
        attempt.mkdir(parents=True)
        model = attempt / 'model.fake'
        model.write_bytes(b'model')
        manifest = attempt / 'fold_manifest.json'
        manifest.write_text(json.dumps({'model_sha256': sha256_file(model)}))
        result = attempt / 'fold_result.json'
        result.write_text(json.dumps({'candidate_id': 'stage2_majority', 'seed': 42, 'fold': fold, 'validation_record_ids': ['r'], 'true_labels': ['a'], 'probabilities': [[1.0, 0.0]], 'model_path': str(model), 'manifest_path': str(manifest)}))
        store.mark_job_complete('stage2_majority', 42, fold, {'model': sha256_file(model), 'manifest': sha256_file(manifest), 'fold_result': sha256_file(result)}, fold_result_path='attempt_1/fold_result.json')

    def test_create_uses_parent_id_and_writes_prepared_protocol(self) -> None:
        store = self._create()

        self.assertTrue(store.run_id.startswith(self.pipeline_id + "__exp_"))
        self.assertEqual(store.state, "prepared")
        self.assertTrue((store.run_dir / "protocol.json").is_file())
        self.assertTrue((store.run_dir / "reference_run.json").is_file())
        self.assertTrue((store.run_dir / "shared_fold_assignments.csv").is_file())
        matrix = json.loads(
            (store.run_dir / "candidate_matrix.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(matrix["jobs"]), 5)

    def test_completed_fold_cannot_be_replaced(self) -> None:
        store = self._create()
        store.mark_running()
        store.mark_job_complete("stage2_majority", 42, 1, {"model": "a" * 64})

        with self.assertRaises(FileExistsError):
            store.mark_job_complete("stage2_majority", 42, 1, {"model": "b" * 64})

    def test_open_requires_matching_config_hash(self) -> None:
        from cryinsight.experiments.runner import ExperimentRunStore, ExperimentStateError

        store = self._create()

        with self.assertRaisesRegex(ExperimentStateError, "config hash"):
            ExperimentRunStore.open(
                store.run_dir,
                expected_config_hash="0" * 64,
            )

    def test_open_requires_matching_assignment_hashes(self) -> None:
        from cryinsight.experiments.runner import ExperimentRunStore, ExperimentStateError

        store = self._create()

        with self.assertRaisesRegex(ExperimentStateError, "assignment hash"):
            ExperimentRunStore.open(
                store.run_dir,
                expected_assignment_hashes={"stage1": "0" * 64},
            )

    def test_complete_run_is_immutable(self) -> None:
        from cryinsight.experiments.runner import ExperimentStateError

        store = self._create()
        store.mark_running()
        for fold in range(1, 6):
            self._complete(store, fold)
        store.finalize(status="complete")

        self.assertEqual(store.state, "complete")
        with self.assertRaisesRegex(ExperimentStateError, "complete"):
            store.mark_running()

    def test_failed_job_can_resume_without_losing_failure_evidence(self) -> None:
        store = self._create()
        store.mark_running()
        failure = store.mark_job_failed(
            "stage2_majority",
            42,
            1,
            error_type="RuntimeError",
            message="interrupted",
        )

        self.assertEqual(store.state, "failed")
        self.assertTrue(failure.is_file())
        store.mark_running()
        pending = store.pending_jobs()
        self.assertIn(1, [job.fold for job in pending])
        self.assertTrue(failure.is_file())


if __name__ == "__main__":
    unittest.main()
