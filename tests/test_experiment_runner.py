from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cryinsight.experiments.contracts import ExperimentConfig, FoldResult
from cryinsight.experiments.fold_data import (
    FrozenRecordSet,
    ReferencePipeline,
    ReferenceStage,
)
from cryinsight.experiments.registry import ExperimentProtocolError
from cryinsight.experiments.runner import (
    ExperimentPreparation,
    prepare_experiment,
    resume_experiment,
    train_experiment,
    summarize_experiment,
    ExperimentRunStore,
    ExperimentStateError,
)
from cryinsight.training.protocol import OriginalRecord, ProtocolViolation
from cryinsight.training.artefacts import sha256_file


class _FakeAdapter:
    def __init__(self, calls: list[int], *, fail_once_at: int | None = None):
        self.calls = calls
        self.fail_once_at = fail_once_at
        self.failed = False

    def fit_predict_fold(self, request):
        self.calls.append(request.fold)
        if request.fold == self.fail_once_at and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic interruption")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "model.fake"
        manifest = request.output_dir / "fold_manifest.json"
        model.write_bytes(b"model")
        manifest.write_text(json.dumps({
            "experiment_run_id": request.experiment_run_id,
            "pipeline_run_id": request.pipeline_run_id,
            "candidate_id": request.candidate.candidate_id, "seed": request.seed,
            "fold": request.fold, "adapter": request.candidate.adapter,
            "label_order": list(request.label_order), "parameter_count": 0,
            "train_record_ids": [r.record_id for r in request.train_records],
            "validation_record_ids": [r.record_id for r in request.validation_records],
            "config_sha256": request.runtime["config_sha256"],
            "assignment_sha256": request.runtime["assignment_sha256"],
            "model_sha256": sha256_file(model),
        }), encoding="utf-8")
        probabilities = []
        for record in request.validation_records:
            probabilities.append([0.9, 0.1] if record.label == "a" else [0.1, 0.9])
        return FoldResult(
            candidate_id=request.candidate.candidate_id,
            seed=request.seed,
            fold=request.fold,
            validation_record_ids=tuple(
                record.record_id for record in request.validation_records
            ),
            true_labels=tuple(record.label for record in request.validation_records),
            probabilities=np.asarray(probabilities),
            model_path=model,
            manifest_path=manifest,
        )


class SharedExperimentRunnerTests(unittest.TestCase):
    pipeline_id = "20260821T164332Z_490383ff"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = tuple(
            self._record(fold, label)
            for fold in range(1, 6)
            for label in ("a", "b")
        )
        self.validation_folds = {
            record.record_id: int(record.record_id.split("_")[1])
            for record in self.records
        }
        self.reference = self._reference()
        self.config = ExperimentConfig(
            schema_version="1.0",
            wave="A",
            seeds=(42,),
            selection_metric="oof_macro_f1",
            candidates=("stage2_majority",),
            continue_on_candidate_failure=False,
        )
        self.request = ExperimentPreparation(
            project_root=self.root,
            pipeline_run_id=self.pipeline_id,
            config=self.config,
            stage_data_roots={
                "stage1": self.root / "train",
                "stage2": self.root / "train",
            },
            runs_root=self.root / "experiment_runs",
            run_id=self.pipeline_id + "__exp_20260822T120000Z_a1b2c3d4",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _record(self, fold: int, label: str) -> OriginalRecord:
        return OriginalRecord(
            record_id=f"rec_{fold}_{label}",
            filepath=self.root / "train" / label / f"{fold}.wav",
            relative_path=f"{label}/{fold}.wav",
            label=label,
            source_label=label,
            source_dataset="fixture",
            group_id=f"group_{fold}_{label}",
            group_rule="fixture",
            sha256=(f"{fold:x}" * 64)[:64],
        )

    def _reference(self) -> ReferencePipeline:
        stages = {}
        for stage in ("stage1", "stage2"):
            run_dir = self.root / stage
            run_dir.mkdir(parents=True)
            assignment = run_dir / "fold_assignments.csv"
            with assignment.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("record_id", "group_id", "label", "validation_fold"),
                )
                writer.writeheader()
                for record in self.records:
                    writer.writerow(
                        {
                            "record_id": record.record_id,
                            "group_id": record.group_id,
                            "label": record.label,
                            "validation_fold": self.validation_folds[record.record_id],
                        }
                    )
            (run_dir / "oof_metrics.json").write_text(
                json.dumps(
                    {
                        "pooled_metrics": {
                            "macro_f1": 0.5,
                            "balanced_accuracy": 0.5,
                            "accuracy": 0.5,
                        },
                        "classification_report": {
                            "a": {"recall": 0.5},
                            "b": {"recall": 0.5},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "verification.json").write_text('{"status":"complete"}')
            (run_dir / "oof_predictions.csv").write_text('fixture')
            from cryinsight.experiments.registry import (
                fold_assignment_sha256,
                load_fold_assignments,
            )

            stages[stage] = ReferenceStage(
                stage=stage,
                run_id=self.pipeline_id,
                project_root=self.root,
                run_dir=run_dir,
                verification_status="complete",
                fold_count=5,
                fold_assignment_path=assignment,
                hashes={
                    "verification": sha256_file(run_dir / "verification.json"),
                    "fold_assignments_file": sha256_file(assignment),
                    "fold_assignments_contract": fold_assignment_sha256(
                        load_fold_assignments(assignment)
                    ),
                    "oof_predictions": sha256_file(run_dir / "oof_predictions.csv"),
                    "oof_metrics": sha256_file(run_dir / "oof_metrics.json"),
                },
            )
        return ReferencePipeline(self.pipeline_id, stages["stage1"], stages["stage2"])

    def _reference_loader(self, _project_root, _pipeline_id):
        return self.reference

    def _record_loader(self, _stage, _data_root):
        return FrozenRecordSet(self.records, self.validation_folds)

    def _prepare(self) -> Path:
        return prepare_experiment(
            self.request,
            reference_loader=self._reference_loader,
            record_loader=self._record_loader,
        )

    def _train(self, run_dir, adapter=None, resume=False):
        return (resume_experiment if resume else train_experiment)(run_dir,
            adapter_resolver=lambda _: adapter or _FakeAdapter([]),
            reference_loader=self._reference_loader, record_loader=self._record_loader,
            bootstrap_iterations=0)

    def test_keyboard_interrupt_marks_failed_and_resumes(self):
        run_dir = self._prepare()
        adapter = _FakeAdapter([])
        with patch.object(adapter, "fit_predict_fold", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self._train(run_dir, adapter)
        self.assertEqual(ExperimentRunStore.open(run_dir).state, "failed")
        self._train(run_dir, resume=True)
        self.assertEqual(ExperimentRunStore.open(run_dir).state, "complete")

    def test_running_after_process_death_can_resume(self):
        run_dir = self._prepare()
        ExperimentRunStore.open(run_dir).mark_running()
        self._train(run_dir, resume=True)
        self.assertEqual(ExperimentRunStore.open(run_dir).state, "complete")

    def test_orphan_result_does_not_confuse_selected_attempt(self):
        run_dir = self._prepare()
        with self.assertRaises(RuntimeError):
            self._train(run_dir, _FakeAdapter([], fail_once_at=3))
        orphan = run_dir / 'candidates/stage2_majority/seed_42/fold_3/attempt_1/fold_result.json'
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text('{}')
        self._train(run_dir, resume=True)
        self.assertTrue(orphan.exists())
        marker = json.loads((orphan.parent.parent / 'complete.json').read_text())
        self.assertEqual(marker['fold_result_path'], 'attempt_2/fold_result.json')

    def test_frozen_metadata_tampering_stops_before_adapter(self):
        for name in ('inputs.json', 'protocol.json', 'resolved_config.json', 'candidate_matrix.json', 'reference_run.json', 'shared_fold_assignments.csv'):
            with self.subTest(name=name):
                self.request = replace(self.request, run_id=self.pipeline_id + '__exp_' + name.replace('.', '_'))
                run_dir = self._prepare()
                path = run_dir / name
                path.write_text(path.read_text() + ' ')
                calls = []
                with self.assertRaises((ExperimentProtocolError, ExperimentStateError)):
                    self._train(run_dir, _FakeAdapter(calls))
                self.assertEqual(calls, [])

    def test_current_reference_metrics_tamper_is_rejected(self):
        run_dir = self._prepare()
        (self.reference.stage2.run_dir / 'oof_metrics.json').write_text('{}')
        with self.assertRaises(ExperimentProtocolError):
            self._train(run_dir)

    def test_completed_report_tamper_is_rejected_without_state_mutation(self):
        run_dir = self._prepare()
        self._train(run_dir)
        state_bytes = (run_dir / 'state.json').read_bytes()
        (run_dir / 'leaderboard.json').write_text('{}')
        with self.assertRaises((ExperimentProtocolError, ExperimentStateError)):
            summarize_experiment(run_dir)
        with self.assertRaises((ExperimentProtocolError, ExperimentStateError)):
            self._train(run_dir)
        self.assertEqual((run_dir / 'state.json').read_bytes(), state_bytes)

    def test_partial_candidate_failure_publishes_only_verified_successes(self):
        self.request = replace(self.request, config=replace(self.config, candidates=('stage2_majority', 'stage2_mfcc_svm'), continue_on_candidate_failure=True))
        run_dir = self._prepare()
        failure = _FakeAdapter([], fail_once_at=1)
        path = train_experiment(run_dir, adapter_resolver=lambda c: failure if c.candidate_id == 'stage2_mfcc_svm' else _FakeAdapter([]), reference_loader=self._reference_loader, record_loader=self._record_loader, bootstrap_iterations=0)
        board = json.loads((path.parent / 'leaderboard.json').read_text())
        self.assertEqual([r['candidate_id'] for r in board['rows']], ['stage2_majority'])
        self.assertEqual(board['exclusions'][0]['candidate_id'], 'stage2_mfcc_svm')
        self.assertFalse((run_dir / 'comparison.md').exists())
        self.assertFalse((run_dir / 'verification.json').exists())

    def test_protocol_violation_is_run_fatal_even_when_continuing(self):
        self.request = replace(self.request, config=replace(self.config, continue_on_candidate_failure=True))
        run_dir = self._prepare()
        adapter = _FakeAdapter([])
        with patch.object(adapter, 'fit_predict_fold', side_effect=ProtocolViolation('leakage')):
            with self.assertRaises(ProtocolViolation):
                self._train(run_dir, adapter)
        self.assertEqual(ExperimentRunStore.open(run_dir).state, 'failed')

    def test_report_interruption_can_resume_without_overwrite(self):
        run_dir = self._prepare()
        with patch('cryinsight.experiments.reporting.write_leaderboard_markdown', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self._train(run_dir)
        self._train(run_dir, resume=True)
        self.assertEqual(ExperimentRunStore.open(run_dir).state, 'complete')
        self.assertTrue((run_dir / 'selection.json').is_file())
        self.assertTrue((run_dir / 'promotion_recommendation.md').is_file())

    def test_actual_svm_five_fold_run(self):
        from cryinsight.experiments.classical import ClassicalAdapter
        self.request = replace(self.request, config=replace(self.config, candidates=('stage2_mfcc_svm',)))
        adapter = ClassicalAdapter(feature_builder=lambda r, c, q: np.asarray([int(r.record_id.split('_')[1]), 1 if r.label == 'a' else -1], dtype=np.float32))
        run_dir = self._prepare()
        self._train(run_dir, adapter)
        self.assertEqual(ExperimentRunStore.open(run_dir).state, 'complete')

    def test_actual_cpu_small_cnn_one_epoch_five_fold_run(self):
        from cryinsight.experiments.neural import NeuralAdapter
        from cryinsight.experiments.registry import experiment_registry
        from tests.test_experiment_neural import NeuralAdapterTests
        spec = replace(experiment_registry()['stage2_logmel_small_cnn'], requires_gpu=False)
        self.request = replace(self.request, config=replace(self.config, candidates=(spec.candidate_id,), parameters={'device': 'cpu', 'epochs': 1, 'batch_size': 4, 'verbose': 0}), candidate_specs={spec.candidate_id: spec})
        run_dir = self._prepare()
        self._train(run_dir, NeuralAdapter(tensor_builder=NeuralAdapterTests._tiny_tensor_builder))
        self.assertEqual(ExperimentRunStore.open(run_dir).state, 'complete')
        seed = run_dir / 'candidates' / spec.candidate_id / 'seed_42'
        for name in ('config.json', 'environment.json'):
            self.assertTrue((seed / name).is_file())
        for fold in range(1, 6):
            attempt = seed / f'fold_{fold}' / 'attempt_1'
            for name in ('predictions.csv', 'metrics.json', 'normalizer_manifest.json', 'augmentation_manifest.json'):
                self.assertTrue((attempt / name).is_file(), name)

    def test_os_lock_blocks_concurrent_resume(self):
        from cryinsight.experiments.runner import _run_lock
        run_dir = self._prepare()
        store = ExperimentRunStore.open(run_dir)
        store.mark_running()
        with _run_lock(store):
            with self.assertRaisesRegex(ExperimentStateError, 'owns'):
                self._train(run_dir, resume=True)
        self.assertEqual(store.state, 'running')

    def test_aggregation_interruption_retains_partial_attempt_then_resumes(self):
        run_dir = self._prepare()
        original = __import__('cryinsight.experiments.selection', fromlist=['write_json_atomic']).write_json_atomic
        def interrupted(path, payload):
            if Path(path).name == 'seed_summary.json':
                raise KeyboardInterrupt()
            return original(path, payload)
        with patch('cryinsight.experiments.selection.write_json_atomic', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self._train(run_dir)
        self._train(run_dir, resume=True)
        seed = run_dir / 'candidates/stage2_majority/seed_42'
        self.assertTrue((seed / 'aggregation_attempt_1/oof_predictions.csv').exists())
        self.assertTrue((seed / 'aggregation_attempt_2/verification.json').exists())

    def test_loaded_record_order_change_rejected_on_resume(self):
        run_dir = self._prepare()
        with self.assertRaises(RuntimeError):
            self._train(run_dir, _FakeAdapter([], fail_once_at=2))
        self.records = tuple(reversed(self.records))
        with self.assertRaises(ExperimentProtocolError):
            self._train(run_dir, resume=True)

    def test_publication_interruption_does_not_leave_partial_root_artifact(self):
        from cryinsight.experiments.runner import _publish_file
        source, destination = self.root / 'source', self.root / 'destination'
        source.write_bytes(b'verified evidence')
        with patch('cryinsight.experiments.runner.os.fsync', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                _publish_file(source, destination)
        self.assertFalse(destination.exists())

    def test_runtime_policy_is_reapplied_on_resume(self):
        from cryinsight.experiments.runner import _write_environment_once
        from cryinsight.experiments.registry import experiment_registry
        import tensorflow as tf
        spec = replace(experiment_registry()['stage2_logmel_small_cnn'], requires_gpu=False)
        config = replace(self.config, candidates=(spec.candidate_id,), parameters={'device': 'cpu'})
        self.request = replace(self.request, config=config, candidate_specs={spec.candidate_id: spec})
        run_dir = self._prepare()
        store = ExperimentRunStore.open(run_dir)
        _write_environment_once(store, config)
        tf.keras.mixed_precision.set_global_policy('float64')
        _write_environment_once(store, config)
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, 'float32')

    def test_summarize_checks_selected_fold_mapping_before_ranking(self):
        from cryinsight.experiments.runner import _verify_seed
        run_dir = self._prepare()
        self._train(run_dir)
        store = ExperimentRunStore.open(run_dir)
        seed = run_dir / 'candidates/stage2_majority/seed_42'
        result_path = seed / 'fold_1/attempt_1/fold_result.json'
        payload = json.loads(result_path.read_text())
        payload['validation_record_ids'] = ['rec_2_a', 'rec_2_b']
        result_path.write_text(json.dumps(payload))
        marker_path = seed / 'fold_1/complete.json'
        marker = json.loads(marker_path.read_text())
        marker['artefact_sha256']['fold_result'] = sha256_file(result_path)
        marker_path.write_text(json.dumps(marker))
        with self.assertRaises(ExperimentProtocolError):
            _verify_seed(store, 'stage2_majority', 42)

    def test_invalid_probability_is_run_fatal_before_next_candidate(self):
        from cryinsight.experiments.selection import ExperimentVerificationError
        self.request = replace(self.request, config=replace(self.config, candidates=('stage2_majority', 'stage2_mfcc_svm'), continue_on_candidate_failure=True))
        run_dir = self._prepare()
        calls = []
        class InvalidProbabilityAdapter(_FakeAdapter):
            def fit_predict_fold(self, request):
                result = super().fit_predict_fold(request)
                return replace(result, probabilities=np.full((len(request.validation_records), 2), .9))
        with self.assertRaises(ExperimentVerificationError):
            self._train(run_dir, InvalidProbabilityAdapter(calls))
        self.assertEqual(calls, [1])
        self.assertEqual(ExperimentRunStore.open(run_dir).state, 'failed')

    def test_parent_derived_api_prepare_rejects_missing_parent_evidence(self):
        self.request = replace(self.request, config=replace(self.config, wave='B_features'))
        with self.assertRaises(ExperimentProtocolError):
            self._prepare()

    def test_reference_stage1_metrics_are_used_for_stage1_report(self):
        self.request = replace(self.request, config=replace(self.config, wave='stage1_baselines', candidates=('stage1_majority',)))
        run_dir = self._prepare()
        self._train(run_dir)
        selection = json.loads((run_dir / 'selection.json').read_text())
        self.assertEqual(selection['reference_stage'], 'stage1')

    def test_retry_failure_is_recorded_in_its_own_attempt(self):
        run_dir = self._prepare()
        class InterruptedAdapter(_FakeAdapter):
            def fit_predict_fold(self, request):
                super().fit_predict_fold(request)
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self._train(run_dir, InterruptedAdapter([]))
        with self.assertRaises(RuntimeError):
            self._train(run_dir, _FakeAdapter([], fail_once_at=1), resume=True)
        fold = run_dir / 'candidates/stage2_majority/seed_42/fold_1'
        self.assertTrue((fold / 'attempt_2/failure.json').exists())
        self.assertFalse((fold / 'attempt_1/failure.json').exists())

    def test_tiny_majority_run_completes_five_folds_and_oof(self) -> None:
        run_dir = self._prepare()
        calls: list[int] = []
        train_experiment(
            run_dir,
            adapter_resolver=lambda _candidate: _FakeAdapter(calls),
            reference_loader=self._reference_loader,
            record_loader=self._record_loader,
            bootstrap_iterations=0,
        )
        verification = json.loads((run_dir / "verification.json").read_text())
        self.assertEqual(verification["status"], "complete")
        self.assertEqual(verification["expected_jobs"], 5)
        self.assertEqual(verification["completed_jobs"], 5)
        self.assertEqual(calls, [1, 2, 3, 4, 5])
        self.assertTrue(
            (run_dir / "candidates/stage2_majority/seed_42/oof_metrics.json").is_file()
        )

    def test_protocol_failure_stops_entire_run(self) -> None:
        run_dir = self._prepare()
        with self.reference.stage2.fold_assignment_path.open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("corruption,corruption,a,1\n")
        with self.assertRaises(ExperimentProtocolError):
            train_experiment(
                run_dir,
                adapter_resolver=lambda _candidate: _FakeAdapter([]),
                reference_loader=self._reference_loader,
                record_loader=self._record_loader,
                bootstrap_iterations=0,
            )
        state = json.loads((run_dir / "state.json").read_text())
        self.assertEqual(state["status"], "failed")

    def test_resume_skips_verified_complete_folds(self) -> None:
        run_dir = self._prepare()
        first_calls: list[int] = []
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            train_experiment(
                run_dir,
                adapter_resolver=lambda _candidate: _FakeAdapter(
                    first_calls, fail_once_at=3
                ),
                reference_loader=self._reference_loader,
                record_loader=self._record_loader,
                bootstrap_iterations=0,
            )
        self.assertEqual(first_calls, [1, 2, 3])
        resumed_calls: list[int] = []
        resume_experiment(
            run_dir,
            adapter_resolver=lambda _candidate: _FakeAdapter(resumed_calls),
            reference_loader=self._reference_loader,
            record_loader=self._record_loader,
            bootstrap_iterations=0,
        )
        self.assertEqual(resumed_calls, [3, 4, 5])

    def test_continue_on_failure_remains_resumable_to_complete_verification(self) -> None:
        self.config = replace(self.config, continue_on_candidate_failure=True)
        self.request = replace(self.request, config=self.config)
        run_dir = self._prepare()
        first_calls: list[int] = []
        failure_summary = train_experiment(
            run_dir,
            adapter_resolver=lambda _candidate: _FakeAdapter(
                first_calls, fail_once_at=3
            ),
            reference_loader=self._reference_loader,
            record_loader=self._record_loader,
            bootstrap_iterations=0,
        )
        self.assertEqual(failure_summary.parent.name, "summary_attempt_1")
        self.assertEqual(failure_summary.name, "summary.json")
        self.assertFalse((run_dir / "verification.json").exists())
        resumed_calls: list[int] = []
        resume_experiment(
            run_dir,
            adapter_resolver=lambda _candidate: _FakeAdapter(resumed_calls),
            reference_loader=self._reference_loader,
            record_loader=self._record_loader,
            bootstrap_iterations=0,
        )
        verification = json.loads((run_dir / "verification.json").read_text())
        self.assertEqual(verification["status"], "complete")
        self.assertEqual(resumed_calls, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
