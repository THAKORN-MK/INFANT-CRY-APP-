from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import json
from dataclasses import asdict, replace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "Models_dbl" / "experiments" / "run_experiments.py"
WAVE_A = (
    PROJECT_ROOT
    / "Models_dbl"
    / "experiments"
    / "configs"
    / "stage2_wave_a.json"
)


def _load_cli():
    spec = importlib.util.spec_from_file_location("shared_experiment_cli", CLI_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load experiment CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = _load_cli()
        self.temporary = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self.temporary.name) / "runs"
        self.runs_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            self.cli.main(["--audit-only", "--train"])

    def test_train_requires_pipeline_id_config_and_experiment_id(self) -> None:
        with self.assertRaises(SystemExit):
            self.cli.main(["--train"])

    def test_audit_is_side_effect_free_and_does_not_import_tensorflow(self) -> None:
        tensorflow_was_loaded = "tensorflow" in sys.modules
        before = set(self.runs_dir.glob("*"))
        output = io.StringIO()
        with redirect_stdout(output):
            code = self.cli.main(
                [
                    "--audit-only",
                    "--config",
                    str(WAVE_A),
                    "--runs-dir",
                    str(self.runs_dir),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(before, set(self.runs_dir.glob("*")))
        self.assertIn('"training_started": false', output.getvalue())
        if not tensorflow_was_loaded:
            self.assertNotIn("tensorflow", sys.modules)

    def test_cli_has_no_test_dataset_argument(self) -> None:
        with self.assertRaises(SystemExit):
            self.cli.main(
                [
                    "--train",
                    "--test-data",
                    "data_set_dbl_split/test",
                ]
            )

    def test_wave_a_contains_exact_screening_candidates(self) -> None:
        from cryinsight.experiments.registry import load_experiment_config

        config = load_experiment_config(WAVE_A)
        self.assertEqual(config.seeds, (42,))
        self.assertEqual(
            config.candidates,
            (
                "stage2_majority",
                "stage2_mfcc_svm",
                "stage2_logmel_small_cnn",
                "stage2_cnn_only",
                "stage2_cnn_bilstm",
                "stage2_corrected_attention",
                "stage2_multi_branch_attention",
            ),
        )

    def test_feature_wave_selects_best_compatible_neural_not_overall_winner(self):
        from cryinsight.experiments.registry import experiment_registry
        specs = experiment_registry()
        rows = [{'rank': i, 'candidate_id': name} for i, name in enumerate(('stage2_majority', 'stage2_mfcc_svm', 'stage2_logmel_small_cnn', 'stage2_multi_branch_attention', 'stage2_cnn_only'), 1)]
        selected = self.cli._select_parent_candidates('B_features', rows, specs)
        self.assertEqual(selected, ['stage2_multi_branch_attention'])
        self.assertEqual(rows[0]['rank'], 1)

    def test_feature_wave_rejects_no_compatible_architecture(self):
        from cryinsight.experiments.registry import experiment_registry, ExperimentProtocolError
        with self.assertRaises(ExperimentProtocolError):
            self.cli._select_parent_candidates('B_features', [{'rank': 1, 'candidate_id': 'stage2_majority'}], experiment_registry())

    def test_confirmation_allows_one_eligible_candidate_and_never_more_than_two(self):
        from cryinsight.experiments.registry import experiment_registry
        specs = experiment_registry()
        rows = [{'rank': i, 'candidate_id': name} for i, name in enumerate(('stage2_cnn_only', 'stage2_multi_branch_attention', 'stage2_corrected_attention'), 1)]
        self.assertEqual(self.cli._select_parent_candidates('C', rows[:1], specs), ['stage2_cnn_only'])
        self.assertEqual(self.cli._select_parent_candidates('C', rows, specs), ['stage2_cnn_only', 'stage2_multi_branch_attention'])

    def test_train_rejects_full_source_config_mismatch(self):
        from tests.test_experiment_runner import SharedExperimentRunnerTests
        from cryinsight.experiments.registry import ExperimentProtocolError
        fixture = SharedExperimentRunnerTests('test_actual_svm_five_fold_run')
        fixture.setUp()
        try:
            run = fixture._prepare()
            config = run.parent / 'changed.json'
            config.write_text(json.dumps(asdict(replace(fixture.config, candidates=('stage2_mfcc_svm',)))))
            with patch.object(self.cli, 'train_experiment', return_value=None):
                with self.assertRaises(ExperimentProtocolError):
                    self.cli.main(['--train', '--pipeline-run-id', fixture.pipeline_id, '--config', str(config), '--runs-dir', str(run.parent), '--experiment-run-id', run.name])
        finally:
            fixture.tearDown()

    def test_parent_wave_requires_same_pipeline_and_immediate_predecessor(self):
        from cryinsight.experiments.registry import ExperimentProtocolError
        for wave, parent_wave in (('B_features', 'B_loss'), ('B_augmentation', 'A'), ('B_loss', 'B_features'), ('C', 'A')):
            with self.subTest(wave=wave), self.assertRaises(ExperimentProtocolError):
                self.cli._validate_parent_wave(wave, 'p', {'pipeline_run_id': 'p', 'config': {'wave': parent_wave}})
        with self.assertRaises(ExperimentProtocolError):
            self.cli._validate_parent_wave('B_features', 'other', {'pipeline_run_id': 'p', 'config': {'wave': 'A'}})

    def test_training_rejects_explicit_runtime_override(self):
        from cryinsight.experiments.registry import ExperimentProtocolError
        args = self.cli.build_parser().parse_args(['--train', '--device', 'cpu'])
        with self.assertRaises(ExperimentProtocolError):
            self.cli._check_runtime_request(args, {'parameters': {'device': 'gpu'}})

    def test_prepare_preserves_runtime_requirements_declared_in_source_config(self):
        from cryinsight.experiments.contracts import ExperimentConfig
        config = ExperimentConfig('1.0', 'A', (42,), 'oof_macro_f1', ('stage2_majority',), parameters={'require_gpu': True, 'mixed_precision': True})
        args = self.cli.build_parser().parse_args(['--prepare-only'])
        frozen = self.cli._runtime_config(config, args)
        self.assertTrue(frozen.parameters['require_gpu'])
        self.assertTrue(frozen.parameters['mixed_precision'])

    def test_loss_variants_keep_anchor_and_collapse_equivalent_crossentropy(self):
        from cryinsight.experiments.registry import experiment_registry
        anchor = experiment_registry()['stage2_corrected_attention']
        variants = {'categorical_crossentropy': {'loss': 'categorical_crossentropy'}, 'anchor': {}, 'class_balanced_crossentropy': {'loss': 'class_balanced_crossentropy'}}
        resolved = self.cli._derive_unique_variants(anchor, variants)
        self.assertEqual(list(resolved), [anchor.candidate_id, anchor.candidate_id + '__class_balanced_crossentropy'])

    def test_confirmation_skips_equivalent_parent_definitions_without_renumbering(self):
        from cryinsight.experiments.registry import experiment_registry, derive_candidate
        registry = experiment_registry()
        anchor = registry['stage2_corrected_attention']
        duplicate = derive_candidate(anchor, 'categorical_crossentropy', {'loss': 'categorical_crossentropy'})
        different = derive_candidate(anchor, 'class_balanced_crossentropy', {'loss': 'class_balanced_crossentropy'})
        specs = {spec.candidate_id: spec for spec in (anchor, duplicate, different)}
        rows = [{'rank': rank, 'candidate_id': spec.candidate_id} for rank, spec in enumerate((anchor, duplicate, different), 1)]
        self.assertEqual(self.cli._select_parent_candidates('C', rows, specs), [anchor.candidate_id, different.candidate_id])
        self.assertEqual(rows[2]['rank'], 3)

    def test_documented_wave_a_command_matches_parser(self) -> None:
        readme = (
            PROJECT_ROOT / "Models_dbl" / "experiments" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("--pipeline-run-id 20260821T164332Z_490383ff", readme)
        self.assertIn(
            "--config Models_dbl/experiments/configs/stage2_wave_a.json",
            readme,
        )
        self.assertNotIn("--test-data", readme)
        self.assertNotIn("--train --pipeline-run-id", readme)


if __name__ == "__main__":
    unittest.main()
