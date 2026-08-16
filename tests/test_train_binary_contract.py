from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "Models_dbl" / "binary" / "train_binary_dbl.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("cryinsight_train_binary_contract", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not create import spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage1StaticContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_import_is_side_effect_free_without_tensorflow(self) -> None:
        module = load_script_module()

        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.build_binary_model))
        self.assertEqual(module.LABEL_ORDER, ("not_baby", "baby"))

    def test_defaults_use_locked_train_and_test_directories(self) -> None:
        module = load_script_module()
        args = module.build_parser().parse_args(["--train"])

        self.assertEqual(
            args.train_data_dir,
            PROJECT_ROOT / "data_set_dbl_split" / "train",
        )
        self.assertEqual(
            args.test_data_dir,
            PROJECT_ROOT / "data_set_dbl_split" / "test",
        )
        self.assertEqual(
            module.PreprocessingConfig.stage1_binary().feature_shape,
            (120, 128, 1),
        )

    def test_stage1_semantics_and_split_before_augmentation_are_explicit(self) -> None:
        self.assertIn("discover_stage1_candidates", self.source)
        self.assertIn("resolve_stage1_records", self.source)
        self.assertIn("esc50_source_file", self.source)
        self.assertNotIn("AUG_BABY", self.source)
        self.assertNotIn("AUG_NOT_BABY", self.source)
        self.assertNotIn("best_fold_accuracy", self.source)
        self.assertNotIn("best_fold_as_publication_result", self.source)
        module = load_script_module()
        main_source = inspect.getsource(module.main)
        training_source = inspect.getsource(module.run_training)
        self.assertIn("split = assign_grouped_folds(", main_source)
        self.assertIn("augmentation_plan = build_target_augmentation_plan(", training_source)
        self.assertLess(
            main_source.index("split = assign_grouped_folds("),
            main_source.index("metrics = run_training("),
        )

    def test_fold_bundle_and_checkpoint_contract_are_present(self) -> None:
        self.assertIn('monitor="val_loss"', self.source)
        self.assertIn('checkpoint_monitor="val_loss"', self.source)
        self.assertIn("load_model", self.source)
        self.assertIn("save_normalizer", self.source)
        self.assertIn("load_normalizer", self.source)
        self.assertIn("expected_run_id=run_dir.name", self.source)
        self.assertIn("expected_fold=fold", self.source)
        self.assertIn("build_fold_manifest", self.source)
        self.assertIn("oof_predictions.csv", self.source)
        self.assertIn('"root_labels": run_dir / "labels_binary_dbl.json"', self.source)
        self.assertIn(
            '"root_preprocessing_config": run_dir / "preprocessing_config.json"',
            self.source,
        )

    def test_fold_models_and_single_final_deployment_model_have_explicit_names(self) -> None:
        self.assertIn(
            'model_path = fold_dir / f"fold_{fold}_binary_dbl.keras"',
            self.source,
        )
        self.assertNotIn(
            'model_path = fold_dir / "best_model_binary_dbl.keras"',
            self.source,
        )
        module = load_script_module()
        final_source = inspect.getsource(module.train_final_model)
        self.assertIn('best_model_binary_dbl.keras', final_source)
        self.assertIn("select_final_refit_epoch(", self.source)
        self.assertNotIn("select_best_fold_for_deployment(", self.source)
        self.assertNotIn("create_selected_fold_deployment_bundle(", self.source)
        self.assertIn('"final_refit_manifest"', self.source)

    def test_stage1_architecture_and_augmentation_are_not_removed(self) -> None:
        module = load_script_module()
        model_source = inspect.getsource(module.build_binary_model)
        training_source = inspect.getsource(module.run_training)

        self.assertIn("Conv2D", model_source)
        self.assertIn("Bidirectional", model_source)
        self.assertIn("LSTM", model_source)
        self.assertIn("AttentionLayer", model_source)
        self.assertIn("build_target_augmentation_plan(", training_source)
        self.assertNotIn("mixup_batch(", training_source)

    def test_locked_test_is_evaluated_once_after_all_folds(self) -> None:
        module = load_script_module()
        training_source = inspect.getsource(module.run_training)
        final_source = inspect.getsource(module.train_final_model)
        fold_section = training_source[
            training_source.index("for fold in range(1, N_FOLDS + 1):") :
            training_source.index("coverage = assert_exact_oof_coverage(")
        ]

        self.assertNotIn("heldout", fold_section)
        self.assertEqual(training_source.count("train_final_model("), 1)
        self.assertIn("final_test_predictions.csv", final_source)
        self.assertIn("final_test_metrics.json", final_source)
        self.assertEqual(final_source.count("aggregate_heldout_metrics("), 1)
        self.assertIn('fold="final_refit"', final_source)

    def test_training_failure_is_recorded_as_incomplete_and_reraised(self) -> None:
        main_source = inspect.getsource(load_script_module().main)

        self.assertIn("write_incomplete_run_verification", self.source)
        self.assertIn("except Exception as exc:", main_source)
        self.assertIn("write_incomplete_run_verification(", main_source)
        self.assertIn("raise", main_source)


class Stage1AuditOnlyTests(unittest.TestCase):
    def test_audit_excludes_target_20_and_groups_negatives_by_source(self) -> None:
        module = load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data_set_dbl"
            stage_root = root / "models_dbl" / "binary"
            for index, label in enumerate(module.INFANT_LABELS):
                label_directory = data / label
                label_directory.mkdir(parents=True)
                (label_directory / f"sample_{index}.wav").write_bytes(
                    f"unique-{label}".encode("utf-8")
                )
            negatives = data / "not_baby"
            negatives.mkdir(parents=True)
            (negatives / "1-211527-A-20.wav").write_bytes(b"crying-baby")
            (negatives / "1-100210-A-36.wav").write_bytes(b"negative-a")
            (negatives / "1-100210-B-36.wav").write_bytes(b"negative-b")

            resolution, audit = module.audit_stage1(data)
            negative = [row for row in resolution.eligible if row.label == "not_baby"]

            self.assertEqual(audit["esc50_target20_excluded_count"], 1)
            self.assertIn("source_confounding_limitation", audit)
            self.assertIn("InfantCry-DBL", audit["source_confounding_limitation"])
            self.assertIn("ESC-50", audit["source_confounding_limitation"])
            self.assertEqual({row.group_id for row in negative}, {"esc50_source:100210"})
            with redirect_stdout(io.StringIO()):
                exit_code = module.main(
                    [
                        "--audit-only",
                        "--data-dir",
                        str(data),
                        "--stage-root",
                        str(stage_root),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertFalse(stage_root.exists())


if __name__ == "__main__":
    unittest.main()
