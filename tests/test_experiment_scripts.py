from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "Models_dbl" / "experiments"

EXPECTED_SCRIPTS = {
    "baselines/stage1/majority.py": "stage1_majority",
    "baselines/stage1/mfcc_svm.py": "stage1_mfcc_svm",
    "baselines/stage1/logmel_cnn.py": "stage1_logmel_small_cnn",
    "baselines/stage2/majority.py": "stage2_majority",
    "baselines/stage2/mfcc_svm.py": "stage2_mfcc_svm",
    "baselines/stage2/logmel_cnn.py": "stage2_logmel_small_cnn",
    "baselines/stage2/yamnet_transfer.py": "stage2_yamnet_linear",
    "ablations/cnn_only.py": "stage2_cnn_only",
    "ablations/without_attention.py": "stage2_cnn_bilstm",
    "ablations/feature_ablation.py": "stage2_feature_blocks",
    "ablations/augmentation_ablation.py": "stage2_augmentation_mixup",
}


def load_script(path: Path):
    module_name = "experiment_contract_" + "_".join(path.parts[-4:]).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentScriptContractTests(unittest.TestCase):
    def test_every_approved_script_has_a_runnable_side_effect_free_audit(self) -> None:
        tensorflow_was_loaded = "tensorflow" in sys.modules

        for relative_path, expected_id in EXPECTED_SCRIPTS.items():
            with self.subTest(script=relative_path):
                path = EXPERIMENT_ROOT / relative_path
                module = load_script(path)
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = module.main(["--audit-only"])
                payload = json.loads(output.getvalue())

                self.assertEqual(exit_code, 0)
                self.assertEqual(payload["experiment_id"], expected_id)
                self.assertEqual(payload["status"], "definition_ready")
                self.assertFalse(payload["training_started"])

        if not tensorflow_was_loaded:
            self.assertNotIn("tensorflow", sys.modules)

    def test_each_script_resolves_to_the_shared_registry(self) -> None:
        from cryinsight.training.experiments import experiment_registry

        registry = experiment_registry()
        for relative_path, expected_id in EXPECTED_SCRIPTS.items():
            with self.subTest(script=relative_path):
                module = load_script(EXPERIMENT_ROOT / relative_path)
                definition = module.experiment_definition()

                self.assertEqual(definition["experiment_id"], expected_id)
                self.assertEqual(definition["registry"], registry[expected_id].__dict__)
                self.assertTrue(definition["implementation"]["factory"])

    def test_ablation_scripts_expose_concrete_comparison_variants(self) -> None:
        expected_variants = {
            "ablations/cnn_only.py": {"cnn_only"},
            "ablations/without_attention.py": {"cnn_bilstm"},
            "ablations/feature_ablation.py": {
                "all_features",
                "without_delta",
                "without_delta2",
                "without_log_mel",
                "without_chroma",
            },
            "ablations/augmentation_ablation.py": {
                "none",
                "waveform_only",
                "waveform_plus_mixup",
            },
        }
        for relative_path, expected in expected_variants.items():
            with self.subTest(script=relative_path):
                module = load_script(EXPERIMENT_ROOT / relative_path)
                self.assertEqual(set(module.experiment_variants()), expected)


if __name__ == "__main__":
    unittest.main()
