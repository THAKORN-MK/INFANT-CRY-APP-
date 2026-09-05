from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ExperimentRegistryTests(unittest.TestCase):
    def test_wave_policy_rejects_invalid_seed_and_parent_sequences(self):
        from cryinsight.experiments.registry import load_experiment_config, ExperimentProtocolError
        cases = [
            {'wave': 'C', 'seeds': [42], 'candidate_source': 'parent_top_2', 'candidates': []},
            {'wave': 'A', 'seeds': [123], 'candidates': ['stage2_majority']},
            {'wave': 'B_features', 'seeds': [42], 'candidates': ['stage2_cnn_only']},
            {'wave': 'unknown', 'seeds': [42], 'candidates': ['stage2_majority']},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for case in cases:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps({'schema_version': '1.0', 'selection_metric': 'oof_macro_f1', **case}))
                with self.subTest(case=case), self.assertRaises(ExperimentProtocolError):
                    load_experiment_config(path)
    def test_registry_has_required_baselines_and_multi_branch_candidate(self) -> None:
        from cryinsight.experiments.registry import experiment_registry

        registry = experiment_registry()
        required = {
            "stage1_majority",
            "stage1_mfcc_svm",
            "stage1_logmel_small_cnn",
            "stage2_majority",
            "stage2_mfcc_svm",
            "stage2_logmel_small_cnn",
            "stage2_yamnet_linear",
            "stage2_yamnet_mlp",
            "stage2_cnn_only",
            "stage2_cnn_bilstm",
            "stage2_corrected_attention",
            "stage2_multi_branch_attention",
        }

        self.assertTrue(required.issubset(registry))
        self.assertEqual(
            registry["stage2_multi_branch_attention"].parameters["architecture"],
            "corrected_multi_branch",
        )

    def test_config_rejects_final_test_selection_metric(self) -> None:
        from cryinsight.experiments.registry import (
            ExperimentProtocolError,
            load_experiment_config,
        )

        payload = {
            "schema_version": "1.0",
            "wave": "A",
            "seeds": [42],
            "selection_metric": "final_test_accuracy",
            "candidates": ["stage2_majority"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ExperimentProtocolError, "Test|held-out"):
                load_experiment_config(path)

    def test_config_rejects_nested_heldout_input(self) -> None:
        from cryinsight.experiments.registry import (
            ExperimentProtocolError,
            load_experiment_config,
        )

        payload = {
            "schema_version": "1.0",
            "wave": "A",
            "seeds": [42],
            "selection_metric": "oof_macro_f1",
            "candidates": ["stage2_majority"],
            "parameters": {"stage2_majority": {"heldout_path": "somewhere"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ExperimentProtocolError, "Test|held-out"):
                load_experiment_config(path)

    def test_derived_candidate_has_deterministic_identity_and_merged_parameters(self) -> None:
        from cryinsight.experiments.registry import derive_candidate, experiment_registry

        anchor = experiment_registry()["stage2_corrected_attention"]
        overrides = {
            "feature_view": "feature_block_subset",
            "parameters": {"blocks": ["mfcc", "delta", "delta2", "log_mel"]},
        }

        first = derive_candidate(anchor, "without_chroma", overrides)
        second = derive_candidate(anchor, "without_chroma", overrides)

        self.assertEqual(first, second)
        self.assertEqual(
            first.candidate_id,
            "stage2_corrected_attention__without_chroma",
        )
        self.assertEqual(first.parameters["architecture"], "corrected_single_branch")
        self.assertEqual(
            first.parameters["blocks"],
            ["mfcc", "delta", "delta2", "log_mel"],
        )

    def test_anchor_variant_returns_original_candidate(self) -> None:
        from cryinsight.experiments.registry import derive_candidate, experiment_registry

        anchor = experiment_registry()["stage2_corrected_attention"]
        self.assertIs(derive_candidate(anchor, "anchor", {}), anchor)

    def test_explicit_config_loads_known_candidates(self) -> None:
        from cryinsight.experiments.registry import load_experiment_config

        payload = {
            "schema_version": "1.0",
            "wave": "A",
            "seeds": [42],
            "selection_metric": "oof_macro_f1",
            "candidate_source": "explicit",
            "continue_on_candidate_failure": True,
            "candidates": ["stage2_majority"],
            "parameters": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_experiment_config(path)

        self.assertEqual(config.candidates, ("stage2_majority",))
        self.assertEqual(config.seeds, (42,))


if __name__ == "__main__":
    unittest.main()
