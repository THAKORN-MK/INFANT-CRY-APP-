# Shared Experiment Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe Shared Experiment Engine that trains classical and neural Stage 1/Stage 2 candidates on the frozen grouped folds from pipeline run `20260821T164332Z_490383ff`, ranks verified OOF results, and recommends—but never automatically promotes—a Stage 2 candidate for training run 3.

**Architecture:** A config-driven runner owns fold loading, provenance, state, artefacts, OOF aggregation, ranking, and reporting. Classical and neural adapters implement one common fold contract, while existing baseline/ablation scripts remain thin side-effect-free definitions. The held-out Test path is absent from the engine API and explicitly rejected at every configuration/path boundary.

**Tech Stack:** Python 3.10, NumPy, librosa 0.10.2.post1, scikit-learn, TensorFlow 2.21/Keras 3 on WSL2, existing `cryinsight.training` protocol/artefact/cache/checkpoint modules, `unittest`, JSON/CSV/Markdown artefacts.

**Spec:** `docs/superpowers/specs/2026-08-22-shared-experiment-engine-design.md`

**Continuation 2026-09-03:** Existing code is being repaired and verified in place under renewed Inline Execution approval. Current evidence and remaining blockers are tracked in [Inline Execution Status](2026-09-03-inline-execution-status.md). Historical task checkboxes below describe the original plan, not current test evidence. Do not recreate completed work, run full-data training, or execute incidental commit/push commands automatically.

## Global Constraints

- Preserve completed Stage 1 and Stage 2 run `20260821T164332Z_490383ff`; never write inside either run directory.
- Use the exact `fold_assignments.csv` files and their SHA-256 values from that reference run; never create new folds.
- Candidate selection uses grouped OOF metrics only; the engine must not read `data_set_dbl_split/test` or any Final Test artefact.
- OOF validation contains original records only; augmentation and normalization are training-fold-only.
- Screening uses seed 42; confirmation uses exactly seeds 42, 123, and 2026 with unchanged fold assignments.
- Primary selection metric is OOF Macro F1; follow the deterministic tie-break and promotion rules in the spec.
- TensorFlow candidates use the existing native-Linux checkpoint staging mechanism before verified publication to `/mnt/d`.
- Completed experiment artefacts are immutable; resume may fill only incomplete Candidate/Seed/Fold jobs with matching hashes.
- Do not alter `Models_dbl/Main/train_main_dbl.py`, `Models_dbl/binary/train_binary_dbl.py`, datasets, Webapp code, or deployment bundles in this implementation.
- Do not start full-dataset experiment training; only unit and tiny synthetic integration/smoke tests are allowed.
- Preserve unrelated user changes in the dirty worktree; stage only files named in each task.
- Python imports used by `--audit-only` must remain side-effect-free and must not import TensorFlow.

## File Structure

### New core package

- `cryinsight/experiments/__init__.py`: public imports only.
- `cryinsight/experiments/contracts.py`: immutable IDs, candidate/fold/result dataclasses, adapter protocol, JSON canonicalization.
- `cryinsight/experiments/registry.py`: candidate definitions, config parsing, forbidden-Test validation, registry payload.
- `cryinsight/experiments/fold_data.py`: reference-run verification, frozen assignment loading, record reconstruction, fold isolation.
- `cryinsight/experiments/feature_views.py`: feature view construction, YAMNet embedding extraction, feature-view-aware cache keys.
- `cryinsight/experiments/classical.py`: Majority, MFCC-SVM, YAMNet Linear/MLP adapters and safe estimator serialization.
- `cryinsight/experiments/neural.py`: TensorFlow candidate factories, losses, fold training, Keras checkpoint staging.
- `cryinsight/experiments/runner.py`: run state, immutable artefact store, prepare/train/resume/summarize orchestration.
- `cryinsight/experiments/selection.py`: probability verification, OOF aggregation, ranking, repeated-seed aggregation, promotion decision.
- `cryinsight/experiments/reporting.py`: deterministic CSV/Markdown leaderboard and experiment report generation.

### Existing files modified

- `cryinsight/training/experiments.py`: compatibility re-export to the new registry/fold hash functions.
- `cryinsight/training/feature_cache.py`: include feature-view identity in cache key without breaking existing callers.
- `Models_dbl/experiments/run_experiments.py`: full CLI lifecycle.
- `Models_dbl/experiments/script_support.py`: definitions resolve to `CandidateSpec` while retaining current audit JSON.
- `Models_dbl/experiments/baselines/**/*.py`: register factories/variants without owning training loops.
- `Models_dbl/experiments/ablations/*.py`: register concrete variant IDs/factories.
- `Models_dbl/experiments/configs/*.json`: Wave A/B/C and Stage 1 baseline configurations.
- `Models_dbl/experiments/README.md`, `README.md`, `Architecture.md`, `Report/report.md`, `file.txt`: workflow documentation.

### New tests

- `tests/test_experiment_contracts.py`
- `tests/test_experiment_registry.py`
- `tests/test_experiment_fold_data.py`
- `tests/test_experiment_feature_views.py`
- `tests/test_experiment_run_store.py`
- `tests/test_experiment_classical.py`
- `tests/test_experiment_neural.py`
- `tests/test_experiment_selection.py`
- `tests/test_experiment_runner.py`
- `tests/test_experiment_reporting.py`
- `tests/test_experiment_cli.py`

---

### Task 1: Contracts and Single-Source Candidate Registry

**Files:**
- Create: `cryinsight/experiments/__init__.py`
- Create: `cryinsight/experiments/contracts.py`
- Create: `cryinsight/experiments/registry.py`
- Modify: `cryinsight/training/experiments.py`
- Test: `tests/test_experiment_contracts.py`
- Test: `tests/test_experiment_registry.py`
- Modify test: `tests/test_experiment_protocol.py`

**Interfaces:**
- Produces: `CandidateSpec`, `FoldRequest`, `FoldResult`, `CandidateAdapter`, `ExperimentConfig`, `experiment_registry()`, `derive_candidate(anchor, variant_id, overrides)`, `load_experiment_config(path)`, `validate_selection_metric(metric)`, `fold_assignment_sha256(rows)`.
- Compatibility: `cryinsight.training.experiments` re-exports `ExperimentProtocolError`, `ExperimentSpec` as an alias of `CandidateSpec`, registry/fold helpers, and `registry_payload`.

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_experiment_contracts.py
import unittest

from cryinsight.experiments.contracts import CandidateSpec


class ExperimentContractTests(unittest.TestCase):
    def test_candidate_identity_is_stable_and_config_is_immutable(self):
        spec = CandidateSpec(
            candidate_id="stage2_majority",
            stage="stage2",
            family="baseline",
            feature_view="labels_only",
            adapter="classical",
            model="dummy_most_frequent",
            augmentation="none",
            normalization="none",
            loss="not_applicable",
        )
        self.assertEqual(spec.selection_metric, "oof_macro_f1")
        with self.assertRaises((AttributeError, TypeError)):
            spec.model = "changed"
```

```python
# tests/test_experiment_registry.py
import json
import tempfile
import unittest
from pathlib import Path

from cryinsight.experiments.registry import (
    ExperimentProtocolError,
    experiment_registry,
    load_experiment_config,
)


class ExperimentRegistryTests(unittest.TestCase):
    def test_registry_has_multi_branch_and_required_baselines(self):
        registry = experiment_registry()
        required = {
            "stage1_majority", "stage1_mfcc_svm", "stage1_logmel_small_cnn",
            "stage2_majority", "stage2_mfcc_svm", "stage2_logmel_small_cnn",
            "stage2_yamnet_linear", "stage2_yamnet_mlp",
            "stage2_cnn_only", "stage2_cnn_bilstm",
            "stage2_corrected_attention", "stage2_multi_branch_attention",
        }
        self.assertTrue(required.issubset(registry))

    def test_config_rejects_any_test_selection_field(self):
        payload = {
            "schema_version": "1.0", "wave": "A", "seeds": [42],
            "selection_metric": "final_test_accuracy",
            "candidates": ["stage2_majority"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ExperimentProtocolError, "Test|held-out"):
                load_experiment_config(path)

    def test_derived_candidate_has_deterministic_identity(self):
        anchor = experiment_registry()["stage2_corrected_attention"]
        first = derive_candidate(anchor, "without_chroma", {"feature_view": "feature_block_subset"})
        second = derive_candidate(anchor, "without_chroma", {"feature_view": "feature_block_subset"})
        self.assertEqual(first, second)
        self.assertEqual(first.candidate_id, "stage2_corrected_attention__without_chroma")
```

- [ ] **Step 2: Run the tests and confirm the new package is missing**

Run:

```bash
python -m unittest tests.test_experiment_contracts tests.test_experiment_registry -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cryinsight.experiments'`.

- [ ] **Step 3: Implement immutable contracts**

Create dataclasses with these exact public fields and signatures:

```python
# cryinsight/experiments/contracts.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    stage: str
    family: str
    feature_view: str
    adapter: str
    model: str
    augmentation: str
    normalization: str
    loss: str
    selection_metric: str = "oof_macro_f1"
    requires_gpu: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    wave: str
    seeds: tuple[int, ...]
    selection_metric: str
    candidates: tuple[str, ...]
    parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    candidate_source: str = "explicit"
    continue_on_candidate_failure: bool = True


@dataclass(frozen=True)
class FoldRequest:
    experiment_run_id: str
    pipeline_run_id: str
    candidate: CandidateSpec
    seed: int
    fold: int
    train_records: tuple[Any, ...]
    validation_records: tuple[Any, ...]
    label_order: tuple[str, ...]
    output_dir: Path
    runtime: Mapping[str, Any]


@dataclass(frozen=True)
class FoldResult:
    candidate_id: str
    seed: int
    fold: int
    validation_record_ids: tuple[str, ...]
    true_labels: tuple[str, ...]
    probabilities: Any
    model_path: Path
    manifest_path: Path


class CandidateAdapter(Protocol):
    def fit_predict_fold(self, request: FoldRequest) -> FoldResult: ...
```

Validate non-empty IDs, `stage in {'stage1', 'stage2'}`, `adapter in {'classical', 'neural'}`, seeds unique/non-negative, selection metric starting with `oof_`, and `candidate_source in {'explicit', 'parent_rank_1', 'parent_top_2'}`. Explicit configs require non-empty known Candidate IDs. Parent-derived configs require an empty `candidates` list at template-load time and are resolved to concrete IDs during `--prepare-only` from a verified complete parent experiment.

- [ ] **Step 4: Move the registry source of truth and preserve compatibility**

Implement `experiment_registry()` in `cryinsight/experiments/registry.py` with every existing candidate plus `stage2_multi_branch_attention`; implement recursive forbidden-key/value scanning for `test`, `heldout`, and `held-out`. Implement `derive_candidate()` with an allowlist of override fields (`feature_view`, `augmentation`, `normalization`, `loss`, `parameters`) and Candidate ID `<anchor_id>__<variant_id>`; merge nested `parameters` over the anchor parameters. The reserved variant `anchor` with empty overrides returns the original anchor unchanged. Reject override attempts to change Stage, family, adapter, or selection metric. Make `cryinsight/training/experiments.py` import and re-export those functions rather than defining a second registry.

```python
# compatibility shape
from cryinsight.experiments.contracts import CandidateSpec as ExperimentSpec
from cryinsight.experiments.registry import (
    ExperimentProtocolError,
    experiment_registry,
    fold_assignment_sha256,
    load_fold_assignments,
    registry_payload,
    validate_selection_metric,
)
```

- [ ] **Step 5: Run focused and legacy registry tests**

```bash
python -m unittest tests.test_experiment_contracts tests.test_experiment_registry tests.test_experiment_protocol tests.test_experiment_scripts -v
```

Expected: PASS; importing audit scripts does not add `tensorflow` to `sys.modules`.

- [ ] **Step 6: Commit only Task 1 files**

```bash
git add cryinsight/experiments/__init__.py cryinsight/experiments/contracts.py cryinsight/experiments/registry.py cryinsight/training/experiments.py tests/test_experiment_contracts.py tests/test_experiment_registry.py tests/test_experiment_protocol.py
git commit -m "feat: define shared experiment contracts"
```

---

### Task 2: Reference-Run Provenance, Frozen Folds, and Test Lock

**Files:**
- Create: `cryinsight/experiments/fold_data.py`
- Test: `tests/test_experiment_fold_data.py`

**Interfaces:**
- Consumes: `ExperimentProtocolError`, `fold_assignment_sha256` from Task 1; `OriginalRecord`, `assert_fold_integrity` from `cryinsight.training.protocol`; `sha256_file` from `cryinsight.training.artefacts`.
- Produces: `ReferenceStage`, `ReferencePipeline`, `load_reference_pipeline(project_root, pipeline_run_id)`, `load_reference_records(reference, data_root)`, `build_fold_dataset(records, assignments, fold)`, `reject_test_path(path, project_root)`.

- [ ] **Step 1: Write failing provenance and leakage tests**

Create temp reference Stage directories with `verification.json`, `fold_assignments.csv`, `oof_predictions.csv`, and `oof_metrics.json`. Cover:

```python
def test_reference_requires_both_complete_stages_with_same_id(self):
    reference = load_reference_pipeline(self.root, "20260821T164332Z_490383ff")
    self.assertEqual(reference.pipeline_run_id, "20260821T164332Z_490383ff")
    self.assertEqual(reference.stage2.fold_count, 5)

def test_reference_rejects_incomplete_verification(self):
    self.write_verification("stage2", status="incomplete")
    with self.assertRaisesRegex(ExperimentProtocolError, "complete"):
        load_reference_pipeline(self.root, self.run_id)

def test_test_directory_is_rejected_after_path_resolution(self):
    with self.assertRaisesRegex(ExperimentProtocolError, "Test dataset"):
        reject_test_path(self.root / "data_set_dbl_split" / "test", self.root)

def test_fold_dataset_has_no_record_group_or_hash_overlap(self):
    dataset = build_fold_dataset(self.records, self.assignments, fold=1)
    self.assertTrue(dataset.train_records)
    self.assertTrue(dataset.validation_records)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_fold_data -v
```

Expected: FAIL because `fold_data` does not exist.

- [ ] **Step 3: Implement reference loading and immutable evidence**

Use exact stage roots:

```python
stage_roots = {
    "stage1": project_root / "Models_dbl" / "binary" / "runs" / pipeline_run_id,
    "stage2": project_root / "Models_dbl" / "Main" / "runs" / pipeline_run_id,
}
```

`ReferenceStage` must contain `stage`, `run_dir`, `verification_status`, `fold_count`, `fold_assignment_path`, and a `hashes` mapping for verification, assignments, OOF predictions, and OOF metrics. Parse verification JSON, require `status == 'complete'`, require folds 1-5, hash every evidence file, and never open a Final Test file.

- [ ] **Step 4: Reconstruct records and fold views**

Load assignment CSV fields `record_id`, `filepath`, `relative_path`, `label`, `validation_fold`, `source_dataset`, `group_id`, `group_rule`, `sha256`. Require the resolved filepath to be inside the approved development root, reject paths inside `data_set_dbl_split/test`, and compare the file SHA-256 to the assignment SHA-256. Return `OriginalRecord` objects plus the frozen validation fold mapping.

```python
@dataclass(frozen=True)
class FoldDataset:
    fold: int
    train_records: tuple[OriginalRecord, ...]
    validation_records: tuple[OriginalRecord, ...]
```

Call `assert_fold_integrity()` for every fold and require folds exactly `{1,2,3,4,5}`.

- [ ] **Step 5: Run focused tests**

```bash
python -m unittest tests.test_experiment_fold_data tests.test_experiment_protocol tests.test_training_protocol -v
```

Expected: PASS.

- [ ] **Step 6: Commit only Task 2 files**

```bash
git add cryinsight/experiments/fold_data.py tests/test_experiment_fold_data.py
git commit -m "feat: lock experiments to verified reference folds"
```

---

### Task 3: Feature Views and Feature-View-Aware Cache Keys

**Files:**
- Create: `cryinsight/experiments/feature_views.py`
- Modify: `cryinsight/training/feature_cache.py`
- Test: `tests/test_experiment_feature_views.py`
- Modify test: `tests/test_feature_cache.py`

**Interfaces:**
- Consumes: `PreprocessingConfig`, `load_preprocessed_waveform`, `extract_features`; existing `FeatureCache`.
- Produces: `FEATURE_BLOCK_SLICES`, `select_feature_blocks(features, blocks)`, `mfcc_summary(features)`, `log_mel_view(features)`, `extract_yamnet_embedding(waveform, model)`, `prepare_yamnet_model(archive, destination)`, `build_feature_view(record, view_id, config, cache)`.

- [ ] **Step 1: Write failing shape/cache/YAMNet-output tests**

```python
class FeatureViewTests(unittest.TestCase):
    def test_feature_block_boundaries_match_stage2_contract(self):
        features = np.arange(196 * 128, dtype=np.float32).reshape(196, 128, 1)
        selected = select_feature_blocks(features, ("mfcc", "delta", "delta2", "chroma"))
        self.assertEqual(selected.shape, (132, 128, 1))

    def test_mfcc_summary_has_mean_and_std_for_40_coefficients(self):
        features = np.ones((196, 128, 1), dtype=np.float32)
        self.assertEqual(mfcc_summary(features).shape, (80,))

    def test_yamnet_embedding_averages_patch_embeddings(self):
        model = FakeYamnet(embeddings=np.array([[1., 3.], [3., 5.]], dtype=np.float32))
        np.testing.assert_allclose(
            extract_yamnet_embedding(np.zeros(16000, dtype=np.float32), model),
            np.array([2., 4.], dtype=np.float32),
        )

    def test_cache_key_changes_when_feature_view_changes(self):
        common = dict(source_sha256="a" * 64, preprocessing={}, augmentation=None,
                      dtype="float32", shape=(196, 128, 1))
        self.assertNotEqual(
            build_feature_cache_key(feature_view="all_blocks", **common),
            build_feature_cache_key(feature_view="log_mel", **common),
        )
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_feature_views tests.test_feature_cache -v
```

Expected: FAIL for missing feature-view functions/signature.

- [ ] **Step 3: Implement exact feature block mapping**

```python
FEATURE_BLOCK_SLICES = {
    "mfcc": slice(0, 40),
    "delta": slice(40, 80),
    "delta2": slice(80, 120),
    "log_mel": slice(120, 184),
    "chroma": slice(184, 196),
}

def select_feature_blocks(features, blocks):
    arrays = [np.asarray(features)[FEATURE_BLOCK_SLICES[name], :, :] for name in blocks]
    result = np.concatenate(arrays, axis=0)
    if not np.isfinite(result).all():
        raise FeatureViewError("Feature view contains non-finite values")
    return np.asarray(result, dtype=np.float32)
```

`mfcc_summary` concatenates mean and standard deviation across time for only the first 40 MFCC bins. `log_mel_view` returns rows 120:184. `all_blocks` returns the original tensor unchanged.

- [ ] **Step 4: Add YAMNet preparation/extraction without a network dependency**

Use the local `yamnet-tensorflow2-yamnet-v1.tar.gz`; validate archive SHA-256; safely reject absolute/parent-traversal tar members; extract to a content-addressed experiment cache; load with `tf.saved_model.load`. Resample audio to 16 kHz, pass a mono float32 waveform, accept mapping output key `embedding`/`embeddings` or the second element of a three-item tuple, and average patch embeddings to one clip vector.

```python
def extract_yamnet_embedding(waveform, model):
    outputs = model(np.asarray(waveform, dtype=np.float32))
    if isinstance(outputs, Mapping):
        patches = outputs.get("embeddings", outputs.get("embedding"))
    else:
        patches = outputs[1]
    values = np.asarray(patches, dtype=np.float32)
    return values.mean(axis=0, dtype=np.float64).astype(np.float32)
```

- [ ] **Step 5: Extend cache key compatibly**

Add keyword `feature_view: str = "all_blocks"` to `build_feature_cache_key()` and include it in the canonical payload. Existing callers retain their old behavior through the default. Add the feature view and extraction config to `build_feature_view()` cache calls.

- [ ] **Step 6: Run focused tests**

```bash
python -m unittest tests.test_experiment_feature_views tests.test_feature_cache tests.test_feature_contract -v
```

Expected: PASS without importing TensorFlow unless a YAMNet model is explicitly prepared.

- [ ] **Step 7: Commit only Task 3 files**

```bash
git add cryinsight/experiments/feature_views.py cryinsight/training/feature_cache.py tests/test_experiment_feature_views.py tests/test_feature_cache.py
git commit -m "feat: add reusable experiment feature views"
```

---

### Task 4: Experiment Run Store, State Machine, and Resume Guards

**Files:**
- Modify: `cryinsight/experiments/runner.py` (create initial state/store implementation)
- Test: `tests/test_experiment_run_store.py`

**Interfaces:**
- Consumes: `write_json_atomic`, `sha256_file`, `ReferencePipeline`, `ExperimentConfig`.
- Produces: `ExperimentRunStore.create(...)`, `ExperimentRunStore.open(...)`, `job_dir(candidate_id, seed, fold)`, `mark_job_complete(...)`, `mark_job_failed(...)`, `pending_jobs(...)`, `finalize(...)`.

- [ ] **Step 1: Write failing run-store tests**

Cover exact state transitions:

```python
def test_create_uses_parent_pipeline_id_in_experiment_id(self):
    store = ExperimentRunStore.create(self.runs, self.pipeline_id, self.config, self.reference)
    self.assertTrue(store.run_id.startswith(self.pipeline_id + "__exp_"))
    self.assertEqual(store.state, "prepared")

def test_completed_fold_cannot_be_replaced(self):
    store.mark_job_complete("stage2_majority", 42, 1, {"model": "a" * 64})
    with self.assertRaises(FileExistsError):
        store.mark_job_complete("stage2_majority", 42, 1, {"model": "b" * 64})

def test_resume_requires_matching_config_and_assignment_hashes(self):
    with self.assertRaisesRegex(ExperimentStateError, "hash"):
        ExperimentRunStore.open(self.run_dir, expected_config_hash="0" * 64)

def test_complete_run_is_immutable(self):
    store.finalize(status="complete")
    with self.assertRaisesRegex(ExperimentStateError, "complete"):
        store.mark_running()
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_run_store -v
```

Expected: FAIL because `ExperimentRunStore` is absent.

- [ ] **Step 3: Implement the state store**

Use `state.json` as the only mutable control file while a run is `prepared`, `running`, or `failed`; update it through write-to-temp plus `os.replace`. All scientific artefacts (`protocol.json`, assignments, configs, manifests, predictions, metrics) remain write-once.

```python
ALLOWED_TRANSITIONS = {
    "prepared": {"running", "failed"},
    "running": {"complete", "failed"},
    "failed": {"running"},
    "complete": set(),
}
```

Generate IDs with UTC timestamp and eight lowercase hex characters. `pending_jobs()` returns only jobs without a verified `complete.json`; a failed job retains `failure.json` and gets a new `attempt_<n>` work directory on resume. Never delete a prior failure artefact.

- [ ] **Step 4: Write protocol/reference snapshots**

At create time write once:

- `protocol.json`: config, selection scope, no-Test declaration, software schema.
- `reference_run.json`: parent ID and evidence hashes.
- `shared_fold_assignments.csv`: exact Stage 1/Stage 2 snapshot.
- `candidate_matrix.json`: expanded Candidate/Seed/Fold jobs.

Store `config_sha256` and per-stage assignment hashes in state and every job manifest.

- [ ] **Step 5: Run focused tests**

```bash
python -m unittest tests.test_experiment_run_store tests.test_training_protocol tests.test_checkpoint_staging -v
```

Expected: PASS.

- [ ] **Step 6: Commit only Task 4 files**

```bash
git add cryinsight/experiments/runner.py tests/test_experiment_run_store.py
git commit -m "feat: add resumable immutable experiment runs"
```

---

### Task 5: Classical Candidate Adapter

**Files:**
- Create: `cryinsight/experiments/classical.py`
- Test: `tests/test_experiment_classical.py`

**Interfaces:**
- Consumes: `FoldRequest`, `FoldResult`, feature views from Task 3.
- Produces: `ClassicalAdapter(factory_resolver, feature_builder)`, `build_classical_estimator(candidate, seed)`, `save_estimator_bundle(path, estimator, metadata)`.

- [ ] **Step 1: Write failing tiny-fold tests**

```python
class ClassicalAdapterTests(unittest.TestCase):
    def test_majority_returns_probabilities_in_declared_label_order(self):
        result = self.adapter("stage2_majority").fit_predict_fold(self.request)
        self.assertEqual(result.probabilities.shape, (len(self.validation), 5))
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)

    def test_svm_scaler_is_fitted_only_on_training_features(self):
        result = self.adapter("stage2_mfcc_svm").fit_predict_fold(self.request)
        metadata = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["normalizer_fit_record_ids"], self.train_ids)
        self.assertTrue(set(self.validation_ids).isdisjoint(metadata["normalizer_fit_record_ids"]))

    def test_serialized_estimator_round_trip_preserves_probabilities(self):
        result = self.adapter("stage2_mfcc_svm").fit_predict_fold(self.request)
        loaded = joblib.load(result.model_path)
        np.testing.assert_allclose(loaded.predict_proba(self.validation_x), result.probabilities)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_classical -v
```

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement estimator mapping and fold lifecycle**

Use:

```python
def build_classical_estimator(candidate, seed):
    if candidate.model == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    if candidate.model == "rbf_svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=10.0, gamma="scale", kernel="rbf",
                               probability=True, class_weight="balanced",
                               random_state=seed)),
        ])
    if candidate.model == "linear_softmax":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000,
                                               class_weight="balanced",
                                               random_state=seed)),
        ])
    if candidate.model == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                         random_state=seed)),
        ])
    raise ExperimentProtocolError(f"Unsupported classical model: {candidate.model}")
```

Map estimator `classes_` probabilities into `request.label_order`; error if a training fold lacks any declared class. Save a joblib bundle and a JSON manifest containing factory parameters, train/validation record IDs, label order, library versions, model SHA-256, feature-view ID, config hash, and assignment hash.

Record `parameter_count` deterministically: Dummy = 0; SVM = total fitted support-vector, dual-coefficient, and intercept scalar count; Logistic Regression = coefficient plus intercept scalar count; MLP = all fitted weight and bias scalar counts.

- [ ] **Step 4: Run focused tests**

```bash
python -m unittest tests.test_experiment_classical tests.test_experiment_feature_views -v
```

Expected: PASS using only tiny in-memory data.

- [ ] **Step 5: Commit only Task 5 files**

```bash
git add cryinsight/experiments/classical.py tests/test_experiment_classical.py
git commit -m "feat: train classical candidates through shared folds"
```

---

### Task 6: Neural Candidate Adapter, Architecture Factories, and Loss Variants

**Files:**
- Create: `cryinsight/experiments/neural.py`
- Modify: `Models_dbl/experiments/baselines/stage1/logmel_cnn.py`
- Modify: `Models_dbl/experiments/baselines/stage2/logmel_cnn.py`
- Modify: `Models_dbl/experiments/ablations/cnn_only.py`
- Modify: `Models_dbl/experiments/ablations/without_attention.py`
- Test: `tests/test_experiment_neural.py`
- Modify test: `tests/test_model_architecture.py`

**Interfaces:**
- Consumes: `FoldRequest`, `FoldResult`, existing Stage 1/2 model builders, `CheckpointStaging`, normalizer/augmentation helpers, TensorFlow runtime manifest.
- Produces: `LossBundle`, `NeuralAdapter`, `build_neural_model(candidate, input_shape, num_classes, tf)`, `build_loss(candidate, class_counts, label_order, tf)`, `effective_number_class_weights(counts, beta)`, `categorical_focal_loss(gamma, alpha, tf)`.

- [ ] **Step 1: Write failing architecture/loss/checkpoint tests**

```python
class NeuralAdapterTests(unittest.TestCase):
    def test_factory_maps_corrected_and_multi_branch_models(self):
        single = build_neural_model(self.spec("stage2_corrected_attention"), (196, 128, 1), 5, tf)
        multi = build_neural_model(self.spec("stage2_multi_branch_attention"), (196, 128, 1), 5, tf)
        self.assertIn("corrected_single_branch", single.name)
        self.assertIn("corrected_multi_branch", multi.name)

    def test_effective_number_weights_are_finite_and_mean_one(self):
        weights = effective_number_class_weights({"a": 10, "b": 100}, beta=0.999)
        self.assertAlmostEqual(sum(weights.values()) / 2.0, 1.0, places=6)
        self.assertGreater(weights["a"], weights["b"])

    def test_tiny_neural_fold_publishes_loadable_keras_checkpoint(self):
        result = self.adapter.fit_predict_fold(self.tiny_request(epochs=1))
        loaded = tf.keras.models.load_model(result.model_path, custom_objects=self.custom_objects)
        self.assertEqual(loaded.output_shape[-1], 5)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_neural -v
```

Expected: FAIL because `neural.py` does not exist.

- [ ] **Step 3: Implement lazy TensorFlow factory resolution**

Do not import TensorFlow at module import time. Resolve factories only inside `fit_predict_fold()`.

```python
def build_neural_model(candidate, input_shape, num_classes, tf):
    if candidate.model == "small_cnn":
        return build_small_cnn(tf, input_shape, num_classes)
    if candidate.model == "cnn_only":
        return build_cnn_only(tf, input_shape, num_classes)
    if candidate.model == "cnn_bilstm":
        return build_cnn_bilstm(tf, input_shape, num_classes)
    if candidate.model == "cnn_bilstm_attention":
        architecture = str(candidate.parameters.get("architecture", "corrected_single_branch"))
        return build_stage2_model(tf, input_shape, num_classes, architecture=architecture)
    raise ExperimentProtocolError(f"Unsupported neural model: {candidate.model}")
```

Move reusable model bodies from thin definition scripts into `neural.py`; scripts call/re-export the shared factory for audit compatibility. Keep final classifier dtype float32 under mixed precision.

- [ ] **Step 4: Implement exact loss variants**

Define the return contract so the adapter does not infer whether weighting belongs in the loss or `model.fit()`:

```python
@dataclass(frozen=True)
class LossBundle:
    loss: Any
    class_weight: Mapping[int, float] | None
    metadata: Mapping[str, Any]
```

- `categorical_crossentropy`: standard Keras categorical cross-entropy.
- `class_balanced_crossentropy`: standard loss plus `class_weight` indexed in declared label order from effective-number weights with `beta=0.999`, normalized to mean 1.
- `focal`: categorical focal loss with `gamma=2.0` and per-class alpha from the same normalized effective-number weights; return `class_weight=None` because alpha is already inside the loss.

Clip predicted probabilities to `[1e-7, 1 - 1e-7]` inside focal loss only. Save the full loss parameters and training class counts in the fold manifest.

- [ ] **Step 5: Implement train-only augmentation and fold training**

Use existing `extract_fold_tensors()` for original and waveform-augmented tensors. For `augmentation == 'none'`, pass an empty plan. For `waveform_only`, use target-based train-fold augmentation but no Mixup. For `waveform_plus_mixup`, create deterministic Mixup from training tensors with `alpha=0.3` and seed derived from Candidate/Seed/Fold. Validation always comes from `extract_original_tensors()`.

Callbacks:

```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(staging.mutable_path, monitor="val_loss",
                                       mode="min", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10,
                                     restore_best_weights=False),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5,
                                        patience=4, min_lr=1e-6),
]
```

Publish through `CheckpointStaging.publish_once()`, verify SHA-256 and Keras reload, then predict validation probabilities. No final refit occurs.

Save `model.count_params()` as `parameter_count` in every neural fold manifest and require the value to agree across all five folds of a Candidate/Seed.

- [ ] **Step 6: Run CPU tiny-fold and architecture tests**

```bash
python -m unittest tests.test_experiment_neural tests.test_model_architecture tests.test_checkpoint_staging -v
```

Expected: PASS; tiny test uses one epoch and synthetic arrays.

- [ ] **Step 7: Run the existing GPU smoke test in WSL**

```bash
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
cd "/mnt/d/INFANT CRY"
python tests/smoke_gpu_models.py
```

Expected: GPU detected, forward/backward/save/load succeeds; PTX JIT warning is allowed.

- [ ] **Step 8: Commit only Task 6 files**

```bash
git add cryinsight/experiments/neural.py Models_dbl/experiments/baselines/stage1/logmel_cnn.py Models_dbl/experiments/baselines/stage2/logmel_cnn.py Models_dbl/experiments/ablations/cnn_only.py Models_dbl/experiments/ablations/without_attention.py tests/test_experiment_neural.py tests/test_model_architecture.py
git commit -m "feat: train neural experiment candidates safely"
```

---

### Task 7: Probability Verification and OOF Metric Aggregation

**Files:**
- Create: `cryinsight/experiments/selection.py` (probability/OOF section first)
- Test: `tests/test_experiment_selection.py`

**Interfaces:**
- Consumes: `FoldResult`, `OofPrediction`, `aggregate_oof_metrics`, `assert_exact_oof_coverage`.
- Produces: `VerifiedProbabilities`, `verify_probabilities(values, tolerance=1e-5)`, `aggregate_candidate_seed(fold_results, expected_records, label_order, output_dir)`.

- [ ] **Step 1: Write failing probability and OOF tests**

```python
def test_small_probability_sum_error_is_normalized_and_reported(self):
    values = np.array([[0.2, 0.3, 0.5000004]], dtype=np.float32)
    checked = verify_probabilities(values, tolerance=1e-5)
    np.testing.assert_allclose(checked.values.sum(axis=1), 1.0, atol=1e-12)
    self.assertGreater(checked.max_sum_deviation, 0.0)

def test_large_probability_sum_error_fails(self):
    with self.assertRaisesRegex(ExperimentVerificationError, "sum"):
        verify_probabilities(np.array([[0.2, 0.2]], dtype=np.float32), tolerance=1e-5)

def test_oof_aggregation_rejects_duplicate_validation_record(self):
    with self.assertRaisesRegex(ExperimentVerificationError, "duplicate"):
        aggregate_candidate_seed(self.duplicate_results, self.records, self.labels, self.output)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_selection -v
```

Expected: FAIL for missing verification functions.

- [ ] **Step 3: Implement strict float64 probability boundary**

```python
def verify_probabilities(values, tolerance=1e-5):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ExperimentVerificationError("Probabilities must be finite 2-D values")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ExperimentVerificationError("Probabilities must be within [0, 1]")
    sums = matrix.sum(axis=1, dtype=np.float64)
    deviation = float(np.max(np.abs(sums - 1.0)))
    if deviation > tolerance or np.any(sums <= 0.0):
        raise ExperimentVerificationError("Probability rows do not sum to one")
    return VerifiedProbabilities(matrix / sums[:, None], deviation)
```

This eliminates the prior scikit-learn `y_prob values do not sum to one` warning while failing genuinely invalid outputs.

- [ ] **Step 4: Aggregate exactly five folds**

Require folds `{1,2,3,4,5}`, unique record IDs, original `sample_kind`, exact expected support, and label order. Write once:

- `oof_predictions.csv`
- `oof_metrics.json`
- `seed_summary.json`
- `verification.json`

Use existing `aggregate_oof_metrics()` and include `max_probability_sum_deviation`, per-class metrics, fold metrics, and all relevant artefact hashes.

- [ ] **Step 5: Run focused and existing artefact tests**

```bash
python -m unittest tests.test_experiment_selection tests.test_training_artefacts tests.test_publication_evaluation -v
```

Expected: PASS without probability-sum warnings.

- [ ] **Step 6: Commit only Task 7 files**

```bash
git add cryinsight/experiments/selection.py tests/test_experiment_selection.py
git commit -m "feat: verify and aggregate experiment OOF results"
```

---

### Task 8: Shared Runner Lifecycle and Safe Resume

**Files:**
- Modify: `cryinsight/experiments/runner.py`
- Test: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes: Tasks 1-7 contracts, registry, fold loader, run store, adapters, OOF aggregation.
- Produces: `prepare_experiment(...)`, `train_experiment(...)`, `resume_experiment(...)`, `summarize_experiment(...)`, `resolve_adapter(candidate)`.

- [ ] **Step 1: Write failing end-to-end tiny runner tests**

```python
def test_tiny_majority_run_completes_five_folds_and_oof(self):
    run_dir = prepare_experiment(self.request)
    train_experiment(run_dir, adapter_resolver=self.fake_adapter_resolver)
    verification = json.loads((run_dir / "verification.json").read_text())
    self.assertEqual(verification["status"], "complete")
    self.assertEqual(verification["expected_jobs"], 5)
    self.assertEqual(verification["completed_jobs"], 5)

def test_protocol_failure_stops_entire_run(self):
    self.corrupt_assignment_hash()
    with self.assertRaises(ExperimentProtocolError):
        train_experiment(self.run_dir)
    self.assertEqual(self.state()["status"], "failed")

def test_resume_skips_verified_complete_fold(self):
    self.complete_folds(1, 2)
    called = resume_experiment(self.run_dir, adapter_resolver=self.recording_resolver)
    self.assertEqual(called, [3, 4, 5])
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_runner -v
```

Expected: FAIL for missing lifecycle functions.

- [ ] **Step 3: Implement adapter resolution and job loop**

```python
def resolve_adapter(candidate):
    if candidate.adapter == "classical":
        return ClassicalAdapter(...)
    if candidate.adapter == "neural":
        return NeuralAdapter(...)
    raise ExperimentProtocolError(f"Unsupported adapter: {candidate.adapter}")
```

For each Candidate → Seed → Fold:

1. Revalidate config/reference/assignment hashes.
2. Write the root `environment.json` once at transition to `running`; include Python/platform and requested device, then include TensorFlow/GPU details when any selected Candidate is neural.
3. Skip only a verified complete job.
4. Build `FoldDataset` and `FoldRequest`.
5. Execute adapter inside an attempt directory.
6. Validate result IDs and model/manifest hashes.
7. Atomically write complete/failure marker.
8. Aggregate Candidate/Seed OOF only after all five folds complete.

- [ ] **Step 4: Separate candidate failure from protocol failure**

Catch model/dependency/resource failures, write `failure.json`, mark the job failed, and continue to the next Candidate if Config says `continue_on_candidate_failure: true`. Do not catch `ExperimentProtocolError` arising from fold hash, Test path, leakage, or OOF identity; mark the whole run failed and stop.

- [ ] **Step 5: Implement summarize gate**

`summarize_experiment()` may run after all expected jobs finish or fail. It accepts only Candidate/Seed directories whose verification status is complete; failed/incomplete items go to an exclusions payload. Finalize the run only after leaderboard/report writers succeed.

- [ ] **Step 6: Run runner and integration tests**

```bash
python -m unittest tests.test_experiment_runner tests.test_experiment_run_store tests.test_experiment_classical tests.test_experiment_selection -v
```

Expected: PASS.

- [ ] **Step 7: Commit only Task 8 files**

```bash
git add cryinsight/experiments/runner.py tests/test_experiment_runner.py
git commit -m "feat: orchestrate shared fold experiments"
```

---

### Task 9: Deterministic Ranking and Promotion Recommendation

**Files:**
- Modify: `cryinsight/experiments/selection.py`
- Modify test: `tests/test_experiment_selection.py`

**Interfaces:**
- Consumes: verified Candidate/Seed OOF metric payloads and reference OOF metrics.
- Produces: `rank_screening_candidates(rows)`, `aggregate_repeated_seeds(rows)`, `rank_confirmation_candidates(rows)`, `promotion_decision(winner, reference)`.

- [ ] **Step 1: Add failing ranking/tie-break tests**

```python
def test_screening_tie_uses_minimum_class_recall_then_parameters(self):
    ranked = rank_screening_candidates([
        self.row("large", macro_f1=.900, min_recall=.80, params=500000),
        self.row("small", macro_f1=.897, min_recall=.85, params=100000),
    ])
    self.assertEqual(ranked[0]["candidate_id"], "small")

def test_confirmation_requires_exactly_three_seeds(self):
    with self.assertRaisesRegex(ExperimentSelectionError, "42, 123, 2026"):
        aggregate_repeated_seeds([self.seed_row(42), self.seed_row(123)])

def test_promotion_requires_one_point_macro_f1_gain(self):
    decision = promotion_decision(
        self.winner(macro_f1=.8977, balanced=.89, min_recall=.80),
        self.reference(macro_f1=.8878, balanced=.8872, min_recall=.79),
    )
    self.assertEqual(decision["status"], "no_promotion_recommended")
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_selection -v
```

Expected: FAIL because ranking functions are absent.

- [ ] **Step 3: Implement Wave A/B comparator**

Sort by Macro F1 descending. Values within `0.005` of the current best use minimum per-class recall descending; if within `0.005` again, use parameter count ascending, then Candidate ID ascending. Keep the raw metric columns so the tie-break can be audited.

- [ ] **Step 4: Implement Wave C aggregation/comparator**

Require seeds exactly `{42,123,2026}`. Compute mean, sample standard deviation, minimum, and maximum for OOF Macro F1, Balanced Accuracy, Accuracy, and minimum per-class recall. Apply the spec order: mean Macro F1, mean minimum recall, Macro F1 standard deviation, parameter count, Candidate ID.

- [ ] **Step 5: Implement promotion decision with explicit reasons**

```python
checks = {
    "three_seed_confirmation": winner["seeds"] == [42, 123, 2026],
    "macro_f1_gain_at_least_0_01": winner["mean_oof_macro_f1"] >= reference["oof_macro_f1"] + 0.01,
    "balanced_accuracy_not_lower": winner["mean_oof_balanced_accuracy"] >= reference["oof_balanced_accuracy"],
    "minimum_recall_drop_within_0_02": winner["mean_minimum_class_recall"] >= reference["minimum_class_recall"] - 0.02,
    "verification_complete": winner["verification_status"] == "complete",
}
status = "recommended_for_training_3" if all(checks.values()) else "no_promotion_recommended"
```

- [ ] **Step 6: Run focused tests**

```bash
python -m unittest tests.test_experiment_selection -v
```

Expected: PASS with deterministic order across repeated calls.

- [ ] **Step 7: Commit only Task 9 files**

```bash
git add cryinsight/experiments/selection.py tests/test_experiment_selection.py
git commit -m "feat: rank OOF candidates and gate promotion"
```

---

### Task 10: Leaderboard and Experiment Reports

**Files:**
- Create: `cryinsight/experiments/reporting.py`
- Create: `Report/experiments/report.md`
- Test: `tests/test_experiment_reporting.py`

**Interfaces:**
- Consumes: verified ranking rows, exclusions, promotion decision, reference evidence.
- Produces: `write_leaderboard_csv`, `write_leaderboard_markdown`, `write_promotion_recommendation`, `write_experiment_report`.

- [ ] **Step 1: Write failing deterministic report tests**

```python
def test_report_excludes_final_test_metrics_and_states_oof_scope(self):
    write_experiment_report(self.output, self.payload)
    text = self.output.read_text(encoding="utf-8")
    self.assertIn("grouped OOF only", text)
    self.assertNotIn("final_test_accuracy", text)
    self.assertNotIn("92.409", text)

def test_failed_candidate_is_listed_but_not_ranked(self):
    write_leaderboard_markdown(self.output, self.valid_rows, self.failures)
    text = self.output.read_text(encoding="utf-8")
    self.assertIn("Excluded/failed candidates", text)
    self.assertNotIn("| 1 | broken_candidate |", text)

def test_report_write_refuses_overwrite(self):
    self.output.write_text("existing", encoding="utf-8")
    with self.assertRaises(FileExistsError):
        write_experiment_report(self.output, self.payload)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_reporting -v
```

Expected: FAIL because `reporting.py` is absent.

- [ ] **Step 3: Implement deterministic OOF-only writers**

Leaderboard columns:

```text
rank,candidate_id,wave,seeds,oof_macro_f1_mean,oof_macro_f1_std,
oof_balanced_accuracy_mean,oof_accuracy_mean,minimum_class_recall_mean,
parameter_count,verification_status
```

The report contains reference OOF metrics, Candidate table, per-class OOF comparison, stability across seeds, exclusions/failures, promotion checks, limitations, and an explicit statement that Final Test was unavailable to ranking. Use write-once UTF-8 output. Write the run-local comparison to `<experiment_run>/comparison.md` and the linked report to `Report/experiments/<experiment_run_id>.md`; both are generated from the same verified payload and include its SHA-256.

- [ ] **Step 4: Build the report hub**

Create `Report/experiments/report.md` as a menu page explaining the difference between pipeline runs and experiment runs. Do not add a fabricated run link; future completed runs append links through a separately approved documentation update.

- [ ] **Step 5: Run report tests**

```bash
python -m unittest tests.test_experiment_reporting -v
```

Expected: PASS.

- [ ] **Step 6: Commit only Task 10 files**

```bash
git add cryinsight/experiments/reporting.py Report/experiments/report.md tests/test_experiment_reporting.py
git commit -m "feat: report verified experiment comparisons"
```

---

### Task 11: CLI Lifecycle, Wave Configs, and Thin Definition Adapters

**Files:**
- Modify: `Models_dbl/experiments/run_experiments.py`
- Modify: `Models_dbl/experiments/script_support.py`
- Modify: `Models_dbl/experiments/baselines/stage1/*.py`
- Modify: `Models_dbl/experiments/baselines/stage2/*.py`
- Modify: `Models_dbl/experiments/ablations/*.py`
- Create: `Models_dbl/experiments/configs/stage1_baselines.json`
- Create: `Models_dbl/experiments/configs/stage2_wave_a.json`
- Create: `Models_dbl/experiments/configs/stage2_wave_b_features.json`
- Create: `Models_dbl/experiments/configs/stage2_wave_b_augmentation.json`
- Create: `Models_dbl/experiments/configs/stage2_wave_b_loss.json`
- Create: `Models_dbl/experiments/configs/stage2_wave_c.json`
- Test: `tests/test_experiment_cli.py`
- Modify test: `tests/test_experiment_scripts.py`

**Interfaces:**
- Consumes: runner lifecycle from Task 8 and config schema from Task 1.
- Produces: CLI modes `--audit-only`, `--prepare-only`, `--train`, `--resume`, `--summarize`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_modes_are_mutually_exclusive(self):
    with self.assertRaises(SystemExit):
        main(["--audit-only", "--train"])

def test_train_requires_pipeline_id_and_config(self):
    with self.assertRaises(SystemExit):
        main(["--train"])

def test_audit_is_side_effect_free_and_does_not_import_tensorflow(self):
    before = set(self.runs_dir.glob("*"))
    code = main(["--audit-only", "--config", str(self.wave_a)])
    self.assertEqual(code, 0)
    self.assertEqual(before, set(self.runs_dir.glob("*")))
    self.assertNotIn("tensorflow", sys.modules)

def test_cli_has_no_test_dataset_argument(self):
    with self.assertRaises(SystemExit):
        main(["--train", "--test-data", "data_set_dbl_split/test"])
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m unittest tests.test_experiment_cli tests.test_experiment_scripts -v
```

Expected: FAIL because the full CLI and configs are absent.

- [ ] **Step 3: Implement the parser and dispatch**

Required common options:

```text
--pipeline-run-id
--parent-experiment-run-id
--config
--runs-dir
--experiment-run-id
--device auto|gpu|cpu
--require-gpu
--mixed-precision
--feature-cache-dir
--continue-on-candidate-failure
```

`--prepare-only` requires pipeline ID/config and creates a prepared run. Wave B/C also require a verified complete `--parent-experiment-run-id`. `--train` requires pipeline ID, config, and the prepared `--experiment-run-id`; it verifies all hashes and resolved Candidate IDs before transitioning that same run to `running`. `--resume` and `--summarize` require experiment run ID. The parser deliberately has no Test/held-out argument. Audit validates Registry, Config, local YAMNet archive availability, and required dependencies without creating files or importing TensorFlow.

- [ ] **Step 4: Write concrete Wave configs**

Wave A must contain seed 42 and these IDs exactly:

```json
{
  "schema_version": "1.0",
  "wave": "A",
  "seeds": [42],
  "selection_metric": "oof_macro_f1",
  "continue_on_candidate_failure": true,
  "candidates": [
    "stage2_majority",
    "stage2_mfcc_svm",
    "stage2_logmel_small_cnn",
    "stage2_yamnet_linear",
    "stage2_yamnet_mlp",
    "stage2_cnn_only",
    "stage2_cnn_bilstm",
    "stage2_corrected_attention",
    "stage2_multi_branch_attention"
  ]
}
```

Wave B runs sequentially: `features` uses Wave A as parent, `augmentation` uses the completed feature run as parent, and `loss` uses the completed augmentation run as parent. Every Wave B config includes the unchanged parent anchor as one Candidate plus its one-factor variants, so a change must outperform the anchor rather than being accepted automatically.

The feature template is concrete and contains no unknown Candidate name:

```json
{
  "schema_version": "1.0",
  "wave": "B_features",
  "seeds": [42],
  "selection_metric": "oof_macro_f1",
  "candidate_source": "parent_rank_1",
  "candidates": [],
  "parameters": {
    "variants": {
      "anchor": {},
      "without_chroma": {"feature_view": "feature_block_subset", "parameters": {"blocks": ["mfcc", "delta", "delta2", "log_mel"]}},
      "without_log_mel": {"feature_view": "feature_block_subset", "parameters": {"blocks": ["mfcc", "delta", "delta2", "chroma"]}},
      "mfcc_derivatives_only": {"feature_view": "feature_block_subset", "parameters": {"blocks": ["mfcc", "delta", "delta2"]}}
    }
  }
}
```

The augmentation template variants are `anchor`, `none`, `waveform_only`, and `waveform_plus_mixup`. The loss template variants are `anchor`, `categorical_crossentropy`, `class_balanced_crossentropy`, and `focal_gamma_2`. Resolve them through `derive_candidate()`.

Wave C uses this policy:

```json
{
  "schema_version": "1.0",
  "wave": "C",
  "seeds": [42, 123, 2026],
  "selection_metric": "oof_macro_f1",
  "candidate_source": "parent_top_2",
  "candidates": [],
  "parameters": {}
}
```

`--prepare-only` resolves no more than two concrete IDs from the verified parent Leaderboard. The resolved IDs and their full Candidate parameters are written to immutable `resolved_config.json` before training. Unknown, failed, or incomplete parent Candidates are rejected. Stage 1 baseline config contains its three baseline IDs, `candidate_source: "explicit"`, and seed 42.

- [ ] **Step 5: Update thin scripts without adding training loops**

Each script keeps `experiment_definition()` and `main(["--audit-only"])`. Replace local duplicate model/estimator bodies with imports from the shared adapter factory where applicable, while retaining concrete variant payloads. Add a multi-branch definition under the corrected attention registration rather than creating another full trainer.

- [ ] **Step 6: Run CLI/definition tests**

```bash
python -m unittest tests.test_experiment_cli tests.test_experiment_scripts tests.test_experiment_registry -v
```

Expected: PASS and no TensorFlow import during audit.

- [ ] **Step 7: Run real audit-only commands in WSL**

```bash
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
cd "/mnt/d/INFANT CRY"
python Models_dbl/experiments/run_experiments.py --audit-only --config Models_dbl/experiments/configs/stage2_wave_a.json
python Models_dbl/experiments/baselines/stage2/yamnet_transfer.py --audit-only
```

Expected: JSON shows Wave A Candidate IDs, `selection_scope: grouped_oof_only`, YAMNet archive available, `training_started: false`; no run directory is created.

- [ ] **Step 8: Commit only Task 11 files**

```bash
git add Models_dbl/experiments/run_experiments.py Models_dbl/experiments/script_support.py Models_dbl/experiments/baselines Models_dbl/experiments/ablations Models_dbl/experiments/configs tests/test_experiment_cli.py tests/test_experiment_scripts.py
git commit -m "feat: expose shared experiment wave lifecycle"
```

---

### Task 12: Documentation and Command Handoff

**Files:**
- Modify: `Models_dbl/experiments/README.md`
- Modify: `README.md`
- Modify: `Architecture.md`
- Modify: `Report/report.md`
- Modify: `file.txt`

**Interfaces:**
- Consumes: final CLI and directory contract from Tasks 1-11.
- Produces: user-facing preparation, training, resume, summarize, and report-navigation instructions.

- [ ] **Step 1: Add documentation assertions to CLI tests**

```python
def test_documented_wave_a_command_matches_parser(self):
    readme = (PROJECT_ROOT / "Models_dbl/experiments/README.md").read_text(encoding="utf-8")
    self.assertIn("--pipeline-run-id 20260821T164332Z_490383ff", readme)
    self.assertIn("--config Models_dbl/experiments/configs/stage2_wave_a.json", readme)
    self.assertNotIn("--test-data", readme)
```

- [ ] **Step 2: Run and verify documentation test failure**

```bash
python -m unittest tests.test_experiment_cli -v
```

Expected: FAIL until documentation contains the exact commands.

- [ ] **Step 3: Update Experiment README**

Document:

1. Difference between pipeline run ID and experiment run ID.
2. Why the current Final Test is unavailable for ranking.
3. Wave A → Wave B → Wave C → separate Promotion approval flow.
4. Commands for audit, prepare, train, resume, summarize.
5. Artefact tree and meaning of `complete`, `failed`, and `no_promotion_recommended`.
6. PTX JIT warning and native-Linux checkpoint staging.

- [ ] **Step 4: Update root documentation**

- `README.md`: link to Experiment README, Architecture section, and `Report/experiments/report.md`; state Python 3.10/WSL GPU commands.
- `Architecture.md`: add Shared Engine data flow and OOF-only selection; state `train_main_dbl.py` remains unchanged until promotion.
- `Report/report.md`: add Experiment hub link without fabricated metrics.
- `file.txt`: add exact WSL commands below.

```bash
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
cd "/mnt/d/INFANT CRY"
python Models_dbl/experiments/run_experiments.py --audit-only --config Models_dbl/experiments/configs/stage2_wave_a.json
python Models_dbl/experiments/run_experiments.py --prepare-only --pipeline-run-id 20260821T164332Z_490383ff --config Models_dbl/experiments/configs/stage2_wave_a.json
```

Do not document a `--train` command as safe to paste until `--prepare-only` output has been checked and the user explicitly approves starting Wave A.

- [ ] **Step 5: Run documentation and CLI tests**

```bash
python -m unittest tests.test_experiment_cli tests.test_experiment_reporting -v
```

Expected: PASS.

- [ ] **Step 6: Commit only Task 12 files**

```bash
git add Models_dbl/experiments/README.md README.md Architecture.md Report/report.md file.txt tests/test_experiment_cli.py
git commit -m "docs: explain shared experiment workflow"
```

---

### Task 13: Full Verification Without Starting Training

**Files:**
- Modify only if a verification failure exposes a defect in files from Tasks 1-12.
- Do not modify datasets, completed runs, `train_main_dbl.py`, or `train_binary_dbl.py`.

**Interfaces:**
- Consumes: complete Shared Experiment Engine.
- Produces: test/audit evidence and a handoff containing the prepare-only command; no full experiment run.

- [ ] **Step 1: Run the full unit suite in the WSL Python 3.10 GPU environment**

```bash
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
cd "/mnt/d/INFANT CRY"
python -m unittest discover -s tests -v
```

Expected: all tests PASS. Allowed warnings: TensorFlow PTX JIT warning for compute capability 12.0a. Not allowed: probability-sum warning, leakage warning, missing GPU when GPU smoke is requested, or writes inside completed pipeline runs.

- [ ] **Step 2: Run checkpoint and GPU smoke tests**

```bash
python tests/smoke_checkpoint_staging.py
python tests/smoke_gpu_models.py
```

Expected: native-Linux checkpoint writes, verified copy to the workspace test destination, Keras reload, and GPU device detection all succeed.

- [ ] **Step 3: Run Wave A audit only**

```bash
python Models_dbl/experiments/run_experiments.py --audit-only --config Models_dbl/experiments/configs/stage2_wave_a.json
```

Expected: all nine Wave A candidates resolve; selection is grouped OOF only; training is false; no new experiment run directory appears.

- [ ] **Step 4: Verify protected files and runs were not changed**

```bash
git diff -- Models_dbl/Main/train_main_dbl.py Models_dbl/binary/train_binary_dbl.py
git status --short Models_dbl/Main/runs/20260821T164332Z_490383ff Models_dbl/binary/runs/20260821T164332Z_490383ff
```

Expected: no new diff caused by this implementation and no modified tracked artefact inside either completed run. Existing untracked run directories remain untouched.

- [ ] **Step 5: Review the final diff for Test leakage and placeholders**

```bash
rg -n -i "final_test|heldout|held-out" cryinsight/experiments Models_dbl/experiments tests/test_experiment_*.py
git diff --check
```

Expected: no incomplete implementation markers; any Test/held-out matches are rejection rules, scope declarations, or tests proving rejection; no whitespace errors.

- [ ] **Step 6: Present handoff and stop before full training**

Report:

- full test count and pass/fail result,
- GPU/checkpoint smoke result,
- Wave A audit result,
- files changed,
- confirmation that pipeline run 2 and both trainers were untouched,
- exact `--prepare-only` command.

Do not run `--prepare-only` or `--train` until the user explicitly approves that next action.
