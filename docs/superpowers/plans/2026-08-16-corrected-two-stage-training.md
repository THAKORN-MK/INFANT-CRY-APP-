# CryInsight Corrected Two-Stage Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-resistant, traceable grouped five-fold training entry points for CryInsight Stage 2 and Stage 1 without altering legacy artefacts.

**Architecture:** A small shared `cryinsight.training` package owns original-record identity, duplicate resolution, grouped fold assignment, target-based augmentation planning, integrity assertions, shared audio features, manifests, OOF aggregation, and hashing. `train_main_dbl.py` and `train_binary_dbl.py` remain the executable orchestration points and retain their respective model architectures.

**Tech Stack:** Python 3.10-compatible code, TensorFlow/Keras 2.15, NumPy, librosa, scikit-learn, matplotlib, seaborn, standard-library `unittest` for dependency-light protocol tests.

**Execution status (2026-08-16):** Tasks 1-6 are implemented. The project Python 3.10
environment has the complete ML stack, 39 tests passed, both model serialization
smoke checks passed, and real-data grouped-fold feasibility was confirmed. The first
Stage 1 attempt stopped before epoch 1 on the unversioned short-audio boundary. The
user authorized the repair and both full five-fold runs; remaining work is specified
in `docs/superpowers/plans/2026-08-16-publication-training-execution.md`.

## Global Constraints

- Project and model root remains exactly `D:/INFANT CRY`.
- Do not delete, move, or overwrite existing datasets, models, normalizers, logs, or reports.
- New artefacts must be written only below a new `models_dbl/<stage>/runs/<run_id>/` directory.
- Assign original records to folds before augmentation; Validation contains original records only.
- Use five folds and seed 42 by default.
- Stage 2 labels are `belly_pain, burping, discomfort, hungry, tired`; Stage 1 labels are `not_baby, baby`.
- Exclude ESC-50 target 20 (`crying_baby`) from Stage 1 negatives and group ESC-50 by source file.
- Do not claim independent/external validation and do not estimate accuracy from legacy metrics.
- Run full five-fold training only after the newly approved regression and smoke
  checks pass; user approval was received on 2026-08-16.

---

### Task 1: Dependency-light record protocol

**Files:**
- Create: `cryinsight/__init__.py`
- Create: `cryinsight/training/__init__.py`
- Create: `cryinsight/training/protocol.py`
- Create: `tests/test_training_protocol.py`

**Interfaces:**
- Produces: `OriginalRecord`, `AuditRow`, `parse_esc50_filename()`, `discover_stage2_candidates()`, `discover_stage1_candidates()`, `resolve_stage2_records()`, `resolve_stage1_records()`, `build_target_augmentation_plan()`, `assert_fold_integrity()`, and `assert_exact_oof_coverage()`.
- Consumes: filesystem `.wav` paths only; no TensorFlow/librosa imports.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_stage1_excludes_esc50_crying_baby():
    parsed = parse_esc50_filename("1-187207-A-20.wav")
    assert parsed.target == 20
    assert parsed.category_rule == "exclude_crying_baby"

def test_target_plan_balances_training_originals_exactly():
    counts = {"a": 2, "b": 4}
    plan = build_target_augmentation_plan(records, fold=1, seed=42)
    assert plan.target == 4
    assert plan.generated_by_label == {"a": 2, "b": 0}

def test_oof_coverage_rejects_duplicates(self):
    with self.assertRaises(ProtocolViolation):
        assert_exact_oof_coverage(["r1", "r2"], ["r1", "r1", "r2"])
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `python -m unittest tests.test_training_protocol -v`
Expected: FAIL because `cryinsight.training.protocol` does not exist.

- [ ] **Step 3: Implement records, audit resolution, target planning, and invariants**

Implement frozen dataclasses and deterministic functions with lazy scikit-learn imports. Cross-label Stage 2 hashes exclude the whole hash group; same-label duplicates retain the lexicographically first path. Stage 1 canonicalizes infant hashes and excludes ESC-50 target 20 before splitting.

- [ ] **Step 4: Run protocol tests**

Run: `python -m unittest tests.test_training_protocol -v`
Expected: all dependency-light tests PASS.

### Task 2: Grouped fold assignment and manifests

**Files:**
- Modify: `cryinsight/training/protocol.py`
- Modify: `tests/test_training_protocol.py`

**Interfaces:**
- Produces: `assign_grouped_folds(records, n_folds=5, seed=42) -> SplitResult`, `write_record_audit_csv()`, `write_fold_assignments_csv()`, and `write_json_atomic()`.
- Consumes: eligible canonical `OriginalRecord` values from Task 1.

- [ ] **Step 1: Add tests for deterministic assignments and zero overlap**

Create synthetic grouped records with five validation-supporting groups per class. Assert repeated assignment with seed 42 is identical, each record has one validation fold, and every fold passes record/group/hash overlap checks.

- [ ] **Step 2: Run the focused tests**

Run: `python -m unittest tests.test_training_protocol.GroupedFoldTests -v`
Expected: FAIL because grouped assignment is not implemented.

- [ ] **Step 3: Implement splitter selection and CSV/JSON writers**

Use `StratifiedGroupKFold`, then `GroupKFold`, and allow `StratifiedKFold` only when the caller explicitly states that no reliable group exists. Record splitter name, seed, group rule, and per-fold class support.

- [ ] **Step 4: Re-run focused and full protocol tests**

Run: `python -m unittest tests.test_training_protocol -v`
Expected: PASS when scikit-learn is available; otherwise grouped-split tests report SKIP while dependency-light tests PASS.

### Task 3: Shared audio and feature contract

**Files:**
- Create: `cryinsight/audio/__init__.py`
- Create: `cryinsight/audio/features.py`
- Create: `tests/test_feature_contract.py`

**Interfaces:**
- Produces: `PreprocessingConfig`, `load_preprocessed_waveform()`, `apply_augmentation()`, `extract_features()`, `extract_fold_tensors()`, and `mixup_batch()`.
- Consumes: original records and deterministic augmentation-plan rows.

- [ ] **Step 1: Add shape, finite-value, deterministic-noise, and validation-original-only tests**

Use generated waveforms; mark librosa-dependent tests skipped when librosa is unavailable. Verify `(196, 128, 1)`, `float32`, finite values, identical seeded augmentation, and no augmented validation row.

- [ ] **Step 2: Run feature tests**

Run: `python -m unittest tests.test_feature_contract -v`
Expected: FAIL because the feature module does not exist.

- [ ] **Step 3: Implement the frozen preprocessing and feature order**

Import librosa lazily inside audio operations, validate empty/non-finite waveforms, apply bounded deterministic transformations, normalize derivatives, pad/truncate frames, and fail on wrong shape or non-finite output.

- [ ] **Step 4: Re-run feature tests**

Run: `python -m unittest tests.test_feature_contract -v`
Expected: PASS with librosa; otherwise dependency check and skip behavior PASS.

### Task 4: Shared fold artefacts and OOF metrics

**Files:**
- Create: `cryinsight/training/artefacts.py`
- Create: `tests/test_training_artefacts.py`

**Interfaces:**
- Produces: `create_run_directory()`, `save_normalizer()`, `load_normalizer()`, `sha256_file()`, `build_fold_manifest()`, `aggregate_oof_metrics()`, and `group_bootstrap_intervals()`.
- Consumes: fold tensors, predictions, labels, assignments, and selected checkpoints.

- [ ] **Step 1: Add tests for immutable runs, normalizer metadata, hash validation, and exact OOF support**

Run directories must reject collisions. Normalizers must include matching run/fold IDs and finite nonzero std. A modified artefact must fail hash verification. OOF aggregation must reject augmented rows and duplicate/missing record IDs.

- [ ] **Step 2: Run artefact tests**

Run: `python -m unittest tests.test_training_artefacts -v`
Expected: FAIL because the artefact module does not exist.

- [ ] **Step 3: Implement immutable artefact and metric functions**

Write temporary files and atomically replace only inside the new run directory. Compute metrics from original OOF rows with explicit evaluation labels and group-level bootstrap sampling.

- [ ] **Step 4: Re-run artefact tests**

Run: `python -m unittest tests.test_training_artefacts -v`
Expected: PASS for dependency-light cases; scikit-learn/matplotlib cases SKIP when unavailable.

### Task 5: Correct Stage 2 entry point

**Files:**
- Modify: `Models_dbl/Main/train_main_dbl.py`
- Create: `tests/test_train_main_contract.py`

**Interfaces:**
- Consumes: Tasks 1-4 shared APIs.
- Produces: `build_main_model()`, CLI commands `--audit-only` and `--prepare-only`, five-fold training, per-fold bundles, and pooled original OOF outputs.

- [ ] **Step 1: Add static and dependency-light Stage 2 contract tests**

Assert the source contains no `AUG_TIMES`, calls original-record assignment before augmentation, uses five folds, writes per-fold normalizers/manifests, reloads checkpoints selected by `val_loss`, and exposes the required CLI modes without starting training on import.

- [ ] **Step 2: Run Stage 2 contract tests**

Run: `python -m unittest tests.test_train_main_contract -v`
Expected: FAIL against the current script.

- [ ] **Step 3: Rewrite Stage 2 orchestration while preserving model architecture**

Use the full original five-class corpus, corrected grouped assignments, per-fold target balancing, scalar fold normalizers, configurable deterministic 500-sample Mixup, `val_loss` checkpoint selection, original-only validation predictions, fold manifests, and pooled OOF aggregation. Do not publish a best fold or label the legacy 20% split independent.

- [ ] **Step 4: Run Stage 2 contract, compilation, and audit-only checks**

Run: `python -m unittest tests.test_train_main_contract -v`
Run: `python -m py_compile Models_dbl/Main/train_main_dbl.py`
Run: `python Models_dbl/Main/train_main_dbl.py --audit-only`
Expected: tests and compile PASS; audit reports 1,551 Stage 2 candidates, duplicate/conflict exclusions, and no model training.

### Task 6: Correct Stage 1 entry point

**Files:**
- Create: `Models_dbl/binary/train_binary_dbl.py`
- Create: `tests/test_train_binary_contract.py`

**Interfaces:**
- Consumes: Tasks 1-4 shared APIs and the legacy Stage 1 architecture as a compatibility reference.
- Produces: `build_binary_model()`, CLI audit/prepare modes, corrected grouped five-fold Stage 1 training, fold bundles, and pooled binary OOF outputs.

- [ ] **Step 1: Add Stage 1 contract tests**

Assert `crying_baby` target 20 is excluded, ESC-50 is grouped by source file, augmentation occurs after fold assignment on Training only, per-fold normalizers are saved, the binary label order is fixed, and importing the script does not train.

- [ ] **Step 2: Run Stage 1 contract tests**

Run: `python -m unittest tests.test_train_binary_contract -v`
Expected: FAIL because the active script does not exist.

- [ ] **Step 3: Implement corrected Stage 1 using the legacy architecture**

Preserve the CNN/BiLSTM/Attention topology and Stage 1 threshold-independent softmax output. Use canonical infant hashes and ESC-50 source groups, exact per-fold target balancing, no Mixup by default, `val_loss` checkpoint selection, and original-only OOF metrics.

- [ ] **Step 4: Run Stage 1 contract, compilation, and audit-only checks**

Run: `python -m unittest tests.test_train_binary_contract -v`
Run: `python -m py_compile Models_dbl/binary/train_binary_dbl.py`
Run: `python Models_dbl/binary/train_binary_dbl.py --audit-only`
Expected: tests and compile PASS; audit identifies and excludes all 40 ESC-50 target-20 files without model training.

### Task 7: Verification and handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-08-16-corrected-two-stage-training.md`
- Create: `docs/corrected_training_verification.md`

**Interfaces:**
- Consumes: all tests and entry points.
- Produces: an evidence table of PASS, FAIL, and NOT RUN checks plus commands for the user's ML environment.

- [ ] **Step 1: Run the full dependency-light suite and static searches**

Run: `python -m unittest discover -s tests -v`
Run: `python -m compileall -q cryinsight Models_dbl/Main/train_main_dbl.py Models_dbl/binary/train_binary_dbl.py`
Search for forbidden active-protocol terms and constructs: fixed `AUG_TIMES`, independent-test claims, best-fold publication selection, and validation augmentation.

- [ ] **Step 2: Verify the user working tree and legacy artefacts remain untouched**

Compare `git status --short` and hashes of pre-existing model artefacts against the audit baseline. Only source/tests/docs files from this plan may be newly modified.

- [ ] **Step 3: Record unavailable ML checks honestly**

Mark TensorFlow/librosa imports, audio feature tests, `--prepare-only`, mini-training, five-fold training, corrected accuracy, and OOF metrics `NOT RUN` when the required environment is absent. Include the exact commands to run after activating the TensorFlow 2.15 environment.

- [ ] **Step 4: Review against the master prompt**

Confirm split-before-augmentation, group/hash isolation, exact target counts, per-fold normalizers and models, immutable run directories, original-only OOF contracts, no external-validation claim, and Stage 1 semantic exclusion are all represented by code and tests.
