# Stage 2 WSL Checkpoint Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent repeated Keras checkpoint overwrites on the WSL-mounted Windows drive from aborting Stage 2 before all five folds complete.

**Architecture:** ModelCheckpoint writes its repeatedly replaced best model to a run-scoped directory on the native Linux filesystem. After `model.fit` finishes, one verified copy is published to the immutable Windows-backed run folder and then loaded for validation. Temporary Linux checkpoint data is removed by a context-managed staging object on success or failure.

**Tech Stack:** Python 3.10, TensorFlow/Keras, pathlib, tempfile, shutil, SHA-256, unittest, WSL 2.

**Spec:** User approval in the active task and the recorded `OSError: [Errno 22] Invalid argument` from Stage 2 Fold 1 Epoch 22.

## Global Constraints

- Do not modify or delete the completed Stage 1 run.
- Delete only `Models_dbl/Main/runs/20260821T164332Z_490383ff` after confirming its verification status is `incomplete`.
- Preserve immutable publication artefacts: a completed run path is never overwritten.
- Do not use held-out Test data for checkpoint or epoch selection.
- Checkpoint staging must work without importing TensorFlow in audit-only mode.

---

### Task 1: Native Linux checkpoint staging

**Files:**
- Create: `cryinsight/training/checkpoint_staging.py`
- Test: `tests/test_checkpoint_staging.py`

**Interfaces:**
- Produces: `CheckpointStaging(run_id: str, fold_name: str, filename: str, staging_root: Path | None = None)` context manager.
- Produces: `CheckpointStaging.local_path: Path` for Keras ModelCheckpoint.
- Produces: `CheckpointStaging.publish(destination: Path) -> dict[str, str | int]` which copies once and verifies size plus SHA-256.

- [ ] **Step 1: Write the failing test**

```python
def test_publish_copies_selected_checkpoint_and_verifies_identity():
    with CheckpointStaging("run1", "fold_1", "model.keras", staging_root=root) as staging:
        staging.local_path.write_bytes(b"selected checkpoint")
        manifest = staging.publish(destination)
    assert destination.read_bytes() == b"selected checkpoint"
    assert manifest["sha256"] == hashlib.sha256(b"selected checkpoint").hexdigest()
    assert not staging.directory.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_checkpoint_staging -v`

Expected: FAIL because `cryinsight.training.checkpoint_staging` does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement a context manager backed by `tempfile.mkdtemp` under `/home` or an explicit test root. Validate identifiers, require a non-empty source checkpoint, copy to a sibling temporary file, verify SHA-256 and size, then atomically replace the destination once.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_checkpoint_staging -v`

Expected: PASS with temporary staging directory removed.

### Task 2: Use staging in Stage 2 folds and final refit

**Files:**
- Modify: `Models_dbl/Main/train_main_dbl.py`
- Modify: `tests/test_train_main_contract.py`

**Interfaces:**
- Consumes: `CheckpointStaging` from Task 1.
- Produces: each `fold_N/fold_N_main_dbl.keras` only after `model.fit` has selected the best Linux-local checkpoint.
- Produces: `best_model/best_model_main_dbl.keras` through the same one-time publication path.

- [ ] **Step 1: Write the failing behavior test**

Add a contract test which imports the Stage 2 trainer and verifies its checkpoint-path resolver rejects a `/mnt/*` staging root and returns a native Linux path when running under WSL.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m unittest tests.test_checkpoint_staging tests.test_train_main_contract -v`

Expected: FAIL before the trainer consumes `CheckpointStaging`.

- [ ] **Step 3: Replace direct run-folder ModelCheckpoint writes**

For every fold, pass `staging.local_path` to Keras `ModelCheckpoint`, publish once to `model_path` after fit, record publication metadata, then load the published file. Apply the same staging boundary to the final-refit model save.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_checkpoint_staging tests.test_train_main_contract tests.test_training_artefacts -v`

Expected: PASS.

### Task 3: Verification and retry readiness

**Files:**
- Modify: `README.md`
- Modify: `Architecture.md`
- Modify: `file.txt`

**Interfaces:**
- Documents: Linux-local checkpoint staging and one-time publication into immutable run artefacts.

- [ ] **Step 1: Run Stage 2 non-training contracts**

Run: `python -m unittest tests.test_run_pairing tests.test_stage_run_pairing_integration tests.test_train_main_contract tests.test_checkpoint_staging -v`

Expected: all tests PASS.

- [ ] **Step 2: Run a small real Keras checkpoint smoke test in WSL**

Save and reload a small model through `CheckpointStaging` without starting full training.

Expected: published `.keras` loads and predicts with the expected output shape.

- [ ] **Step 3: Confirm retry path is free**

Verify the completed Binary run remains present and the matching Stage 2 run directory does not exist.

- [ ] **Step 4: Document the exact rerun command**

Use the existing GPU Stage 2 command without `--prepare-only`; automatic pairing must resolve `20260821T164332Z_490383ff`.
