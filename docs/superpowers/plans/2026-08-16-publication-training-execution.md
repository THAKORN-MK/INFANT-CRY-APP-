# CryInsight Publication Training Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active Stage 1 and Stage 2 trainers complete the approved corrected grouped five-fold publication runs and produce traceable pooled original-record OOF evidence.

**Architecture:** The active entry points under `Models_dbl` retain the architectures and pre-specified hyperparameters found in `models_dbl_OLD`. Shared `cryinsight.audio` and `cryinsight.training` modules own the versioned feature boundary, grouped protocol, immutable run status, fold artefacts, and OOF evaluation so Stage 1 and Stage 2 cannot silently diverge.

**Tech Stack:** Python 3.10.11, TensorFlow/Keras 2.15.0, NumPy 1.26.4, librosa 0.11.0, scikit-learn 1.7.2, matplotlib 3.10.8, standard-library `unittest`.

## Global Constraints

- Project root remains exactly `D:/INFANT CRY`.
- `models_dbl_OLD` is read-only reference material; do not copy its models, logs, post-hoc metrics, or 99.x% claims into active publication outputs.
- Active trainers remain `Models_dbl/Main/train_main_dbl.py` and `Models_dbl/binary/train_binary_dbl.py`.
- Preserve the legacy CNN/BiLSTM/Attention topologies, feature composition, label orders, and pre-specified Stage-specific hyperparameters.
- Use five folds and seed 42; assign eligible originals/groups before all augmentation.
- Validation contains original records only; augmentation, Mixup, and normalizer fitting are Training-only.
- Every new attempt writes a distinct immutable child directory below the applicable
  `runs` directory and never resumes or overwrites an old run.
- Report only corrected pooled OOF metrics from all five selected checkpoints; do not publish a best-fold score.
- The user authorized full Stage 1 and Stage 2 training after regression, dependency, model, and data checks pass.
- Do not delete the failed Stage 1 run `20260815T181927Z_571bfde3`; retain it as incomplete evidence.
- Do not claim independent/external validation, calibrated confidence, clinical validity, or subject-independent validation.

---

### Task 1: Version and repair the short-audio feature boundary

**Files:**
- Modify: `tests/test_feature_contract.py`
- Modify: `cryinsight/audio/features.py`

**Interfaces:**
- Produces: `PreprocessingConfig.delta_width: int`, `PreprocessingConfig.short_audio_policy: str`, and `PreprocessingConfig.minimum_waveform_samples: int`.
- Preserves: `extract_features(waveform, sample_rate=22050, config=None) -> np.ndarray` with shape `(196, 128, 1)` and dtype `float32`.

- [ ] **Step 1: Add the failing regression and serialized-contract assertions**

Add tests equivalent to:

```python
def test_short_nonzero_waveform_is_right_padded_before_delta(self):
    config = PreprocessingConfig()
    t = np.arange(3072, dtype=np.float32) / config.sample_rate
    waveform = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    features = extract_features(
        waveform,
        sample_rate=config.sample_rate,
        config=config,
    )
    self.assertEqual(features.shape, (196, 128, 1))
    self.assertEqual(features.dtype, np.float32)
    self.assertTrue(np.isfinite(features).all())

def test_short_audio_policy_is_versioned(self):
    config = PreprocessingConfig()
    self.assertEqual(config.version, "cryinsight_features_v2")
    self.assertEqual(config.delta_width, 9)
    self.assertEqual(config.minimum_waveform_samples, 4096)
    self.assertEqual(config.short_audio_policy, "right_zero_pad_to_delta_width")
    self.assertEqual(config.to_dict()["minimum_waveform_samples"], 4096)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 -m unittest tests.test_feature_contract.LibrosaFeatureTests -v
```

Expected: the 3,072-sample test fails with `Audio is too short for the delta feature contract`, and the versioned fields are absent or still version 1.

- [ ] **Step 3: Implement the minimum shared waveform padding**

Change `PreprocessingConfig` to declare `delta_width=9`,
`short_audio_policy="right_zero_pad_to_delta_width"`, and version
`cryinsight_features_v2`. Derive the minimum with:

```python
@property
def minimum_waveform_samples(self) -> int:
    return (self.delta_width - 1) * self.hop_length
```

After `_validated_waveform()` and before MFCC/Mel/Chroma extraction, right-pad only
when needed:

```python
if audio.size < active.minimum_waveform_samples:
    audio = np.pad(
        audio,
        (0, active.minimum_waveform_samples - audio.size),
        mode="constant",
    ).astype(np.float32, copy=False)
```

Pass `width=active.delta_width` explicitly to both Delta calls and serialize the
derived minimum in `to_dict()`. Empty, non-finite, and zero-energy validation remains
unchanged.

- [ ] **Step 4: Run focused and complete feature tests and verify GREEN**

Run the focused command from Step 2, then:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 -m unittest tests.test_feature_contract -v
```

Expected: all feature contract tests pass without skipping the librosa cases.

### Task 2: Record contextual audio failures and immutable incomplete runs

**Files:**
- Modify: `tests/test_feature_contract.py`
- Modify: `tests/test_training_artefacts.py`
- Modify: `tests/test_train_main_contract.py`
- Modify: `tests/test_train_binary_contract.py`
- Modify: `cryinsight/audio/features.py`
- Modify: `cryinsight/training/artefacts.py`
- Modify: `Models_dbl/Main/train_main_dbl.py`
- Modify: `Models_dbl/binary/train_binary_dbl.py`

**Interfaces:**
- Produces: `write_incomplete_run_verification(run_dir, *, stage, error) -> Path`.
- Preserves: immutable write-once `verification.json`; successful runs still write `status="complete"` only after all five folds and exact OOF coverage finish.

- [ ] **Step 1: Add failing tests for source context and incomplete status**

Create a temporary zero-energy WAV-backed `OriginalRecord` and assert that
`extract_fold_tensors()` raises an `AudioContractError` containing its `record_id`
and absolute source path. Add an artefact test equivalent to:

```python
with tempfile.TemporaryDirectory() as directory:
    run_dir = Path(directory) / "run_a"
    (run_dir / "fold_1").mkdir(parents=True)
    write_json_atomic(run_dir / "fold_1" / "fold_manifest.json", {"fold": 1})
    path = write_incomplete_run_verification(
        run_dir,
        stage="stage1_binary_baby_gate",
        error=AudioContractError("bad record"),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    self.assertEqual(payload["status"], "incomplete")
    self.assertEqual(payload["folds_completed"], 1)
    self.assertEqual(payload["error_type"], "AudioContractError")
    self.assertEqual(payload["accuracy_status"], "NOT_AVAILABLE_INCOMPLETE_RUN")
```

Extend both entry-point contract tests to require a guarded `run_training()` call
that invokes `write_incomplete_run_verification()` before re-raising.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 -m unittest tests.test_feature_contract tests.test_training_artefacts tests.test_train_main_contract tests.test_train_binary_contract -v
```

Expected: tests fail because contextual wrapping and the incomplete writer do not
exist.

- [ ] **Step 3: Implement contextual errors and incomplete verification**

Wrap original, augmented, and validation feature materialization errors with
`sample_kind`, `sample_id`, and `source_path`, preserving the original exception with
`raise contextual_error from exc`.

Implement `write_incomplete_run_verification()` in `artefacts.py`. Count completed
folds only from existing `fold_*/fold_manifest.json` files, write UTC completion time,
stage, error type/message, `training_started=true`, and
`accuracy_status="NOT_AVAILABLE_INCOMPLETE_RUN"` using the existing write-once JSON
writer.

In both active entry points, wrap only the `run_training()` call:

```python
try:
    metrics = run_training(
        run_dir=run_dir,
        source_snapshot=source_snapshot,
        resolution=resolution,
        split=split,
        preprocessing=preprocessing,
        args=args,
    )
except Exception as exc:
    write_incomplete_run_verification(
        run_dir,
        stage="stage1_binary_baby_gate",  # Stage 2 uses its own stage identifier
        error=exc,
    )
    raise
```

- [ ] **Step 4: Re-run the focused suite and verify GREEN**

Run the command from Step 2. Expected: every focused test passes.

### Task 3: Verify the active trainers against the legacy reference and real audio

**Files:**
- Modify only if a check fails: `Models_dbl/Main/train_main_dbl.py`
- Modify only if a check fails: `Models_dbl/binary/train_binary_dbl.py`
- Modify: `docs/corrected_training_verification.md`

**Interfaces:**
- Consumes: the corrected shared contracts from Tasks 1-2.
- Produces: evidence that active model shapes, label orders, Stage-specific defaults,
  serialization, and the three known short records are training-ready.

- [ ] **Step 1: Run dependency, compilation, and complete unit checks**

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -m pip check
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 -m compileall -q cryinsight tests Models_dbl/Main/train_main_dbl.py Models_dbl/binary/train_binary_dbl.py
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 -m unittest discover -s tests -v
```

Expected: no broken requirements, compilation exit 0, and the complete suite passes.

- [ ] **Step 2: Check the three real short records with feature contract v2**

Load, trim, normalize, and extract all three records below with the shared API:

```text
data_set_dbl/not_baby/3-216284-A-39.wav
data_set_dbl/not_baby/3-216281-A-39.wav
data_set_dbl/not_baby/2-83688-A-34.wav
```

Expected for each: finite `float32` tensor `(196, 128, 1)` with no skip or exception.

- [ ] **Step 3: Build and serialize both legacy-compatible topologies**

Build Main with `(196, 128, 1) -> 5` and Binary with `(196, 128, 1) -> 2`, save each
to a temporary `.keras` path, reload with the registered Attention custom object, and
complete one zero-batch forward pass. Do not call `fit()`.

Expected: input/output shapes match the legacy reference and both reloaded models
return finite scores with the correct output dimension.

- [ ] **Step 4: Update verification evidence without inventing metrics**

Record exact commands, package versions, test count, feature-v2 evidence, and model
smoke results. Keep corrected OOF accuracy as `NOT AVAILABLE` until Tasks 4-5 finish.

### Task 4: Execute and verify the corrected Stage 1 publication run

**Files:**
- Create at runtime: one generated immutable child below `Models_dbl/binary/runs/`
- Modify after completion: `docs/corrected_training_verification.md`

**Interfaces:**
- Consumes: 3,311 eligible originals, 2,857 groups, label order `not_baby, baby`.
- Produces: five selected Binary checkpoints with matching normalizers and pooled
  original OOF predictions/metrics.

- [ ] **Step 1: Start a new immutable Stage 1 run**

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 Models_dbl/binary/train_binary_dbl.py --train
```

Do not reuse `20260815T181927Z_571bfde3`. Monitor fold/epoch output; an unchanged long
epoch is not a failure.

- [ ] **Step 2: Stop on any failed or incomplete Stage 1 run**

If the process exits nonzero, verify the new run has `verification.json` with
`status="incomplete"`, preserve the run, diagnose from the recorded source context,
and do not begin Stage 2.

- [ ] **Step 3: Verify the complete Stage 1 artefact contract**

Require `verification.json` status `complete`, fold manifests 1-5, matching file
hashes, five selected checkpoints and normalizers, OOF support exactly 3,311 unique
original record IDs, zero augmented OOF rows, and no missing/duplicate OOF IDs.

- [ ] **Step 4: Record only the actual Stage 1 pooled OOF result**

Read metrics from the completed run's `oof_metrics.json`; record run ID, sample count,
pooled metrics, per-fold metrics, mean/SD, group-bootstrap intervals, confusion matrix
paths, and the internal-validation-only limitation.

### Task 5: Execute and verify the corrected Stage 2 publication run

**Files:**
- Create at runtime: one generated immutable child below `Models_dbl/Main/runs/`
- Modify after completion: `docs/corrected_training_verification.md`

**Interfaces:**
- Consumes: 1,348 eligible originals, label order
  `belly_pain, burping, discomfort, hungry, tired`.
- Produces: five selected Main checkpoints with matching normalizers and pooled
  original OOF predictions/metrics.

- [ ] **Step 1: Start a new immutable Stage 2 run after Stage 1 passes**

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -X utf8 Models_dbl/Main/train_main_dbl.py --train
```

- [ ] **Step 2: Stop on any failed or incomplete Stage 2 run**

If the process exits nonzero, preserve its immutable run, verify explicit incomplete
status, diagnose the recorded source context, and do not report corrected accuracy.

- [ ] **Step 3: Verify the complete Stage 2 artefact contract**

Require `verification.json` status `complete`, fold manifests 1-5, matching hashes,
five selected checkpoints and normalizers, OOF support exactly 1,348 unique original
record IDs, zero augmented OOF rows, and no missing/duplicate OOF IDs.

- [ ] **Step 4: Record only the actual Stage 2 pooled OOF result**

Read the completed run's `oof_metrics.json`; record run identity, pooled/per-fold
metrics, mean/SD, group-bootstrap intervals, per-class report, confusion matrix paths,
and the limitation that infant subject/session identities were unavailable.

### Task 6: Publish verified documentation and final evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/corrected_training_verification.md`

**Interfaces:**
- Consumes: completed Stage 1 and Stage 2 run IDs and immutable metrics files.
- Produces: publication-ready current-state documentation with no legacy-result
  substitution.

- [ ] **Step 1: Replace legacy README claims with verified current state**

Keep the 2-Stage concept and architecture explanation, but remove 99.x% claims,
nonexistent active file paths, `independent test` wording, calibrated-confidence
wording, and bare `python train_*.py` commands. Document corrected grouped five-fold
internal validation, split-before-augmentation, target-based Training-only
augmentation, per-fold bundles, pooled OOF evaluation, operational labels, and
medical/generalizability limitations.

- [ ] **Step 2: Add run-specific evidence**

Reference only the two completed run IDs, their `verification.json`,
`oof_metrics.json`, fold manifests, OOF predictions, and confusion matrices. Clearly
separate legacy post-hoc files and the failed Stage 1 attempt from corrected results.

- [ ] **Step 3: Run final verification**

Re-run `pip check`, compilation, the complete unit suite, file-hash verification for
all ten fold bundles, exact OOF coverage checks, and `git diff --check`. Confirm no
legacy model/log/dataset was modified or restored.

- [ ] **Step 4: Deliver the final research-training report**

Report outcome, source resolution, files changed, before/after flow, data evidence,
artefact evidence, verification checks, actual metrics, limitations, and smallest
next action. Explicitly state that results are corrected internal validation and that
independent/external validation, calibration, and subject-independent validation have
not been performed.
