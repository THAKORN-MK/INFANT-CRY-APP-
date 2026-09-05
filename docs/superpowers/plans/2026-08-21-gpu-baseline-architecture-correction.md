# GPU Baseline Architecture Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox.

**Goal:** Correct the Stage 1/Stage 2 time axis, add an auditable TensorFlow GPU runtime, leakage-safe normalization and feature caching, and provide reproducible baseline/ablation/evaluation tooling without starting a full training run.

**Architecture:** Keep the corrected grouped five-fold and final-refit protocol intact. Move model construction and runtime policy into shared lazy-import modules, preserve time through asymmetric pooling, explicitly transpose CNN output to time-major form before BiLSTM, and make every new experiment consume immutable shared fold assignments. Extend normalizer artefacts compatibly so existing scalar bundles remain readable while new runs use train-only per-feature-bin statistics.

**Tech Stack:** Python 3.10, TensorFlow 2.21/Keras 3, NumPy 2.2, librosa 0.11, scikit-learn 1.7, matplotlib, unittest, WSL2 Ubuntu 22.04, RTX 5070 Ti.

**Spec:** `docs/superpowers/specs/2026-08-21-gpu-baseline-architecture-correction-design.md`

## Global Constraints

- [ ] Do not start full fold training; only unit tests and tiny synthetic forward/backward/save-load smoke tests are allowed.
- [ ] Do not modify, delete, or overwrite any existing immutable run or Dataset file.
- [ ] Preserve split-before-augmentation, original-only Validation/OOF, exact OOF coverage, and final-refit deployment selection.
- [ ] Do not implement external-audio input contracts, windowing, quality rejection, codec handling, or Webapp routing (deferred item 8).
- [ ] Preserve the user's existing uncommitted edits in `README.md`, `Architecture.md`, and `Report/runs/report_01_20260816.md`.
- [ ] Keep trainer imports side-effect-free when TensorFlow is unavailable.

---

## Task 1: Lock Current Behavior and GPU Runtime Contract

**Files:**

- Create: `tests/test_runtime_device.py`
- Create: `cryinsight/runtime/__init__.py`
- Create: `cryinsight/runtime/device.py`
- Modify: `Models_dbl/binary/train_binary_dbl.py`
- Modify: `Models_dbl/Main/train_main_dbl.py`

- [ ] Add failing tests for `auto|gpu|cpu`, `--require-gpu` fail-fast behavior, explicit mixed precision, memory growth, and a JSON-safe runtime manifest.
- [ ] Run `python -m unittest tests.test_runtime_device` and confirm failure because the runtime module is absent.
- [ ] Implement a TensorFlow-lazy `configure_tensorflow_runtime()` returning device, versions, GPU names, compute capability when available, and precision policy.
- [ ] Add `--device`, `--require-gpu`, and `--mixed-precision` to both trainers and validate contradictory flags before any run directory is created.
- [ ] Persist `environment.json` inside every new Stage run and include its hash in verification/manifest inputs.
- [ ] Run runtime tests and both trainer import-contract tests.
- [ ] Commit: `feat: add auditable TensorFlow GPU runtime`

## Task 2: Correct Stage 1 and Stage 2 Time Sequence Architecture

**Files:**

- Create: `cryinsight/models/__init__.py`
- Create: `cryinsight/models/attention.py`
- Create: `cryinsight/models/stage1_model.py`
- Create: `cryinsight/models/stage2_model.py`
- Create: `tests/test_model_architecture.py`
- Modify: `Models_dbl/binary/train_binary_dbl.py`
- Modify: `Models_dbl/Main/train_main_dbl.py`

- [ ] Add TensorFlow-optional tests asserting Stage 1 input `(120,128,1)` becomes time sequence length 32 after pooling `(2,2),(2,2),(2,1)`.
- [ ] Add tests asserting Stage 2 input `(196,128,1)` becomes time sequence length 32 after pooling `(2,2),(2,2),(2,1),(2,1)`.
- [ ] Assert the explicit transpose order maps CNN `[feature,time,channel]` to `[time,feature,channel]` before reshape/BiLSTM.
- [ ] Implement serializable shared Attention and lazy model builders; force the final Softmax output to float32 under mixed precision.
- [ ] Retain trainer wrapper functions `build_binary_model()` and `build_main_model()` for compatibility while delegating to shared builders.
- [ ] Add Stage 2 `corrected_single_branch` and `corrected_multi_branch` builders with documented feature block boundaries `(0:120, 120:184, 184:196)`.
- [ ] Run architecture, trainer contract, Keras build, forward/backward, and `.keras` save/load tests.
- [ ] Commit: `fix: make BiLSTM consume the time axis`

## Task 3: Add Leakage-Safe Per-Feature Normalization

**Files:**

- Modify: `cryinsight/training/artefacts.py`
- Create: `tests/test_feature_normalizer.py`
- Modify: `tests/test_training_artefacts.py`
- Modify: `Models_dbl/binary/train_binary_dbl.py`
- Modify: `Models_dbl/Main/train_main_dbl.py`

- [ ] Add failing tests for `global_scalar` backward compatibility and `per_feature_bin` statistics with mean/std shape `(feature_bins,1,1)`.
- [ ] Add a test proving Validation/Test values do not influence fitted statistics.
- [ ] Implement `fit_normalizer()` and `apply_normalizer()` with finite/shape checks and zero-variance protection.
- [ ] Extend normalizer metadata with mode, value shape, feature order/block boundaries, preprocessing version, and SHA-256 while retaining old scalar bundle loading.
- [ ] Replace direct scalar fitting/arithmetic in all fold and final-refit paths with the shared helpers.
- [ ] Verify old fixture normalizers still load and new array normalizers round-trip.
- [ ] Commit: `feat: add train-only per-feature normalization`

## Task 4: Add Immutable Feature Cache

**Files:**

- Create: `cryinsight/training/feature_cache.py`
- Create: `tests/test_feature_cache.py`
- Modify: `cryinsight/audio/features.py`
- Modify: `Models_dbl/binary/train_binary_dbl.py`
- Modify: `Models_dbl/Main/train_main_dbl.py`

- [ ] Add failing tests for keys containing source SHA-256, preprocessing hash/version, augmentation parameters/seed, dtype, and shape.
- [ ] Add tests that corrupted arrays, metadata hash mismatch, or incompatible config are rejected rather than silently reused.
- [ ] Implement atomic, immutable cache writes and verified reads without changing record/group identity.
- [ ] Add optional `--feature-cache-dir` and `--no-feature-cache`; keep augmented entries separated by derivative plan identity.
- [ ] Record cache configuration/hash summary in run environment and fold manifests.
- [ ] Run cache, feature contract, and protocol tests.
- [ ] Commit: `feat: cache verified audio features`

## Task 5: Add Shared Fold Baseline and Ablation Framework

**Files:**

- Create: `Models_dbl/experiments/README.md`
- Create: `Models_dbl/experiments/configs/baselines.json`
- Create: `Models_dbl/experiments/configs/ablations.json`
- Create: `Models_dbl/experiments/run_experiments.py`
- Create: `cryinsight/training/experiments.py`
- Create: `tests/test_experiment_protocol.py`

- [ ] Add failing tests that all experiments consume the same original record IDs, group assignments, label order, fold count, and fold-assignment SHA-256.
- [ ] Implement immutable experiment directories and protocol manifests with `--audit-only`, `--experiment`, `--seed`, and repeated `--seeds` support.
- [ ] Register Stage 1 baselines: majority, MFCC-summary SVM, Log-Mel small CNN.
- [ ] Register Stage 2 baselines: majority, MFCC-summary SVM, Log-Mel small CNN, local YAMNet embedding plus linear/MLP head.
- [ ] Register ablations: CNN only, CNN+BiLSTM, corrected Attention model, feature blocks, normalization, waveform augmentation, and Mixup.
- [ ] Ensure candidates are selected by grouped OOF only and held-out Test is unavailable to experiment ranking.
- [ ] Run experiment protocol tests and an audit-only experiment command; do not train.
- [ ] Commit: `feat: add shared-fold baseline and ablation registry`

## Task 6: Add Publication Evaluation Outputs

**Files:**

- Create: `cryinsight/evaluation/__init__.py`
- Create: `cryinsight/evaluation/curves.py`
- Create: `cryinsight/evaluation/cascade.py`
- Create: `tests/test_publication_evaluation.py`
- Modify: `cryinsight/training/artefacts.py`
- Modify: `Models_dbl/binary/train_binary_dbl.py`
- Modify: `Models_dbl/Main/train_main_dbl.py`

- [ ] Add tests for finite renormalized class probabilities to eliminate scikit-learn probability-sum warnings.
- [ ] Add tests for binary/multiclass ROC and PR table/plot generation from immutable prediction CSVs.
- [ ] Add tests for end-to-end Stage 1 gate plus Stage 2 route aggregation, including Stage 1 false rejects.
- [ ] Implement prediction-probability validation/renormalization within numeric tolerance before log-loss/Brier/ECE calculations.
- [ ] Save ROC/PR CSV/PNG artefacts and hashes in completed run verification.
- [ ] Add a standalone cascade evaluator that accepts completed Stage 1/Stage 2 run directories and never retrains models.
- [ ] Run publication evaluation and existing artefact metric tests with warnings treated as errors.
- [ ] Commit: `feat: add publication curves and cascade evaluation`

## Task 7: Reproducible GPU Environment and Commands

**Files:**

- Create: `requirements/gpu-py310.txt`
- Create: `requirements/gpu-environment-lock.txt`
- Modify: `file.txt`
- Create: `tests/test_documented_commands.py`

- [ ] Record direct Python dependencies separately from the full verified WSL environment lock.
- [ ] Include commands for WSL launch, VS Code WSL, venv activation, GPU verification, unit tests, audit-only Stage 1/2, experiment audit, and optional smoke test.
- [ ] State that CUDA libraries come from `tensorflow[and-cuda]`, the NVIDIA Windows driver remains host-managed, and PTX JIT on compute capability 12.0a can make the first run slow.
- [ ] Ensure no documented command starts full training unintentionally.
- [ ] Run command/document tests.
- [ ] Commit: `docs: lock WSL TensorFlow GPU environment`

## Task 8: Update README, Architecture, and Report Hub

**Files:**

- Modify: `README.md`
- Modify: `Architecture.md`
- Modify: `Report/report.md`
- Create: `Report/experiments/baseline_report.md`
- Create: `Report/experiments/ablation_report.md`
- Do not modify: `README_OLD.md`

- [ ] Update Python 3.10/WSL2/GPU setup and verified runtime without claiming that GPU improves accuracy.
- [ ] Replace architecture diagrams/text with the explicit time-major transpose and time-preserving pooling shapes.
- [ ] Document single-branch versus multi-branch candidates, per-feature normalizer, cache, baselines, ablations, multiple seeds, and cascade evaluation.
- [ ] Add Report hub links to baseline/ablation reports and preserve the link to the first historical report.
- [ ] Keep publication limitations: no verified subject/session IDs, source confounding, no external validation, operational labels, and source ethics/license checks.
- [ ] Explicitly state that external-audio support remains undecided and unimplemented.
- [ ] Commit: `docs: describe corrected GPU experiment architecture`

## Task 9: Final Verification Without Full Training

**Files:**

- Modify only if verification exposes a defect in files above.

- [ ] Run the complete `python -m unittest discover -s tests -v` suite in the WSL GPU venv.
- [ ] Run TensorFlow GPU detection and confirm at least one physical GPU.
- [ ] Run tiny Stage 1 and Stage 2 synthetic forward/backward/save-load smoke tests and capture actual device placement.
- [ ] Run Stage 1 and Stage 2 `--audit-only` against the configured data without creating run directories.
- [ ] Scan for `TODO|TBD|placeholder`, external-audio modules, and accidental training processes.
- [ ] Review `git diff --check`, `git status`, and all documentation links.
- [ ] Confirm no Dataset or previous run changed and no Full Train was launched.
- [ ] Commit any test-only fixes, then report exact passing/skipped test counts and remaining publication limitations.

## Spec Coverage Review

- [ ] GPU runtime/device manifest: Tasks 1, 7, 9.
- [ ] Time-axis and time-preserving pooling: Task 2.
- [ ] Single/multi-branch Stage 2: Task 2.
- [ ] Train-only normalizer and manifests: Task 3.
- [ ] Feature cache and invalidation: Task 4.
- [ ] Shared-fold baseline/ablation and multiple seeds: Task 5.
- [ ] Probability diagnostics, ROC/PR, and end-to-end cascade: Task 6.
- [ ] README/Architecture/Report/file commands: Tasks 7-8.
- [ ] Immutable runs, no leakage, no Full Train: Global Constraints and Task 9.
- [ ] Deferred external-audio item 8 remains absent: Global Constraints, Tasks 8-9.
