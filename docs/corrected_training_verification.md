# CryInsight Corrected Training Verification

Date: 2026-08-16  
Scope: Stage 2 `train_main_dbl.py` and Stage 1 `train_binary_dbl.py`

## Outcome

Both entry points now implement the corrected grouped five-fold internal-validation
contract. They discover and hash original records before splitting, resolve exact
duplicates, assign validation folds before augmentation, augment Training only to a
dynamic per-fold target, fit one normalizer per fold from Training features only,
select every fold checkpoint with `val_loss`, and pool predictions from original OOF
records only.

The scripts do not replace stable legacy model files. New artefacts are written only
under a newly created `models_dbl/<stage>/runs/<run_id>/` directory.

## Verification evidence

| Check | Result | Evidence |
|---|---|---|
| Python compilation | PASS | `python -m compileall -q cryinsight tests Models_dbl/Main/train_main_dbl.py Models_dbl/binary/train_binary_dbl.py` exited 0 |
| Full unit suite in project Python 3.10 | PASS | 39 tests ran and passed; 0 skipped |
| Stage 2 real-data audit | PASS | 1,551 candidates; 1,348 eligible; 197 same-label duplicate rows and 6 cross-label conflict rows excluded |
| Stage 1 real-data audit | PASS | 3,551 candidates; 3,311 eligible; all 40 ESC-50 target-20 files excluded; 200 same-label duplicate rows excluded |
| Legacy leakage constructs in active scripts | PASS | No fixed augmentation multiplier, legacy split path, publication best-fold selection, or class-weight fit path found |
| Audit-only write isolation | PASS | Contract tests confirm audit mode creates no stage/run directory |
| New publication run created | NOT RUN | Audit mode only; no `runs/<run_id>` was created |
| Grouped split execution | PASS | Deterministic grouped five-fold tests ran with scikit-learn 1.7.2 |
| Librosa feature extraction | PASS | The real feature extractor returned the frozen `(196, 128, 1)` float32 tensor with librosa 0.11.0 |
| TensorFlow model construction and serialization | PASS | TensorFlow 2.15.0 built both models, saved/reloaded temporary `.keras` checkpoints, and completed one forward pass without training |
| Full corrected five-fold training | NOT RUN | Requires separate explicit approval because it is time/GPU intensive |
| Corrected pooled OOF accuracy | NOT AVAILABLE | It may be reported only after all five selected checkpoints produce exact-once original OOF predictions |
| Independent/external validation | NOT PERFORMED | No independent cohort is established in the current repository |

The complete suite ran under
`C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe` (Python
3.10.11). The environment contains TensorFlow 2.15.0, librosa 0.11.0,
scikit-learn 1.7.2, and NumPy 1.26.4. Protocol, grouped splitting, real feature
extraction, metric calculation, duplicate handling, augmentation planning,
normalizer identity, OOF support, immutable writers, both CLI contracts, and
audit-only behavior all ran in this environment.

## Real-data audit details

### Stage 2

Eligible original counts after the frozen duplicate policy:

- `belly_pain`: 150
- `burping`: 230
- `discomfort`: 129
- `hungry`: 185
- `tired`: 654

The only reliable infant grouping evidence currently available is exact-content
SHA-256. No verified subject/session identifier was found. The result must therefore
be described as clip-level corrected internal validation with exact-content leakage
protection, not subject-independent validation.

### Stage 1

Eligible original counts after semantic exclusion and exact deduplication:

- `not_baby`: 1,960
- `baby`: 1,351

ESC-50 negatives are grouped with the second filename component (`source_file`).
ESC-50 target 20 (`crying_baby`) is always excluded from `not_baby` before fold
assignment.

## Accuracy status

There is no defensible corrected accuracy estimate yet. Existing legacy results were
produced by a different, leakage-prone protocol and must not be used as an estimate of
the corrected grouped-OOF result. The new scripts write the actual pooled accuracy,
balanced accuracy, macro/weighted metrics, ROC-AUC where defined, per-class report,
confusion matrix, per-fold estimates, mean/SD, and group-bootstrap intervals only after
all five folds complete.

## Commands for the project ML environment

The verified Python executable is shown explicitly because `python` is not currently
available on the Codex PowerShell `PATH`.

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --audit-only
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --audit-only
```

To create an inspection-only immutable fold manifest run:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --prepare-only --run-id stage2_prepare_20260816
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --prepare-only --run-id stage1_prepare_20260816
```

A prepared-only run is immutable and is not resumed for training. Use a new run ID for
an explicitly authorized full run:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --train --run-id stage2_corrected_20260816
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --train --run-id stage1_corrected_20260816
```

## Working-tree caveat

During final verification, Git reported tracked legacy Stage 2 fold models, logs,
labels, and normalizers as deleted. No delete, move, restore, or checkout command was
issued as part of this implementation. Those external deletions were preserved and
are not presented as output from the corrected scripts.
