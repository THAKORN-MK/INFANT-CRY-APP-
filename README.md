# CryInsight — Two-stage Infant Cry Classification

CryInsight เป็นระบบวิเคราะห์ไฟล์เสียง `.wav` แบบ 2 Stage:

1. **Stage 1 — Binary Baby Gate:** ตรวจว่าเป็นเสียงทารกหรือไม่
2. **Stage 2 — Infant State Classifier:** เมื่อเป็นเสียงทารก จึงจำแนกเป็น 5 กลุ่ม

การเปรียบเทียบ Baseline/Ablation รุ่นใหม่ใช้ [Shared Experiment Engine](./Models_dbl/experiments/README.md) ซึ่งตรึง fold assignments จาก Pipeline Run `20260821T164332Z_490383ff` และจัดอันดับด้วย grouped OOF เท่านั้น ดูสถานะที่ [Experiment Report Hub](./Report/experiments/report.md) ส่วน trainer หลักยังไม่ถูกเปลี่ยนตาม Candidate ใดจนกว่าจะผ่าน Wave C และได้รับอนุมัติ Promotion แยกต่างหาก

สถานะการเก็บงานโค้ดและผลตรวจล่าสุดอยู่ใน [Inline Execution Status](./docs/superpowers/plans/2026-09-03-inline-execution-status.md) แยกจากรายงานผลทดลองจริง การผ่าน CPU tests ไม่ใช้แทนการตรวจ GPU/WSL

## Pipeline Run 1 — GPU baseline

ผลหลักปัจจุบันของโครงการคือ Pipeline Run `20260821T164332Z_490383ff` ซึ่งฝึก Stage 1 และ Stage 2 บน WSL2 Ubuntu 22.04 ด้วย GPU และ mixed precision จริง หลักฐาน runtime อยู่ใน `environment.json` ของทั้งสอง Stage และรายงานฉบับเต็มอยู่ที่ [Pipeline Run 1 Report](./Report/runs/report_01_20260821.md)

| รายการ | Stage 1 — Binary Baby Gate | Stage 2 — Five-class Infant State |
|---|---:|---:|
| OOF accuracy | 98.93% | 91.10% |
| Locked internal Test accuracy | 99.42% | 92.41% |
| Locked internal Test Macro F1 | 99.41% | 91.02% |
| Final Test support | 694 | 303 |

ตัวเลขนี้เป็นผลจาก immutable run ที่ `verification.json` มีสถานะ `complete` และเป็น internal held-out evaluation จาก corpus เดียวกัน ไม่ใช่ external validation ผลของ baseline/ablation จะรายงานแยกเมื่อ Experiment Run เทรนและ verification ครบแล้ว

## System overview

```mermaid
flowchart TB
    A(["ไฟล์เสียง .wav"]) --> B["Audio preprocessing<br/>22,050 Hz · Mono · Trim · Normalize"]
    B --> C["Stage 1 — Binary Baby Gate<br/>MFCC + Delta + Delta2<br/>CNN → BiLSTM → Attention"]
    C --> D{"ผลการจำแนก Stage 1"}
    D -->|"not_baby"| E["หยุดการประมวลผล<br/>แจ้งว่าไม่ใช่เสียงทารก"]
    D -->|"baby"| F["Stage 2 — Infant State Classifier<br/>MFCC + Delta + Delta2 + Log-Mel + Chroma<br/>CNN → BiLSTM → Attention"]
    F --> G(["ผลลัพธ์ 5 กลุ่ม<br/>belly_pain · burping · discomfort · hungry · tired"])

    classDef input fill:#F8FAFC,stroke:#334155,color:#0F172A,stroke-width:1.5px;
    classDef process fill:#F1F5F9,stroke:#475569,color:#0F172A,stroke-width:1.5px;
    classDef stage fill:#EFF6FF,stroke:#1D4ED8,color:#172554,stroke-width:2px;
    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#431407,stroke-width:1.5px;
    classDef stop fill:#FEF2F2,stroke:#B91C1C,color:#450A0A,stroke-width:1.5px;
    classDef result fill:#F0FDF4,stroke:#15803D,color:#052E16,stroke-width:2px;

    class A input;
    class B process;
    class C,F stage;
    class D decision;
    class E stop;
    class G result;

    linkStyle default stroke:#64748B,stroke-width:1.5px;
```

แผนภาพนี้แสดง inference flow ของระบบ ส่วน augmentation, 5-fold validation และ held-out testing เป็นขั้นตอนเฉพาะระหว่างการพัฒนาโมเดล

ผลการฝึกและประเมินโมเดลแต่ละครั้งรวบรวมอยู่ที่ [Experiment Report Hub](./Report/report.md) ส่วนรายละเอียดเชิงเทคนิค เหตุผลของ feature แต่ละชนิด โครงสร้าง neural network และข้อจำกัดสำหรับงานวิจัยอยู่ใน [Architecture.md](./Architecture.md)

## Environment หลัก

โปรเจกต์นี้กำหนด **Python 3.10** เป็น environment หลัก

| รายการ | เวอร์ชันหลัก |
|---|---|
| Python | **3.10** |
| TensorFlow / Keras | **2.21.0 / 3.12.4** |
| NumPy | **2.2.6** |
| librosa | 0.11.0 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| GPU ที่ตรวจแล้ว | NVIDIA GeForce RTX 5070 Ti ผ่าน WSL2 |
| Pipeline Run 1 runtime | WSL2 Ubuntu 22.04 · GPU:0 · mixed precision |

เหตุผลที่ใช้ Python 3.10:

- เป็นเวอร์ชันที่ตรวจสอบร่วมกับ TensorFlow 2.21.0/Keras 3 ในโปรเจกต์นี้แล้ว
- ลดปัญหาความเข้ากันได้ระหว่าง TensorFlow, Keras, NumPy และไลบรารีเสียง
- ทำให้ environment สำหรับฝึกและโหลดโมเดลมี baseline เดียวกัน

ไม่ควรเปลี่ยนไปใช้ Python เวอร์ชันอื่นโดยไม่รัน unit tests, model-build smoke test และตรวจการ save/load โมเดลใหม่ทั้งหมด

## Environment สำหรับ GPU Training

GPU Training ใช้ WSL2 Ubuntu 22.04 และ venv ต่อไปนี้เป็น environment หลัก ไม่ใช้ TensorFlow ฝั่ง Windows สำหรับการเทรนรอบใหม่:

```bash
wsl -d Ubuntu-22.04 -u adminuser
cd "/mnt/d/INFANT CRY"
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
python -m pip install -r requirements/gpu-py310.txt
```

เลือก Python interpreter ใน VS Code เป็น:

```text
/home/adminuser/.venvs/audio-ml-gpu/bin/python
```

ตรวจ GPU ก่อนเทรน:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

ต้องพบ `GPU:0` ระบบใช้ CUDA/cuDNN ที่มากับ `tensorflow[and-cuda]`; NVIDIA driver อยู่ฝั่ง Windows ไม่ต้องติดตั้ง CUDA Toolkit แยกเพื่อโปรเจกต์นี้ RTX 5070 Ti อาจแสดง PTX JIT warning สำหรับ compute capability 12.0a และการเริ่มครั้งแรกอาจช้ากว่าปกติ

Pipeline Run 1 บันทึกว่าใช้ Python 3.10.12, TensorFlow 2.21.0, Keras 3.12.4, CUDA 12.5.1 และ cuDNN 9 โดยมี `requested_device: gpu`, `require_gpu: true`, `selected_device: gpu` และ `precision_policy: mixed_float16` ทั้ง Stage 1 และ Stage 2

รายการ dependency โดยตรงอยู่ใน `requirements/gpu-py310.txt` และ environment ที่ตรวจจริงทั้งหมดอยู่ใน `requirements/gpu-environment-lock.txt`

## โครงสร้างโปรเจกต์ที่ใช้งานอยู่

```text
INFANT CRY/
├── Architecture.md
├── README.md
├── Report/
│   ├── report.md                   # Experiment Report Hub
│   ├── runs/                       # รายงานถาวรของแต่ละการทดลอง
│   ├── experiments/                # สถานะ Baseline/Ablation
│   └── assets/                     # ภาพประกอบรายงาน
├── requirements/                   # GPU direct dependencies + environment lock
├── split_audio.py
│
├── data_set_dbl/                  # ข้อมูลต้นฉบับ
├── data_set_dbl_split/
│   ├── train/                     # ใช้สร้าง 5 folds และฝึก
│   └── test/                      # locked held-out test
│
├── Models_dbl/
│   ├── binary/
│   │   └── train_binary_dbl.py    # Stage 1
│   ├── Main/
│   │   └── train_main_dbl.py      # Stage 2
│   └── experiments/
│       ├── configs/               # registry configurations
│       ├── baselines/
│       │   ├── stage1/             # Majority, MFCC-SVM, Log-Mel CNN
│       │   └── stage2/             # Majority, MFCC-SVM, Log-Mel CNN
│       ├── ablations/              # CNN, Attention, features, augmentation
│       ├── runs/                   # immutable experiment runs เมื่อมีการรันจริง
│       └── run_experiments.py      # shared-fold experiment protocol
│
├── cryinsight/
│   ├── audio/features.py
│   ├── models/                     # corrected time-major model builders
│   ├── runtime/device.py           # GPU/CPU/mixed-precision policy
│   ├── evaluation/                 # ROC/PR และ cascade evaluation
│   └── training/
│       ├── protocol.py
│       ├── artefacts.py
│       ├── experiments.py
│       └── feature_cache.py
│
├── models_dbl_OLD/                # โค้ด/artefact รุ่นเดิม ใช้อ้างอิงเท่านั้น
└── tests/
```

trainer รุ่นปัจจุบันจะไม่เขียนทับโมเดลใน `models_dbl_OLD`

## Dataset

โฟลเดอร์ `data_set_dbl` มี 6 label directories:

```text
belly_pain/
burping/
discomfort/
hungry/
tired/
not_baby/
```

- Stage 1 รวม 5 infant labels เป็น `baby` และใช้ `not_baby` เป็นคลาสลบ
- Stage 2 ใช้เฉพาะ 5 infant labels
- ESC-50 หมวด `crying_baby` (target 20) ไม่ถูกนำมาเป็น `not_baby`

แหล่งข้อมูล [InfantCry-DBL Version 1](https://data.mendeley.com/datasets/x493z8nmwc/1) (DOI: `10.17632/x493z8nmwc.1`) ระบุ `metadata.csv` จำนวน 1,551 clips สำหรับ 5 infant labels ไฟล์เสียงในโครงการมีจำนวนตรงกันดังนี้:

| Label | จำนวนไฟล์ต้นฉบับ |
|---|---:|
| `belly_pain` | 170 |
| `burping` | 280 |
| `discomfort` | 168 |
| `hungry` | 217 |
| `tired` | 716 |
| **รวม InfantCry-DBL** | **1,551** |

เมื่อรวม `not_baby` จาก ESC-50 จำนวน 2,000 ไฟล์ โฟลเดอร์ `data_set_dbl` จึงมีทั้งหมด 3,551 ไฟล์ การแบ่ง 80/20 ปัจจุบันเก็บ candidate files ครบ 3,551 ไฟล์ ไม่มีไฟล์หายระหว่างการ split

## การแบ่งข้อมูล 80/20

สคริปต์ [split_audio.py](./split_audio.py) สุ่มแบ่งไฟล์รายคลาสจาก `data_set_dbl` ไปยัง:

```text
data_set_dbl_split/train
data_set_dbl_split/test
```

ค่าเริ่มต้นคือ train 80%, test 20% และ seed 42

หากยังไม่มี split:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' split_audio.py
```

หากปลายทางมีข้อมูล สคริปต์จะหยุดเพื่อป้องกันการเขียนทับ หากตั้งใจสร้างใหม่จริงให้ตรวจสอบปลายทางก่อนแล้วจึงใช้:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' split_audio.py --overwrite
```

`--overwrite` จะลบ `data_set_dbl_split` เดิมก่อนสร้างใหม่ จึงไม่ควรใช้ระหว่างหรือหลังเริ่มการทดลองที่ต้องอ้างอิง split เดิม

การแบ่งนี้เป็น stratified split ระดับไฟล์ ไม่ใช่ระดับทารกหรือ recording session เนื่องจากยังไม่มี `subject_id/session_id` ที่ตรวจสอบได้ครบถ้วน ตัว trainer จึงตรวจ SHA-256 และ group provenance เพิ่มเติม พร้อมสงวนรายการ Test ที่ชนกับ Train ออกจากการฝึก แต่ยังไม่สามารถรับรอง subject-independent evaluation ได้

สำหรับ Pipeline Run 1 Stage 2 `20260821T164332Z_490383ff` จำนวน candidate 1,551 records ถูกกระทบยอดเป็น eligible originals 1,348 records ดังนี้:

```text
1,551 candidates
  - 117 same-label exact duplicate records
  -   6 cross-label exact duplicate records (3 conflicting SHA-256 groups)
  -  80 Train records ที่มี exact SHA-256 ตรงกับ Test
= 1,348 eligible originals (Train 1,045 + Final Test 303)
```

ดังนั้น 203 records ไม่ได้หายและไม่ได้ถูกตัดด้วย quality filtering แต่ถูก exclude ตามกฎ exact-content deduplication และ leakage prevention โดยไม่มีการลบไฟล์เสียงจริง รายละเอียดรายคลาสและ audit artifacts อยู่ใน [Pipeline Run 1 Report](./Report/runs/report_01_20260821.md)

## Feature และ architecture

### Stage 1 — Binary Baby Gate

```text
MFCC(40) + Delta(40) + Delta2(40)
                 │
                 ▼
       Input (120, 128, 1)
                 │
                 ▼
CNN → BiLSTM → Attention → Softmax(2)
                 │
                 ▼
          not_baby / baby
```

Stage 1 ไม่ใช้ Log-Mel, Chroma หรือ Mixup

CNN ใช้ pooling `(2,2) → (2,2) → (2,1)` ทำให้ feature map เป็น `(15,32)` จากนั้น transpose จาก `[feature,time,channel]` เป็น `[time,feature,channel]` อย่างชัดเจนก่อน reshape ดังนั้น BiLSTM อ่านลำดับเวลา 32 steps ไม่ใช่อ่านแกน feature

### Stage 2 — Five-class Infant State Classifier

```text
MFCC(40) + Delta(40) + Delta2(40) + Log-Mel(64) + Chroma(12)
                              │
                              ▼
                    Input (196, 128, 1)
                              │
                              ▼
              CNN → BiLSTM → Attention → Softmax(5)
```

Stage 2 ใช้ Mixup ค่าเริ่มต้น 500 ตัวอย่างต่อ fold และ `alpha=0.3`

candidate หลัก `corrected_single_branch` ใช้ pooling `(2,2) → (2,2) → (2,1) → (2,1)` ได้ `(12,32)` ก่อน transpose เป็น time-major ส่วน `corrected_multi_branch` แยก MFCC derivatives, Log-Mel และ Chroma เป็น CNN branches แล้วรวม sequence ที่เวลา 32 steps การเลือก candidate ต้องใช้ grouped OOF ไม่ใช้ Test

## Augmentation

augmentation ถูกสร้างเฉพาะ training partition ของแต่ละ fold หลังแบ่ง train/validation แล้ว

วิธีที่ training plan ใช้:

- Gaussian noise
- Pitch shift
- Time stretch
- Time shift

Stage 2 มี Mixup เพิ่มหลัง normalization ส่วน validation และ `data_set_dbl_split/test` ไม่มี augmentation

รายการ augmentation ทุกตัวอย่างถูกบันทึกใน:

```text
fold_N/augmentation_manifest.csv
```

## Training protocol

1. อ่านและ hash ข้อมูล `train/test` เพื่อ audit provenance และตรวจ overlap โดยยังไม่ใช้ Test คำนวณผล
2. ให้ locked Test มีสิทธิ์ก่อน และไม่นำ training record ที่ group หรือ SHA-256 ชนกับ Test เข้าเทรน
3. แบ่ง training originals เป็น grouped 5-fold ก่อน augmentation
4. ใน Fold 1–5 สร้าง augmentation เฉพาะ Training partition
5. fit per-feature-bin normalizer จาก Training features ของแต่ละ Fold เท่านั้น และใช้ค่าเดิมกับ Validation
6. เลือก checkpoint จาก `val_loss` ของ Fold validation
7. สร้าง OOF prediction จาก original validation records และไม่ประเมิน Test ภายใน Fold
8. สรุปจำนวน Epoch สุดท้ายด้วยค่ามัธยฐานของ `best_epoch` ทั้ง 5 Fold
9. สร้าง Final-refit model ใหม่จาก Train 80% ทั้งหมด พร้อม final-refit normalizer และ augmentation plan ของตนเอง
10. ประเมิน Final-refit model บน locked internal Test 20% เพียงครั้งเดียว

สำหรับ WSL ระบบไม่เขียนทับไฟล์ `.keras` ซ้ำ ๆ โดยตรงบน `/mnt/d` ระหว่าง Epoch อีกต่อไป Mutable best checkpoint จะอยู่ใน native Linux temporary storage (`/tmp/cryinsight_checkpoints`) เมื่อ Fold จบจึงคัดลอกเข้า Run folder เพียงครั้งเดียว ตรวจขนาดและ SHA-256 แล้วจึงโหลดประเมิน ไฟล์ `checkpoint_publication.json` ของแต่ละ Fold และ Final-refit บันทึกหลักฐานการเผยแพร่ดังกล่าว

Test ไม่ถูกใช้เลือก architecture, hyperparameters, Epoch, augmentation, normalizer หรือ Final Model ผล OOF ใช้ประเมินกระบวนการพัฒนา ส่วนผล `final_test_*` ใช้ประเมิน Final-refit artefact ภายใน corpus เดียวกัน

## ตรวจข้อมูลโดยไม่เทรน

Stage 1:

```bash
python Models_dbl/binary/train_binary_dbl.py --audit-only
```

Stage 2:

```bash
python Models_dbl/Main/train_main_dbl.py --audit-only
```

โหมด `--audit-only` ไม่สร้าง run และไม่เริ่ม TensorFlow training

## เตรียม run โดยไม่เทรน

```bash
python Models_dbl/binary/train_binary_dbl.py --prepare-only

python Models_dbl/Main/train_main_dbl.py --prepare-only
```

โหมดนี้สร้าง audit, fold assignments และ configuration แต่ไม่สร้างโมเดล

## เทรน Stage 1

```bash
python Models_dbl/binary/train_binary_dbl.py --train --device gpu --require-gpu --mixed-precision
```

ค่าเริ่มต้น:

- Epoch สูงสุด: 200
- Batch size: 32
- Learning rate: 0.001
- AdamW weight decay: `1e-4`
- Label smoothing: 0.05
- Early-stopping patience: 25
- Mixup: ไม่ใช้

## เทรน Stage 2

ควรเทรน Stage 1 ให้เสร็จก่อนหากต้องการนำระบบ 2 Stage ไปใช้งานครบ pipeline

```bash
python Models_dbl/Main/train_main_dbl.py --train --device gpu --require-gpu --mixed-precision --architecture corrected_single_branch
```

Stage 2 จะอ่าน Run ล่าสุดใน `Models_dbl/binary/runs/` และใช้ `run_id` เดียวกับ Stage 1 โดยอัตโนมัติ เฉพาะเมื่อ `verification.json` ของ Stage 1 มี `status: complete` เท่านั้น หาก Run ล่าสุดยังเทรนอยู่ ล้มเหลว หรือยังไม่มี verification ระบบจะหยุดและจะไม่ย้อนกลับไปจับคู่กับ Run เก่า หลักฐานการจับคู่และ SHA-256 ของ verification จะถูกบันทึกใน `protocol.json` และ `run_config.json` ของ Stage 2

ค่าเริ่มต้น:

- Epoch สูงสุด: 200
- Batch size: 32
- Learning rate: 0.001
- AdamW weight decay: `1e-4`
- Label smoothing: 0.10
- Early-stopping patience: 30
- Mixup: 500 samples/fold, alpha 0.3

สามารถระบุ run ID ได้ เช่น:

```bash
python Models_dbl/binary/train_binary_dbl.py --train --device gpu --require-gpu --mixed-precision --run-id stage1_seed42
```

run ID ต้องไม่ซ้ำ เพราะ run directories เป็น immutable

## ผลลัพธ์หลังเทรน

ผลที่เป็นทางการในปัจจุบันคือ Pipeline Run 1 `20260821T164332Z_490383ff`; ดู metric, confidence interval, confusion matrix และข้อจำกัดได้ที่ [Pipeline Run 1 Report](./Report/runs/report_01_20260821.md) ส่วนผลที่เกิดจากคำสั่งเทรนครั้งถัดไปต้องอ่านจาก immutable run ใหม่ ไม่เขียนทับผลนี้

Stage 1:

```text
Models_dbl/binary/runs/<run_id>/
```

Stage 2:

```text
Models_dbl/Main/runs/<run_id>/
```

โครงสร้างสำคัญ:

```text
runs/<run_id>/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
├── fold_5/
├── best_model/
│   ├── best_model_*_dbl.keras
│   ├── norm_stats_*_dbl.npy
│   ├── norm_stats_*_dbl.npy.metadata.json
│   ├── labels_*_dbl.json
│   ├── preprocessing_config.json
│   ├── augmentation_manifest.csv
│   ├── class_counts.json
│   ├── history.csv
│   ├── final_refit_manifest.json
│   └── deployment_manifest.json
├── oof_metrics.json
├── oof_predictions.csv
├── fold_metrics.csv
├── final_test_predictions.csv
├── final_test_metrics.json
├── final_test_confusion_matrix.csv
├── final_test_confusion_matrix.png
├── final_test_manifest.json
└── verification.json
```

ภายในแต่ละ `fold_N` มี:

- `fold_N_binary_dbl.keras` หรือ `fold_N_main_dbl.keras`
- normalizer และ metadata
- training history
- validation predictions/metrics
- augmentation manifest
- fold manifest

โมเดลสำหรับนำไปใช้งานอยู่ที่:

```text
best_model/best_model_binary_dbl.keras
best_model/best_model_main_dbl.keras
```

`best_model` ไม่ใช่สำเนาของ Fold ที่คะแนนดีที่สุด แต่เป็น Final-refit model ที่เทรนใหม่ด้วย Train 80% ทั้งหมดตามจำนวน Epoch มัธยฐานจาก 5 Fold ต้องใช้ model พร้อม normalizer, labels และ `preprocessing_config.json` จาก bundle เดียวกัน

## Metrics

ระบบบันทึก:

- Accuracy
- Balanced accuracy
- Macro precision/recall/F1
- Weighted F1
- ROC-AUC เมื่อคำนวณได้
- Sensitivity และ Specificity สำหรับ Stage 1
- Log loss, Brier score และ Expected Calibration Error (ECE) เพื่อวินิจฉัย calibration
- Classification report
- Confusion matrix
- ROC และ Precision-Recall curves ทั้ง CSV/PNG จาก immutable OOF predictions
- 95% group-bootstrap confidence intervals เมื่อ group structure รองรับ

ตาราง Pipeline Run 1 ด้านบนแสดงเฉพาะค่าที่มาจาก run ที่ฝึกเสร็จจริงและ `verification.json` มีสถานะ `complete` เท่านั้น ห้ามใช้ค่าใดจาก run ที่ `prepared`, `running` หรือ `failed` เป็นผลวิจัย

ผล validation ภายใน 5-fold อยู่ใน `oof_metrics.json` ส่วนผลประเมิน Final-refit model บน locked internal Test ครั้งเดียวอยู่ใน `final_test_metrics.json`

ก่อนคำนวณ Log loss/Brier/ECE ระบบตรวจว่า score เป็นค่าจำกัด ไม่ติดลบ และผลรวมอยู่ใน tolerance จากนั้น renormalize แต่ละแถวให้รวมเป็น 1.0 อย่างแม่นยำเพื่อหลีกเลี่ยงคำเตือนของ scikit-learn โดยไม่เปลี่ยนคลาส argmax

## Baseline และ Ablation

registry อยู่ใน `Models_dbl/experiments/` และกำหนด Majority, MFCC-summary SVM, Log-Mel small CNN ตลอดจน CNN/BiLSTM/Attention, feature, normalization และ augmentation ablations ทุก candidate ต้องใช้ fold assignments และ SHA-256 เดียวกับ proposed model และจัดอันดับด้วย grouped OOF เท่านั้น

ตรวจ registry โดยไม่สร้าง run และไม่เทรน:

```bash
python Models_dbl/experiments/run_experiments.py --audit-only --config Models_dbl/experiments/configs/stage2_wave_a.json
```

สคริปต์นิยามที่มีอยู่จริง:

```text
Models_dbl/experiments/
├── baselines/
│   ├── stage1/
│   │   ├── majority.py
│   │   ├── mfcc_svm.py
│   │   └── logmel_cnn.py
│   └── stage2/
│       ├── majority.py
│       ├── mfcc_svm.py
│       ├── logmel_cnn.py
├── ablations/
│   ├── cnn_only.py
│   ├── without_attention.py
│   ├── feature_ablation.py
│   └── augmentation_ablation.py
└── runs/
```

แต่ละไฟล์รันด้วย `--audit-only` ได้เพื่อแสดง model factory, input contract และ comparison variants โดยไม่โหลด TensorFlow ไม่สร้าง Run และไม่เทรน ตัวอย่าง:

```bash
python Models_dbl/experiments/baselines/stage1/mfcc_svm.py --audit-only
python Models_dbl/experiments/ablations/feature_ablation.py --audit-only
```

ไฟล์เหล่านี้เป็นนิยามโมเดล/ตัวแปรทดลองที่ตรวจสอบได้ ส่วนการฝึกครบทุก fold และการสร้าง metrics ต้องผ่าน shared-fold experiment orchestration หลัง proposed Stage 1 และ Stage 2 เสร็จแล้ว จึงยังห้ามเรียกสถานะ `definition_ready` ว่าเป็นผลทดลอง

สถานะผล baseline/ablation อยู่ใน [Baseline Report](./Report/experiments/baseline_report.md) และ [Ablation Report](./Report/experiments/ablation_report.md) ปัจจุบัน Stage 1 baseline ถูก `prepared` ภายใต้ Pipeline Run 1 แล้ว แต่ยังไม่มี metric เพราะยังไม่เริ่ม `--train`; ค่าใดที่ยังไม่เทรนจะไม่มี metric ประมาณ

## รัน unit tests

```bash
python -m unittest discover -s tests -v
```

ต้องรันใน WSL GPU venv ตามหัวข้อ Environment และตรวจผลล่าสุดจากคำสั่งจริง ไม่ยึดจำนวน tests ที่เขียนค้างไว้ในเอกสาร

## ปัญหาที่พบบ่อย

### `ModuleNotFoundError: No module named 'librosa'`

ตรวจว่าใช้ Python 3.10 environment ที่ติดตั้ง dependencies แล้ว:

```powershell
python -m pip install librosa==0.11.0
```

### `ModuleNotFoundError: No module named 'sklearn'`

```powershell
python -m pip install scikit-learn==1.7.2
```

### TensorFlow import ไม่ได้

ตรวจว่า interpreter เป็น Python 3.10 และติดตั้ง TensorFlow ใน interpreter เดียวกัน:

```powershell
python --version
python -m pip show tensorflow
```

### Audio สั้นเกินไป

feature contract ปัจจุบันจะเติมศูนย์ทางขวาจนถึงจำนวน sample ขั้นต่ำสำหรับ Delta window หากยังเกิด error ให้ดู `record_id` และ `filepath` ที่ trainer รายงานเพื่อตรวจว่าไฟล์เสียหรือถอดรหัสไม่ได้

### ไม่พบ accuracy

accuracy จะไม่มีจนกว่า training run จะจบครบ 5 folds ตรวจ `verification.json` หากสถานะเป็น `incomplete` ต้องใช้ run ใหม่ ไม่ควรเขียนทับ run เดิม

### `OSError: [Errno 22] Invalid argument` ตอนบันทึก `.keras` ผ่าน `/mnt/d`

Stage 2 รุ่นปัจจุบันป้องกันกรณีนี้โดยให้ Keras เขียน mutable checkpoint ใน Linux `/tmp` แล้วเผยแพร่ไฟล์ที่เลือกเข้า Run folder เพียงครั้งเดียว หากต้องย้าย staging root ให้กำหนด `CRYINSIGHT_CHECKPOINT_STAGING_DIR` เป็น path บน native Linux filesystem ห้ามกำหนดเป็น `/mnt/c` หรือ `/mnt/d`

## การตีความผลวิจัย

- `data_set_dbl_split/test` เป็น locked internal held-out test สำหรับ run ใหม่ ไม่ใช่ external validation เพราะมาจาก corpus เดียวกับ Train และ corpus นี้เคยถูกใช้พัฒนาโมเดลเก่ามาก่อน
- ยังไม่มี subject/session identifiers ที่ยืนยันครบ จึงไม่ควรอ้างว่าเป็น subject-independent evaluation
- Stage 1 ใช้เสียงทารกและ `not_baby` จากคนละแหล่งข้อมูล จึงมีความเสี่ยงต่อ dataset-source bias
- Stage 2 มี class imbalance สูง และบางคลาสต้องพึ่ง synthetic augmentation จำนวนมาก จึงควรมี ablation study ก่อนกล่าวอ้างว่า augmentation ช่วยเพิ่มผลลัพธ์
- คะแนน Softmax เป็น uncalibrated model score ไม่ใช่ calibrated probability หรือโอกาสถูกต้องทางการแพทย์
- label ทั้ง 5 ของ Stage 2 เป็นกลุ่มตาม dataset ไม่ใช่การวินิจฉัยทางการแพทย์
- มีตัวรวม cascade จาก prediction artefacts แล้ว แต่การประเมินแบบเต็มต้องมี Stage 2 prediction สำหรับทุกไฟล์ที่ Stage 1 route ผ่าน; run เก่าที่ยังไม่มี support ดังกล่าวต้องไม่อ้างว่าเป็น full end-to-end evaluation
- InfantCry-DBL ต้องอ้างอิง Mendeley Data Version 1 และ DOI `10.17632/x493z8nmwc.1` พร้อมระบุ annotation provenance, consent และ ethics ตามเอกสารต้นฉบับในการส่งตีพิมพ์

การรองรับเสียงจากแหล่งภายนอก เช่น input contract ใหม่, multi-window inference, quality rejection และ Webapp routing ยังอยู่ระหว่างตัดสินใจและยังไม่ได้ implement ในรอบนี้

## License

[InfantCry-DBL Version 1](https://data.mendeley.com/datasets/x493z8nmwc/1) ระบุสัญญาอนุญาต CC BY 4.0 ส่วน source code, ESC-50 และข้อมูลจากแหล่งอื่นต้องตรวจและอ้างอิงสิทธิ์แยกกันก่อนแจกจ่ายหรือตีพิมพ์ผล
