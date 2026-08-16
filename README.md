# CryInsight — Two-stage Infant Cry Classification

CryInsight เป็นระบบวิเคราะห์ไฟล์เสียง `.wav` แบบ 2 Stage:

1. **Stage 1 — Binary Baby Gate:** ตรวจว่าเป็นเสียงทารกหรือไม่
2. **Stage 2 — Infant State Classifier:** เมื่อเป็นเสียงทารก จึงจำแนกเป็น 5 กลุ่ม

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

รายละเอียดเชิงเทคนิค เหตุผลของ feature แต่ละชนิด โครงสร้าง neural network และข้อจำกัดสำหรับงานวิจัยอยู่ใน [Architecture.md](./Architecture.md)

## Environment หลัก

โปรเจกต์นี้กำหนด **Python 3.10** เป็น environment หลัก

| รายการ | เวอร์ชันหลัก |
|---|---|
| Python | **3.10** |
| TensorFlow | 2.15.0 |
| NumPy | 1.26.4 |
| librosa | 0.11.0 |
| scikit-learn | 1.7.2 |
| matplotlib | เวอร์ชันที่รองรับ Python 3.10 |

เหตุผลที่ใช้ Python 3.10:

- เป็นเวอร์ชันที่ตรวจสอบร่วมกับ TensorFlow 2.15.0 ในโปรเจกต์นี้แล้ว
- ลดปัญหาความเข้ากันได้ระหว่าง TensorFlow, Keras, NumPy และไลบรารีเสียง
- ทำให้ environment สำหรับฝึกและโหลดโมเดลมี baseline เดียวกัน

ไม่ควรเปลี่ยนไปใช้ Python เวอร์ชันอื่นโดยไม่รัน unit tests, model-build smoke test และตรวจการ save/load โมเดลใหม่ทั้งหมด

## การติดตั้งบน Windows

ตรวจสอบ Python 3.10:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' --version
```

ผลที่ต้องได้ควรขึ้นต้นด้วย:

```text
Python 3.10
```

### สร้าง virtual environment

รันจากโฟลเดอร์ `D:\INFANT CRY`:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -m venv .venv310
& '.\.venv310\Scripts\Activate.ps1'
python -m pip install --upgrade pip
```

### ติดตั้งไลบรารีหลัก

```powershell
python -m pip install tensorflow==2.15.0 numpy==1.26.4 librosa==0.11.0 scikit-learn==1.7.2 matplotlib
```

ตรวจสอบ environment:

```powershell
python -c "import sys, tensorflow, numpy, librosa, sklearn; print(sys.version); print('TensorFlow', tensorflow.__version__); print('NumPy', numpy.__version__); print('librosa', librosa.__version__); print('scikit-learn', sklearn.__version__)"
```

> หากไม่ได้ activate virtual environment ให้ใช้ path ของ Python 3.10 แบบเต็มในทุกคำสั่งตามตัวอย่างส่วนการฝึกโมเดล

## โครงสร้างโปรเจกต์ที่ใช้งานอยู่

```text
INFANT CRY/
├── Architecture.md
├── README.md
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
│   └── Main/
│       └── train_main_dbl.py      # Stage 2
│
├── cryinsight/
│   ├── audio/features.py
│   └── training/
│       ├── protocol.py
│       └── artefacts.py
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

1. อ่านและ hash ข้อมูล `train/test`
2. ให้ locked test มีสิทธิ์ก่อน
3. ไม่นำ training records ที่ group หรือ SHA-256 ชนกับ test เข้าเทรน
4. แบ่ง training originals เป็น grouped 5-fold
5. สร้าง augmentation จาก training partition ของแต่ละ fold เท่านั้น
6. fit normalizer จาก training features ของ fold เท่านั้น
7. เลือก checkpoint จาก `val_loss` ต่ำที่สุด
8. ประเมิน fold validation
9. ประเมิน locked test หลังจบแต่ละ fold
10. เลือก `best_model` จาก validation loss ไม่ใช่ test accuracy

held-out test ถูกประเมินหลังทุก fold ตาม protocol ปัจจุบัน แต่ไม่ถูกใช้เลือก epoch, hyperparameter หรือ deployment fold

## ตรวจข้อมูลโดยไม่เทรน

Stage 1:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --audit-only
```

Stage 2:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --audit-only
```

โหมด `--audit-only` ไม่สร้าง run และไม่เริ่ม TensorFlow training

## เตรียม run โดยไม่เทรน

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --prepare-only

& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --prepare-only
```

โหมดนี้สร้าง audit, fold assignments และ configuration แต่ไม่สร้างโมเดล

## เทรน Stage 1

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --train
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

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --train
```

ค่าเริ่มต้น:

- Epoch สูงสุด: 200
- Batch size: 32
- Learning rate: 0.001
- AdamW weight decay: `1e-4`
- Label smoothing: 0.10
- Early-stopping patience: 30
- Mixup: 500 samples/fold, alpha 0.3

สามารถระบุ run ID ได้ เช่น:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --train --run-id stage1_seed42
```

run ID ต้องไม่ซ้ำ เพราะ run directories เป็น immutable

## ผลลัพธ์หลังเทรน

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
├── oof_metrics.json
├── oof_predictions.csv
├── fold_metrics.csv
├── heldout_test_fold_metrics.csv
├── heldout_test_summary.json
└── verification.json
```

ภายในแต่ละ `fold_N` มี:

- `fold_N_binary_dbl.keras` หรือ `fold_N_main_dbl.keras`
- normalizer และ metadata
- training history
- validation predictions/metrics
- held-out test predictions/metrics
- confusion matrix
- augmentation manifest
- fold manifest

โมเดลสำหรับนำไปใช้งานอยู่ที่:

```text
best_model/best_model_binary_dbl.keras
best_model/best_model_main_dbl.keras
```

ต้องใช้ model พร้อมกับ normalizer, labels และ `preprocessing_config.json` จาก `best_model` ชุดเดียวกัน ห้ามนำไฟล์จากคนละ run หรือคนละ fold มาผสมกัน

## Metrics

ระบบบันทึก:

- Accuracy
- Balanced accuracy
- Macro precision/recall/F1
- Weighted F1
- ROC-AUC เมื่อคำนวณได้
- Classification report
- Confusion matrix

README นี้ไม่ระบุค่า accuracy ล่วงหน้า เพราะค่าที่รายงานต้องมาจาก run ที่ฝึกเสร็จจริงและ `verification.json` มีสถานะ `complete`

ผล validation ภายใน 5-fold อยู่ใน `oof_metrics.json` ส่วนผล test ของแต่ละ fold และสรุปรวมอยู่ใน `heldout_test_metrics.json` และ `heldout_test_summary.json`

## รัน unit tests

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -m unittest discover -s tests -v
```

baseline ที่ตรวจล่าสุดผ่าน 59 tests

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

## การตีความผลวิจัย

- `data_set_dbl_split/test` เป็น locked internal held-out test ไม่ใช่ external validation เพราะมาจาก corpus เดียวกับ train
- ยังไม่มี subject/session identifiers ที่ยืนยันครบ จึงไม่ควรอ้างว่าเป็น subject-independent evaluation
- test ชุดเดียวกันถูกประเมินด้วยโมเดลทั้ง 5 folds ผลจึงมีความสัมพันธ์กัน
- คะแนน Softmax เป็น model confidence และไม่ใช่ calibrated probability โดยอัตโนมัติ
- label ทั้ง 5 ของ Stage 2 เป็นกลุ่มตาม dataset ไม่ใช่การวินิจฉัยทางการแพทย์

## License

ตรวจสอบสิทธิ์ของ source code และ dataset แต่ละแหล่งก่อนแจกจ่ายหรือตีพิมพ์ผล เนื่องจาก license ของ dataset อาจแตกต่างจาก license ของตัวโปรเจกต์
