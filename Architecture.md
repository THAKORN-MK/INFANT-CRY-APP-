# CryInsight Model Architecture and Training Protocol

เอกสารนี้อธิบายสถาปัตยกรรม การเตรียมข้อมูล การเพิ่มข้อมูล (augmentation) วิธีฝึก และวิธีประเมินโมเดล CryInsight ตาม implementation ปัจจุบัน โดยระบบแบ่งเป็น 2 ขั้นเพื่อแยกคำถามว่า “เป็นเสียงทารกหรือไม่” ออกจากคำถามว่า “เสียงทารกอยู่ในกลุ่มอารมณ์ใด”

## 1. ภาพรวมระบบ

```text
ไฟล์เสียง .wav
        │
        ▼
Stage 1: Binary Baby Gate
CNN + MFCC/Delta/Delta2 + BiLSTM + Attention
"เป็นเสียงทารกหรือไม่?"
        │
        ├── not_baby ──► หยุดและแจ้งว่าไม่ใช่เสียงทารก
        │
        └── baby
              │
              ▼
Stage 2: Five-class Infant State Classifier
CNN + MFCC/Delta/Delta2/Log-Mel/Chroma + BiLSTM + Attention
"เสียงทารกอยู่ในกลุ่มใด?"
              │
              ▼
belly_pain / burping / discomfort / hungry / tired
พร้อมคะแนนความมั่นใจจาก Softmax
```

เหตุผลที่แบ่งเป็น 2 Stage คือให้ Stage 1 ทำหน้าที่กรองเสียงสิ่งแวดล้อมก่อน เพื่อไม่บังคับให้ Stage 2 เลือกหนึ่งใน 5 กลุ่มทารกเมื่ออินพุตไม่ใช่เสียงทารก โครงสร้างนี้ทำให้หน้าที่ของแต่ละโมเดลชัดเจนและนำไปเชื่อมกับ Web application ได้ตรงไปตรงมา

## 2. ข้อมูลที่ใช้

ข้อมูลต้นฉบับอยู่ใน `data_set_dbl` และถูกสุ่มแยกแบบรายคลาสประมาณ 80/20 เป็น:

```text
data_set_dbl_split/
├── train/   # ใช้สร้าง 5 folds และฝึกโมเดลเท่านั้น
└── test/    # locked held-out test ไม่ใช้ปรับโมเดล
```

ตัวแบ่งข้อมูลใช้ random seed เท่ากับ `42` เพื่อให้ทำซ้ำได้ อย่างไรก็ตาม การสุ่มระดับไฟล์อาจทำให้ไฟล์ที่มีเนื้อหาเหมือนกันหรือไฟล์จากแหล่งกำเนิดเดียวกันกระจายข้าม `train/test` ได้ ดังนั้น trainer จะตรวจ SHA-256 และ group identity อีกครั้งก่อนฝึก โดยให้ชุด `test` มีสิทธิ์ก่อนและไม่นำรายการใน `train` ที่ชนกับ test เข้าเทรน ทั้งนี้ไม่มีการลบไฟล์เสียงจริง

ผล audit ของ split ปัจจุบัน:

| Stage | Training หลังคัดกรอง | Held-out test หลังคัดกรอง | Group overlap หลังคัดกรอง | SHA-256 overlap หลังคัดกรอง |
|---|---:|---:|---:|---:|
| Stage 1 | 2,432 | 694 | 0 | 0 |
| Stage 2 | 1,045 | 303 | 0 | 0 |

จำนวนนี้อาจเปลี่ยนหากสร้าง split ใหม่ด้วย seed หรือข้อมูลต้นฉบับที่ต่างออกไป

### กฎเฉพาะ Stage 1

- คลาสทารกทั้ง 5 กลุ่มถูกรวมเป็น `baby`
- เสียงสิ่งแวดล้อมเป็น `not_baby`
- ESC-50 หมวด `crying_baby` (target 20) ถูกตัดออกจาก `not_baby` เพราะมีความหมายชนกับคลาสบวก
- เสียงทารกจัดกลุ่มด้วย exact-content SHA-256
- เสียง ESC-50 จัดกลุ่มด้วย `source_file` จากชื่อไฟล์ เพื่อไม่ให้หลาย take จากต้นเสียงเดียวกันกระจายข้าม fold

### กฎเฉพาะ Stage 2

- ใช้เฉพาะ `belly_pain`, `burping`, `discomfort`, `hungry` และ `tired`
- ไม่ใช้ `not_baby`
- exact duplicate ในคลาสเดียวกันเก็บ canonical record เพียงหนึ่งรายการ
- exact duplicate ที่พบข้ามคลาสถูกตัดทั้งกลุ่ม เพราะ label ขัดแย้งกัน

## 3. Audio preprocessing

ทั้งสอง Stage ใช้สัญญา preprocessing เดียวกันในส่วนเสียง:

| รายการ | ค่า | เหตุผล |
|---|---:|---|
| Sample rate | 22,050 Hz | เพียงพอสำหรับโครงสร้างเสียงร้องและลดภาระเมื่อเทียบกับ sample rate ที่สูงกว่า |
| Channel | Mono | ทำให้อินพุตทุกไฟล์มีรูปแบบเดียวกัน |
| Silence trimming | `top_db=20` | ลดช่วงเงียบที่ไม่ช่วยจำแนก |
| Waveform normalization | Peak normalization | ลดความต่างของระดับความดังระหว่างไฟล์ |
| FFT size | 2,048 samples | ให้รายละเอียดความถี่เหมาะกับงานเสียงทั่วไป |
| Hop length | 512 samples | กำหนดระยะเลื่อนระหว่างเฟรม |
| Maximum frames | 128 | ทำให้ทุกตัวอย่างมีขนาดคงที่สำหรับ CNN |
| Short-audio policy | Right zero-padding | ทำให้ไฟล์สั้นรองรับ Delta window ได้โดยไม่ทิ้งไฟล์ |
| Minimum samples | 4,096 | มาจาก `(delta_width - 1) × hop_length` เมื่อ `delta_width=9` |
| Data type | `float32` | เหมาะกับ TensorFlow และใช้หน่วยความจำน้อยกว่า float64 |

หาก feature มีน้อยกว่า 128 เฟรมจะเติมศูนย์ทางขวา หากยาวกว่า 128 เฟรมจะตัดให้เหลือ 128 เฟรม

## 4. Acoustic features

### MFCC

MFCC (Mel-frequency cepstral coefficients) จำนวน 40 coefficients สรุปลักษณะ spectral envelope ในสเกลที่ใกล้กับการรับรู้เสียงของมนุษย์ จึงเหมาะกับการแทน timbre และรูปแบบเสียงร้อง

### Delta และ Delta2

- `Delta` จำนวน 40 bins แทนการเปลี่ยนแปลงของ MFCC ตามเวลาในอันดับหนึ่ง
- `Delta2` จำนวน 40 bins แทนการเปลี่ยนแปลงอันดับสองหรือความเร่งของ MFCC

เสียงร้องไม่ได้มีเพียงลักษณะความถี่ ณ จุดใดจุดหนึ่ง แต่มีการไต่ระดับ การสั่น และการเปลี่ยนรูปตามเวลา Delta และ Delta2 จึงเพิ่มข้อมูลด้านพลวัตที่ MFCC แบบคงที่ไม่มี

### Log-Mel spectrogram

Log-Mel จำนวน 64 bands แสดงพลังงานตามเวลาและความถี่ใน Mel scale โดยเก็บรายละเอียด spectro-temporal มากกว่า MFCC เหมาะกับ Stage 2 ซึ่งต้องแยกความแตกต่างระหว่างกลุ่มเสียงทารกที่ใกล้เคียงกัน

### Chroma

Chroma จำนวน 12 bins รวมพลังงานตาม pitch class ช่วยเพิ่มมุมมองเกี่ยวกับโครงสร้างระดับเสียงและ harmonic content แม้เสียงร้องทารกไม่ใช่ดนตรี แต่ Chroma อาจช่วยแยกรูปแบบ pitch/harmonic ที่ MFCC และ Mel แทนได้ไม่เหมือนกัน

> การกล่าวว่า feature ใดช่วยเพิ่ม accuracy ต้องยืนยันด้วย ablation study การเลือก feature ในระบบนี้เป็นสมมติฐานการออกแบบตามโครงสร้างที่กำหนด ไม่ใช่หลักฐานว่า feature ชุดนี้ดีที่สุดสำหรับทุก corpus

## 5. Feature contract ของแต่ละ Stage

| Stage | Feature order | การคำนวณจำนวน bins | Input shape |
|---|---|---:|---|
| Stage 1 | MFCC + Delta + Delta2 | 40 + 40 + 40 = 120 | `(120, 128, 1)` |
| Stage 2 | MFCC + Delta + Delta2 + Log-Mel + Chroma | 40 + 40 + 40 + 64 + 12 = 196 | `(196, 128, 1)` |

Stage 1 ใช้ representation ที่กะทัดรัดกว่า เพราะเป้าหมายเป็น binary gate ระหว่าง `baby/not_baby` ส่วน Stage 2 เพิ่ม Log-Mel และ Chroma เพื่อให้โมเดลเห็นรายละเอียด spectral และ pitch เพิ่มเติมสำหรับปัญหา 5 คลาส

### Feature extraction flow

```mermaid
flowchart TB
    A(["Waveform .wav"]) --> B["Audio preprocessing<br/>Resample · Mono · Trim · Peak normalize"]
    B --> C["MFCC<br/>40 bins"]
    C --> D["Delta<br/>40 bins"]
    C --> E["Delta2<br/>40 bins"]
    B --> F["Log-Mel spectrogram<br/>64 bins"]
    B --> G["Chroma<br/>12 bins"]

    C --> H["Stage 1 feature concatenation<br/>MFCC + Delta + Delta2"]
    D --> H
    E --> H
    H --> I(["Tensor<br/>120 × 128 × 1"])

    C --> J["Stage 2 feature concatenation<br/>MFCC + Delta + Delta2 + Log-Mel + Chroma"]
    D --> J
    E --> J
    F --> J
    G --> J
    J --> K(["Tensor<br/>196 × 128 × 1"])

    classDef input fill:#F8FAFC,stroke:#334155,color:#0F172A,stroke-width:1.5px;
    classDef process fill:#F1F5F9,stroke:#475569,color:#0F172A,stroke-width:1.5px;
    classDef feature fill:#FAF5FF,stroke:#7E22CE,color:#3B0764,stroke-width:1.5px;
    classDef stage1 fill:#EFF6FF,stroke:#1D4ED8,color:#172554,stroke-width:2px;
    classDef stage2 fill:#ECFDF5,stroke:#047857,color:#022C22,stroke-width:2px;

    class A input;
    class B process;
    class C,D,E,F,G feature;
    class H,I stage1;
    class J,K stage2;

    linkStyle default stroke:#64748B,stroke-width:1.5px;
```

Delta และ Delta2 คำนวณต่อจาก MFCC ขณะที่ Log-Mel และ Chroma คำนวณจาก waveform หลัง preprocessing จากนั้นจึงเรียงและต่อ feature ตาม contract ของแต่ละ Stage

## 6. Stage 1 architecture

```text
Input (120, 128, 1)
  │
  ├─ Conv2D 32 → BatchNorm → ReLU
  ├─ Conv2D 32 → BatchNorm → ReLU
  ├─ MaxPool → Dropout 0.25
  │
  ├─ Conv2D 64 → BatchNorm → ReLU
  ├─ Conv2D 64 → BatchNorm → ReLU
  ├─ MaxPool → Dropout 0.25
  │
  ├─ Conv2D 128 → BatchNorm → ReLU
  ├─ Conv2D 128 → BatchNorm → ReLU
  ├─ MaxPool → Dropout 0.30
  │
  ├─ Reshape CNN map เป็นลำดับเวลา
  ├─ Bidirectional LSTM 128
  ├─ Dropout 0.30
  ├─ Bidirectional LSTM 64
  ├─ Dropout 0.30
  ├─ Attention
  ├─ Dense 128 → BatchNorm → Dropout 0.40
  ├─ Dense 64 → Dropout 0.40
  └─ Softmax 2 classes
```

หน้าที่ของแต่ละส่วน:

- CNN เรียนรู้ลวดลายเฉพาะพื้นที่บน feature map เช่น formant, harmonics และการเปลี่ยนพลังงานระยะสั้น
- Batch normalization ช่วยให้การกระจาย activation มีเสถียรภาพระหว่างการฝึก
- Max pooling ลดขนาดข้อมูลและเพิ่มความทนต่อการเลื่อนเล็กน้อย
- BiLSTM อ่านบริบททั้งทิศทางก่อนหน้าและถัดไปของลำดับที่ได้จาก CNN
- Attention เรียนรู้น้ำหนักของช่วงเวลาที่สำคัญ แทนการให้ทุกเฟรมมีผลเท่ากัน
- Dropout ลดความเสี่ยง overfitting
- Softmax ให้คะแนนสำหรับ `not_baby` และ `baby`

## 7. Stage 2 architecture

```text
Input (196, 128, 1)
  │
  ├─ CNN blocks: 32 → 64 → 128 → 256 filters
  │    Conv2D + BatchNorm + ReLU + MaxPool + Dropout
  │
  ├─ Reshape CNN map เป็นลำดับเวลา
  ├─ Bidirectional LSTM 128 → Dropout 0.30
  ├─ Bidirectional LSTM 64  → Dropout 0.30
  ├─ Attention
  ├─ Dense 256 → BatchNorm → Dropout 0.40
  ├─ Dense 128 → BatchNorm → Dropout 0.40
  ├─ Dense 64 → Dropout 0.30
  └─ Softmax 5 classes
```

Stage 2 มี CNN block และ Dense capacity มากกว่า Stage 1 เพราะต้องแยก 5 คลาสที่มีรูปแบบใกล้กัน และรับ feature map ที่กว้างกว่า อย่างไรก็ตาม capacity ที่มากขึ้นเพิ่มความเสี่ยง overfitting จึงใช้ Batch normalization, Dropout, augmentation, Mixup และ early stopping ร่วมกัน

## 8. Data augmentation

augmentation ใช้เฉพาะ training partition ภายในแต่ละ fold หลังแบ่ง train/validation แล้ว จึงไม่มี augmented sample ใน validation หรือ held-out test

ตัววางแผน augmentation จะเพิ่มตัวอย่างของแต่ละคลาสให้เท่ากับจำนวน original ของคลาสที่ใหญ่ที่สุดใน training partition ของ fold นั้น การ transform ถูกสร้างแบบ deterministic จาก seed และบันทึกใน `augmentation_manifest.csv`

Waveform augmentation ที่ trainer เลือกใช้:

| วิธี | หลักการ | เหตุผล |
|---|---|---|
| Gaussian noise | เติม noise ระดับต่ำ | เพิ่มความทนต่อ background noise |
| Pitch shift | เลื่อน pitch ขึ้นหรือลง | จำลองความแตกต่างของระดับเสียงระหว่างผู้ร้อง |
| Time stretch | ยืดหรือบีบเวลาโดยไม่เปลี่ยน label | จำลองความเร็วและระยะเวลาการร้องที่ต่างกัน |
| Time shift | เลื่อน waveform ตามเวลา | ลดการยึดติดกับตำแหน่งเริ่มต้นของเหตุการณ์เสียง |

ระบบ feature utility รองรับ amplitude scaling แต่ training augmentation plan ปัจจุบันไม่ได้เลือก transform นี้

### Stage 1

- ใช้ waveform augmentation เพื่อช่วย balance คลาสภายในแต่ละ fold
- ไม่ใช้ Mixup

### Stage 2

- ใช้ waveform augmentation เพื่อ balance คลาสภายในแต่ละ fold
- หลัง fit normalizer จาก training features แล้ว สร้าง Mixup เพิ่มค่าเริ่มต้น 500 ตัวอย่างต่อ fold
- ค่า Mixup alpha เริ่มต้นเท่ากับ `0.3`
- Mixup ผสมทั้ง feature tensors และ one-hot labels ของตัวอย่าง training เท่านั้น

## 9. Normalization และการป้องกัน leakage

แต่ละ fold มี normalizer ของตัวเอง โดยคำนวณ mean และ standard deviation จาก training features ของ fold นั้นเท่านั้น จากนั้นนำค่าสถิติเดียวกันไปแปลง training, validation และ held-out test

```text
fit mean/std: fold training originals + training augmentation
apply mean/std: fold training, fold validation, locked test
```

ห้าม fit normalizer จาก validation หรือ test เพราะจะทำให้ข้อมูลเกี่ยวกับ distribution ของชุดประเมินรั่วเข้าสู่กระบวนการฝึก ไฟล์ normalizer จึงถูกบันทึกพร้อม metadata ที่ระบุ `run_id`, fold, feature shape และ SHA-256

## 10. Five-fold training protocol

หลังสงวน test และตัด overlap จาก train แล้ว ข้อมูล train จะถูกแบ่ง 5 folds แบบ grouped stratification:

```text
Fold 1: train folds 2–5, validate fold 1
Fold 2: train folds 1,3–5, validate fold 2
...
Fold 5: train folds 1–4, validate fold 5
```

### Training and evaluation flow

```mermaid
flowchart TB
    A["data_set_dbl_split/train"] --> C["Dataset audit<br/>SHA-256 + group identity"]
    B["data_set_dbl_split/test<br/>Locked test"] --> C
    C --> D["Test-priority reservation<br/>ไม่นำรายการ train ที่ชนกับ test เข้าเทรน"]
    D --> E["Grouped stratified 5-fold<br/>จาก training originals เท่านั้น"]

    subgraph FOLD["กระบวนการภายในแต่ละ Fold"]
        direction TB
        F["Fold training partition"] --> G["Training-only augmentation<br/>Balance classes"]
        G --> H["Extract features"]
        H --> I["Fit normalizer<br/>จาก training features เท่านั้น"]
        I --> J["Train CNN + BiLSTM + Attention"]
        J --> K["Checkpoint<br/>Minimum validation loss"]

        L["Fold validation<br/>Original only"] --> M["Apply fold normalizer"]
        M --> N["Validation metrics<br/>และ OOF prediction"]
        K --> N

        O["Locked held-out test<br/>Original only · No augmentation"] --> P["Apply fold normalizer"]
        P --> Q["Held-out metrics<br/>บันทึกแยกสำหรับ Fold"]
        K --> Q
    end

    E --> F
    E --> L
    D --> O
    N --> R["รวม OOF metrics<br/>หลังครบ 5 folds"]
    Q --> S["สรุป held-out metrics<br/>ทั้ง 5 fold models"]
    K --> T["เลือก deployment fold<br/>ด้วย validation loss เท่านั้น"]
    T --> U(["best_model bundle"])

    classDef data fill:#F8FAFC,stroke:#334155,color:#0F172A,stroke-width:1.5px;
    classDef audit fill:#FFF7ED,stroke:#C2410C,color:#431407,stroke-width:1.5px;
    classDef train fill:#EFF6FF,stroke:#1D4ED8,color:#172554,stroke-width:1.5px;
    classDef validation fill:#FAF5FF,stroke:#7E22CE,color:#3B0764,stroke-width:1.5px;
    classDef test fill:#F0FDFA,stroke:#0F766E,color:#042F2E,stroke-width:1.5px;
    classDef result fill:#F0FDF4,stroke:#15803D,color:#052E16,stroke-width:2px;

    class A,B data;
    class C,D audit;
    class E,F,G,H,I,J,K train;
    class L,M,N,R validation;
    class O,P,Q,S test;
    class T,U result;

    linkStyle default stroke:#64748B,stroke-width:1.5px;
```

เส้นทาง validation และ held-out test ใช้ checkpoint และ normalizer ของ fold เดียวกัน แต่ทั้งสองชุดไม่มี augmentation และไม่มีส่วนในการ fit normalizer โดยเฉพาะ held-out metrics จะถูกใช้เพื่อรายงานเท่านั้น ไม่ใช่เกณฑ์เลือก deployment fold

หลักการสำคัญ:

- Original record/group ถูกแบ่งก่อนสร้าง augmentation
- กลุ่มเดียวกันและ SHA-256 เดียวกันต้องไม่อยู่ทั้ง train และ validation
- Validation ใช้ original records เท่านั้น
- ทุก eligible training record ต้องเป็น validation exactly once เมื่อรวม OOF ทั้ง 5 folds
- ใช้ random seed `42` เป็นค่าเริ่มต้น
- checkpoint ของแต่ละ fold เลือก epoch ที่มี `val_loss` ต่ำที่สุด
- Early stopping และ ReduceLROnPlateau เฝ้าดู `val_loss`
- ใช้ AdamW ค่า learning rate เริ่มต้น `0.001` และ weight decay `1e-4`

ค่าหลักที่ต่างกัน:

| ค่า | Stage 1 | Stage 2 |
|---|---:|---:|
| Epoch สูงสุด | 200 | 200 |
| Batch size | 32 | 32 |
| Label smoothing | 0.05 | 0.10 |
| Early-stopping patience | 25 | 30 |
| LR-reduction patience | 10 | 12 |
| Mixup samples/fold | 0 | 500 |
| Mixup alpha | ไม่ใช้ | 0.3 |

## 11. Held-out test หลังแต่ละ fold

เมื่อ train ของ fold เสร็จ ระบบโหลด checkpoint ที่มี `val_loss` ต่ำที่สุดของ fold นั้น แล้วประเมิน locked test ชุดเดียวกันทันที:

```text
Fold N training
   │
   ├─ เลือก checkpoint ด้วย fold validation val_loss
   ├─ ประเมิน fold validation
   └─ ประเมิน data_set_dbl_split/test
```

ผล test ไม่ถูกใช้เพื่อ:

- เลือก epoch
- ปรับ hyperparameters
- สร้าง augmentation
- fit normalizer
- เลือก best deployment fold

การประเมิน test หลังทุก fold ให้ข้อมูลเปรียบเทียบความเสถียรระหว่างโมเดล 5 ตัว แต่ทุกโมเดลใช้ test records ชุดเดียวกัน ดังนั้นผลทั้ง 5 folds มีความสัมพันธ์กันและไม่ใช่ independent experiments

## 12. การเลือก best model

หลังครบ 5 folds ระบบเลือก deployment fold ตามกฎ:

1. เลือก fold ที่มี `selected_checkpoint_val_loss` ต่ำที่สุด
2. หากค่าเท่ากัน ให้เลือกหมายเลข fold ต่ำกว่า
3. ไม่ใช้ test accuracy หรือ test F1 ในการเลือก

โมเดลของ fold ที่ชนะถูกคัดลอกพร้อม normalizer, labels, preprocessing contract, metrics และ manifests ไปยังโฟลเดอร์ `best_model`

- Stage 1: `best_model_binary_dbl.keras`
- Stage 2: `best_model_main_dbl.keras`

เหตุผลที่ไม่ใช้ test metric เลือก best model คือการเลือกจาก test จะเปลี่ยน test ให้กลายเป็น validation โดยปริยายและทำให้ค่าประเมินเอนเอียง

## 13. Metrics

ระบบบันทึก metrics ต่อไปนี้:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- ROC-AUC สำหรับ Stage 1
- Macro/weighted one-vs-rest ROC-AUC สำหรับ Stage 2 เมื่อคำนวณได้
- Classification report
- Confusion matrix ทั้ง CSV และ PNG

OOF metrics รวม validation predictions ที่ record แต่ละรายการถูกประเมินหนึ่งครั้ง ส่วน held-out metrics ถูกบันทึกแยกทุก fold และสรุปค่าเฉลี่ย/SD ของโมเดลทั้ง 5 ตัว

Accuracy จริงไม่ควรประมาณหรือกรอกล่วงหน้า ต้องอ่านจาก artefact หลังการฝึกเสร็จ คะแนน Softmax เป็น model confidence และไม่ควรเรียกว่า calibrated probability เว้นแต่มีการทดสอบ calibration เพิ่มเติม

## 14. โครงสร้างผลลัพธ์

```text
Models_dbl/<stage>/runs/<run_id>/
├── protocol.json
├── run_config.json
├── dataset_audit.json
├── heldout_dataset_audit.json
├── heldout_reservation.json
├── record_audit.csv
├── heldout_record_audit.csv
├── fold_assignments.csv
├── preprocessing_config.json
├── labels_*.json
│
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
├── fold_5/
│   ├── fold_N_*.keras
│   ├── norm_stats_*.npy
│   ├── norm_stats_*.npy.metadata.json
│   ├── augmentation_manifest.csv
│   ├── class_counts.json
│   ├── history.csv
│   ├── validation_predictions.csv
│   ├── metrics.json
│   ├── heldout_test_predictions.csv
│   ├── heldout_test_metrics.json
│   ├── heldout_test_confusion_matrix.csv
│   ├── heldout_test_confusion_matrix.png
│   ├── heldout_test_manifest.json
│   └── fold_manifest.json
│
├── best_model/
│   ├── best_model_*_dbl.keras
│   ├── norm_stats_*.npy
│   ├── preprocessing_config.json
│   ├── labels_*.json
│   ├── heldout_test_metrics.json
│   ├── source_fold_manifest.json
│   └── deployment_manifest.json
│
├── oof_predictions.csv
├── oof_metrics.json
├── oof_confusion_matrix.csv
├── oof_confusion_matrix.png
├── fold_metrics.csv
├── heldout_test_fold_metrics.csv
├── heldout_test_summary.json
└── verification.json
```

แต่ละ run เป็น immutable run: หาก `run_id` มีอยู่แล้ว trainer จะไม่เขียนทับ เพื่อป้องกัน artefact จากการทดลองคนละรอบปะปนกัน

## 15. วิธีใช้งาน

ตรวจข้อมูลโดยไม่เทรน:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --audit-only

& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --audit-only
```

ฝึก Stage 1:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/binary/train_binary_dbl.py --train
```

ฝึก Stage 2:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' Models_dbl/Main/train_main_dbl.py --train
```

## 16. ข้อจำกัดสำหรับการตีพิมพ์

- Train และ test มาจาก corpus เดียวกัน จึงเรียกผล test นี้ว่า `locked internal held-out test` ไม่ใช่ external validation
- ยังไม่มี subject/session identifiers ที่ตรวจสอบได้ครบถ้วน การป้องกัน leakage ของเสียงทารกจึงรับประกันระดับ exact-content family แต่ยังไม่รับประกัน subject-independent evaluation
- การประเมิน test ชุดเดิมหลังทุก fold ทำให้ค่าทั้ง 5 รายการสัมพันธ์กัน
- Augmentation และ Mixup เป็นวิธีลด overfitting/imbalance แต่ไม่ได้รับประกันว่า accuracy จะสูงขึ้น ต้องยืนยันจากผลทดลองหรือ ablation study
- กลุ่มทั้ง 5 ของ Stage 2 เป็น label ตาม dataset ไม่ควรตีความเป็นการวินิจฉัยทางการแพทย์
- ก่อนรายงานผลควรใช้ metrics จาก immutable run ที่ `verification.json` มีสถานะ `complete` เท่านั้น

## 17. ไฟล์ implementation ที่เกี่ยวข้อง

- `Models_dbl/binary/train_binary_dbl.py` — Stage 1 training และ evaluation
- `Models_dbl/Main/train_main_dbl.py` — Stage 2 training และ evaluation
- `cryinsight/audio/features.py` — preprocessing, feature extraction, augmentation และ Mixup
- `cryinsight/training/protocol.py` — dataset audit, grouping, leakage checks และ fold assignment
- `cryinsight/training/artefacts.py` — normalizer, metrics, manifests, best-model bundle และ verification
- `split_audio.py` — สุ่มแบ่ง dataset แบบรายคลาส 80/20
