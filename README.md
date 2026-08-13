# 🍼 Baby Cry Detector — CNN+MFCC 2-Stage

ระบบวิเคราะห์เสียงร้องของทารก ด้วย Deep Learning แบบ **2-Stage Pipeline**  
เทรนเองทั้งหมด 100% โดยใช้ **CNN + MFCC + Mel + Chroma + BiLSTM + Attention**  
จำแนกอารมณ์ทารกได้ 5 ประเภทตาม **Dunstan Baby Language**:

| อารมณ์ | เสียง DBL | ความหมาย |
|--------|----------|---------|
| 😣 ปวดท้อง | eairh | ลมในช่องท้องส่วนล่าง |
| 😮 เรอ | eh | ลมในหน้าอก ต้องการเรอ |
| 😰 ไม่สบาย | heh | ผิวหนังระคายเคือง |
| 🍼 หิว | neh | Sucking reflex |
| 😴 เหนื่อย | owh | ง่วงนอน |

---

## 🔄 สถาปัตยกรรม 2-Stage Pipeline

```
เสียงร้องทารก (.wav)
        ↓
┌─────────────────────────────────────┐
│  Stage 1: Binary Classifier         │
│  CNN + MFCC + BiLSTM + Attention    │
│  "เสียงเด็กหรือไม่?"                │
└─────────────────────────────────────┘
        ├── ❌ ไม่ใช่ → หยุด แจ้งผล
        └── ✅ ใช่
                  ↓
    ┌─────────────────────────────────────┐
    │  Stage 2: Emotion Classifier        │
    │  CNN + MFCC + Mel + Chroma          │
    │       + BiLSTM + Attention          │
    │  "อารมณ์คืออะไร?"                  │
    └─────────────────────────────────────┘
                  ↓
         ผลลัพธ์ 5 อารมณ์ + ความมั่นใจ %
```

---

## 🧠 สถาปัตยกรรมโมเดล

### Stage 1 — Binary Classifier
```
Input (196, 128, 1)
  → CNN Block 1: Conv2D(32×2) + BN + ReLU + MaxPool + Dropout(0.25)
  → CNN Block 2: Conv2D(64×2) + BN + ReLU + MaxPool + Dropout(0.25)
  → CNN Block 3: Conv2D(128×2) + BN + ReLU + MaxPool + Dropout(0.3)
  → Reshape
  → BiLSTM(128) + Dropout(0.3)
  → BiLSTM(64)  + Dropout(0.3)
  → Attention Layer
  → Dense(128) + BN + Dropout(0.4)
  → Dense(64)  + Dropout(0.4)
  → Dense(2, softmax) → [not_baby, baby]
```

### Stage 2 — Emotion Classifier
```
Input (196, 128, 1)
  → CNN Block 1: Conv2D(32×2) + BN + ReLU + MaxPool + Dropout(0.25)
  → CNN Block 2: Conv2D(64×2) + BN + ReLU + MaxPool + Dropout(0.25)
  → CNN Block 3: Conv2D(128×2) + BN + ReLU + MaxPool + Dropout(0.3)
  → CNN Block 4: Conv2D(256)   + BN + ReLU + MaxPool + Dropout(0.3)
  → Reshape
  → BiLSTM(128) + Dropout(0.3)
  → BiLSTM(64)  + Dropout(0.3)
  → Attention Layer
  → Dense(256) + BN + Dropout(0.4)
  → Dense(128) + BN + Dropout(0.4)
  → Dense(64)  + Dropout(0.3)
  → Dense(5, softmax) → 5 อารมณ์
```

**Feature:** MFCC(40) + Delta(40) + Delta²(40) + Mel(64) + Chroma(12) = **196 channels**  
**Loss:** CategoricalCrossentropy (label_smoothing=0.1)  
**Optimizer:** AdamW (lr=0.001, weight_decay=1e-4)

---

## 📊 ผลการเทรน

| โมเดล | Val Accuracy | เทคนิค |
|-------|-------------|--------|
| Stage 1 Binary | 99.94% | CNN+MFCC+BiLSTM+Attention, K-Fold |
| Stage 2 Emotion | 99.67% | CNN+MFCC+Mel+Chroma+BiLSTM+Attention, Mixup |

---

## 📁 โครงสร้างโปรเจกต์

```
INFANT CRY/
├── data_set_dbl/              ← Dataset InfantCry-DBL
│   ├── belly_pain/   (170 ไฟล์)
│   ├── burping/      (280 ไฟล์)
│   ├── discomfort/   (168 ไฟล์)
│   ├── hungry/       (217 ไฟล์)
│   ├── tired/        (716 ไฟล์)
│   └── not_baby/     (2,000 ไฟล์ จาก ESC-50)
│
├── models_dbl/
│   ├── binary/                ← Stage 1
│   │   ├── train_binary_dbl.py
│   │   ├── best_model_binary_dbl.keras
│   │   ├── labels_binary_dbl.json
│   │   ├── norm_stats_binary_dbl.npy
│   │   ├── classification_report.txt
│   │   └── confusion_matrix.png
│   │
│   ├── main/                  ← Stage 2
│   │   ├── train_main_dbl.py
│   │   ├── best_model_main_dbl.keras
│   │   ├── labels_main_dbl.json
│   │   ├── norm_stats_main_dbl.npy
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── training_log.csv
│   │
│   └── app/                   ← แอปพลิเคชัน
│       └── app_dbl.py
```

---

## 🖥️ การติดตั้งสำหรับนักพัฒนา

### 1. ติดตั้ง Python 3.10
ดาวน์โหลด: https://www.python.org/downloads/release/python-3100/

> ✅ ติ๊ก **"Add Python to PATH"** ตอนติดตั้ง  
> ⚠️ Python 3.11+ อาจมีปัญหากับ TensorFlow 2.15

### 2. ติดตั้ง Library

```bash
pip install tensorflow==2.15.0
pip install librosa scikit-learn numpy matplotlib seaborn
```

### 3. เทรนโมเดล Stage 1 (Binary)

```bash
cd models_dbl/binary
python train_binary_dbl.py
```

### 4. เทรนโมเดล Stage 2 (Emotion)

```bash
cd models_dbl/main
python train_main_dbl.py
```

### 5. รัน App

```bash
cd models_dbl/app
python app_dbl.py
```

---

## 📱 การติดตั้งสำหรับผู้ใช้ทั่วไป

### สิ่งที่ต้องมี

| รายการ | รายละเอียด |
|--------|-----------|
| Python | 3.10 |
| Stage 1 Model | `best_model_binary_dbl.keras` |
| Stage 1 Labels | `labels_binary_dbl.json` |
| Stage 1 Norm | `norm_stats_binary_dbl.npy` |
| Stage 2 Model | `best_model_main_dbl.keras` |
| Stage 2 Labels | `labels_main_dbl.json` |
| Stage 2 Norm | `norm_stats_main_dbl.npy` |
| App | `app_dbl.py` |

### ขั้นตอน

**1. ติดตั้ง Library**
```bash
pip install tensorflow==2.15.0 librosa numpy matplotlib
```

**2. รัน App**
```bash
python app_dbl.py
```

---

## 📦 Library ที่ใช้

| Library | Version | หน้าที่ |
|---------|---------|--------|
| `tensorflow` | 2.15.0 | สร้างและรันโมเดล |
| `librosa` | ≥ 0.11 | แปลงเสียงเป็น MFCC, Mel, Chroma |
| `numpy` | ≥ 1.23 | คำนวณ array |
| `scikit-learn` | ≥ 1.3 | Metrics, LabelEncoder, Class Weight |
| `matplotlib` | ≥ 3.7 | กราฟใน App |
| `seaborn` | latest | Confusion Matrix |
| `tkinter` | built-in | GUI Application |

---

## 🗂️ Dataset ที่ใช้

### InfantCry-DBL
- **ผู้สร้าง:** Mohammed Tawfik (2026)
- **จำนวน:** 1,551 ไฟล์ (54.4 นาที)
- **License:** CC BY 4.0
- **Cohen's κ:** 0.89 — ความน่าเชื่อถือสูง

### ESC-50 (Not Baby Samples)
- **ผู้สร้าง:** Karol J. Piczak (2015)
- **จำนวน:** 2,000 ไฟล์
- **ใช้เป็น:** Negative samples สำหรับ Stage 1

---

## ⚙️ เทคนิคที่ใช้เพิ่มความแม่นยำ

| เทคนิค | รายละเอียด |
|--------|-----------|
| Data Augmentation | Pitch shift, Time stretch, Noise, Volume |
| Mixup | ผสมเสียง 2 ไฟล์สร้างข้อมูลใหม่ |
| Class Weight | แก้ปัญหา class imbalance |
| Label Smoothing | ป้องกัน overconfidence |
| Attention Mechanism | โฟกัสส่วนสำคัญของเสียง |
| Bidirectional LSTM | จับ temporal pattern ทั้ง 2 ทิศทาง |
| Batch Normalization | เทรนเสถียรและเร็วขึ้น |
| AdamW Optimizer | Adam + Weight Decay ป้องกัน overfit |
| EarlyStopping | หยุดเทรนเมื่อไม่ดีขึ้น |
| ReduceLROnPlateau | ลด learning rate อัตโนมัติ |

---

## ❓ แก้ปัญหาที่พบบ่อย

**`ModuleNotFoundError: No module named 'librosa'`**
```bash
pip install librosa
```

**`Cannot deserialize object of type 'AttentionLayer'`**  
ตรวจสอบว่า `app_dbl.py` มีการลงทะเบียน `AttentionLayer` ด้วย `@tf.keras.saving.register_keras_serializable()`

**`Could not load the model`**  
ตรวจสอบว่าไฟล์ `.keras`, `.json`, `.npy` อยู่ใน folder ที่ถูกต้อง

**`tensorflow` ติดตั้งไม่ได้บน Python 3.11+`**  
ใช้ Python 3.10 ครับ

---

## 📄 License

MIT License
