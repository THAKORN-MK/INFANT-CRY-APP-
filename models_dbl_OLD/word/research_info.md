# ข้อมูลที่เกี่ยวข้อง — Baby Cry Emotion Detection

## 1. ภาพรวมระบบ

ระบบตรวจจับและวิเคราะห์อารมณ์จากเสียงร้องของทารก โดยใช้สถาปัตยกรรม **2-Stage Pipeline** ที่เทรนเองทั้งหมด ประกอบด้วย:

```
เสียงร้องทารก (.wav)
        ↓
┌─────────────────────────────────┐
│  Stage 1: Binary Classifier     │
│  CNN + MFCC + BiLSTM + Attention│
│  "เสียงเด็กหรือไม่?"            │
└─────────────────────────────────┘
        ├── ❌ ไม่ใช่ → หยุด
        └── ✅ ใช่
                  ↓
    ┌─────────────────────────────────┐
    │  Stage 2: Emotion Classifier    │
    │  CNN + MFCC + Mel + Chroma      │
    │       + BiLSTM + Attention      │
    │  "อารมณ์คืออะไร?"              │
    └─────────────────────────────────┘
                  ↓
    5 อารมณ์ตาม Dunstan Baby Language
```

---

## 2. Dataset ที่ใช้

### 2.1 InfantCry-DBL (Dataset หลัก)
- **ชื่อเต็ม:** InfantCry-DBL: A Two-Tier Annotated Corpus of Infant Cries Labelled with Dunstan Baby Language Categories
- **ผู้สร้าง:** Mohammed Tawfik (2026)
- **License:** CC BY 4.0
- **จำนวนไฟล์:** 1,551 ไฟล์ (54.4 นาที)
- **Cohen's κ:** 0.89 (95% CI 0.85–0.93) — ความน่าเชื่อถือของการ annotate สูงมาก

| Tier | รายละเอียด | จำนวน |
|------|-----------|-------|
| Tier 1 — Dunstan-Core | Studio quality จากวิดีโอต้นฉบับ Dunstan | 337 ไฟล์ |
| Tier 2 — InfantCry-1214 | Real-world จาก CryCeleb, YouTube, บ้าน | 1,214 ไฟล์ |

**การกระจายของ class:**

| Class | ความหมาย | จำนวน |
|-------|---------|-------|
| `neh` (hungry) | หิว — Sucking reflex | 217 |
| `owh` (tired) | ง่วง — Yawn-like | 716 |
| `heh` (discomfort) | ไม่สบาย — Skin stimulus | 168 |
| `eairh` (belly_pain) | ปวดท้อง — Lower wind | 170 |
| `eh` (burping) | เรอ — Upper wind | 280 |

### 2.2 ESC-50 (Negative Samples สำหรับ Binary Classifier)
- **ชื่อเต็ม:** ESC: Dataset for Environmental Sound Classification
- **ผู้สร้าง:** Karol J. Piczak (2015)
- **จำนวน:** 2,000 ไฟล์ — เสียงสิ่งแวดล้อม 50 ประเภท
- **ใช้เป็น:** Not_baby samples สำหรับ Stage 1

---

## 3. Dunstan Baby Language (DBL)

ทฤษฎีที่พัฒนาโดย Priscilla Dunstan (2006) อธิบายว่าทารกทั่วโลกมีเสียงร้องพื้นฐาน 5 แบบ ซึ่งเกิดจาก **Reflex ก่อนที่เสียงจะออกมา**

| เสียง | ความหมาย | Reflex |
|-------|---------|--------|
| **neh** | หิว | Sucking reflex กดเพดานปากอ่อน |
| **owh** | ง่วงนอน | Yawn — ปากเปิดกลม |
| **heh** | ไม่สบาย | ผิวหนังรู้สึกระคายเคือง |
| **eairh** | ปวดท้อง/ลม | ลมในช่องท้องส่วนล่าง |
| **eh** | ต้องการเรอ | ลมในหน้าอก |

---

## 4. Feature Extraction

### 4.1 MFCC (Mel-Frequency Cepstral Coefficients)
- เลียนแบบการรับรู้เสียงของหูมนุษย์
- แปลงเสียงเป็น frequency domain ด้วย Mel scale
- ใช้ **40 coefficients** ต่อ frame

### 4.2 Delta และ Delta-Delta
- **Delta (Δ):** อนุพันธ์อันดับ 1 ของ MFCC — บอกการเปลี่ยนแปลงของเสียง
- **Delta-Delta (ΔΔ):** อนุพันธ์อันดับ 2 — บอกความเร่งของการเปลี่ยนแปลง
- รวมกับ MFCC ได้ **120 features** ต่อ frame

### 4.3 Mel-Spectrogram
- แสดงพลังงานเสียงในแต่ละความถี่ตามเวลา
- ใช้ **64 Mel bins**
- แปลงเป็น dB scale ด้วย `power_to_db`

### 4.4 Chroma Features
- แสดงการกระจายพลังงานใน **12 โน้ตดนตรี**
- ช่วยจับ pitch pattern ของเสียงร้อง

### 4.5 Feature รวม
```
MFCC     : 40 features
Delta    : 40 features
Delta2   : 40 features
Mel      : 64 features
Chroma   : 12 features
─────────────────────
รวม      : 196 features × 128 frames → (196, 128, 1)
```

---

## 5. สถาปัตยกรรมโมเดล

### 5.1 Stage 1 — Binary Classifier
**วัตถุประสงค์:** แยกเสียงเด็กออกจากเสียงอื่น

```
Input (196, 128, 1)
  → CNN Block 1: Conv2D(32×2) → BN → ReLU → MaxPool → Dropout(0.25)
  → CNN Block 2: Conv2D(64×2) → BN → ReLU → MaxPool → Dropout(0.25)
  → CNN Block 3: Conv2D(128×2) → BN → ReLU → MaxPool → Dropout(0.3)
  → Reshape
  → BiLSTM(128) → Dropout(0.3)
  → BiLSTM(64)  → Dropout(0.3)
  → Attention Layer
  → Dense(128) → BN → Dropout(0.4)
  → Dense(64)  → Dropout(0.4)
  → Output: Dense(2, softmax) → [not_baby, baby]
```

### 5.2 Stage 2 — Emotion Classifier
**วัตถุประสงค์:** จำแนก 5 อารมณ์

```
Input (196, 128, 1)
  → CNN Block 1: Conv2D(32×2) → BN → ReLU → MaxPool → Dropout(0.25)
  → CNN Block 2: Conv2D(64×2) → BN → ReLU → MaxPool → Dropout(0.25)
  → CNN Block 3: Conv2D(128×2) → BN → ReLU → MaxPool → Dropout(0.3)
  → CNN Block 4: Conv2D(256)   → BN → ReLU → MaxPool → Dropout(0.3)
  → Reshape
  → BiLSTM(128) → Dropout(0.3)
  → BiLSTM(64)  → Dropout(0.3)
  → Attention Layer
  → Dense(256) → BN → Dropout(0.4)
  → Dense(128) → BN → Dropout(0.4)
  → Dense(64)  → Dropout(0.3)
  → Output: Dense(5, softmax) → 5 อารมณ์
```

---

## 6. เทคนิคเพิ่มความแม่นยำ

### 6.1 Data Augmentation
สร้างข้อมูลเพิ่มจากไฟล์เดิมเพื่อแก้ปัญหาข้อมูลน้อย

| เทคนิค | รายละเอียด |
|--------|-----------|
| Gaussian Noise | เพิ่ม noise เบาๆ (σ = 0.004) |
| Pitch Shift ±1.5 | เปลี่ยน pitch ±1.5 semitones |
| Pitch Shift ±2.0 | เปลี่ยน pitch ±2.0 semitones |
| Time Stretch ×1.1/×0.9 | เร่ง/ช้า 10% |
| Time Stretch ×1.2/×0.8 | เร่ง/ช้า 20% |
| Volume Shift | ปรับ volume ±20% |

**จำนวน augment ต่อ class:**
```
belly_pain : 170 × 12 = 2,040
burping    : 280 ×  8 = 2,240
discomfort : 168 × 12 = 2,016
hungry     : 217 × 10 = 2,170
tired      : 716 ×  3 = 2,148
```

### 6.2 Mixup Augmentation
ผสมข้อมูลสองตัวอย่างเพื่อสร้างข้อมูลใหม่
```
X_mix = λ × X₁ + (1-λ) × X₂
y_mix = λ × y₁ + (1-λ) × y₂
โดย λ ~ Beta(0.3, 0.3)
จำนวน: 500 samples
```

### 6.3 Class Weight Balancing
ปรับน้ำหนัก loss ของแต่ละ class เพื่อแก้ class imbalance โดยอัตโนมัติ

### 6.4 Label Smoothing
ป้องกัน overconfidence ของโมเดล โดยแทนที่ [0,1,0,0,0] ด้วย [0.02, 0.92, 0.02, 0.02, 0.02]
```
loss = CategoricalCrossentropy(label_smoothing=0.1)
```

### 6.5 Attention Mechanism
ให้โมเดลโฟกัสเฉพาะส่วนสำคัญของเสียง
```
score  = tanh(x·W + b)
α      = softmax(score·u)
output = Σ(α × x)
```

### 6.6 Bidirectional LSTM
อ่านข้อมูลทั้งไปข้างหน้าและย้อนกลับ จับ temporal pattern ได้ดีกว่า LSTM ทิศทางเดียว

### 6.7 Batch Normalization
ทำให้การเทรนเสถียรและเร็วขึ้น โดย normalize output ของแต่ละ layer

### 6.8 AdamW Optimizer
Adam + Weight Decay ป้องกัน overfitting
```
learning_rate = 0.001
weight_decay  = 1e-4
```

---

## 7. Callbacks ระหว่างเทรน

| Callback | หน้าที่ |
|---------|--------|
| EarlyStopping | หยุดเทรนเมื่อ val_loss ไม่ดีขึ้นใน 30 epoch |
| ReduceLROnPlateau | ลด learning rate ×0.3 เมื่อ val_loss ค้างใน 12 epoch |
| ModelCheckpoint | บันทึกโมเดลที่ดีที่สุดเท่านั้น |
| CSVLogger | บันทึก log ทุก epoch เป็นไฟล์ CSV |

---

## 8. การประเมินผล (Evaluation Metrics)

| Metric | สูตร | ความหมาย |
|--------|------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | ความถูกต้องโดยรวม |
| **Precision** | TP/(TP+FP) | ทำนาย Positive แล้วถูกกี่% |
| **Recall** | TP/(TP+FN) | หา Positive จริงเจอกี่% |
| **F1-Score** | 2×(P×R)/(P+R) | ค่าสมดุลของ Precision และ Recall |
| **ROC-AUC** | Area under ROC curve | ความสามารถแยก class (1.0 = สมบูรณ์) |
| **Confusion Matrix** | ตาราง actual vs predicted | เห็นว่าสับสน class ไหน |

---

## 9. Library ที่ใช้

| Library | Version | หน้าที่ |
|---------|---------|--------|
| librosa | 0.11.0 | Feature extraction (MFCC, Mel, Chroma) |
| tensorflow/keras | 2.15.0 | สร้างและเทรนโมเดล |
| scikit-learn | 1.7.2 | Metrics, LabelEncoder, Class Weight |
| numpy | 1.26.4 | จัดการ array |
| matplotlib | 3.10.8 | วาดกราฟ |
| seaborn | latest | Confusion Matrix |
| tkinter | built-in | GUI Application |

---

## 10. การอ้างอิง

```
[1] Tawfik, M. (2026). InfantCry-DBL: A two-tier annotated corpus of newborn
    cries labelled with the five Dunstan Baby Language categories.
    Mendeley Data, V1.

[2] Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.
    Proceedings of the ACM International Conference on Multimedia.
    DOI: 10.1145/2733373.2806390

[3] Dunstan, P. (2006). Dunstan Baby Language.
    Instructional video series.

[4] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
    mixup: Beyond Empirical Risk Minimization.
    ICLR 2018. arXiv:1710.09412

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2015).
    Neural Machine Translation by Jointly Learning to Align and Translate.
    ICLR 2015. arXiv:1409.0473

[6] Schuster, M., & Paliwal, K. K. (1997).
    Bidirectional Recurrent Neural Networks.
    IEEE Transactions on Signal Processing, 45(11), 2673-2681.
```
