# INFANT-CRY-Project-NEW-
# 🍼 Baby Cry Detector

ระบบวิเคราะห์เสียงร้องของทารก ด้วย Deep Learning (CNN + MFCC + Attention + BiLSTM)  
จำแนกอารมณ์ทารกได้ 5 ประเภท: **ปวดท้อง / เรอ / ไม่สบาย / หิว / เหนื่อย**

## 🖥️ การติดตั้งสำหรับนักพัฒนา (Training / Development)

### 1. ติดตั้ง Python

ใช้ **Python 3.10** (แนะนำ — เข้ากันได้กับ TensorFlow 2.x)  
ดาวน์โหลด: https://www.python.org/downloads/release/python-3100/

> ⚠️ Python 3.11+ อาจมีปัญหากับ TensorFlow บางเวอร์ชัน

---

### 2. ติดตั้ง Library ทั้งหมด

```bash
pip install tensorflow==2.13.0
pip install tensorflow-hub
pip install librosa
pip install scikit-learn
pip install numpy
pip install matplotlib
```

หรือติดตั้งทีเดียวด้วย:

```bash
pip install tensorflow==2.13.0 tensorflow-hub librosa scikit-learn numpy matplotlib
```

---

### 3. รันการเทรนโมเดล

```bash
cd models/cnn_mfcc
python train_CNN_MFCC2.py
```

---

### 4. ทดสอบโมเดล

```bash
cd models/cnn_mfcc
python predict_CNN_MFCC2.py
```

---

## 📱 การติดตั้งสำหรับผู้ใช้ทั่วไป (ใช้แค่ App)

### สิ่งที่ต้องมี(ตัวอย่าง)

| รายการ | รายละเอียด |
|---|---|
| Python | 3.10 |
| ไฟล์โมเดล | `best_model_CNN_MFCC2.keras` |
| ไฟล์ labels | `labels_CNN_MFCC2.json` |
| ไฟล์ค่า normalize | `norm_stats_CNN_MFCC2.npy` |
| ไฟล์ app | `app_CNN_MFCC.py` |

> ไฟล์ทั้ง 4 ต้องอยู่ใน **folder เดียวกัน**

---

### ขั้นตอนติดตั้ง

**1. ติดตั้ง Python 3.10**  
https://www.python.org/downloads/release/python-3100/

> ✅ ติ๊ก **"Add Python to PATH"** ตอนติดตั้ง

**2. ติดตั้ง Library ที่จำเป็น**

เปิด Command Prompt แล้วพิมพ์:

```bash
pip install tensorflow==2.13.0 tensorflow-hub librosa numpy matplotlib
```

**3. รัน App**

```bash
python app_CNN_MFCC.py
```

หรือดับเบิลคลิกที่ไฟล์ `app_CNN_MFCC.py` ได้เลย

---

## 📦 Library ที่ใช้ทั้งหมด

| Library | เวอร์ชันแนะนำ | ใช้ทำอะไร |
|---|---|---|
| `tensorflow` | 2.13.0 | โหลดและรันโมเดล |
| `librosa` | ≥ 0.10 | แปลงไฟล์เสียงเป็น MFCC |
| `numpy` | ≥ 1.23 | คำนวณ array |
| `scikit-learn` | ≥ 1.3 | เทรนโมเดล (LabelEncoder, train_test_split) |
| `matplotlib` | ≥ 3.7 | กราฟวงกลมใน App |
| `tensorflow-hub` | ≥ 0.14 | โมเดล YAMNet (ถ้าใช้ branch yamnet) |

---

## 🧠 สถาปัตยกรรมโมเดล

```
Input (120, 128, 1)
    │
    ├── CNN Block 1  →  Conv2D(32) × 2 + BN + ReLU + MaxPool + Dropout
    ├── CNN Block 2  →  Conv2D(64) × 2 + BN + ReLU + MaxPool + Dropout
    ├── CNN Block 3  →  Conv2D(128) × 2 + BN + ReLU + MaxPool + Dropout
    │
    ├── Reshape → (time_steps, features)
    │
    ├── BiLSTM(128) + Dropout
    ├── BiLSTM(64)  + Dropout
    │
    ├── Attention Layer
    │
    ├── Dense(128) + BN + Dropout
    ├── Dense(64)  + Dropout
    │
    └── Dense(5) → Softmax
```

**Feature:** MFCC (40) + Delta (40) + Delta² (40) = 120 channels  
**Loss:** Focal Loss (γ=2.0)  
**Optimizer:** AdamW (lr=0.001, weight_decay=1e-4)

---

## 📊 ผลการเทรน

| โมเดล | Val Accuracy | Best Epoch | หมายเหตุ |
|---|---|---|---|
| CNN_MFCC v1 | 90.66% | 70 | Cross Entropy, AUG ไม่สมดุล |
| CNN_MFCC v2 | **97.16%** | 47 | Focal Loss, AUG สมดุล |

---

## ❓ แก้ปัญหาที่พบบ่อย

**`ModuleNotFoundError: No module named 'librosa'`**
```bash
pip install librosa
```

**`Could not load the model`**  
ตรวจสอบว่าไฟล์ `.keras`, `.json`, `.npy` อยู่ใน folder เดียวกับ `app_CNN_MFCC.py`

**`tensorflow` ติดตั้งไม่ได้บน Python 3.11+`**  
ใช้ Python 3.10 ครับ

---

## 📄 License

MIT License# INFANT-CRY-APP-
