"""
ประเมินผลโมเดล CNN + MFCC + Attention + BiLSTM (Emotion Classifier)
==================================================================
ใช้ไฟล์เสียงดิบ (ไม่ augment) จากโฟลเดอร์ data_set/<emotion>/*.wav
แบ่ง train/test แบบ stratified (80/20, random_state=42) แล้ววัดผลเฉพาะฝั่ง test

หมายเหตุสำคัญ:
- ไฟล์ดิบที่ใช้ทำ test ชุดนี้ อาจเคยถูกนำไป augment แล้วใช้เทรนโมเดลไปแล้ว
  (เพราะตอนเทรนใช้ augmented version ของทุกไฟล์) ดังนั้นตัวเลขที่ได้จะ
  "ดูดีกว่าความเป็นจริงเล็กน้อย" (optimistic bias) — ถ้าต้องการค่าที่แม่นยำ
  100% ควรใช้ไฟล์เสียงชุดใหม่ที่ไม่เคยผ่านการเทรนเลย
- ใช้ norm_stats_CNN_MFCC.npy ตัวเดิมจากตอนเทรน ห้ามคำนวณ mean/std ใหม่
"""

import os, glob, json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── ตั้งค่า font ให้รองรับภาษาไทย (ป้องกันตัวอักษรขึ้นเป็นกล่องเปล่า) ──
_THAI_FONT_FILES = [
    'D:/INFANT CRY/fonts/KANIT-LIGHT.TTF',
    'C:/Windows/Fonts/tahoma.ttf',
    'C:/Windows/Fonts/leelawUI.ttf',
    'C:/Windows/Fonts/angsa.ttf',
]
_thai_font_prop = None
for _fpath in _THAI_FONT_FILES:
    if os.path.exists(_fpath):
        fm.fontManager.addfont(_fpath)
        _thai_font_prop = fm.FontProperties(fname=_fpath)
        _font_name = _thai_font_prop.get_name()
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [_font_name] + plt.rcParams['font.sans-serif']
        break
if _thai_font_prop is None:
    print("⚠️  ไม่พบไฟล์ font ภาษาไทยที่ path มาตรฐาน — กราฟอาจแสดงตัวอักษรไทยเป็นกล่องเปล่า")
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════
#  CONFIG — แก้ path ให้ตรงกับเครื่องคุณ
# ══════════════════════════════════════════
DATA_DIR     = 'D:/INFANT CRY/data_set'
MODEL_PATH   = 'D:/INFANT CRY/models/cnn_mfcc/best_model_CNN_MFCC.keras'
LABELS_PATH  = 'D:/INFANT CRY/models/cnn_mfcc/labels_CNN_MFCC.json'
NORM_PATH    = 'D:/INFANT CRY/models/cnn_mfcc/norm_stats_CNN_MFCC.npy'
RESULT_DIR   = 'D:/INFANT CRY/results'

EMOTIONS = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
SR       = 22050
N_MFCC   = 40
MAX_LEN  = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(RESULT_DIR, exist_ok=True)


# ══════════════════════════════════════════
#  ATTENTION LAYER (ต้องตรงกับตอนเทรนเป๊ะ)
# ══════════════════════════════════════════
@tf.keras.saving.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform', trainable=True, name='attn_W')
        self.b = self.add_weight(shape=(input_shape[-1],),
                                  initializer='zeros', trainable=True, name='attn_b')
        self.u = self.add_weight(shape=(input_shape[-1],),
                                  initializer='glorot_uniform', trainable=True, name='attn_u')
        super().build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.u, axes=1)
        alpha = tf.nn.softmax(score, axis=1)
        alpha = tf.expand_dims(alpha, -1)
        return tf.reduce_sum(x * alpha, axis=1)

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════
#  FEATURE EXTRACTION — เหมือนตอนเทรน (ไม่ augment)
# ══════════════════════════════════════════
def extract_mfcc(audio, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0, 0), (0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]
    return combined[..., np.newaxis].astype(np.float32)


# ══════════════════════════════════════════
#  1) รวบรวมรายชื่อไฟล์ + แบ่ง test set แบบ stratified
# ══════════════════════════════════════════
print("📂 รวบรวมรายชื่อไฟล์เสียง...")
file_paths, file_labels = [], []
for label in EMOTIONS:
    files = sorted(glob.glob(f'{DATA_DIR}/{label}/*.wav'))
    print(f"  {label:15s}: {len(files)} ไฟล์")
    file_paths.extend(files)
    file_labels.extend([label] * len(files))

_, test_paths, _, test_labels = train_test_split(
    file_paths, file_labels,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=file_labels
)
print(f"\n✅ Test set: {len(test_paths)} ไฟล์")
for lbl in EMOTIONS:
    print(f"  {lbl:15s}: {test_labels.count(lbl)}")

# ══════════════════════════════════════════
#  2) Extract feature จากไฟล์ test (ไม่ augment)
# ══════════════════════════════════════════
print("\n🎛️  สกัดคุณลักษณะ MFCC...")
X_test, y_test_labels = [], []
for f, lbl in zip(test_paths, test_labels):
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.util.normalize(audio)
        feat = extract_mfcc(audio, sr=sr)
        X_test.append(feat)
        y_test_labels.append(lbl)
    except Exception as e:
        print(f'    Skip {f}: {e}')

X_test = np.array(X_test)

# ══════════════════════════════════════════
#  3) โหลด labels + normalize ด้วยค่าตอนเทรน
# ══════════════════════════════════════════
with open(LABELS_PATH) as f:
    class_names = json.load(f)  # ลำดับ index ตรงกับตอนเทรน

label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}
y_true = np.array([label_to_idx[lbl] for lbl in y_test_labels])

mean, std = np.load(NORM_PATH)
X_test = (X_test - mean) / std

# ══════════════════════════════════════════
#  4) โหลดโมเดล + ทำนาย
# ══════════════════════════════════════════
print("\n🧠 โหลดโมเดล...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})

print("🔮 ทำนายผล...")
probs = model.predict(X_test, verbose=0)          # shape: (N, 5)
y_pred = np.argmax(probs, axis=1)
confidences = np.max(probs, axis=1)                # ความมั่นใจของคำตอบที่โมเดลเลือก

# ══════════════════════════════════════════
#  5) METRICS
# ══════════════════════════════════════════
acc = accuracy_score(y_true, y_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro', zero_division=0
)
precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted', zero_division=0
)
precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names))
)

try:
    auc_ovr = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
except Exception as e:
    auc_ovr = None
    print(f"⚠️  คำนวณ ROC-AUC ไม่ได้: {e}")

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

# ── Confidence stats ──
correct_mask = (y_pred == y_true)
conf_correct = confidences[correct_mask].mean() if correct_mask.any() else float('nan')
conf_wrong   = confidences[~correct_mask].mean() if (~correct_mask).any() else float('nan')

# ══════════════════════════════════════════
#  6) แสดงผลสรุป
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 สรุปผลการประเมิน (Emotion Model — CNN+MFCC+Attention+BiLSTM)")
print("=" * 60)
print(f"  Accuracy         : {acc*100:.2f}%")
print(f"  Precision (macro): {precision_macro*100:.2f}%   (weighted: {precision_w*100:.2f}%)")
print(f"  Recall    (macro): {recall_macro*100:.2f}%   (weighted: {recall_w*100:.2f}%)")
print(f"  F1-score  (macro): {f1_macro*100:.2f}%   (weighted: {f1_w*100:.2f}%)")
if auc_ovr is not None:
    print(f"  ROC-AUC (macro, OvR): {auc_ovr:.4f}")
print(f"\n  Confidence เฉลี่ยตอนทายถูก : {conf_correct*100:.2f}%")
print(f"  Confidence เฉลี่ยตอนทายผิด : {conf_wrong*100:.2f}%")

print("\n📋 Per-class metrics:")
print(f"  {'Class':15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
for i, cls in enumerate(class_names):
    print(f"  {cls:15s} {precision_pc[i]*100:9.2f}% {recall_pc[i]*100:9.2f}% {f1_pc[i]*100:9.2f}% {support_pc[i]:8d}")

print("\n📄 Classification report (sklearn):")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# ══════════════════════════════════════════
#  7) บันทึกผลเป็นไฟล์
# ══════════════════════════════════════════
summary_path = f'{RESULT_DIR}/eval_CNN_MFCC_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Precision (macro/weighted): {precision_macro*100:.2f}% / {precision_w*100:.2f}%\n")
    f.write(f"Recall (macro/weighted): {recall_macro*100:.2f}% / {recall_w*100:.2f}%\n")
    f.write(f"F1 (macro/weighted): {f1_macro*100:.2f}% / {f1_w*100:.2f}%\n")
    if auc_ovr is not None:
        f.write(f"ROC-AUC (macro, OvR): {auc_ovr:.4f}\n")
    f.write(f"Confidence เฉลี่ย (ถูก/ผิด): {conf_correct*100:.2f}% / {conf_wrong*100:.2f}%\n\n")
    f.write("Per-class:\n")
    for i, cls in enumerate(class_names):
        f.write(f"  {cls:15s} P={precision_pc[i]*100:.2f}% R={recall_pc[i]*100:.2f}% F1={f1_pc[i]*100:.2f}% n={support_pc[i]}\n")
print(f"\n💾 บันทึกสรุปผลไว้ที่: {summary_path}")

# ══════════════════════════════════════════
#  8) Confusion Matrix (heatmap)
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — Emotion Model\nAccuracy: {acc*100:.2f}%')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)
fig.colorbar(im, ax=ax, label='count')
fig.tight_layout()
cm_path = f'{RESULT_DIR}/confusion_matrix_CNN_MFCC_eval.png'
fig.savefig(cm_path, dpi=150)
print(f"💾 บันทึก confusion matrix ไว้ที่: {cm_path}")

# ══════════════════════════════════════════
#  9) Confidence distribution plot
# ══════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.hist(confidences[correct_mask], bins=20, alpha=0.6, label='ทายถูก', color='#34D399')
ax2.hist(confidences[~correct_mask], bins=20, alpha=0.6, label='ทายผิด', color='#FF6B6B')
ax2.set_xlabel('Confidence (max softmax prob)'); ax2.set_ylabel('จำนวนไฟล์')
ax2.set_title('การกระจายตัวของ Confidence')
ax2.legend()
fig2.tight_layout()
conf_path = f'{RESULT_DIR}/confidence_distribution_CNN_MFCC.png'
fig2.savefig(conf_path, dpi=150)
print(f"💾 บันทึกกราฟ confidence ไว้ที่: {conf_path}")

print("\n✅ เสร็จสิ้น")