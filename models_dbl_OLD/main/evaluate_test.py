import os, glob, json
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════
# CONFIG — ต้องตรงกับ train_main_dbl.py
# ══════════════════════════════════════════
DATA_DIR  = 'D:/INFANT CRY/data_set_dbl'
SAVE_DIR  = 'D:/INFANT CRY/models_dbl/main'
EMOTIONS  = ['belly_pain','burping','discomfort','hungry','tired']
SR=22050; N_MFCC=40; N_MELS=64; N_CHROMA=12; MAX_LEN=128

# ══════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════
def extract_features(audio, sr=SR, max_len=MAX_LEN):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel    = librosa.power_to_db(librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA, n_fft=2048, hop_length=512)
    combined = np.vstack([mfcc, delta, delta2, mel, chroma])
    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0,0),(0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]
    return combined[..., np.newaxis].astype(np.float32)

# ══════════════════════════════════════════
# โหลด ORIGINAL FILES (ไม่ augment)
# ══════════════════════════════════════════
print("\n📂 โหลด original dataset (ไม่ augment)...")
X_raw, y_raw = [], []

for label in EMOTIONS:
    files = glob.glob(f'{DATA_DIR}/{label}/*.wav')
    print(f"  {label:15s}: {len(files)} ไฟล์")
    for f in files:
        try:
            audio, sr = librosa.load(f, sr=SR, mono=True)
            audio, _  = librosa.effects.trim(audio, top_db=20)
            audio     = librosa.util.normalize(audio).astype(np.float32)
            X_raw.append(extract_features(audio))
            y_raw.append(label)
        except Exception as e:
            print(f'    Skip: {e}')

print(f"\n✅ รวม: {len(X_raw)} ไฟล์ (original เท่านั้น)")

# ══════════════════════════════════════════
# แยก TEST SET 20%
# ══════════════════════════════════════════
X_raw = np.array(X_raw)
le    = LabelEncoder()
y_int = le.fit_transform(y_raw)

# แยก test set ด้วย random_state เดิม
_, X_test, _, y_test = train_test_split(
    X_raw, y_int,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)

print(f"\n📊 Test set: {len(X_test)} ไฟล์")
for i, lbl in enumerate(le.classes_):
    count = np.sum(y_test == i)
    print(f"  {lbl:15s}: {count} ไฟล์")

# ══════════════════════════════════════════
# Normalize ด้วย norm_stats ของโมเดล
# ══════════════════════════════════════════
norm = np.load(f'{SAVE_DIR}/norm_stats_main_dbl.npy')
mean, std = norm[0], norm[1]
X_test_n = (X_test - mean) / std

# ══════════════════════════════════════════
# โหลดโมเดล
# ══════════════════════════════════════════
@tf.keras.saving.register_keras_serializable()
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
    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.u, axes=1)
        alpha = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * tf.expand_dims(alpha, -1), axis=1)

print("\n📦 โหลดโมเดล...")
model = tf.keras.models.load_model(
    f'{SAVE_DIR}/best_model_main_dbl.keras',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# ══════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════
print("🔍 กำลัง predict...")
probs   = model.predict(X_test_n, verbose=1)
y_pred  = np.argmax(probs, axis=1)

# ══════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════
label_names = le.classes_.tolist()
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred,
                                target_names=label_names, digits=4)

print(f"\n{'='*55}")
print(f"📊 Independent Test Results")
print(f"{'='*55}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"\n{report}")

# ROC-AUC
try:
    y_true_bin = label_binarize(y_test, classes=list(range(5)))
    roc_auc    = roc_auc_score(y_true_bin, probs,
                                multi_class='ovr', average='macro')
    print(f"  ROC-AUC (macro, OvR): {roc_auc:.4f}")
except Exception as e:
    roc_auc = 0
    print(f"  ROC-AUC skip: {e}")

# ══════════════════════════════════════════
# บันทึก Report
# ══════════════════════════════════════════
with open(f'{SAVE_DIR}/test_confusion_matrix_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Independent Test Results\n")
    f.write(f"Model: best_model_main_dbl.keras\n")
    f.write(f"Features: MFCC + Delta + Delta2 + Mel + Chroma\n")
    f.write(f"Test Set: 20% of original files (no augmentation)\n\n")
    f.write(f"Test Accuracy : {acc*100:.2f}%\n\n")
    f.write(report)
    f.write(f"\nROC-AUC (macro, OvR): {roc_auc:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_names,
            yticklabels=label_names,
            cmap='Blues', linewidths=0.5)
plt.title('Confusion Matrix — Independent Test Set', fontsize=13, pad=12)
plt.ylabel('Actual', fontsize=11)
plt.xlabel('Predicted', fontsize=11)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/test_confusion_matrix.png', dpi=150)
plt.close()

print(f"\n✅ บันทึกผลแล้ว!")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  Report        : {SAVE_DIR}/test_report.txt")
print(f"  Confusion     : {SAVE_DIR}/test_confusion_matrix.png")