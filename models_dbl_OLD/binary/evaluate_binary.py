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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
DATA_DIR     = 'D:/INFANT CRY/data_set_dbl'
SAVE_DIR     = 'D:/INFANT CRY/models_dbl/binary'
BABY_CLASSES = ['belly_pain','burping','discomfort','hungry','tired']
NOT_BABY_DIR = f'{DATA_DIR}/not_baby'
SR=22050; N_MFCC=40; N_MELS=64; N_CHROMA=12; MAX_LEN=128

# ══════════════════════════════════════════
#  FEATURE EXTRACTION
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
#  โหลด DATASET (ไม่ augment)
# ══════════════════════════════════════════
print("\n📂 โหลด dataset (original เท่านั้น)...")
X_raw, y_raw = [], []

# Positive: เสียงเด็ก
baby_files = []
for cls in BABY_CLASSES:
    baby_files += glob.glob(f'{DATA_DIR}/{cls}/*.wav')
print(f"  เสียงเด็ก (positive): {len(baby_files)} ไฟล์")
for f in baby_files:
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _  = librosa.effects.trim(audio, top_db=20)
        audio     = librosa.util.normalize(audio).astype(np.float32)
        X_raw.append(extract_features(audio))
        y_raw.append(1)
    except Exception as e:
        print(f'    Skip: {e}')

# Negative: ไม่ใช่เสียงเด็ก
not_baby_files = glob.glob(f'{NOT_BABY_DIR}/*.wav')
print(f"  ไม่ใช่เสียงเด็ก (negative): {len(not_baby_files)} ไฟล์")
for f in not_baby_files:
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _  = librosa.effects.trim(audio, top_db=20)
        audio     = librosa.util.normalize(audio).astype(np.float32)
        X_raw.append(extract_features(audio))
        y_raw.append(0)
    except Exception as e:
        print(f'    Skip: {e}')

print(f"\n✅ รวม: {len(X_raw)} ไฟล์")
print(f"  เสียงเด็ก      : {y_raw.count(1)}")
print(f"  ไม่ใช่เสียงเด็ก: {y_raw.count(0)}")

# ══════════════════════════════════════════
#  แยก TEST SET 20%
# ══════════════════════════════════════════
X_raw = np.array(X_raw)
y_int = np.array(y_raw)

_, X_test, _, y_test = train_test_split(
    X_raw, y_int,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)

print(f"\n📊 Test set: {len(X_test)} ไฟล์")
print(f"  เสียงเด็ก      : {np.sum(y_test == 1)} ไฟล์")
print(f"  ไม่ใช่เสียงเด็ก: {np.sum(y_test == 0)} ไฟล์")

# ══════════════════════════════════════════
#  Normalize
# ══════════════════════════════════════════
norm = np.load(f'{SAVE_DIR}/norm_stats_binary_dbl.npy')
mean, std  = norm[0], norm[1]
X_test_n   = (X_test - mean) / std

# ══════════════════════════════════════════
#  โหลดโมเดล
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
    f'{SAVE_DIR}/best_model_binary_dbl.keras',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# ══════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════
print("🔍 กำลัง predict...")
probs  = model.predict(X_test_n, verbose=1)
y_pred = np.argmax(probs, axis=1)

# ══════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════
label_names = ['not_baby', 'baby']
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred,
                                target_names=label_names, digits=4)

print(f"\n{'='*55}")
print(f"📊 Binary Classifier — Independent Test Results")
print(f"{'='*55}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"\n{report}")

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_test, probs[:, 1])
    print(f"  ROC-AUC: {roc_auc:.4f}")
except Exception as e:
    roc_auc = 0
    print(f"  ROC-AUC skip: {e}")

# ══════════════════════════════════════════
#  บันทึก Report
# ══════════════════════════════════════════
with open(f'{SAVE_DIR}/test_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Binary Classifier — Independent Test Results\n")
    f.write(f"Model: best_model_binary_dbl.keras\n")
    f.write(f"Features: MFCC + Delta + Delta2 + Mel + Chroma\n")
    f.write(f"Test Set: 20% of original files (no augmentation)\n\n")
    f.write(f"Test Set Size:\n")
    f.write(f"  เสียงเด็ก      : {np.sum(y_test == 1)} ไฟล์\n")
    f.write(f"  ไม่ใช่เสียงเด็ก: {np.sum(y_test == 0)} ไฟล์\n")
    f.write(f"  รวม            : {len(X_test)} ไฟล์\n\n")
    f.write(f"Test Accuracy : {acc*100:.2f}%\n\n")
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.4f}\n")

# ══════════════════════════════════════════
#  Confusion Matrix
# ══════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_names,
            yticklabels=label_names,
            cmap='Blues', linewidths=0.5,
            annot_kws={'size': 14})
plt.title('Confusion Matrix — Binary Classifier\n(Independent Test Set)',
          fontsize=13, pad=12)
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
