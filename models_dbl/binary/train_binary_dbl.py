import os, glob, json
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
BASE_DIR   = 'D:/INFANT CRY'
DATA_DIR   = f'{BASE_DIR}/data_set_dbl'
SAVE_DIR   = f'{BASE_DIR}/models_dbl/binary'
RESULT_DIR = f'{BASE_DIR}/models_dbl/binary'

BABY_CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
NOT_BABY_DIR = f'{DATA_DIR}/not_baby'

SR      = 22050
N_MFCC  = 40
N_MELS  = 64
N_CHROMA= 12
MAX_LEN = 128
BATCH   = 32
EPOCHS  = 200
N_FOLDS = 5

# augment ต่อ class
AUG_BABY     = 3   # เสียงเด็ก (1,551 × 3 = ~4,653)
AUG_NOT_BABY = 2   # ไม่ใช่เสียงเด็ก (2,000 × 2 = ~4,000)

os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════
#  FEATURE EXTRACTION
#  MFCC + Delta + Delta2 + Mel + Chroma
# ══════════════════════════════════════════
def extract_features(audio, sr=SR, max_len=MAX_LEN):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel    = librosa.power_to_db(
                librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                               n_fft=2048, hop_length=512), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA,
                                          n_fft=2048, hop_length=512)
    combined = np.vstack([mfcc, delta, delta2, mel, chroma])  # (196, frames)

    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0,0),(0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]

    return combined[..., np.newaxis].astype(np.float32)

# ══════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════
def augment_audio(y, sr, n=3):
    results = [y]
    results.append(y + 0.004 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    results.append(librosa.effects.time_stretch(y, rate=1.1))
    results.append(librosa.effects.time_stretch(y, rate=0.9))
    results.append(y * np.random.uniform(0.8, 1.2))
    return results[:n]

# ══════════════════════════════════════════
#  โหลด DATASET
# ══════════════════════════════════════════
print("\n📂 โหลด dataset...")
X, y_labels = [], []

# Positive: เสียงเด็ก (label = 1)
baby_files = []
for cls in BABY_CLASSES:
    baby_files += glob.glob(f'{DATA_DIR}/{cls}/*.wav')

print(f"  เสียงเด็ก (positive): {len(baby_files)} ไฟล์")
for f in baby_files:
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _  = librosa.effects.trim(audio, top_db=20)
        audio     = librosa.util.normalize(audio)
        for aug in augment_audio(audio, sr, n=AUG_BABY):
            aug = librosa.util.normalize(aug.astype(np.float32))
            X.append(extract_features(aug))
            y_labels.append(1)
    except Exception as e:
        print(f'    Skip: {e}')

# Negative: ไม่ใช่เสียงเด็ก (label = 0)
not_baby_files = glob.glob(f'{NOT_BABY_DIR}/*.wav')
print(f"  ไม่ใช่เสียงเด็ก (negative): {len(not_baby_files)} ไฟล์")
for f in not_baby_files:
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _  = librosa.effects.trim(audio, top_db=20)
        audio     = librosa.util.normalize(audio)
        for aug in augment_audio(audio, sr, n=AUG_NOT_BABY):
            aug = librosa.util.normalize(aug.astype(np.float32))
            X.append(extract_features(aug))
            y_labels.append(0)
    except Exception as e:
        print(f'    Skip: {e}')

print(f"\n✅ รวม: {len(X)} ไฟล์")
print(f"  เสียงเด็ก      : {y_labels.count(1)}")
print(f"  ไม่ใช่เสียงเด็ก: {y_labels.count(0)}")

# ══════════════════════════════════════════
#  ENCODE
# ══════════════════════════════════════════
X     = np.array(X)
y_int = np.array(y_labels)
y_enc = to_categorical(y_int, num_classes=2)

print(f"\n  Input shape: {X.shape[1:]}")

# ══════════════════════════════════════════
#  ATTENTION LAYER
# ══════════════════════════════════════════
class AttentionLayer(layers.Layer):
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
        alpha = tf.expand_dims(alpha, -1)
        return tf.reduce_sum(x * alpha, axis=1)

# ══════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════
def build_binary(input_shape, num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # CNN Block 1
    x = layers.Conv2D(32, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # CNN Block 2
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # CNN Block 3
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    # Reshape → LSTM
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # Attention
    x = AttentionLayer()(x)

    # Classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Binary_CNN_MFCC_BiLSTM_Attention')

# ══════════════════════════════════════════
#  K-FOLD TRAINING
# ══════════════════════════════════════════
print(f"\n🔀 เริ่ม {N_FOLDS}-Fold Cross Validation...")

skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_models  = []
fold_scores  = []
all_val_true = []
all_val_pred = []
all_val_prob = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_int)):
    print(f"\n{'='*50}")
    print(f"  Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*50}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_enc[train_idx], y_enc[val_idx]

    # Normalize
    mean = X_train.mean()
    std  = X_train.std()
    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std

    if fold == 0:
        np.save(f'{SAVE_DIR}/norm_stats_binary_dbl.npy', [mean, std])

    # Class weight
    cw_int = np.argmax(y_train, axis=1)
    cws    = compute_class_weight('balanced', classes=np.unique(cw_int), y=cw_int)
    cw     = dict(enumerate(cws))

    # Build + Compile
    model = build_binary(input_shape=X_train_n.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.3, patience=10, min_lr=1e-7, verbose=0),
        ModelCheckpoint(f'{SAVE_DIR}/fold_{fold+1}_binary.keras',
                        save_best_only=True, verbose=0),
        CSVLogger(f'{SAVE_DIR}/fold_{fold+1}_log.csv')
    ]

    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )

    best_val = max(history.history['val_accuracy'])
    fold_scores.append(best_val)
    fold_models.append(model)

    probs = model.predict(X_val_n, verbose=0)
    all_val_true.extend(np.argmax(y_val, axis=1))
    all_val_pred.extend(np.argmax(probs, axis=1))
    all_val_prob.extend(probs)

    print(f"\n  Fold {fold+1} Best Val Accuracy: {best_val*100:.2f}%")

# ══════════════════════════════════════════
#  บันทึก Best Model
# ══════════════════════════════════════════
best_fold = int(np.argmax(fold_scores))
import shutil
shutil.copy(f'{SAVE_DIR}/fold_{best_fold+1}_binary.keras',
            f'{SAVE_DIR}/best_model_binary_dbl.keras')

with open(f'{SAVE_DIR}/labels_binary_dbl.json', 'w') as f:
    json.dump(['not_baby', 'baby'], f)

print(f"\n⭐ Best Fold: {best_fold+1} ({fold_scores[best_fold]*100:.2f}%)")

# ══════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════
label_names = ['not_baby', 'baby']
report = classification_report(all_val_true, all_val_pred,
                                target_names=label_names, digits=4)

print(f"\n{'='*50}")
print("📊 Classification Report:")
print(f"{'='*50}")
print(report)

# ROC-AUC
all_val_prob = np.array(all_val_prob)
try:
    roc_auc = roc_auc_score(all_val_true, all_val_prob[:, 1])
    print(f"🎯 ROC-AUC: {roc_auc:.4f}")
except Exception as e:
    roc_auc = 0
    print(f"ROC-AUC skip: {e}")

# บันทึก report
with open(f'{SAVE_DIR}/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Binary Classifier — {N_FOLDS}-Fold Cross Validation\n")
    f.write(f"Features: MFCC + Delta + Delta2 + Mel + Chroma\n")
    f.write(f"Model: CNN + BiLSTM + Attention\n\n")
    f.write(f"Fold Scores: {[f'{s*100:.2f}%' for s in fold_scores]}\n")
    f.write(f"Mean Accuracy: {np.mean(fold_scores)*100:.2f}%\n")
    f.write(f"Std: {np.std(fold_scores)*100:.2f}%\n\n")
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(all_val_true, all_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_names,
            yticklabels=label_names,
            cmap='Blues', linewidths=0.5)
plt.title('Confusion Matrix — Binary Classifier', fontsize=13, pad=12)
plt.ylabel('Actual', fontsize=11)
plt.xlabel('Predicted', fontsize=11)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/confusion_matrix.png', dpi=150)
plt.close()
print(f"✅ บันทึก confusion_matrix.png")

print(f"\n{'='*50}")
print(f"✅ เทรนเสร็จ!")
print(f"  Fold Scores       : {[f'{s*100:.2f}%' for s in fold_scores]}")
print(f"  Mean Accuracy     : {np.mean(fold_scores)*100:.2f}%")
print(f"  Best Fold         : {best_fold+1} ({fold_scores[best_fold]*100:.2f}%)")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  โมเดล             : {SAVE_DIR}/best_model_binary_dbl.keras")
print(f"  Labels            : {SAVE_DIR}/labels_binary_dbl.json")
print(f"  Norm stats        : {SAVE_DIR}/norm_stats_binary_dbl.npy")
print(f"  Report            : {SAVE_DIR}/classification_report.txt")
print(f"  Confusion Matrix  : {SAVE_DIR}/confusion_matrix.png")
print(f"{'='*50}")