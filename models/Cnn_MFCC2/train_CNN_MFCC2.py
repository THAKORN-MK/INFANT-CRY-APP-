import os, glob, json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, regularizers
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════

DATA_DIR   = 'D:/INFANT CRY/data_set'
SAVE_DIR   = '.'
RESULT_DIR = '../../results'

EMOTIONS  = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
SR        = 22050
N_MFCC    = 40
MAX_LEN   = 128
BATCH     = 16
EPOCHS    = 200

# ✅ ปรับให้ทุก class มี ~380 ไฟล์ เท่ากับ hungry
AUG_TIMES = {
    'belly_pain': 24,   # 16 × 24 = 384
    'burping':    48,   #  8 × 48 = 384
    'discomfort': 15,   # 27 × 15 = 405
    'hungry':      1,   # 382 × 1 = 382
    'tired':       16,  # 24 × 16 = 384
}

# ══════════════════════════════════════════
#  DEBUG
# ══════════════════════════════════════════
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
for label in EMOTIONS:
    path = f'{DATA_DIR}/{label}'
    files = glob.glob(path + '/*.wav')
    print(f"  {label}: exists={os.path.exists(path)}, "
          f"files={len(files)} → ~{len(files)*AUG_TIMES[label]} หลัง augment")

# ══════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════
def extract_mfcc(audio, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])

    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0,0),(0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]

    return combined[..., np.newaxis].astype(np.float32)

# ══════════════════════════════════════════
#  AUGMENTATION — เพิ่ม variation ให้ class ที่มีไฟล์น้อย
# ══════════════════════════════════════════
def augment_audio(y, sr, n=6):
    """
    มี 7 version ให้เลือก (index 0 = ต้นฉบับ)
    n คือจำนวนที่ต้องการ ถ้า n > 7 จะวนซ้ำพร้อม random noise
    """
    base = [
        y,                                                        # 0: ต้นฉบับ
        y + 0.004 * np.random.randn(len(y)),                      # 1: noise เบา
        librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5),       # 2: pitch +
        librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5),      # 3: pitch -
        librosa.effects.time_stretch(y, rate=1.1),                 # 4: เร็วขึ้น
        librosa.effects.time_stretch(y, rate=0.9),                 # 5: ช้าลง
        y * np.random.uniform(0.8, 1.2),                          # 6: volume
    ]
    results = base[:min(n, len(base))]

    # ถ้าต้องการมากกว่า 7 → วนซ้ำพร้อมเพิ่ม noise ให้ต่างกัน
    while len(results) < n:
        src = base[len(results) % len(base)]
        results.append(src + 0.002 * np.random.randn(len(src)))

    return results[:n]

def mixup(X, y, alpha=0.3, n_mix=200):
    X_mix, y_mix = [], []
    for _ in range(n_mix):
        i, j = np.random.choice(len(X), 2, replace=False)
        lam   = np.random.beta(alpha, alpha)
        X_mix.append(lam * X[i] + (1-lam) * X[j])
        y_mix.append(lam * y[i] + (1-lam) * y[j])
    return np.array(X_mix), np.array(y_mix)

# ══════════════════════════════════════════
#  FOCAL LOSS — โฟกัส class ที่ยากขึ้น
# ══════════════════════════════════════════
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss: ลงโทษ class ที่โมเดลมั่นใจผิด มากกว่า class ที่ถูก
    ช่วย class ที่มีข้อมูลน้อย (belly_pain, burping) ให้โมเดลใส่ใจมากขึ้น
    gamma=2.0: ยิ่งสูง ยิ่งโฟกัส hard example
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn

# ══════════════════════════════════════════
#  โหลด DATASET
# ══════════════════════════════════════════
print("\n📂 โหลด dataset...")
X, y_labels = [], []

for label in EMOTIONS:
    files = glob.glob(f'{DATA_DIR}/{label}/*.wav')
    aug   = AUG_TIMES[label]
    print(f"  {label:15s}: {len(files):4d} ไฟล์ → ~{len(files)*aug} หลัง augment")

    for f in files:
        try:
            audio, sr = librosa.load(f, sr=SR, mono=True)
            audio, _  = librosa.effects.trim(audio, top_db=20)
            audio     = librosa.util.normalize(audio)

            for aug_audio in augment_audio(audio, sr, n=aug):
                aug_audio = librosa.util.normalize(aug_audio.astype(np.float32))
                feat = extract_mfcc(aug_audio, sr=sr)
                X.append(feat)
                y_labels.append(label)
        except Exception as e:
            print(f'    Skip: {e}')

print(f"\n✅ รวมหลัง augment: {len(X)} ไฟล์")
for lbl in EMOTIONS:
    print(f"  {lbl:15s}: {y_labels.count(lbl)}")

# ══════════════════════════════════════════
#  ENCODE + SPLIT
# ══════════════════════════════════════════
X     = np.array(X)
le    = LabelEncoder()
y_int = le.fit_transform(y_labels)
y_enc = to_categorical(y_int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

mean    = X_train.mean()
std     = X_train.std()
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
np.save(f'{SAVE_DIR}/norm_stats_CNN_MFCC2.npy', [mean, std])

print("\n🔀 Mixup augmentation...")
X_mix, y_mix = mixup(X_train, y_train, alpha=0.3, n_mix=300)
X_train = np.concatenate([X_train, X_mix])
y_train = np.concatenate([y_train, y_mix])
print(f"  Training set หลัง mixup: {len(X_train)} ไฟล์")

# class_weight ยังคงไว้เผื่อ focal loss ไม่พอ
y_train_int   = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
cw = dict(enumerate(class_weights))
print("\n⚖️  Class weights:")
for i, lbl in enumerate(le.classes_):
    print(f"  {lbl:15s}: {cw[i]:.3f}")

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
#  MODEL v2
# ══════════════════════════════════════════
def build_CNN_MFCC2(input_shape=(120, 128, 1), num_classes=5):
    inputs = layers.Input(shape=input_shape)

    # ── CNN Block 1 ──
    x = layers.Conv2D(32, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # ── CNN Block 2 ──
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # ── CNN Block 3 ──
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Reshape → Sequential ──
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # ── Bidirectional LSTM ──
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # ── Attention ──
    x = AttentionLayer()(x)

    # ── Classifier ──
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='CNN_MFCC2_Attention_BiLSTM')


model = build_CNN_MFCC2(input_shape=X_train.shape[1:], num_classes=5)
model.summary()

# ✅ Focal Loss แทน CrossEntropy
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

# ══════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════
os.makedirs(RESULT_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=12, min_lr=1e-7, verbose=1),
    ModelCheckpoint(f'{SAVE_DIR}/best_model_CNN_MFCC2.keras', save_best_only=True, verbose=1),
    CSVLogger(f'{RESULT_DIR}/training_log_CNN_MFCC2.csv')
]

# ══════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════
print("\n🚀 เริ่มเทรน CNN_MFCC2 (Balanced AUG + Focal Loss)...")
print(f"  Input shape : {X_train.shape[1:]}")
print(f"  Train size  : {len(X_train)}")
print(f"  Val size    : {len(X_val)}")
print(f"  Classes     : {le.classes_.tolist()}\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=cw
)

# ══════════════════════════════════════════
#  บันทึก
# ══════════════════════════════════════════
with open(f'{SAVE_DIR}/labels_CNN_MFCC2.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)

best_val = max(history.history['val_accuracy'])
print(f"\n✅ เทรนเสร็จ!")
print(f"   Best val accuracy : {best_val*100:.2f}%")
print(f"   โมเดล             : {SAVE_DIR}/best_model_CNN_MFCC2.keras")
print(f"   Labels            : {SAVE_DIR}/labels_CNN_MFCC2.json")
print(f"   Norm stats        : {SAVE_DIR}/norm_stats_CNN_MFCC2.npy")
print(f"   Training log      : {RESULT_DIR}/training_log_CNN_MFCC2.csv")