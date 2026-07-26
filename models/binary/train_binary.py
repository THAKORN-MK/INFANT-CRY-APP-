import os, glob, json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
# Positive = เสียงร้องเด็กทุก class
# Negative = เสียงอื่นจาก ESC-50
BABY_DIR   = 'D:/INFANT CRY/data_set'
ESC50_DIR  = 'D:/INFANT CRY/data_set/not_baby'   # วางไฟล์ ESC-50 ที่นี่
SAVE_DIR   = 'D:/INFANT CRY/models/binary'
RESULT_DIR = 'D:/INFANT CRY/results'

BABY_CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
SR      = 22050
N_MFCC  = 40
MAX_LEN = 128
BATCH   = 16
EPOCHS  = 200

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
#  AUGMENTATION
# ══════════════════════════════════════════
def augment_audio(y, sr, n=6):
    results = [y]
    results.append(y + 0.004 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    results.append(librosa.effects.time_stretch(y, rate=1.1))
    results.append(librosa.effects.time_stretch(y, rate=0.9))
    return results[:n]

# ══════════════════════════════════════════
#  โหลด DATASET
# ══════════════════════════════════════════
print("\n📂 โหลด dataset...")
X, y_labels = [], []

# --- Positive: เสียงเด็ก (label = 1) ---
baby_files = []
for cls in BABY_CLASSES:
    baby_files += glob.glob(f'{BABY_DIR}/{cls}/*.wav')

print(f"  เสียงเด็ก (positive): {len(baby_files)} ไฟล์")
for f in baby_files:
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _  = librosa.effects.trim(audio, top_db=20)
        audio     = librosa.util.normalize(audio)
        for aug in augment_audio(audio, sr, n=4):
            aug = librosa.util.normalize(aug.astype(np.float32))
            X.append(extract_mfcc(aug))
            y_labels.append(1)  # 1 = เสียงเด็ก
    except Exception as e:
        print(f'    Skip: {e}')

# --- Negative: ไม่ใช่เสียงเด็ก (label = 0) ---
not_baby_files = glob.glob(f'{ESC50_DIR}/*.wav')
print(f"  ไม่ใช่เสียงเด็ก (negative): {len(not_baby_files)} ไฟล์")
for f in not_baby_files:
    try:
        audio, sr = librosa.load(f, sr=SR, mono=True)
        audio, _  = librosa.effects.trim(audio, top_db=20)
        audio     = librosa.util.normalize(audio)
        for aug in augment_audio(audio, sr, n=2):
            aug = librosa.util.normalize(aug.astype(np.float32))
            X.append(extract_mfcc(aug))
            y_labels.append(0)  # 0 = ไม่ใช่เสียงเด็ก
    except Exception as e:
        print(f'    Skip: {e}')

print(f"\n✅ รวม: {len(X)} ไฟล์")
print(f"  เสียงเด็ก     : {y_labels.count(1)}")
print(f"  ไม่ใช่เสียงเด็ก: {y_labels.count(0)}")

# ══════════════════════════════════════════
#  ENCODE + SPLIT
# ══════════════════════════════════════════
X       = np.array(X)
y_array = np.array(y_labels)
y_enc   = to_categorical(y_array, num_classes=2)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

mean    = X_train.mean()
std     = X_train.std()
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
np.save(f'{SAVE_DIR}/norm_stats_binary.npy', [mean, std])

y_train_int   = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
cw = dict(enumerate(class_weights))
print(f"\n⚖️  Class weights: {cw}")

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
#  MODEL — Binary CNN + BiLSTM + Attention
# ══════════════════════════════════════════
def build_binary(input_shape=(120, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    x = AttentionLayer()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # 2 class: เสียงเด็ก / ไม่ใช่
    outputs = layers.Dense(2, activation='softmax')(x)
    return Model(inputs, outputs, name='Binary_CNN_MFCC')


model = build_binary(input_shape=X_train.shape[1:])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

os.makedirs(RESULT_DIR, exist_ok=True)
callbacks = [
    EarlyStopping(patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=10, min_lr=1e-7, verbose=1),
    ModelCheckpoint(f'{SAVE_DIR}/best_model_binary.keras', save_best_only=True, verbose=1),
    CSVLogger(f'{RESULT_DIR}/training_log_binary.csv')
]

print("\n🚀 เริ่มเทรน Binary Classifier...")
print(f"  Input shape : {X_train.shape[1:]}")
print(f"  Train size  : {len(X_train)}")
print(f"  Val size    : {len(X_val)}\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=cw
)

with open(f'{SAVE_DIR}/labels_binary.json', 'w') as f:
    json.dump(['not_baby', 'baby'], f)

best_val = max(history.history['val_accuracy'])
print(f"\n✅ เทรนเสร็จ!")
print(f"   Best val accuracy : {best_val*100:.2f}%")
print(f"   โมเดล             : {SAVE_DIR}/best_model_binary.keras")
print(f"   Labels            : {SAVE_DIR}/labels_binary.json")
print(f"   Norm stats        : {SAVE_DIR}/norm_stats_binary.npy")
