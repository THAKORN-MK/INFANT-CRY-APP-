import os, glob, json
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---- ติดตั้ง tensorflow_hub ก่อนถ้ายังไม่มี ----
# pip install tensorflow-hub

DATA_DIR = 'data_set'
EMOTIONS  = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

AUG_TIMES = {
    'belly_pain': 30,
    'burping':    60,
    'discomfort': 18,
    'hungry':      1,
    'tired':       20
}

# ---- Augmentation ----
def augment(y, sr, n=6):
    results = [y]
    results.append(y + 0.005 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    results.append(librosa.effects.time_stretch(y, rate=1.15))
    results.append(librosa.effects.time_stretch(y, rate=0.85))
    results.append(y + 0.002 * np.random.randn(len(y)))
    return results[:n]

# ---- โหลด YAMNet ----
print("โหลด YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("โหลด YAMNet สำเร็จ!")

def get_yamnet_embedding(audio, sr=16000):
    """แปลง audio → YAMNet embedding (1024-dim)"""
    # YAMNet ต้องการ 16kHz mono
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    _, embeddings, _ = yamnet_model(audio)
    # embeddings shape: (frames, 1024) → เฉลี่ยทุก frame
    return np.mean(embeddings.numpy(), axis=0)  # (1024,)

# ---- โหลด Dataset ----
X, y = [], []
for label in EMOTIONS:
    files = glob.glob(f'{DATA_DIR}/{label}/*.wav')
    aug   = AUG_TIMES[label]
    print(f"  {label:15s}: {len(files):4d} ไฟล์ → ~{len(files)*aug} หลัง augment")

    for f in files:
        try:
            audio, sr = librosa.load(f, sr=16000, mono=True)
            audio, _  = librosa.effects.trim(audio, top_db=20)
            audio     = librosa.util.normalize(audio)

            augmented_list = augment(audio, sr=16000, n=aug)
            for aug_audio in augmented_list:
                aug_audio = librosa.util.normalize(aug_audio.astype(np.float32))
                emb = get_yamnet_embedding(aug_audio, sr=16000)
                X.append(emb)
                y.append(label)
        except Exception as e:
            print(f'    Skip {os.path.basename(f)}: {e}')

print(f"\nรวมหลัง augment: {len(X)} ไฟล์")
print("จำนวนต่อ class:")
for label in EMOTIONS:
    print(f"  {label:15s}: {y.count(label)}")

# ---- Encode ----
X   = np.array(X, dtype=np.float32)
le  = LabelEncoder()
y_int = le.fit_transform(y)
y_enc = to_categorical(y_int)

# ---- Split ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ---- Normalize ----
mean, std = X_train.mean(), X_train.std()
X_train   = (X_train - mean) / std
X_val     = (X_val   - mean) / std
np.save('norm_stats_yamnet.npy', [mean, std])

# ---- Class Weight ----
y_train_int   = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:")
for i, lbl in enumerate(le.classes_):
    print(f"  {lbl:15s}: {class_weight_dict[i]:.3f}")

# ---- โมเดล (Dense เพราะ input เป็น 1024-dim vector) ----
def build_classifier(input_dim=1024, num_classes=5):
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='YAMNet_Classifier')

model = build_classifier(input_dim=1024, num_classes=5)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_model_yamnet.keras', save_best_only=True, verbose=1)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=300,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# ---- บันทึก ----
with open('labels_yamnet.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)

print("\n✅ เทรนเสร็จ! โมเดลบันทึกที่ best_model_yamnet.keras")