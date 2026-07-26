import os, glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json

DATA_DIR = 'data_set'
EMOTIONS = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# จำนวน augment ต่อ class — hungry มีเยอะอยู่แล้ว ลดลง
AUG_TIMES = {
    'belly_pain': 6,
    'burping':    6,
    'discomfort': 6,
    'hungry':     2,
    'tired':      6
}

# ---- Augmentation ----
def augment(y, sr):
    results = [y]
    results.append(y + 0.005 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1))
    results.append(librosa.effects.time_stretch(y, rate=1.1))
    results.append(librosa.effects.time_stretch(y, rate=0.9))
    return results

def load_with_augment(file_path, label, X, y_list, aug_times=6):
    audio, sr = librosa.load(file_path, sr=22050)
    audio, _  = librosa.effects.trim(audio, top_db=20)
    audio     = librosa.util.normalize(audio)

    for aug_audio in augment(audio, sr)[:aug_times]:
        aug_audio = librosa.util.normalize(aug_audio)
        mel = librosa.feature.melspectrogram(
            y=aug_audio, sr=sr, n_mels=128,
            n_fft=2048, hop_length=512, fmin=50, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0,0),(0, 128 - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :128]
        X.append(mel_db[..., np.newaxis].astype(np.float32))
        y_list.append(label)

# ---- โหลด + Augment ----
X, y = [], []
for label in EMOTIONS:
    files = glob.glob(f'{DATA_DIR}/{label}/*.wav')
    aug   = AUG_TIMES[label]
    print(f"  {label:15s}: {len(files):4d} ไฟล์ → {len(files)*aug:5d} หลัง augment")
    for f in files:
        try:
            load_with_augment(f, label, X, y, aug_times=aug)
        except Exception as e:
            print(f'  Skip: {e}')

print(f"\nรวมหลัง augment: {len(X)} ไฟล์")

X   = np.array(X)
le  = LabelEncoder()
y_integers = le.fit_transform(y)
y_enc      = to_categorical(y_integers)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

mean, std = X_train.mean(), X_train.std()
X_train   = (X_train - mean) / std
X_val     = (X_val   - mean) / std
np.save('norm_stats.npy', [mean, std])

# ---- Class Weight ----
y_train_int  = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:")
for i, label in enumerate(le.classes_):
    print(f"  {label:15s}: {class_weight_dict[i]:.3f}")

# ---- โมเดล ----
def build_simple_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

model = build_simple_cnn(X_train.shape[1:], num_classes=5)
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=150,
    callbacks=callbacks,
    class_weight=class_weight_dict  # แก้ class imbalance
)

with open('labels.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)

print("\n✅ เทรนเสร็จ!")