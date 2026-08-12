import os, glob, json
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# CONFIG
BASE_DIR  = 'D:/INFANT CRY'
DATA_DIR  = f'{BASE_DIR}/data_set_dbl'
SAVE_DIR  = f'{BASE_DIR}/models_dbl/main'
EMOTIONS  = ['belly_pain','burping','discomfort','hungry','tired']
SR=22050; N_MFCC=40; N_MELS=64; N_CHROMA=12; MAX_LEN=128; BATCH=32; EPOCHS=200; N_FOLDS=5

AUG_TIMES = {
    'belly_pain': 12,
    'burping':    8,
    'discomfort': 12,
    'hungry':     3,
    'tired':      3,
}
os.makedirs(SAVE_DIR, exist_ok=True)

# FEATURE EXTRACTION — MFCC + Delta + Delta2 + Mel + Chroma
def extract_features(audio, sr=SR, max_len=MAX_LEN):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel    = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA, n_fft=2048, hop_length=512)
    combined = np.vstack([mfcc, delta, delta2, mel, chroma])
    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0,0),(0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]
    return combined[..., np.newaxis].astype(np.float32)

# AUGMENTATION
def augment_audio(y, sr, n=6):
    results = [y]
    results.append(y + 0.004 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    results.append(librosa.effects.time_stretch(y, rate=1.1))
    results.append(librosa.effects.time_stretch(y, rate=0.9))
    results.append(y * np.random.uniform(0.8, 1.2))
    results.append(y + 0.002 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0))
    results.append(librosa.effects.time_stretch(y, rate=1.2))
    results.append(librosa.effects.time_stretch(y, rate=0.8))
    return results[:n]

def mixup(X, y, alpha=0.3, n_mix=500):
    X_mix, y_mix = [], []
    for _ in range(n_mix):
        i,j=np.random.choice(len(X),2,replace=False)
        lam=np.random.beta(alpha,alpha)
        X_mix.append(lam*X[i]+(1-lam)*X[j])
        y_mix.append(lam*y[i]+(1-lam)*y[j])
    return np.array(X_mix), np.array(y_mix)

# โหลด DATASET
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
                X.append(extract_features(librosa.util.normalize(aug_audio.astype(np.float32))))
                y_labels.append(label)
        except Exception as e:
            print(f'    Skip: {e}')

print(f"\n✅ รวมหลัง augment: {len(X)} ไฟล์")
for lbl in EMOTIONS:
    print(f"  {lbl:15s}: {y_labels.count(lbl)}")

X=np.array(X)
le=LabelEncoder()
y_int=le.fit_transform(y_labels)
y_enc=to_categorical(y_int)
print(f"\n  Input shape: {X.shape[1:]}")
print(f"  Classes    : {le.classes_.tolist()}")
with open(f'{SAVE_DIR}/labels_main_dbl.json','w') as f:
    json.dump(le.classes_.tolist(), f)

# ATTENTION LAYER
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W=self.add_weight(shape=(input_shape[-1],input_shape[-1]),initializer='glorot_uniform',trainable=True,name='attn_W')
        self.b=self.add_weight(shape=(input_shape[-1],),initializer='zeros',trainable=True,name='attn_b')
        self.u=self.add_weight(shape=(input_shape[-1],),initializer='glorot_uniform',trainable=True,name='attn_u')
    def call(self, x):
        score=tf.nn.tanh(tf.tensordot(x,self.W,axes=1)+self.b)
        score=tf.tensordot(score,self.u,axes=1)
        alpha=tf.nn.softmax(score,axis=1)
        return tf.reduce_sum(x*tf.expand_dims(alpha,-1),axis=1)

# MODEL
def build_main(input_shape, num_classes=5):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(32,(3,3),padding='same')(inputs)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.Conv2D(32,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.MaxPooling2D((2,2))(x); x=layers.Dropout(0.25)(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.MaxPooling2D((2,2))(x); x=layers.Dropout(0.25)(x)
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.MaxPooling2D((2,2))(x); x=layers.Dropout(0.3)(x)
    x=layers.Conv2D(256,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x)
    x=layers.MaxPooling2D((2,2))(x); x=layers.Dropout(0.3)(x)
    shape=x.shape
    x=layers.Reshape((shape[1],shape[2]*shape[3]))(x)
    x=layers.Bidirectional(layers.LSTM(128,return_sequences=True))(x); x=layers.Dropout(0.3)(x)
    x=layers.Bidirectional(layers.LSTM(64,return_sequences=True))(x); x=layers.Dropout(0.3)(x)
    x=AttentionLayer()(x)
    x=layers.Dense(256,activation='relu')(x)
    x=layers.BatchNormalization()(x); x=layers.Dropout(0.4)(x)
    x=layers.Dense(128,activation='relu')(x)
    x=layers.BatchNormalization()(x); x=layers.Dropout(0.4)(x)
    x=layers.Dense(64,activation='relu')(x); x=layers.Dropout(0.3)(x)
    outputs=layers.Dense(num_classes,activation='softmax')(x)
    return Model(inputs,outputs,name='Main_CNN_MFCC_Mel_Chroma_BiLSTM_Attention')

# K-FOLD TRAINING
print(f"\n🔀 เริ่ม {N_FOLDS}-Fold Cross Validation...")
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=42)
fold_models=[]; fold_scores=[]; all_val_true=[]; all_val_pred=[]; all_val_prob=[]

for fold,(train_idx,val_idx) in enumerate(skf.split(X,y_int)):
    print(f"\n{'='*50}\n  Fold {fold+1}/{N_FOLDS}\n{'='*50}")
    X_train,X_val=X[train_idx],X[val_idx]
    y_train,y_val=y_enc[train_idx],y_enc[val_idx]
    mean=X_train.mean(); std=X_train.std()
    X_train_n=(X_train-mean)/std; X_val_n=(X_val-mean)/std
    if fold==0: np.save(f'{SAVE_DIR}/norm_stats_main_dbl.npy',[mean,std])

    # Mixup
    X_mix,y_mix=mixup(X_train_n,y_train,alpha=0.3,n_mix=500)
    X_train_n=np.concatenate([X_train_n,X_mix])
    y_train_f=np.concatenate([y_train,y_mix])

    cw_int=np.argmax(y_train,axis=1)
    cws=compute_class_weight('balanced',classes=np.unique(cw_int),y=cw_int)
    cw=dict(enumerate(cws))

    model=build_main(input_shape=X_train_n.shape[1:], num_classes=5)
    if fold==0: model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001,weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    callbacks=[
        EarlyStopping(patience=30,restore_best_weights=True,verbose=0),
        ReduceLROnPlateau(factor=0.3,patience=12,min_lr=1e-7,verbose=0),
        ModelCheckpoint(f'{SAVE_DIR}/fold_{fold+1}_main.keras',save_best_only=True,verbose=0),
        CSVLogger(f'{SAVE_DIR}/fold_{fold+1}_log.csv')
    ]
    history=model.fit(X_train_n,y_train_f,validation_data=(X_val_n,y_val),
                      batch_size=BATCH,epochs=EPOCHS,callbacks=callbacks,
                      class_weight=cw,verbose=1)
    best_val=max(history.history['val_accuracy'])
    fold_scores.append(best_val); fold_models.append(model)
    probs=model.predict(X_val_n,verbose=0)
    all_val_true.extend(np.argmax(y_val,axis=1))
    all_val_pred.extend(np.argmax(probs,axis=1))
    all_val_prob.extend(probs)
    print(f"\n  Fold {fold+1} Best Val Accuracy: {best_val*100:.2f}%")

# บันทึก Best Model
best_fold=int(np.argmax(fold_scores))
shutil.copy(f'{SAVE_DIR}/fold_{best_fold+1}_main.keras',f'{SAVE_DIR}/best_model_main_dbl.keras')
print(f"\n⭐ Best Fold: {best_fold+1} ({fold_scores[best_fold]*100:.2f}%)")

# METRICS
label_names=le.classes_.tolist()
report=classification_report(all_val_true,all_val_pred,target_names=label_names,digits=4)
print(f"\n{'='*50}\n📊 Classification Report:\n{'='*50}")
print(report)

all_val_prob=np.array(all_val_prob)
try:
    y_true_bin=label_binarize(all_val_true,classes=list(range(5)))
    roc_auc=roc_auc_score(y_true_bin,all_val_prob,multi_class='ovr',average='macro')
    print(f"🎯 ROC-AUC (macro): {roc_auc:.4f}")
except Exception as e:
    roc_auc=0; print(f"ROC-AUC skip: {e}")

with open(f'{SAVE_DIR}/classification_report.txt','w',encoding='utf-8') as f:
    f.write(f"Main Emotion Classifier — {N_FOLDS}-Fold Cross Validation\n")
    f.write(f"Features: MFCC + Delta + Delta2 + Mel + Chroma\n")
    f.write(f"Model: CNN + BiLSTM + Attention + Mixup\n\n")
    f.write(f"Fold Scores: {[f'{s*100:.2f}%' for s in fold_scores]}\n")
    f.write(f"Mean Accuracy: {np.mean(fold_scores)*100:.2f}%\n")
    f.write(f"Std: {np.std(fold_scores)*100:.2f}%\n\n")
    f.write(report)
    f.write(f"\nROC-AUC (macro, OvR): {roc_auc:.4f}\n")

cm=confusion_matrix(all_val_true,all_val_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=label_names,yticklabels=label_names,cmap='Blues',linewidths=0.5)
plt.title('Confusion Matrix — Main Emotion Classifier',fontsize=13,pad=12)
plt.ylabel('Actual',fontsize=11); plt.xlabel('Predicted',fontsize=11)
plt.tight_layout(); plt.savefig(f'{SAVE_DIR}/confusion_matrix.png',dpi=150); plt.close()

print(f"\n{'='*50}")
print(f"✅ เทรนเสร็จ!")
print(f"  Fold Scores   : {[f'{s*100:.2f}%' for s in fold_scores]}")
print(f"  Mean Accuracy : {np.mean(fold_scores)*100:.2f}%")
print(f"  Best Fold     : {best_fold+1} ({fold_scores[best_fold]*100:.2f}%)")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  โมเดล         : {SAVE_DIR}/best_model_main_dbl.keras")
print(f"  Labels        : {SAVE_DIR}/labels_main_dbl.json")
print(f"  Norm stats    : {SAVE_DIR}/norm_stats_main_dbl.npy")
print(f"  Report        : {SAVE_DIR}/classification_report.txt")
print(f"  Confusion     : {SAVE_DIR}/confusion_matrix.png")
print(f"{'='*50}")
