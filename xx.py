import glob

DATA_DIR = 'D:/INFANT CRY/data_set'
EMOTIONS = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

for label in EMOTIONS:
    files = glob.glob(f'{DATA_DIR}/{label}/*.wav')
    print(f"{label:15s}: {len(files)} ไฟล์")