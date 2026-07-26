import tarfile
import os

tar_path = 'D:/INFANT CRY/yamnet-tensorflow2-yamnet-v1.tar.gz'
extract_dir = 'D:/INFANT CRY/models/yamnet'

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)