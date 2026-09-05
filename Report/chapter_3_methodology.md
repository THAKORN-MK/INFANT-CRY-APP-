# บทที่ 3 วิธีการดำเนินงานด้านโมเดล

บทนี้ระบุวิธีพัฒนา ฝึก และประเมินโมเดลจำแนกเสียงร้องทารกของ CryInsight ในขอบเขต Machine Learning เท่านั้น เนื้อหาจึงไม่มีผลการทดลองหรือค่าประสิทธิภาพของโมเดล และใช้เป็นวิธีการที่ทำซ้ำได้ (reproducible methodology) สำหรับส่วนโมเดลของโครงงาน

[กลับ Report Hub](./report.md) · [README](../README.md) · [Architecture](../Architecture.md)

## 3.1 ข้อมูลเสียงและการจัดเตรียมกลุ่มตัวอย่าง

### 3.1.1 ขอบเขตข้อมูลและหน้าที่ของแต่ละ Stage

การศึกษานี้ใช้โมเดลแบบ two-stage cascade เพื่อแยกปัญหาเป็นสองคำถามที่มีขอบเขตต่างกัน

1. **Stage 1 — Binary baby gate** จำแนกไฟล์เสียงเป็น `baby` หรือ `not_baby` โดยใช้เสียงร้องทารกจาก InfantCry-DBL เป็นกลุ่ม `baby` และเสียงสิ่งแวดล้อมจาก ESC-50 เป็นกลุ่ม `not_baby`
2. **Stage 2 — Five-class infant-cry classifier** ทำงานเฉพาะเมื่อ Stage 1 ส่งผล `baby` และจำแนกป้ายกำกับเชิงปฏิบัติการของ Dataset ได้แก่ `belly_pain`, `burping`, `discomfort`, `hungry` และ `tired`

เหตุผลของ cascade คือ Stage 2 ได้รับการฝึกจากเสียงทารกเท่านั้น จึงไม่ควรถูกบังคับให้เลือกหนึ่งในห้าป้ายกำกับเมื่อข้อมูลเข้าเป็นเสียงสิ่งแวดล้อม การมี Stage 1 ช่วยจำกัด domain ของ Stage 2 ให้สอดคล้องกับข้อมูลที่ใช้ฝึก ทั้งสอง Stage เป็นเครื่องมือจำแนกตามป้ายกำกับของ Dataset ไม่ใช่ระบบวินิจฉัยทางการแพทย์

### 3.1.2 การแบ่ง Train และ locked internal Test

ข้อมูลเสียงถูกแบ่งแบบ stratified ระดับไฟล์จาก `data_set_dbl` ไปยัง `data_set_dbl_split/train` และ `data_set_dbl_split/test` ด้วยสคริปต์ `split_audio.py` โดยตรึง random seed และ file manifest ของ split ไว้ การทำ stratification ช่วยให้สัดส่วน class ระหว่าง Train และ Test ใกล้เคียงกันเท่าที่ข้อมูลอนุญาต ขณะที่การตรึง manifest ทำให้การทดลองครั้งถัดไปใช้ Test ชุดเดิมได้โดยไม่เกิดการสุ่มใหม่โดยไม่ตั้งใจ

ชุด `train` เป็น development partition สำหรับการสร้าง grouped five-fold cross-validation, การปรับ hyperparameter, baseline และ ablation เท่านั้น ส่วน `test` เป็น **locked internal held-out test** สำหรับประเมิน final-refit model หลังตัดสินใจเรื่อง feature, augmentation, architecture, loss และจำนวน epoch แล้ว ไม่ใช้คะแนน Test เพื่อเลือก candidate ระหว่างการพัฒนา เพราะจะก่อให้เกิด test-set selection bias

แม้ Test ถูกกันออกจาก Train แต่ยังมาจาก corpus เดียวกัน จึงเรียกว่า internal held-out test ไม่ใช่ independent external validation และไม่ควรใช้แทนหลักฐาน generalizability กับประชากรหรืออุปกรณ์บันทึกเสียงอื่น

### 3.1.3 การป้องกัน data leakage

การสุ่มระดับชื่อไฟล์อย่างเดียวไม่เพียงพอ หากไฟล์ที่มีเนื้อหาเสียงเหมือนกันหรือเป็นสำเนาอยู่คนละฝั่งของ Train และ Validation/Test โมเดลอาจจดจำสัญญาณเดิมและให้ค่าประเมินสูงเกินจริง จึงใช้มาตรการต่อไปนี้

- คำนวณ SHA-256 ของเนื้อหาไฟล์เพื่อหา exact-content family
- ตัด duplicate ที่มีป้ายกำกับเดียวกันตามกฎของ protocol
- ตัด hash family ที่มีป้ายกำกับขัดแย้งข้าม class เพื่อไม่ให้ target กำกวม
- ห้าม exact-content family เดียวกันปรากฏทั้ง development Train และ locked Test
- ภายใน Train ใช้ grouped five-fold cross-validation โดยสมาชิกใน group เดียวกันต้องอยู่ในฝั่ง Train หรือ Validation ของ fold เดียวกันทั้งหมด
- Stage 1 จัดกลุ่ม negative จาก ESC-50 ตาม `source_file` เพื่อป้องกัน segment ที่มาจากไฟล์ต้นทางเดียวกันกระจายข้าม fold

การป้องกันดังกล่าวรับประกันระดับ exact-content family และ provenance ที่มีอยู่ใน manifest เท่านั้น เนื่องจาก Dataset ไม่มี subject/session identifier ที่ตรวจสอบได้ครบถ้วน จึงยังไม่อาจอ้างว่าเป็น subject-independent evaluation ได้

### 3.1.4 การตรวจคุณภาพข้อมูลก่อนฝึก

ก่อนสร้าง run, trainer ตรวจ path, นามสกุลไฟล์, label, audio decoding, duplicate policy, group assignment และการชนกันระหว่าง partition หาก audio ไม่ผ่าน feature contract เช่นสั้นเกินกว่าที่จะสร้าง delta feature ได้ จะถูกบันทึกเป็น exclusion พร้อมเหตุผล ไม่ปล่อยให้ตัวอย่างดังกล่าวทำให้การฝึกของบาง fold ล้มเหลวโดยไม่มีหลักฐาน

ข้อมูลต้นทาง, รายการคัดออก, fold assignment, hash และ configuration ถูกบันทึกเป็น artefacts ของ run เพื่อให้ตรวจสอบย้อนหลังได้ว่าตัวอย่างใดเข้าสู่ขั้นตอนใด หลักการนี้ทำให้การคัดออกเป็นการรักษา protocol ไม่ใช่การเลือกเฉพาะตัวอย่างที่โมเดลทำนายง่าย

## 3.2 เครื่องมือและสภาพแวดล้อมการวิจัย

### 3.2.1 ฮาร์ดแวร์ที่ใช้พัฒนาและฝึกโมเดล

ตารางนี้บันทึกค่าที่ตรวจจากเครื่องพัฒนาจริง ณ สภาพแวดล้อมที่ใช้ฝึกโมเดล เพื่อให้ผู้อื่นประเมินทรัพยากรและทำซ้ำการฝึกได้

| องค์ประกอบ | รายละเอียด | เหตุผลที่ใช้/บันทึก |
|---|---|---|
| เครื่องแม่ข่าย/เครื่องพัฒนา | เครื่อง Desktop ที่ใช้เมนบอร์ด Gigabyte Technology Co., Ltd. X870 AORUS ELITE WIFI7 ICE | ระบุแพลตฟอร์มฮาร์ดแวร์ที่ใช้พัฒนาและฝึกโมเดล |
| CPU | AMD Ryzen 7 9800X3D; 8 cores / 16 threads; maximum reported clock 4.70 GHz | ใช้ audio decoding, preprocessing, feature extraction, hashing, grouping, cache และการจัดการ artefact |
| RAM | ติดตั้ง 32 GB; ระบบตรวจพบ 31.61 GiB | รองรับ waveform, feature tensor, cache และ WSL2 ร่วมกับ Windows |
| GPU | NVIDIA GeForce RTX 5070 Ti; VRAM 16,303 MiB (ประมาณ 16 GB) | เร่ง CNN, BiLSTM, attention, backpropagation และ mixed-precision computation |
| GPU compute capability | 12.0; TensorFlow รายงาน compatibility target เป็น 12.0a | ระบุความเข้ากันได้ของ CUDA kernel กับอุปกรณ์จริง |
| NVIDIA driver | 616.56 (Windows `nvidia-smi`) | ทำให้ WSL2 และ TensorFlow เข้าถึง GPU ได้ |
| Storage | Drive C: NTFS 487.50 GB (ว่าง 194.19 GB); Drive D: NTFS 1.37 TB (ว่าง 652.39 GB) | Drive C ใช้ระบบ/โปรแกรม; Drive D เก็บ Dataset, feature cache, run artefacts และ model bundles |

GPU ใช้ลดระยะเวลาฝึกเท่านั้น ไม่เปลี่ยน train/test split, fold assignment, audio contract หรือ metric หากฝึกด้วย CPU และ GPU ภายใต้ seed, code, dependency และ deterministic setting เดียวกัน ค่าเชิงตัวเลขอาจต่างเล็กน้อยจากลำดับการคำนวณแบบ floating point แต่ protocol และข้อมูลที่ใช้ต้องเหมือนเดิม

### 3.2.2 ระบบปฏิบัติการและ GPU runtime

การพัฒนาใช้ Windows เป็น host และรัน training environment ผ่าน WSL2 เพื่อใช้ Linux runtime ที่ TensorFlow GPU รองรับโดยตรง การแยก host กับ Linux environment ช่วยให้เครื่องมือพัฒนาแบบ Windows และ library ฝั่ง Linux ใช้งานร่วมกันโดยใช้ NVIDIA driver เดียวกัน

| รายการ | เวอร์ชัน/ค่า | เหตุผลที่ใช้ |
|---|---|---|
| Host operating system | Microsoft Windows 11 Pro 64-bit, version 10.0.26200, build 26200 | ระบบหลักของเครื่องพัฒนา |
| Linux environment | Ubuntu 22.04.5 LTS บน WSL2 | runtime สำหรับ Python และ TensorFlow GPU |
| WSL kernel | 6.6.87.2-microsoft-standard-WSL2 | บันทึกสภาวะแวดล้อม Linux ที่ใช้ฝึก |
| Python virtual environment | `/home/adminuser/.venvs/audio-ml-gpu` | แยก dependency ของงาน ML ออกจาก Python ระบบ |
| GPU selection | `--device gpu` และ `--require-gpu` | ให้ trainer หยุดทันทีหากตรวจไม่พบ GPU แทนการ fallback ไป CPU โดยไม่ตั้งใจ |
| GPU memory policy | memory growth | ลดการจอง VRAM ทั้งหมดตั้งแต่เริ่ม process |
| numeric precision | `mixed_float16` เมื่อสั่ง `--mixed-precision`; output layer เป็น `float32` | ลดการใช้หน่วยความจำและเพิ่ม throughput ในส่วนที่ GPU รองรับ โดยคง softmax output ที่เสถียร |

ก่อนเริ่มทุก training run ให้ตรวจทั้ง `nvidia-smi` และคำสั่ง `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` ใน WSL หากไม่ปรากฏ `GPU:0` ต้องแก้ environment ก่อนเริ่ม run เนื่องจากการฝึกที่ fallback เป็น CPU จะมีเวลาและ environment manifest ต่างจาก protocol GPU

TensorFlow ในเครื่องนี้แสดงคำเตือนว่า binary ไม่มี CUDA kernel ที่ precompiled สำหรับ compute capability 12.0a จึงอาจ JIT-compile kernel จาก PTX ในงานแรกหลังเปิด environment ซึ่งอาจใช้เวลานาน คำเตือนนี้เป็นลักษณะ runtime ของ GPU รุ่นใหม่ ไม่ใช่ข้อผิดพลาดของข้อมูลหรือโมเดล และต้องเผื่อเวลา warm-up ก่อนสรุปเวลาการฝึก

### 3.2.3 ซอฟต์แวร์ เครื่องมือพัฒนา และเวอร์ชัน

รายการต่อไปนี้เป็น software ที่เกี่ยวข้องโดยตรงกับการพัฒนา ฝึก ตรวจสอบ และทำซ้ำโมเดล จึงไม่รวม extension หรือ application ที่ไม่เกี่ยวกับงาน ML

| ซอฟต์แวร์/ไลบรารี | เวอร์ชัน | หน้าที่ |
|---|---:|---|
| Visual Studio Code | 1.136.1, commit `a44adf7f53e00964ab890f9f8758a334f1fc15bc`, x64 | แก้ไขโค้ด ตรวจ log และเปิด terminal ของ WSL2 |
| VS Code Remote - WSL | 0.104.3 | เชื่อม workspace Windows กับ Ubuntu/WSL2 เพื่อให้ terminal ใช้ environment เดียวกับ training |
| VS Code Python | 2026.4.0 | เลือก interpreter, run/debug Python และตรวจ Python environment |
| Pylance | 2026.3.1 | static analysis และ type/language support ใน editor |
| Python Debugger (`debugpy`) | 2026.6.0 | debug โค้ด Python ใน WSL2 |
| Python Environments | 1.36.0 | จัดการและตรวจ virtual environment ใน VS Code |
| Python | 3.10.12 | ภาษาหลักของ pipeline |
| TensorFlow | 2.21.0 | สร้าง ฝึก และ serialize neural network รวมถึงการใช้ GPU |
| Keras | 3.12.4 | API สำหรับ layers, callbacks, model checkpoint และไฟล์ `.keras` |
| NumPy | 2.2.6 | tensor/array processing, labels และ numerical operations |
| librosa | 0.11.0 | audio loading, resampling และการสกัด feature |
| scikit-learn | 1.7.2 | baselines แบบ classical ML และ classification/calibration metrics |
| SciPy | 1.15.3 | การคำนวณเชิงวิทยาศาสตร์ที่ dependency ใช้ร่วมกัน |
| SoundFile | 0.14.0 | อ่านและเขียนไฟล์เสียงที่รองรับโดย pipeline |
| audioread | 3.1.0 | backend อ่านไฟล์เสียงที่ librosa ใช้เมื่อเหมาะสม |
| matplotlib | 3.10.9 | สร้าง confusion matrix, ROC/PR และกราฟการเรียนรู้ |
| h5py | 3.14.0 | backend ที่เกี่ยวข้องกับ model/data serialization |
| joblib | 1.5.3 | บันทึก object ของ baseline/preprocessing ที่รองรับ |

TensorFlow build manifest ระบุ CUDA 12.5.1 และ cuDNN major version 9 ขณะที่ virtual environment ติดตั้ง `tensorflow[and-cuda]` พร้อม runtime packages ได้แก่ `nvidia-cuda-runtime-cu12` 12.9.79, `nvidia-cuda-nvcc-cu12` 12.9.86, `nvidia-cudnn-cu12` 9.24.0.43, `nvidia-cublas-cu12` 12.9.2.10, `nvidia-cufft-cu12` 11.4.1.4, `nvidia-curand-cu12` 10.3.10.19, `nvidia-cusolver-cu12` 11.7.5.82, `nvidia-cusparse-cu12` 12.5.10.65, `nvidia-nccl-cu12` 2.31.2 และ `nvidia-nvjitlink-cu12` 12.9.86 การบันทึกทั้ง build manifest และ package runtime สำคัญ เพราะคำว่า “CUDA version” จาก `nvidia-smi`, เวอร์ชันที่ TensorFlow build มา และเวอร์ชันของ Python package อาจไม่ใช่ค่าเดียวกัน

### 3.2.4 การจัดการไฟล์และ artefact เพื่อทำซ้ำการทดลอง

โค้ดฝึกสร้าง immutable run directory และบันทึก artefacts ที่สัมพันธ์กัน ได้แก่ configuration, data audit, fold assignment, preprocessing configuration, normalizer statistics, augmentation manifest, label order, checkpoints, OOF predictions, final-test predictions, metric tables, figures, model bundle และ `verification.json` ที่ระบุสถานะ/แฮชของไฟล์สำคัญ

การเก็บ model เพียง `.keras` ไฟล์เดียวไม่เพียงพอสำหรับ reproducibility หรือ deployment ที่ถูกต้อง เพราะโมเดลต้องใช้ feature contract, label mapping และ normalizer จาก run เดียวกันเสมอ การตรวจ hash และ manifest จึงช่วยป้องกันการนำไฟล์จากคนละ run มาปะปน

## 3.3 การประมวลผลเสียง การสกัดคุณลักษณะ และ augmentation

### 3.3.1 Audio preprocessing contract

ทุกไฟล์ผ่าน preprocessing contract เดียวกันก่อนสกัด feature ทั้งตอนฝึกและตอน inference ได้แก่

1. อ่าน waveform จากไฟล์เสียงและตรวจว่าถอดรหัสได้
2. แปลงเป็น mono เพื่อลดความแตกต่างจากจำนวน channel
3. resample เป็น **22,050 Hz** เพื่อกำหนด temporal/frequency resolution เดียวกันให้ทุกไฟล์ และลดภาระคำนวณเมื่อเทียบกับ sample rate ที่สูงกว่า
4. trim ช่วง silence ตามกติกาเดียวกัน เพื่อเน้นช่วงที่มีสัญญาณเสียง
5. normalize amplitude และ pad/trim ตาม target duration ที่กำหนดใน feature configuration
6. ตรวจจำนวน frame ก่อนสร้าง Delta และ Delta2; ไฟล์ที่สั้นเกิน contract จะไม่ถูกส่งเข้าสู่ model โดยไม่มีการบันทึก

เหตุผลสำคัญของ contract คือโมเดลเรียนรู้ distribution ของ feature ที่สร้างจากกระบวนการนี้ หาก training และ inference resample คนละอัตรา, ใช้ stereo/mono คนละแบบ หรือไม่ pad/trim ด้วยกติกาเดียวกัน feature distribution จะเปลี่ยน แม้ใช้ไฟล์ model เดิม ผลทำนายก็ไม่สามารถเทียบกับการฝึกได้อย่างถูกต้อง

### 3.3.2 คุณลักษณะของ Stage 1

Stage 1 ใช้ **MFCC + Delta + Delta2**

- **MFCC** สรุป spectral envelope ในสเกลที่สัมพันธ์กับการรับรู้เสียง เหมาะกับการแยกโครงสร้างเสียงร้อง/เสียงพูดออกจากเสียงสิ่งแวดล้อมในระดับ feature ที่กะทัดรัด
- **Delta** แทนการเปลี่ยนแปลงของ MFCC ตามเวลา ทำให้โมเดลรับรู้การเปลี่ยนผ่านของเสียงได้
- **Delta2** แทนอัตราการเปลี่ยนของ Delta จึงช่วยแสดงลักษณะการเร่งหรือชะลอของ spectral dynamics

การจำกัด feature ของ Stage 1 ให้กะทัดรัดมีเหตุผลเพราะโจทย์เป็น binary gate ไม่จำเป็นต้องใช้ representation กว้างเท่า Stage 2 และช่วยลดต้นทุนในการคัดกรองก่อนส่งต่อข้อมูล

### 3.3.3 คุณลักษณะของ Stage 2

Stage 2 ใช้ **MFCC + Delta + Delta2 + Log-Mel + Chroma** แล้วรวมตาม feature contract ของ `cryinsight/audio/features.py`

- **Log-Mel spectrogram** เก็บการกระจายพลังงานตามเวลาและย่านความถี่ใน Mel scale การใช้ log ทำให้ dynamic range จัดการได้ดีขึ้น
- **Chroma** รวมพลังงานเชิง pitch class จึงเป็น representation เพิ่มเติมของโครงสร้าง harmonic/tonal ที่อาจต่างกันระหว่างรูปแบบเสียงร้อง
- MFCC, Delta และ Delta2 คงไว้เพื่อให้โมเดลมีทั้ง spectral envelope และลักษณะการเปลี่ยนแปลงตามเวลา

การเพิ่ม Log-Mel และ Chroma เฉพาะ Stage 2 มีเหตุผลจากความซับซ้อนของโจทย์หลายคลาส ซึ่งอาจต้องใช้ representation ที่หลากหลายกว่า binary gate ไม่ได้หมายความว่า Chroma เป็นหลักฐานทางคลินิกหรือเป็นตัวชี้วัดอารมณ์โดยตรง

### 3.3.4 การ normalize feature โดยไม่รั่วข้อมูล

normalizer ของแต่ละ fold ถูก `fit` ด้วย feature จาก training partition ของ fold นั้นเท่านั้น แล้วจึง `transform` validation partition ด้วยสถิติเดียวกัน ใน final refit normalizer จะ fit ด้วย development Train ทั้งหมดก่อน transform locked Test ห้ามรวม Validation หรือ Test เข้าไปในขั้น fit เพราะค่า mean/standard deviation จากข้อมูลประเมินเป็นข้อมูลอนาคต (information leakage)

### 3.3.5 Data augmentation และ Mixup

augmentation ใช้กับ training partition ของแต่ละ fold เท่านั้น ไม่ใช้กับ Validation, OOF หรือ Test เหตุผลคือข้อมูลสังเคราะห์ในฝั่งประเมินอาจทำให้ metric ไม่สะท้อนความสามารถของโมเดลกับข้อมูลต้นฉบับ

augmentation ประกอบด้วย perturbation ที่กำหนด seed และบันทึก manifest เพื่อทำซ้ำได้ เช่นการเปลี่ยน gain, time shift, additive noise, time stretch และ pitch shift ภายใต้ช่วงที่ configuration อนุญาต วัตถุประสงค์คือให้โมเดลทนต่อความแปรผันของระดับความดัง ตำแหน่งเสียง จังหวะ และสภาพบันทึกที่สมเหตุสมผล โดยไม่เปลี่ยนป้ายกำกับโดยเจตนา

สำหรับ Stage 2 ใช้ target-based augmentation เพื่อให้ class ที่มีตัวอย่างน้อยได้รับโอกาสปรากฏใน batch อย่างเหมาะสม และใช้ **Mixup** ใน training batch โดยสร้างตัวอย่างใหม่ตามสมการ

$$
\tilde{x}=\lambda x_i+(1-\lambda)x_j,\qquad
\tilde{y}=\lambda y_i+(1-\lambda)y_j
$$

เมื่อ $\lambda$ มาจาก Beta distribution Mixup ช่วยทำให้ขอบเขตการตัดสินใจราบเรียบขึ้นและลดการจดจำตัวอย่างเดี่ยว แต่ไม่ได้สร้าง independent biological observations ใหม่ จึงต้องรายงานเป็นวิธีลด imbalance/overfitting ไม่ใช่การเพิ่มจำนวนกลุ่มตัวอย่างจริง

## 3.4 การออกแบบโมเดลและ protocol การฝึก

### 3.4.1 สถาปัตยกรรมของโมเดล

ทั้งสอง Stage ใช้แนวคิด **CNN + BiLSTM + Attention** แต่ใช้ feature input ต่างกันตามข้อ 3.3

1. **Convolutional Neural Network (CNN)** เรียนรู้ local pattern จาก time-frequency feature เช่นบริเวณพลังงานหรือการเปลี่ยน spectral ที่อยู่ใกล้กัน
2. **Bidirectional LSTM (BiLSTM)** อ่านลำดับ feature ทั้งทิศทางไปข้างหน้าและย้อนกลับ เพื่อเชื่อมบริบทก่อน–หลังของช่วงเสียง
3. **Attention layer** เรียนรู้น้ำหนักของแต่ละ time step แล้วสรุปลำดับเป็น context vector ทำให้โมเดลไม่จำเป็นต้องให้น้ำหนักทุกช่วงเท่ากัน
4. **Classifier head** ใช้ softmax สำหรับ Stage 2 และ binary probability head สำหรับ Stage 1 โดยกำหนด output เป็น `float32` แม้ใช้ mixed precision เพื่อรักษาเสถียรภาพของ loss และ probability calculation

โค้ด model builder อยู่ใน `cryinsight/models/stage1_model.py` และ `cryinsight/models/stage2_model.py`; attention ที่ serialize ได้อยู่ใน `cryinsight/models/attention.py` การแยก builder ออกจาก trainer ทำให้สามารถทดสอบการสร้างและ save/load model ได้โดยไม่ต้องเริ่มการฝึกเต็มรูปแบบ

### 3.4.2 Grouped five-fold cross-validation

บน development Train ใช้ grouped five-fold cross-validation โดยแต่ละรอบใช้ 4 fold สำหรับฝึก และ 1 fold สำหรับ validation หมุนจนครบทุก fold ทุก original record ที่ eligible ต้องปรากฏใน validation เพียงครั้งเดียว จึงรวมผลทำนายเป็น **pooled out-of-fold (OOF) prediction** ได้

เหตุผลที่ใช้ five folds คือเพิ่มความเสถียรของ development evaluation เมื่อเทียบกับ validation split เดี่ยว และช่วยให้ทุกตัวอย่างใน Train มีโอกาสเป็น validation โดยไม่ต้องเปิด locked Test ระหว่างตัดสินใจ model selection การใช้ grouping สำคัญกว่าเพียงจำนวน folds เพราะเป็นสิ่งที่คุม family leakage ตามข้อ 3.1.3

### 3.4.3 Checkpoint, early stopping และ final refit

ในแต่ละ fold trainer เลือก checkpoint ตาม validation loss โดย `ModelCheckpoint` บันทึกเฉพาะ checkpoint ที่ดีที่สุดของ fold และ `EarlyStopping` หยุดเมื่อ validation loss ไม่ดีขึ้นตาม patience ที่กำหนด หลังจบการฝึก trainer โหลด checkpoint ที่บันทึกไว้กลับมาใช้ทำนาย validation จึงไม่ใช้ weights จาก epoch สุดท้ายโดยอัตโนมัติ วิธีนี้ป้องกันการเลือก epoch จาก training accuracy ซึ่งอาจดีขึ้นต่อเนื่องแม้ generalization แย่ลง

หลังได้ OOF จากทั้งห้า fold จะเลือกจำนวน epoch สำหรับ final refit โดยอิง development validation evidence เท่านั้น จากนั้นเทรน final model ด้วย development Train ทั้งหมด แล้วประเมิน locked Test ตาม protocol ที่ตรึงไว้ final bundle มีเพียง model หลักหนึ่งชุดสำหรับการนำไปใช้ ขณะที่ fold checkpoints เก็บไว้เป็นหลักฐานของ cross-validation ไม่ใช่ไฟล์ที่จะนำไปใช้แทน final model โดยสุ่มเลือก

### 3.4.4 การกำหนด seed, cache และ run identity

trainer ตั้ง seed สำหรับ Python, NumPy และ TensorFlow ตาม configuration และบันทึกค่าไว้ใน run manifest การควบคุม seed ไม่ทำให้ GPU deterministic ทุก operation เสมอไป แต่ทำให้แหล่งสุ่มที่ควบคุมได้สามารถตรวจสอบและทำซ้ำได้มากขึ้น

feature cache มี key ที่ผูกกับ filepath, audio hash และ feature configuration เพื่อป้องกันการนำ feature ที่สร้างด้วย config เก่ามาใช้กับ run ใหม่ หาก config หรือเนื้อหาไฟล์เปลี่ยน cache key ต้องเปลี่ยนตามเสมอ

Stage 1 เป็นจุดเริ่มของ pipeline run identity และ Stage 2 รวมถึง experiments จะอ้างอิง pipeline run เดียวกัน เพื่อให้สามารถระบุได้ว่า model, normalizer, fold assignment และรายงานมาจาก training cycle เดียวกัน

### 3.4.5 การใช้ GPU อย่างตรวจสอบได้

ก่อนสร้าง training artifact runtime เรียก `configure_tensorflow_runtime` ใน `cryinsight/runtime/device.py` เพื่อยืนยัน device policy การใช้ flag `--require-gpu` ทำให้ trainer fail fast หาก TensorFlow ไม่เห็น GPU ซึ่งป้องกันการสร้าง run ที่ผู้ใช้เข้าใจผิดว่าใช้ GPU แต่จริงใช้ CPU

สามารถใช้ `--mixed-precision` เพื่อเปิด `mixed_float16` ใน layer ที่รองรับ และบันทึก policy ลง environment/configuration artifact ไม่ควรเปรียบเทียบ throughput ระหว่าง run ที่เปิดและปิด mixed precision โดยไม่ระบุ setting นี้ เพราะเป็นปัจจัยที่ทำให้เวลาฝึกและการใช้ VRAM ต่างกัน

## 3.5 Protocol และเครื่องมือประเมินโมเดล

### 3.5.1 ลำดับการประเมินที่ป้องกัน test-set selection bias

ลำดับที่กำหนดล่วงหน้าคือ

```text
Development Train
  -> grouped 5-fold training
  -> pooled OOF predictions
  -> เลือก architecture / feature / augmentation / epoch จาก OOF เท่านั้น
  -> final refit ด้วย Development Train ทั้งหมด
  -> เปิด locked internal Test หนึ่งครั้ง
  -> บันทึก artefact และ verification.json
```

baseline และ ablation ใช้ fold assignments เดียวกับ proposed model เพื่อให้การเปรียบเทียบยุติธรรม หากต้องทดลอง candidate จำนวนมาก ให้ใช้ OOF เพื่อคัดเลือกและเก็บ Test ไว้จนกว่า protocol จะ freeze แล้วเท่านั้น

### 3.5.2 Confusion matrix

Confusion matrix แสดงจำนวนตัวอย่างจริงเทียบกับคลาสที่โมเดลทำนาย Stage 1 ใช้ตาราง 2 × 2 เพื่อแยก true positive, true negative, false positive และ false negative ส่วน Stage 2 ใช้ตาราง 5 × 5 เพื่อดูว่าป้ายกำกับใดสับสนกับป้ายกำกับใด Accuracy เพียงค่าเดียวไม่บอกชนิดของความผิดพลาด จึงต้องรายงาน confusion matrix ควบคู่เสมอ

### 3.5.3 Accuracy, precision, recall และ F1-score

ให้ $C$ เป็นจำนวนคลาส, $TP_c$, $FP_c$ และ $FN_c$ เป็นค่าของ class $c$

$$
\mathrm{Accuracy}=\frac{\text{จำนวนที่ทำนายถูกทั้งหมด}}{\text{จำนวนตัวอย่างทั้งหมด}}
$$

$$
\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c},\qquad
\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c}
$$

$$
\mathrm{F1}_c=\frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c}
$$

Accuracy เหมาะเป็นภาพรวม แต่เมื่อ class imbalance สูงอาจสะท้อน class ใหญ่เป็นหลัก จึงใช้ macro average ที่เฉลี่ยคะแนนของทุกคลาสเท่ากัน และ weighted average ที่ถ่วงตามจำนวนตัวอย่างจริง สำหรับ Stage 2 จะใช้ macro F1 เป็นตัวชี้วัดหลักของความสมดุลระหว่างคลาส ส่วน weighted F1 ใช้บอกผลที่ถ่วงตาม distribution จริง

### 3.5.4 Balanced accuracy, sensitivity และ specificity

สำหรับ binary Stage 1

$$
\mathrm{Sensitivity}=\frac{TP}{TP+FN},\qquad
\mathrm{Specificity}=\frac{TN}{TN+FP}
$$

$$
\mathrm{Balanced\ Accuracy}=\frac{\mathrm{Sensitivity}+\mathrm{Specificity}}{2}
$$

Sensitivity บอกสัดส่วนเสียง `baby` ที่ gate ตรวจพบ ส่วน specificity บอกสัดส่วนเสียง `not_baby` ที่คัดออกได้ถูกต้อง Balanced accuracy ลดอิทธิพลของ class ที่มีจำนวนมากกว่า จึงเหมาะกับการสื่อสารผลของ binary classifier ที่ class distribution อาจไม่สมดุล

### 3.5.5 ROC, precision–recall และ threshold analysis

ROC curve แสดงความสัมพันธ์ระหว่าง true-positive rate และ false-positive rate เมื่อเลื่อน threshold ส่วน area under ROC curve (ROC-AUC) ใช้ประเมินการเรียงลำดับคะแนนโดยไม่ยึด threshold เดียว

Precision–recall curve และ PR-AUC ให้ความสำคัญกับ positive class มากขึ้น จึงมีประโยชน์เมื่อ positive มีจำนวนน้อยหรือเมื่อ false positive มีผลต่อการใช้งาน Stage 1 ต้องบันทึก ROC และ PR เพื่อให้พิจารณา trade-off ของ threshold ได้อย่างโปร่งใส อย่างไรก็ตาม threshold ต้องเลือกจาก development/OOF evidence ไม่ใช่ปรับตาม locked Test

### 3.5.6 Log loss, Brier score และ Expected Calibration Error

metric กลุ่มนี้ตรวจคุณภาพของ probability score ไม่ใช่แค่ label ที่ชนะ

$$
\mathrm{LogLoss}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic}\log(p_{ic})
$$

$$
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}(p_{ic}-y_{ic})^2
$$

Expected Calibration Error (ECE) แบ่ง prediction ตามช่วง confidence แล้วเปรียบเทียบ confidence เฉลี่ยกับ accuracy จริงในแต่ละ bin ค่าเหล่านี้มีไว้ตรวจว่า score ของ softmax มีแนวโน้ม overconfident หรือ underconfident เพียงใด ไม่ได้ทำให้ score เป็นความน่าจะเป็นทางการแพทย์ และไม่ควรนำ confidence ไปใช้เป็นคำวินิจฉัย

### 3.5.7 Uncertainty interval และการรายงานผลที่ทำซ้ำได้

รายงาน metrics ของ OOF และ locked Test แยกกัน พร้อม sample support ของแต่ละ class, confusion matrix และ uncertainty interval จาก stratified bootstrap เมื่อใช้ใน trainer Bootstrap resampling ใช้ประเมินความไม่แน่นอนของ metric ภายใต้ชุดข้อมูลประเมินที่มีอยู่ แต่ไม่แก้ข้อจำกัดเรื่อง external validity หรือ subject independence

ก่อนนำผลจาก run ใดไปอ้างอิง ต้องตรวจว่า `verification.json` มีสถานะ `complete`, artefact สำคัญมีอยู่ครบ และ run ระบุ evaluation scope ชัดเจน ผลจาก incomplete run หรือ run ที่ไม่มี verification evidence ไม่ใช้เป็นผลสรุปของงานวิจัย

## 3.6 ขั้นตอนการพัฒนาและการทดสอบโมเดล

### 3.6.1 ขั้นตอนพัฒนาอย่างเป็นลำดับ

1. **กำหนด protocol ก่อนฝึก** — กำหนด labels, split, grouping, feature contract, model families, metrics, seed policy และกฎการเปิด Test ก่อนรัน เพื่อไม่ให้ตัดสินใจจากผล Test ย้อนหลัง
2. **ตรวจข้อมูลต้นทาง** — ตรวจ manifest, decoding, label mapping, duplicate/hash family, source provenance และจำนวน frame ที่ผ่าน feature contract
3. **สร้าง split ที่ตรึงได้** — รัน `split_audio.py` เพียงเมื่อจัดเตรียม split ใหม่ จากนั้นเก็บ manifest และไม่สุ่มทับ split เดิมระหว่างการเปรียบเทียบโมเดล
4. **ตรวจ runtime GPU** — เปิด WSL2, activate `/home/adminuser/.venvs/audio-ml-gpu`, ตรวจ `nvidia-smi` และ TensorFlow `GPU:0`, แล้วใช้ `--require-gpu` ในคำสั่ง train
5. **ทำ preflight/audit** — รัน trainer ในโหมด audit หรือ prepare-only ก่อน training เต็มรูปแบบ เพื่อตรวจ path, data contract, group leakage, cache และความสมบูรณ์ของ configuration
6. **ฝึก Stage 1** — ใช้ `Models_dbl/binary/train_binary_dbl.py` บน development Train ด้วย grouped five folds, checkpoints ต่อ fold และ OOF aggregation
7. **ฝึก Stage 2** — ใช้ `Models_dbl/Main/train_main_dbl.py` ด้วย pipeline run identity เดียวกับ Stage 1, grouped five folds, feature/augmentation ที่กำหนด และ OOF aggregation
8. **สร้าง final refit bundle** — เลือก epoch จาก OOF/validation evidence, refit ด้วย development Train ทั้งหมด, ประเมิน locked Test ตาม protocol และบันทึก bundle หลักเพียงหนึ่งชุดต่อ Stage
9. **ทำ baseline และ ablation** — ใช้ `Models_dbl/experiments/run_experiments.py` และ shared fold engine เพื่อเปรียบเทียบ candidate ภายใต้ข้อมูล/folds เดียวกันโดยไม่เปิด Test เพื่อเลือก candidate
10. **ตรวจ artefact ก่อนรายงาน** — ตรวจ run manifest, normalizer, labels, model bundle, prediction files, figures และ `verification.json` ให้สอดคล้องกันก่อนนำข้อมูลไปเขียนรายงาน

### 3.6.1.1 การเปิดสภาพแวดล้อมและตรวจ GPU ก่อนการเทรน

การฝึกดำเนินการใน Ubuntu บน WSL2 ไม่ใช่ Python ของ Windows โดยเปิด terminal ของ VS Code ที่เชื่อมต่อ WSL2 หรือเปิด Ubuntu terminal แล้วใช้คำสั่งต่อไปนี้จากจุดเริ่มต้นของทุก session

```bash
cd "/mnt/d/INFANT CRY"
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

คำสั่ง `source` ทำให้ shell ใช้ interpreter `/home/adminuser/.venvs/audio-ml-gpu/bin/python` ซึ่งมี Python 3.10.12 และ dependency ที่ตรึงไว้ ส่วน `nvidia-smi` ตรวจว่า WSL2 มองเห็น driver และ RTX 5070 Ti และคำสั่ง TensorFlow ต้องแสดง `GPU:0` ก่อนเริ่มการฝึกจริง หากไม่แสดง device ดังกล่าว ต้องหยุดแก้ environment ไม่ควรรัน train เพราะ `--require-gpu` ถูกกำหนดให้ fail fast เพื่อป้องกัน CPU fallback แบบไม่ตั้งใจ

หลังเปิด environment ครั้งแรก TensorFlow อาจ JIT-compile PTX สำหรับ compute capability 12.0a จึงอาจเกิดช่วงรอนานก่อนงาน GPU แรก การรอดังกล่าวไม่ใช่ epoch ของโมเดลและไม่ควรถูกตีความเป็นความผิดปกติของ trainer

### 3.6.1.2 ขั้นตรวจสอบก่อนสร้าง run (audit และ prepare)

ก่อน training ทุกครั้งต้องทำ preflight ตามลำดับ ไม่ข้ามไปสั่ง `--train` ทันที

```bash
# Stage 1: ตรวจข้อมูลเสียง, labels, hashes และ held-out reservation เท่านั้น
python Models_dbl/binary/train_binary_dbl.py --audit-only

# Stage 1: สร้าง immutable run, audit, grouped fold assignment และ manifest โดยยังไม่ฝึก
python Models_dbl/binary/train_binary_dbl.py --prepare-only
```

`--audit-only` อ่านข้อมูลจาก Train และ locked Test เพื่อตรวจ audio records, duplicate policy, group rule และการกัน content family ของ Test ออกจาก Train แต่ไม่ import TensorFlow และไม่สร้าง training artefact จึงใช้ตรวจ data contract ได้รวดเร็ว

`--prepare-only` ทำ audit เดิมแล้วสร้าง run directory, `fold_assignments.csv`, data audit, source snapshot และ protocol/manifests ที่ตรึงข้อมูลสำหรับ five-fold training โดยยังไม่สกัด feature และไม่ทำ gradient update ต้องตรวจ output ของขั้นนี้ก่อนว่าพาธ Train/Test ถูกต้องและไม่มีข้อผิดพลาดจาก grouping หรือไฟล์เสียง

Stage 2 ใช้คำสั่ง preflight ในรูปแบบเดียวกัน แต่ต้องทำหลังจาก Stage 1 มี completed run แล้ว เพราะ Stage 2 จะจับคู่ pipeline identity กับ Stage 1

```bash
# Stage 2: ตรวจ Dataset 5 classes และจับคู่กับ completed Stage 1 run ล่าสุด
python Models_dbl/Main/train_main_dbl.py --audit-only

# Stage 2: สร้าง immutable run ที่ใช้ run ID เดียวกับ Stage 1
python Models_dbl/Main/train_main_dbl.py --prepare-only
```

หากต้องการระบุ Stage 1 run โดยไม่พึ่งการเลือก run ล่าสุด ให้เพิ่ม `--run-id <completed_stage1_run_id>` ในคำสั่ง Stage 2 การระบุ explicit run ID เหมาะกับการ reproduce งานเก่าหรือเมื่อต้องการหลีกเลี่ยงการจับคู่กับ Stage 1 run ที่เพิ่งสร้างใหม่โดยไม่ตั้งใจ

### 3.6.1.3 คำสั่งฝึก Stage 1 และกลไกภายในแต่ละ fold

เมื่อ audit/prepare ผ่านแล้ว การฝึก Stage 1 ใช้คำสั่งต่อไปนี้ โดย default data paths คือ `data_set_dbl_split/train` และ `data_set_dbl_split/test`

```bash
python Models_dbl/binary/train_binary_dbl.py \
  --train \
  --device gpu \
  --require-gpu \
  --mixed-precision
```

คำสั่งเดียวนี้รันครบทั้งห้า folds ไม่ใช่เพียง fold เดียว ค่า default ที่ใช้ถ้าไม่ได้ override คือ seed ของ trainer, maximum 200 epochs, batch size 32, AdamW learning rate 0.001, weight decay 0.0001, label smoothing 0.05, early-stopping patience 25 epochs, learning-rate patience 10 epochs และ bootstrap iterations 2,000 รอบ ค่าทุกตัวจะถูกเขียนลง configuration/manifest ของ run เพื่อให้ตรวจซ้ำได้

ใน **แต่ละ fold** Stage 1 ดำเนินงานตามลำดับต่อไปนี้

1. เลือก 4 grouped folds เป็น training originals และอีก 1 fold เป็น validation originals โดยตรวจว่า group ไม่ซ้อนกัน
2. สร้าง target-based augmentation plan จาก training originals เท่านั้น; Stage 1 ไม่ใช้ Mixup (`mixup_samples = 0`)
3. สกัด MFCC + Delta + Delta2 จาก training, augmentation และ validation ตาม audio contract เดียวกัน โดยใช้ content-addressed feature cache เมื่อเปิด cache
4. fit normalizer แบบ per-feature-bin จาก training feature เท่านั้น แล้วบันทึก `norm_stats_binary_dbl.npy` และ metadata ก่อน transform validation
5. สร้าง binary CNN + BiLSTM + Attention, compile ด้วย AdamW และ categorical cross-entropy ที่มี label smoothing
6. train ด้วย mini-batch ขนาด 32 โดย monitor `val_loss`; `ReduceLROnPlateau` ลด learning rate ด้วย factor 0.3 เมื่อ validation loss ไม่ดีขึ้น และ `EarlyStopping` หยุดเมื่อเกิน patience
7. `ModelCheckpoint` บันทึกเฉพาะ `.keras` checkpoint ที่ `val_loss` ดีที่สุด จากนั้น trainer โหลด checkpoint นี้เพื่อทำนาย validation originals ของ fold
8. เขียน validation predictions, metrics, history, normalizer, preprocessing config, labels, augmentation manifest และ `fold_manifest.json` ไว้ใน `fold_1` ถึง `fold_5`

เมื่อครบห้า folds trainer ตรวจว่า original record ที่ eligible ทุกตัวปรากฏใน OOF predictions เพียงหนึ่งครั้ง จากนั้นรวม OOF metrics, เลือก final-refit epoch ตามกฎ median ของ best epoch จาก folds, refit บน development Train ทั้งหมด และเปิด locked Test หนึ่งครั้งหลัง final refit เท่านั้น

### 3.6.1.4 คำสั่งฝึก Stage 2 และกลไกภายในแต่ละ fold

Stage 2 ต้องเริ่มหลัง Stage 1 complete เพื่อรักษา pipeline identity เดียวกัน คำสั่งมาตรฐานคือ

```bash
python Models_dbl/Main/train_main_dbl.py \
  --train \
  --device gpu \
  --require-gpu \
  --mixed-precision
```

ค่า default ของ Stage 2 คือ maximum 200 epochs, batch size 32, AdamW learning rate 0.001, weight decay 0.0001, label smoothing 0.10, Mixup 500 synthetic combinations ต่อ fold, Mixup alpha 0.3, early-stopping patience 30 epochs, learning-rate patience 12 epochs และ bootstrap iterations 2,000 รอบ Model architecture default คือ `corrected_single_branch`; การเปลี่ยนเป็น `corrected_multi_branch` ต้องระบุ `--architecture corrected_multi_branch` และถือเป็น candidate ใหม่ที่ต้องเปรียบเทียบด้วย OOF protocol

ในแต่ละ fold Stage 2 ทำงานดังนี้

1. เลือก grouped Train/Validation originals แบบ 4:1 และสร้าง target-based augmentation plan เฉพาะ Train
2. สกัด MFCC + Delta + Delta2 + Log-Mel + Chroma และ fit normalizer จาก Train feature เท่านั้น
3. สร้าง Mixup feature/soft label จำนวนตาม `--mixup-samples` แล้วต่อท้าย training tensors; Validation ไม่ได้รับ augmentation หรือ Mixup
4. สร้าง five-class CNN + BiLSTM + Attention และ compile ด้วย AdamW กับ categorical cross-entropy ที่ใช้ label smoothing
5. train โดย monitor `val_loss` เช่นเดียวกับ Stage 1; checkpoint ถูกบันทึกผ่าน checkpoint staging ก่อน publish ไปยัง run directory เพื่อไม่ให้ไฟล์ checkpoint ที่เขียนไม่ครบถูกใช้เป็น artefact
6. โหลด selected checkpoint เพื่อทำนาย validation originals แล้วบันทึก OOF evidence และ artefacts ต่อ fold
7. เมื่อครบทุก fold ตรวจ OOF coverage, รวม OOF metrics, หา median selected epoch, final-refit ด้วย development Train ทั้งหมด และประเมิน locked Test หนึ่งครั้ง

`fold_1_main_dbl.keras` ถึง `fold_5_main_dbl.keras` เป็นหลักฐานจาก cross-validation ไม่ใช่ไฟล์สำหรับเลือกใช้ในระบบแบบสุ่ม Final bundle ที่สร้างหลัง final refit เป็น model หลักเพียงหนึ่งชุดต่อ Stage และต้องใช้ร่วมกับ normalizer, label mapping และ preprocessing configuration จาก bundle เดียวกัน

### 3.6.1.5 โครงสร้าง artefact หลังการเทรน

ทุก completed run เก็บหลักฐานเป็นโครงสร้างโดยสรุปดังนี้

```text
Models_dbl/<stage>/runs/<run_id>/
├── dataset_audit.json, heldout_dataset_audit.json, record_audit.csv
├── heldout_record_audit.csv, heldout_reservation.json, protocol.json, run_config.json
├── environment.json, preprocessing_config.json, labels_<stage>_dbl.json
├── fold_assignments.csv, fold_metrics.csv, oof_predictions.csv, oof_metrics.json
├── final_test_predictions.csv, final_test_metrics.json, final_test_manifest.json
├── final_test_confusion_matrix.csv/.png, oof_confusion_matrix.csv/.png
├── fold_1/ ... fold_5/
│   ├── fold_<n>_<stage>_dbl.keras
│   ├── norm_stats_<stage>_dbl.npy และ metadata
│   ├── history.csv, validation_predictions.csv, metrics.json
│   ├── augmentation_manifest.csv, class_counts.json
│   └── fold_manifest.json
├── best_model/                         # final-refit deployment bundle เพียงหนึ่งชุด
│   ├── best_model_<stage>_dbl.keras
│   ├── norm_stats_<stage>_dbl.npy, labels_*.json
│   ├── preprocessing_config.json
│   └── final_refit_manifest.json และ deployment_manifest.json
└── verification.json
```

ชื่อไฟล์จริงของ Stage 1 ใช้ `binary` และ Stage 2 ใช้ `main` ตาม trainer แต่หน้าที่เหมือนกัน: folder `fold_n` คือหลักฐาน validation ของ fold; folder `best_model` คือ final-refit bundle สำหรับใช้งานจริง การตรวจ `verification.json` สถานะ `complete` เป็นเงื่อนไขก่อนนำ run ไปใช้รายงานหรือ deployment

### 3.6.1.6 Baseline, ablation และ experiment lifecycle

การทดลองเปรียบเทียบไม่แก้ trainer หลักโดยตรง แต่ใช้ Shared Experiment Engine ที่ตรึง development records และ fold assignments จาก pipeline run หลัก ขั้นตอนคือ audit config, prepare immutable experiment run, train candidate ที่เตรียมแล้ว, resume เมื่อเกิดการหยุดชะงัก และ summarize เมื่อครบ

```bash
# 1) audit config โดยยังไม่สร้าง experiment artefact
python Models_dbl/experiments/run_experiments.py \
  --audit-only \
  --config Models_dbl/experiments/configs/stage1_baselines.json

# 2) prepare experiment: สร้าง experiment run ID และ frozen protocol
python Models_dbl/experiments/run_experiments.py \
  --prepare-only \
  --pipeline-run-id "<completed_stage1_run_id>" \
  --config Models_dbl/experiments/configs/stage1_baselines.json \
  --device gpu --require-gpu --mixed-precision

# 3) train: ใช้ experiment run ID ที่ได้จากขั้น prepare แบบตรงตัว
python Models_dbl/experiments/run_experiments.py \
  --train \
  --pipeline-run-id "<completed_stage1_run_id>" \
  --config Models_dbl/experiments/configs/stage1_baselines.json \
  --experiment-run-id "<experiment_run_id>" \
  --device gpu --require-gpu --mixed-precision
```

placeholder ที่คร่อมด้วย `<...>` ใน block นี้เป็นค่าที่ผู้รันต้องแทนด้วย run ID จริงเท่านั้น ไม่ใช่ชื่อ folder แบบตายตัว `--prepare-only` ไม่ฝึกโมเดล จึงไม่มี checkpoint; ต้องนำ experiment run ID ที่คำสั่งแสดงออกมาไปใช้กับ `--train` ทุกครั้ง การเปรียบเทียบ candidate ใช้ OOF evidence เท่านั้น และไม่เปิด locked Test เพื่อจัดอันดับ candidate

### 3.6.2 รายการโค้ดหลักและความรับผิดชอบ

| พาธ | ความรับผิดชอบ | เหตุผลที่แยกส่วน |
|---|---|---|
| `Models_dbl/binary/train_binary_dbl.py` | orchestration ของ Stage 1 | แยก binary gate จาก multiclass trainer |
| `Models_dbl/Main/train_main_dbl.py` | orchestration ของ Stage 2 | คุม feature/augmentation/labels ของ five-class task |
| `cryinsight/audio/features.py` | audio contract และ feature extraction | ให้ training, evaluation และ inference ใช้ feature เดียวกัน |
| `cryinsight/models/stage1_model.py` | model builder ของ Stage 1 | ทดสอบสถาปัตยกรรมได้แยกจาก training loop |
| `cryinsight/models/stage2_model.py` | model builder ของ Stage 2 | ลดการปนกันของ task-specific configuration |
| `cryinsight/models/attention.py` | attention layer ที่ serialize ได้ | ให้ save/load `.keras` model ทำซ้ำได้ |
| `cryinsight/training/protocol.py` | seed, split, grouped folds และ protocol checks | รวมกฎป้องกัน leakage ไว้จุดเดียว |
| `cryinsight/training/feature_cache.py` | cache ที่ผูกกับ audio hash/config | ลดเวลา extraction โดยไม่ใช้ feature เก่าผิด config |
| `cryinsight/training/artefacts.py` | manifest, predictions, normalizer และ verification | ทำให้ run เป็นหลักฐานตรวจสอบได้ |
| `cryinsight/training/checkpoint_staging.py` | จัดเก็บ checkpoint และ final bundle | แยก evidence ของแต่ละ fold จาก model ใช้งานจริง |
| `cryinsight/runtime/device.py` | GPU detection, memory growth และ precision policy | ป้องกัน silent CPU fallback |
| `Models_dbl/experiments/run_experiments.py` | shared experiment engine | ทำ baseline/ablation ภายใต้ protocol เดียวกัน |

### 3.6.3 แผนการทดสอบเชิงเทคนิค

| ระดับการทดสอบ | วัตถุประสงค์ | ตัวอย่างกรณีทดสอบ | หลักฐานที่ต้องเก็บ |
|---|---|---|---|
| Environment smoke test | ยืนยัน runtime ที่ถูกต้อง | TensorFlow ตรวจพบ `GPU:0`; `--require-gpu` fail เมื่อ GPU ไม่พร้อม | terminal log และ environment manifest |
| Audio contract test | ตรวจ input และ feature consistency | stereo เป็น mono, resample 22,050 Hz, audio สั้นเกินถูก reject อย่างมีเหตุผล | unit-test output และ exclusion record |
| Feature test | ตรวจ shape/finite values/ordering | MFCC, Delta, Delta2, Log-Mel, Chroma ได้ shape ตาม config และไม่มี NaN/Inf | test output และ config manifest |
| Data-integrity test | ป้องกัน leakage | duplicate policy, cross-label conflict, Train/Test overlap และ group fold separation | audit JSON/CSV และ fold manifest |
| Normalization test | ป้องกัน statistics leakage | `fit` ใช้ Train partition เท่านั้น; Validation/Test ใช้ transform | unit/integration test log |
| Model construction test | ตรวจ architecture และ serialization | model build ได้, output dimension ตรง labels, custom attention save/load ได้ | model smoke-test output |
| Fold protocol test | ตรวจ 5-fold CV | validation coverage ครบ, record ไม่ซ้ำใน OOF, checkpoint เลือกจาก validation loss | OOF prediction และ verification artefact |
| Augmentation isolation test | ป้องกันข้อมูลสังเคราะห์รั่ว | augmentation/Mixup เกิดเฉพาะ training batch | augmentation manifest และ test log |
| Final bundle test | ตรวจ deployment consistency | model, normalizer, labels และ feature config มาจาก run เดียวกัน | bundle manifest และ load smoke test |
| Metric regression test | ตรวจสูตร/ชนิด input | probability ถูก normalize ก่อน log loss/Brier/ECE; macro/weighted metric ใช้ label order เดียวกัน | unit-test output |
| Experiment comparability test | เปรียบเทียบ baseline อย่างยุติธรรม | baselines/ablations ใช้ frozen train records, groups และ folds เดียวกับ reference run | experiment protocol และ frozen manifest |

### 3.6.4 Test cases สำคัญ

| รหัส | เงื่อนไขเริ่มต้น | ขั้นตอน | ผลที่คาดหวัง |
|---|---|---|---|
| ML-TC-01 | เปิด WSL2 และ activate venv แล้ว | รันคำสั่งตรวจ TensorFlow GPU | ปรากฏ physical device `GPU:0` |
| ML-TC-02 | GPU ใช้งานไม่ได้หรือ flag ถูกปิด | เริ่ม trainer ด้วย `--require-gpu` | trainer หยุดก่อนสร้าง training run ที่ไม่ตรง policy |
| ML-TC-03 | มีไฟล์เสียง stereo | ส่งเข้า feature extractor | ได้ waveform mono ตาม audio contract |
| ML-TC-04 | มีไฟล์ sample rate อื่น | ส่งเข้า feature extractor | ได้ representation ที่ resample เป็น 22,050 Hz |
| ML-TC-05 | มีไฟล์ audio สั้นเกิน | ส่งเข้า feature extractor | ได้ exclusion/error ที่อธิบายได้ ไม่ใช่ training crash กลาง fold |
| ML-TC-06 | มี duplicate/hash family | สร้าง audit และ grouped folds | family เดียวกันไม่ข้าม Train/Validation/Test ตามกฎ |
| ML-TC-07 | มี fold assignment | รวม OOF predictions | original record ที่ eligible เป็น validation เพียงหนึ่งครั้ง |
| ML-TC-08 | เปิด augmentation | ตรวจ Validation/Test tensors | ไม่มี augmentation หรือ Mixup ในข้อมูลประเมิน |
| ML-TC-09 | มี custom attention model | save แล้ว load `.keras` bundle | โหลดได้และ output shape/label order ตรง manifest |
| ML-TC-10 | มี probability output | คำนวณ calibration metrics | ค่า probability ถูกตรวจ/normalize ตามข้อกำหนด metric |
| ML-TC-11 | มี final bundle | โหลด model พร้อม normalizer, labels และ feature config | artefacts เป็น run เดียวกันและตรวจ mismatch ได้ |
| ML-TC-12 | มี experiment candidate | prepare/train experiment | frozen records และ fold assignments ตรงกับ reference pipeline run |

### 3.6.5 การทดสอบซ้ำและการควบคุมการเปลี่ยนแปลง

การเปลี่ยน feature, augmentation, loss, architecture, random seed, precision policy หรือ dependency version ต้องถือเป็นการเปลี่ยน experiment condition และสร้าง artefact ใหม่ ห้ามแทนที่ manifest หรือผลเดิมใน run เก่า สำหรับการหาสาเหตุของผลต่าง ให้เปลี่ยนปัจจัยครั้งละหนึ่งกลุ่มใน ablation และใช้ shared fold assignments เดิม

เมื่อสิ้นสุดการฝึก ให้ยืนยันความสมบูรณ์จาก `verification.json`, ตรวจว่ามี final bundle หลักหนึ่งชุดต่อ Stage, และเก็บ OOF/Test evidence แยกจากกัน การรายงานผลในบทผลการทดลองจึงสามารถระบุเงื่อนไขของ run, environment, sample support, metric และข้อจำกัดได้โดยย้อนตรวจ artefact จริง ไม่อาศัยเพียงข้อความบันทึกด้วยมือ
