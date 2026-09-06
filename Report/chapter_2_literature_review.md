# บทที่ 2 เอกสารและงานวิจัยที่เกี่ยวข้อง

บทนี้รวบรวมทฤษฎี หลักการ และงานวิจัยที่เกี่ยวข้องกับการจำแนกเสียงร้องทารกด้วยการประมวลผลสัญญาณเสียงและการเรียนรู้ของเครื่อง เพื่อใช้เป็นฐานเหตุผลสำหรับโครงงาน CryInsight ซึ่งประกอบด้วย Stage 1 สำหรับตรวจว่าไฟล์เสียงเป็นเสียงทารกหรือไม่ และ Stage 2 สำหรับจำแนกเสียงทารกออกเป็น 5 ป้ายกำกับเชิงปฏิบัติการของ Dataset

> เอกสารนี้เป็น **narrative literature review** ไม่ใช่ systematic review แบบ PRISMA กล่าวคือ ผู้จัดทำคัดเลือกเอกสารต้นฉบับ (primary studies), review papers, เอกสาร Dataset และงานวิจัยด้าน evaluation ที่เกี่ยวข้องโดยตรงกับสถาปัตยกรรมและ protocol ของโครงงาน ณ วันที่เข้าถึง 5 กันยายน 2026 การอ้างอิงผลของงานอื่นมีไว้เพื่ออธิบายบริบท ไม่ใช้เปรียบเทียบคะแนนกับ CryInsight โดยตรง เพราะ Dataset, labels, unit of split, preprocessing และวิธีประเมินแตกต่างกัน

[กลับ Report Hub](./report.md) · [บทที่ 3 วิธีการดำเนินงาน](./chapter_3_methodology.md) · [สถาปัตยกรรมโมเดล](../Architecture.md)

## 2.1 ทฤษฎีและหลักการที่เกี่ยวข้อง

### 2.1.1 แนวคิดของโครงการและขอบเขตการตีความเสียงร้องทารก

เสียงร้องเป็นสัญญาณเสียงที่เปลี่ยนแปลงตามเวลา (time-varying acoustic signal) และได้รับอิทธิพลจากแหล่งกำเนิดเสียง, ทางเดินเสียง, การหายใจ, สภาพแวดล้อม, อุปกรณ์บันทึก และกระบวนการตัดต่อไฟล์ งานทบทวนของ Ji และคณะชี้ว่า infant-cry research ครอบคลุมตั้งแต่การตรวจจับเสียงร้อง, การแยกสาเหตุของการร้อง, จนถึงการศึกษาสัญญาณที่สัมพันธ์กับภาวะทางพยาธิวิทยา และแต่ละโจทย์ใช้ข้อมูล/ป้ายกำกับ/วิธีประเมินต่างกันอย่างมีนัยสำคัญ [2]

ดังนั้น CryInsight จำกัดหน้าที่ไว้ที่ **audio classification ตามป้ายกำกับที่มีในข้อมูลฝึก** ไม่ตีความว่า output เป็นการยืนยันความต้องการจริง, ภาวะสุขภาพ, ความเจ็บปวด หรือการวินิจฉัยทางการแพทย์ ป้ายกำกับของ Stage 2 คือ operational labels ที่ repository ใช้จัดกลุ่มไฟล์เสียง ไม่ใช่ ground truth ทางคลินิกที่ตรวจยืนยันสำหรับการใช้งานกับเด็กทุกราย

ข้อมูลหลักของโครงงานมาจาก InfantCry-DBL Version 1 ซึ่งเผยแพร่พร้อม manifest ระดับคลิปและระบุ metadata เช่น class, duration, sample rate, channel count และ bit depth [1] คุณสมบัตินี้มีประโยชน์ต่อการตรวจ provenance และ reproducibility แต่ไม่ได้ทำให้ผลการประเมินเป็น external validation โดยอัตโนมัติ การนำโมเดลไปใช้กับเสียงที่บันทึกจากอุปกรณ์ สถานที่ ภาษา อายุ หรือประชากรอื่น ต้องได้รับการประเมินแยกต่างหาก

### 2.1.2 สัญญาณดิจิทัลและการเตรียมเสียง

ไฟล์ `.wav` แทนเสียงเป็นลำดับตัวอย่าง amplitude ตามเวลา กระบวนการสกัด feature ต้องกำหนดรูปแบบอินพุตให้คงที่ เพื่อไม่ให้โมเดลเรียนรู้ความต่างทางเทคนิคแทนลักษณะเสียงที่สนใจ หลักการที่เกี่ยวข้องมีดังนี้

| ขั้นตอน | หลักการ | เหตุผลต่อโครงงาน |
|---|---|---|
| Decoding | แปลงไฟล์เป็น waveform และตรวจว่าถอดรหัสได้ | แยกไฟล์เสีย/ไม่รองรับออกก่อนสร้าง feature |
| Mono conversion | รวมหรือเลือก channel ให้เหลือหนึ่งช่องสัญญาณ | ลดความต่างจากการบันทึก stereo/mono และทำให้ input shape คงที่ |
| Resampling | เปลี่ยนอัตราสุ่มเป็นอัตราเดียว | ทำให้แกนเวลาและความถี่ของทุก record อยู่บนมาตรฐานเดียวกัน |
| Silence trimming | ลดช่วงที่ไม่มีสัญญาณสำคัญตามกฎที่ตรึงไว้ | ลดผลของความยาว silence ที่แตกต่างกันระหว่างไฟล์ |
| Amplitude normalization | ปรับระดับสัญญาณด้วยกติกาเดียวกัน | ลดความต่างจาก gain หรือระยะไมโครโฟน โดยไม่อ้างว่าแก้ noise ได้ทั้งหมด |
| Framing และ padding/truncation | แบ่ง waveform เป็นเฟรมซ้อนทับและกำหนดจำนวนเฟรมเป้าหมาย | ทำให้ tensor ที่ป้อน CNN มีขนาดคงที่ |

การใช้ sample rate 22,050 Hz, mono, silence trimming, peak normalization, `n_fft=2,048`, `hop_length=512` และ target 128 frames ในโครงงานเป็น **feature contract ของ implementation** ไม่ใช่ค่ามาตรฐานเดียวที่ถูกต้องเสมอไป ค่าดังกล่าวต้องใช้เหมือนกันใน training, validation, final test และ inference มิฉะนั้น distribution ของ feature จะไม่ตรงกัน งานทบทวนในสาขานี้ชี้ว่า data acquisition, preprocessing, feature extraction และ classification เป็นขั้นที่เชื่อมโยงกัน; ความแปรผันจากการบันทึกจริงและเสียงรบกวนเป็นข้อจำกัดสำคัญ [2]

### 2.1.3 การแปลงจาก waveform เป็น time–frequency representation

เสียงร้องไม่ใช่สัญญาณคงที่ตลอดคลิป จึงนิยมวิเคราะห์ในช่วงเวลาสั้น (short-time analysis) โดยใช้ Short-Time Fourier Transform (STFT) เพื่อดูการกระจายพลังงานตามเวลาและความถี่ จากนั้นจึงสร้าง representation ที่เหมาะกับโมเดล เช่น Mel spectrogram หรือ MFCC

#### (1) Mel spectrogram และ Log-Mel spectrogram

Mel spectrogram คำนวณพลังงานจาก STFT แล้วรวมลงใน Mel filter bank ซึ่งเป็นการจัดแกนความถี่เชิง perceptual การแปลงเป็น log scale ช่วยบีบ dynamic range ของพลังงาน จึงทำให้ representation มีความเหมาะสมต่อการเรียนรู้จากพลังงานที่แตกต่างกันมากระหว่างเฟรม

ใน CryInsight Stage 2 ใช้ Log-Mel 64 bands ร่วมกับ feature อื่น ไม่ถือว่า Log-Mel เพียงอย่างเดียวสามารถระบุ “อารมณ์” หรือสาเหตุของการร้องได้ งานทบทวนระบุว่า spectrogram และ feature เชิง acoustic/prosodic หลายชนิดถูกใช้ในงาน infant cry อย่างแพร่หลาย แต่การเลือก feature ต้องขึ้นกับ Dataset และ task [2]

#### (2) Mel-Frequency Cepstral Coefficients (MFCC)

MFCC เป็น cepstral representation ที่คำนวณจาก log Mel spectrum แล้วผ่าน discrete cosine transform (DCT) เพื่อสรุปลักษณะ spectral envelope ของเสียง งานคลาสสิกของ Davis และ Mermelstein เปรียบเทียบ parametric acoustic representations สำหรับงานรู้จำคำพูด และเป็นรากฐานสำคัญของการใช้ MFCC ในงาน speech/audio recognition [3] เอกสาร `librosa` ยังระบุชัดว่า MFCC ถูกคำนวณจาก audio time series หรือ log-power Mel spectrogram และสามารถกำหนดจำนวน coefficients, FFT, hop length และ sampling rate ได้ [4]

CryInsight ใช้ MFCC 40 coefficients เป็นส่วนประกอบทั้งสอง Stage เพราะเป็น representation ที่กะทัดรัดและสอดคล้องกับงาน infant-cry ก่อนหน้า อย่างไรก็ตาม การใช้ MFCC ไม่ได้ลดความจำเป็นของ baseline หรือ ablation: ต้องตรวจด้วย protocol เดียวกันว่า feature ชุดใดมีประโยชน์กับ Dataset นี้จริง

#### (3) Delta และ Delta2

MFCC ของเฟรมหนึ่งบอกลักษณะ spectral ณ ช่วงเวลาหนึ่ง ส่วน Delta เป็นอนุพันธ์เชิงเวลาลำดับหนึ่ง และ Delta2 เป็นอนุพันธ์ลำดับสอง จึงเพิ่มข้อมูลเกี่ยวกับทิศทางและอัตราการเปลี่ยนของ spectral envelope สำหรับเสียงร้องที่มี onset, sustain, pause และการเปลี่ยน pitch/energy ตามเวลา feature เหล่านี้ทำให้โมเดลเห็นพลวัตมากกว่า MFCC แบบ static เพียงอย่างเดียว

ในระบบนี้ Delta และ Delta2 คำนวณต่อจาก MFCC ด้วย window ที่ตรึงไว้ ไฟล์ที่สั้นเกินกว่าจะสร้าง derivative ตาม contract ต้องถูก pad หรือถูกบันทึกเป็น exclusion ตามกฎเดียวกัน ไม่ควรปล่อยให้แต่ละ fold จัดการความยาวสัญญาณต่างกัน

#### (4) Chroma

Chroma รวมพลังงานเชิง pitch class เป็น 12 bins ต่อเฟรม แม้ถูกใช้มากใน music information retrieval แต่สามารถเป็น representation เพิ่มเติมของโครงสร้าง harmonic และระดับเสียงได้ ใน CryInsight ใช้ Chroma เฉพาะ Stage 2 เพื่อทดสอบการผสาน representation หลายชนิด; ไม่มีข้ออ้างว่า Chroma มีความหมายเชิงคลินิกหรือพิสูจน์ว่าเป็น feature ที่ดีที่สุด

แนวคิดของการผสาน MFCC, Chromagram และ Mel-scaled spectrogram สอดคล้องกับงานของ Ting, Choo และ Kamar ที่ศึกษาการใช้ hybrid speech features สำหรับโจทย์แยก asphyxia กับกลุ่มเปรียบเทียบ [10] แต่ task, Dataset และ labels ในงานดังกล่าวต่างจาก CryInsight จึงใช้เป็นเหตุผลเชิงแนวคิด ไม่ใช่หลักฐานโดยตรงว่าการเพิ่ม Chroma จะเพิ่มผลของโครงงานนี้

### 2.1.4 หลักการเรียนรู้แบบมีผู้สอนและการจำแนกหลายคลาส

CryInsight เป็น supervised learning: แต่ละ record มี input feature tensor $x_i$ และป้ายกำกับ $y_i$ โมเดลเรียนรู้พารามิเตอร์ $\theta$ เพื่อลด loss ระหว่าง probability ที่ทำนาย $p_\theta(y\mid x)$ กับป้ายกำกับที่ใช้ฝึก

สำหรับ Stage 2 ใช้ softmax เพื่อทำให้คะแนนของ $C$ classes เป็น distribution

$$
p_c = \frac{\exp(z_c)}{\sum_{j=1}^{C}\exp(z_j)}
$$

โดย $z_c$ คือ logit ของ class $c$ และ $p_c$ คือ score ที่รวมกันเป็น 1 ในแต่ละตัวอย่าง ระหว่างฝึกใช้ categorical cross-entropy และ label smoothing ตาม configuration ของ trainer เพื่อลดการผลัก target แบบ one-hot ให้สุดโต่งเกินไป Score จาก softmax เป็นคะแนนของโมเดลภายใต้ distribution ที่ใช้ฝึก ไม่ใช่ calibrated probability ทางการแพทย์

Stage 1 เป็น binary classifier ที่แยก `baby`/`not_baby` ก่อน Stage 2 การออกแบบแบบ cascade มีเหตุผลด้าน domain control: Stage 2 เรียนรู้เพียง 5 labels ของ infant audio จึงไม่ควรถูกบังคับให้ทำนายหนึ่งในห้าคลาสเมื่อ input เป็นเสียงสิ่งแวดล้อม งานด้าน baby-monitoring ที่ศึกษา binary detection แยก task การตรวจจับเสียงร้องออกจากการตีความเนื้อหาของเสียง และเปรียบเทียบ representation อย่าง MFCC กับ spectrogram [14] ซึ่งสนับสนุนการกำหนด task boundary ที่ชัดเจน แม้ implementation และ Dataset จะต่างจากโครงงานนี้

### 2.1.5 Convolutional Neural Network (CNN)

CNN ใช้ convolution kernels เรียนรู้ local patterns ที่เลื่อนตำแหน่งได้บน feature map เมื่อนำ time–frequency feature มาจัดเป็น tensor CNN จึงสามารถเรียนรู้บริเวณพลังงาน, spectral edge, การเปลี่ยนแปลงระยะสั้น และ pattern ที่เกิดซ้ำในละแวกใกล้กัน หลักการ gradient-based convolutional learning ได้รับการอธิบายอย่างเป็นระบบโดย LeCun และคณะ [5]

ใน CryInsight CNN ทำหน้าที่สร้าง representation ระดับสูงจาก feature tensor ก่อนส่งต่อให้ sequence model การเลือก CNN ไม่ได้หมายความว่า convolution เพียงอย่างเดียวเพียงพอต่อทุก task; งานของ Abbaskhah, Sedighi และ Marvi เปรียบเทียบ SVM, MLP และ CNN สำหรับ 5 classes จาก MFCC และใช้การจัดการ class imbalance เป็นส่วนหนึ่งของการศึกษา [11] จึงเป็นเหตุผลให้โครงงานคง baseline และ ablation แยกจาก proposed pipeline

### 2.1.6 LSTM และ Bidirectional LSTM (BiLSTM)

Recurrent neural networks ประมวลผลข้อมูลตามลำดับเวลา แต่ RNN แบบดั้งเดิมมีปัญหา gradient ที่ลดทอนเมื่อเรียนรู้ dependency ระยะยาว Long Short-Term Memory (LSTM) ใช้ memory cell และ gates เพื่อควบคุมการเก็บ ลืม และเผยแพร่ข้อมูล ทำให้เหมาะกับลำดับข้อมูลที่ต้องพึ่งบริบทหลายช่วงเวลา [6]

Bidirectional RNN อ่าน sequence ทั้งทิศทางไปข้างหน้าและย้อนกลับ เพื่อให้ representation ณ time step หนึ่งอาศัยบริบทจากก่อนหน้าและภายหลังได้ [7] ในงานนี้ BiLSTM ทำงานหลัง CNN เพื่อ model temporal dependencies ของ frame-level feature โดยใช้ลำดับเวลาเป็นแกน sequence ไม่ใช่เรียง feature bins อย่างไม่ตั้งใจ

งานของ Maghfira, Basaruddin และ Krisnadhi ใช้ CNN–RNN กับการจัดกลุ่ม 5 categories บน Dunstan Baby Language derivative และรายงานว่าการผสาน CNN สำหรับ spectrogram representation กับ RNN สำหรับข้อมูลเชิงเวลามีความเป็นไปได้สำหรับโจทย์หลายคลาส [8] ขณะที่ Liang และคณะเปรียบเทียบ CNN และ LSTM จาก MFCC บน hospital recordings ที่มี labels แตกต่างกัน [9] สองงานนี้สนับสนุนการพิจารณา CNN และ sequence modeling ร่วมกัน แต่ไม่ทำให้สถาปัตยกรรมเดียวกันใช้ได้โดยไม่ตรวจใน Dataset ใหม่

### 2.1.7 Attention mechanism

Attention เป็นกลไกที่คำนวณน้ำหนักให้แต่ละ time step แล้วรวม hidden states เป็น context vector แทนที่จะลดลำดับด้วยการเฉลี่ยอย่างเท่ากัน แนวคิด attention ใน encoder–decoder ได้รับการเสนอโดย Bahdanau, Cho และ Bengio เพื่อให้โมเดลเลือกส่วนของ input ที่สัมพันธ์กับ output มากกว่า [12] และถูกนำไปประยุกต์กับ speech recognition โดย Chorowski และคณะ [13]

สำหรับ CryInsight Attention ต่อจาก BiLSTM เพื่อให้โมเดลเรียนรู้น้ำหนักของช่วงเวลาเอง อย่างไรก็ตาม attention weight ไม่ใช่คำอธิบายเชิงสาเหตุหรือหลักฐานว่าส่วนของเสียงนั้นเป็นสัญญาณทางการแพทย์ หากต้องอ้างเรื่อง interpretability ต้องมีการศึกษา explanation method และการตรวจจากผู้เชี่ยวชาญแยกต่างหาก

### 2.1.8 Data augmentation, class imbalance และ Mixup

ข้อมูลเสียงร้องทารกมักมีจำนวนจำกัดและจำนวนตัวอย่างต่อ class ไม่สมดุล การทำ augmentation เช่น gain perturbation, time shift, noise injection, time stretch และ pitch shift สามารถเพิ่มความหลากหลายของสัญญาณฝึกได้ถ้าช่วงการเปลี่ยนยังรักษาป้ายกำกับเดิมอย่างสมเหตุสมผล งานล่าสุดด้าน infant-cry detection ยังศึกษาการเพิ่มความหลากหลายผ่านเวลา ความเร็ว pitch และ noise เพื่อรับมือข้อมูลจำกัดและความแปรผันของเสียง [15]

ในโครงงาน augmentation เกิดเฉพาะ training partition ของแต่ละ fold เท่านั้น ไม่ใช้กับ validation หรือ locked Test เพื่อไม่ให้ synthetic sample ปนในข้อมูลประเมิน Target-based augmentation ใช้ปรับโอกาสการปรากฏของ classes ที่มีตัวอย่างน้อย แต่ไม่สร้าง independent biological observations ใหม่

Mixup สร้าง feature และ soft label ใหม่จากตัวอย่างสองรายการตามสมการ

$$
\tilde{x}=\lambda x_i+(1-\lambda)x_j,\qquad
\tilde{y}=\lambda y_i+(1-\lambda)y_j
$$

โดย $\lambda$ สุ่มจาก Beta distribution Zhang และคณะเสนอ Mixup เป็นวิธีเพิ่มข้อมูลที่ส่งเสริม decision boundary ที่ราบเรียบกว่า empirical-risk minimization แบบใช้ตัวอย่างเดี่ยว [16] CryInsight ใช้ Mixup เฉพาะ Stage 2 และบันทึก configuration/manifest เพื่อให้ตรวจสอบได้ การใช้ Mixup เป็นสมมติฐานการ regularize จึงควรยืนยันผลด้วย augmentation ablation และ repeated-seed experiments ไม่ใช่อ้างประสิทธิผลจากหลักการเพียงอย่างเดียว

### 2.1.9 Train/validation/test, grouped cross-validation และการป้องกัน leakage

การฝึกโมเดลต้องแยกบทบาทของข้อมูลให้ชัดเจน

| ส่วนข้อมูล | หน้าที่ที่ถูกต้องในโครงงาน |
|---|---|
| Development Train | สร้าง grouped folds, fit normalizer, ฝึก model, ปรับ candidate และสร้าง OOF predictions |
| Validation fold | เลือก checkpoint/epoch และวัด development performance โดยไม่ใช้ record นั้นฝึกใน fold เดียวกัน |
| Locked internal Test | ประเมิน final-refit model หลัง freeze การตัดสินใจด้าน model development แล้ว |

หาก record ที่มีเนื้อหาเหมือนกัน หรือ segments จาก source เดียวกัน กระจายอยู่ทั้ง training และ validation/test โมเดลอาจจำสัญญาณเดิมแทนที่จะ generalize สถานการณ์นี้เรียกว่า data leakage ด้วยเหตุนี้ CryInsight ใช้ SHA-256 exact-content families, source-file grouping สำหรับ ESC-50 negatives, held-out reservation และ grouped five-fold cross-validation

แนวทาง evaluation ที่เข้มงวดควรแยก preprocessing fit, feature selection, model selection และ hyperparameter tuning ออกจาก test data เพราะการใช้ holdout ซ้ำเพื่อเลือก candidate ทำให้ estimate เกิด bias [17][18] ในระบบนี้ normalizer จึง fit ด้วย training portion ของแต่ละ fold เท่านั้น; OOF evidence ใช้เลือก candidate; locked Test ใช้หลัง final refit เท่านั้น ผล Test ของ corpus เดียวกันยังเป็น internal validation ไม่ใช่ external validation หรือ subject-independent validation

### 2.1.10 Metrics การประเมินและการรายงานอย่างรับผิดชอบ

Accuracy คือสัดส่วนตัวอย่างที่ทำนายถูกทั้งหมด แต่ไม่เพียงพอเมื่อ class imbalance สูง จึงควรอ่านร่วมกับ precision, recall, F1-score, macro average, weighted average และ confusion matrix

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},\qquad
\mathrm{F1}=\frac{2(\mathrm{Precision})(\mathrm{Recall})}{\mathrm{Precision}+\mathrm{Recall}}
$$

สำหรับ Stage 1 ควรมี sensitivity และ specificity เพื่อแยกความผิดพลาดของการรับเสียงทารก (`false negative`) ออกจากการส่งเสียงที่ไม่ใช่ทารกเข้าสู่ Stage 2 (`false positive`) สำหรับ Stage 2 ควรรายงาน macro F1 เพราะทำให้น้ำหนักแต่ละ class เท่ากัน และ weighted F1 เพื่อสะท้อน distribution ที่เกิดขึ้นจริง

นอกจาก label metrics แล้ว CryInsight เก็บ log loss, Brier score และ Expected Calibration Error (ECE) เพื่อวินิจฉัยความน่าเชื่อถือของ probability score การบันทึก metric เหล่านี้ไม่ได้ทำให้ softmax score เป็น probability ทางการแพทย์ ต้องสื่อสารว่าเป็น model confidence ที่ยังไม่ผ่าน clinical calibration

## 2.2 งานวิจัยที่เกี่ยวข้อง

### 2.2.1 หลักเกณฑ์การคัดเลือกและวิธีอ่านผลของงานเดิม

เลือกเอกสารที่เกี่ยวข้องเมื่อมีอย่างน้อยหนึ่งข้อดังนี้: (1) จำแนกเสียงร้องทารกหรือการตรวจจับเสียงร้อง, (2) ใช้ representation ที่สัมพันธ์กับ MFCC, Mel/spectrogram, Chroma หรือ hybrid features, (3) ใช้ CNN, recurrent model, attention หรือ baseline แบบ classical ML, หรือ (4) อธิบายข้อจำกัดด้าน class imbalance, data leakage และ evaluation protocol

ไม่เปรียบเทียบ accuracy/F1 ของงานในตารางข้างล่างกับ CryInsight ตรง ๆ เพราะมีความแตกต่างอย่างน้อยหนึ่งด้าน ได้แก่ จำนวน/แหล่งข้อมูล, age range, definition ของ labels, preprocessing, unit ของ split, data augmentation, model selection และ whether test set ถูกใช้ซ้ำ การวิเคราะห์จึงเน้น “สิ่งที่งานเดิมสนับสนุน” และ “สิ่งที่ยังสรุปไม่ได้”

### 2.2.2 ตารางสรุปงานวิจัยที่เกี่ยวข้อง

| ลำดับ | งานวิจัย | ข้อมูล/โจทย์ | วิธีหลัก | ประเด็นที่นำมาใช้กับ CryInsight | ข้อแตกต่างและข้อควรระวัง |
|---:|---|---|---|---|---|
| 1 | Ji et al. (2021) [2] | งานทบทวน infant cry analysis/classification | ทบทวน data acquisition, signal processing, features และ classifiers | สนับสนุนการมอง pipeline เป็นข้อมูล → preprocessing → feature → classification → evaluation | เป็น review ไม่ใช่หลักฐานว่า architecture ใดชนะบน Dataset ของ CryInsight |
| 2 | Maghfira et al. (2020) [8] | 5 cry-need categories จาก Dunstan Baby Language derivative | CNN–RNN จาก spectrogram, 5-fold CV และ held-out test | ใกล้กับ Stage 2 ในเชิงโจทย์ 5 classes และแนวคิด CNN + sequence model | จำนวนข้อมูล, การจัดคลิป, labels และ protocol ต่างจาก InfantCry-DBL; ห้ามเทียบคะแนนข้ามงานโดยตรง |
| 3 | Liang et al. (2022) [9] | hospital recordings; healthy/sick และ cry-needs ที่ต่างจากโครงงาน | MFCC เปรียบเทียบ ANN, CNN และ LSTM | สนับสนุนการเก็บ CNN/LSTM baseline และความสำคัญของ class balance | labels ถูกกำหนดจากบริบทโรงพยาบาลและขนาด sample/recording protocol ต่างกัน |
| 4 | Ting et al. (2022) [10] | Baby Chillanto; normal/non-asphyxia/asphyxia | MFCC, Chromagram, Mel spectrogram และ feature fusion กับ DNN/CNN | สนับสนุนสมมติฐานว่าหลาย acoustic representations อาจมีข้อมูลเสริมกัน | เป็น pathological-cry task ไม่ใช่ cry-needs task; ไม่ควรแปลผลเป็น diagnostic capability ของ CryInsight |
| 5 | Abbaskhah et al. (2023) [11] | 5 infant-cry classes | MFCC และเปรียบเทียบ SVM, MLP, CNN รวมถึงการจัดการ imbalance | สนับสนุน baseline SVM/CNN และ ablation ของ imbalance treatment | ผลขึ้นกับวิธี split และ SMOTE/non-SMOTE ของงานนั้น ต้องใช้ grouped OOF ของโครงงานเอง |
| 6 | Qiao et al. (2024) [19] | Dunstan Baby Language, Donate a Cry และ Baby Cry datasets | hybrid features, graph structure และ attention-based model | สนับสนุนการศึกษาการผสาน feature และ attention เป็น candidate ใน experiment | วิธี graph construction และ corpora ต่างจาก feature contract ของ CryInsight |
| 7 | Herlea et al. (2025) [14] | binary infant-cry detection ในบริบท baby monitoring | เปรียบเทียบ CNN architectures บน spectrogram/MFCC และศึกษาการเพิ่มข้อมูล | สนับสนุนการแยก Stage 1 เป็น cry gate และการประเมิน robustness กับเสียงอื่น | Dataset ที่มี noise/silence/adult speech และ objective ต่างจาก Stage 1 ที่ใช้ ESC-50 negative set |

### 2.2.3 การวิเคราะห์เชิงเปรียบเทียบ

#### (1) งานเดิมแบ่งเป็นหลาย family ของปัญหา

งาน infant-cry ไม่ใช่โจทย์เดียวกันทั้งหมด บางงานตรวจจับเสียงร้องกับเสียงอื่น (detection), บางงานแยกความต้องการ/สภาวะการดูแล (cry-needs), บางงานแยก normal/pathological cry และบางงานใช้ emotion labels ความแตกต่างนี้มีผลโดยตรงต่อความหมายของ positive/negative class, risk ของ false positive/false negative และ metric ที่ควรรายงาน

CryInsight จึงแยก binary baby gate ออกจาก five-class classifier แทนการฝึก multiclass model ให้ตอบทุกชนิดของเสียงในครั้งเดียว การออกแบบนี้สอดคล้องกับหลักการจำกัด domain ของ classifier แต่ยังต้องมี end-to-end cascade evaluation ในอนาคต เพื่อดูว่าความผิดพลาด Stage 1 ส่งผลต่อ Stage 2 เท่าใด

#### (2) Feature fusion เป็นแนวทางที่สมเหตุสมผล แต่ไม่ใช่ข้อพิสูจน์ล่วงหน้า

ทั้ง Ting et al. [10] และ Qiao et al. [19] ศึกษาการผสาน feature หลาย domain ส่วน CryInsight ใช้ MFCC + Delta + Delta2 ใน Stage 1 และเพิ่ม Log-Mel + Chroma ใน Stage 2 เหตุผลคือแต่ละ representation เน้นมุมมองต่างกัน: spectral envelope, temporal dynamics, time–frequency energy และ harmonic structure

อย่างไรก็ตาม feature ที่มากขึ้นเพิ่มจำนวน input bins และความเสี่ยง overfitting โดยเฉพาะเมื่อข้อมูลมีจำกัด จึงต้องใช้ feature ablation ภายใต้ grouped folds เดียวกันก่อนกล่าวว่าการเพิ่ม Log-Mel หรือ Chroma ทำให้ดีขึ้นจริง

#### (3) CNN และ sequence model มีบทบาทเสริมกัน แต่ต้องเทียบกับ baseline

CNN เหมาะกับ local time–frequency patterns ขณะที่ LSTM/BiLSTM เหมาะกับ dependency ตามเวลา งานของ Maghfira et al. [8] และ Liang et al. [9] แสดงให้เห็นการใช้ convolutional และ recurrent approaches ในโจทย์ infant cry แต่ผลของแต่ละ architecture ไม่คงที่ข้าม Dataset

เหตุผลนี้ทำให้โครงงานจัด `majority`, MFCC-summary SVM, Log-Mel CNN และ ablation ของ CNN/BiLSTM/Attention ไว้ใน Shared Experiment Engine โดยมีเป้าหมายให้ proposed architecture ถูกประเมินเทียบกับทางเลือกที่เรียบง่ายกว่าและทางเลือกที่ใช้ representation ต่างกัน

#### (4) Class imbalance และ augmentation ต้องประเมินอย่างแยกปัจจัย

งานของ Abbaskhah et al. [11] และ Herlea et al. [14] แสดงให้เห็นว่างานในสาขานี้พิจารณา class imbalance และ data augmentation อยู่เสมอ แต่ augmentation อาจช่วยบาง class ขณะลดความคล้ายกับข้อมูลจริงได้หาก perturbation ไม่เหมาะสม CryInsight จึงใช้ target-based augmentation และ Mixup เฉพาะ Train partition พร้อมวาง augmentation ablation และ repeated seeds ไว้เป็นการทดลองที่ต้องทำ ไม่ควรสรุปว่าการเพิ่มข้อมูลสังเคราะห์แทนที่ข้อมูลจากทารกจริงได้

#### (5) การรายงานผลต้องคุม leakage และความสามารถในการ generalize

งานหลายชิ้นใช้ Dataset ขนาดเล็กหรือ Dataset ที่แบ่ง segment จาก record เดียวกัน จึงมีความเป็นไปได้ที่ split level จะกระทบ estimate ของ performance งานทบทวนย้ำว่าข้อมูล infant cry เก็บยากและมีจำนวนจำกัด [2] ทำให้การป้องกัน duplicate, source overlap และ patient/subject leakage สำคัญมาก

CryInsight ดำเนินการที่ระดับ exact-content hash และ source-file grouping เท่าที่ metadata รองรับ ผลที่ได้จึงต้องรายงานเป็น grouped internal validation ไม่ใช่ subject-independent หรือ external validation จนกว่าจะมี subject/session identifiers และ dataset ภายนอกที่เหมาะสม

### 2.2.4 ตำแหน่งของ CryInsight เมื่อเทียบกับวรรณกรรม

| ประเด็น | สิ่งที่วรรณกรรมสนับสนุน | การตอบสนองใน CryInsight | สิ่งที่ยังต้องพิสูจน์ |
|---|---|---|---|
| Cry/no-cry detection | binary detection เป็น task แยกที่พบในระบบ monitoring [14] | Stage 1 ใช้ binary baby gate ก่อน Stage 2 | ความทนทานต่อ soundscape ภายนอก ESC-50 และ audio จากอุปกรณ์จริง |
| Cry-need classification | CNN–RNN ถูกใช้กับ 5 categories ใน DBL derivative [8] | Stage 2 ใช้ CNN + BiLSTM + Attention | ความหมายของ labels และ generalization ข้ามเด็ก/สถานที่ |
| Feature representation | MFCC, spectrogram และ hybrid features ปรากฏในหลายงาน [2][10][19] | Stage 1/2 มี feature contract ต่างกันตาม task | ประโยชน์ส่วนเพิ่มของแต่ละ block ด้วย feature ablation |
| Imbalanced data | งานก่อนหน้าใช้ balancing/augmentation [9][11][14] | target-based augmentation; Mixup ใน Stage 2 | ผลของแต่ละ augmentation และ seed-to-seed stability |
| Evaluation | small datasets ทำให้ validation protocol สำคัญ [2][17] | grouped 5-fold OOF + locked internal Test | external and subject-independent validation |
| Clinical interpretation | งาน pathology ใช้ labels และ protocol เฉพาะทาง [10][20] | จำกัด output เป็น Dataset label ไม่ใช่ diagnosis | clinical utility, calibration และ prospective validation |

## เอกสารอ้างอิง

1. Tawfik, M. (2026). *InfantCry-DBL: A Two-Tier Annotated Corpus of Infant Cries Labelled with Dunstan Baby Language Categories*. Mendeley Data, V1. [https://doi.org/10.17632/x493z8nmwc.1](https://doi.org/10.17632/x493z8nmwc.1)
2. Ji, C., Mudiyanselage, T. B., Gao, Y., & Pan, Y. (2021). A review of infant cry analysis and classification. *EURASIP Journal on Audio, Speech, and Music Processing*, 2021, 8. [https://doi.org/10.1186/s13636-021-00197-5](https://doi.org/10.1186/s13636-021-00197-5)
3. Davis, S. B., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366. [https://doi.org/10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
4. McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–25. [https://doi.org/10.25080/Majora-7b98e3ed-003](https://doi.org/10.25080/Majora-7b98e3ed-003) และ [librosa MFCC documentation](https://librosa.org/doc/0.10.2/generated/librosa.feature.mfcc.html)
5. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. [https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
7. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681. [https://doi.org/10.1109/78.650093](https://doi.org/10.1109/78.650093)
8. Maghfira, T. N., Basaruddin, T., & Krisnadhi, A. (2020). Infant cry classification using CNN–RNN. *Journal of Physics: Conference Series*, 1528, 012019. [https://doi.org/10.1088/1742-6596/1528/1/012019](https://doi.org/10.1088/1742-6596/1528/1/012019)
9. Liang, Y.-C., Wijaya, I., Yang, M.-T., Cuevas Juarez, J. R., & Chang, H.-T. (2022). Deep learning for infant cry recognition. *International Journal of Environmental Research and Public Health*, 19(10), 6311. [https://doi.org/10.3390/ijerph19106311](https://doi.org/10.3390/ijerph19106311)
10. Ting, H.-N., Choo, Y.-M., & Kamar, A. A. (2022). Classification of asphyxia infant cry using hybrid speech features and deep learning models. *Expert Systems with Applications*, 208, 118064. [https://doi.org/10.1016/j.eswa.2022.118064](https://doi.org/10.1016/j.eswa.2022.118064)
11. Abbaskhah, A., Sedighi, H., & Marvi, H. (2023). Infant cry classification by MFCC feature extraction with MLP and CNN structures. *Biomedical Signal Processing and Control*, 86, 105261. [https://doi.org/10.1016/j.bspc.2023.105261](https://doi.org/10.1016/j.bspc.2023.105261)
12. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *International Conference on Learning Representations*. [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
13. Chorowski, J., Bahdanau, D., Serdyuk, D., Cho, K., & Bengio, Y. (2015). Attention-based models for speech recognition. [https://arxiv.org/abs/1506.07503](https://arxiv.org/abs/1506.07503)
14. Herlea, D. M., Iancu, B., & Ardelean, E.-R. (2025). A study of deep learning models for audio classification of infant crying in a baby monitoring system. *Informatics*, 12(2), 50. [https://doi.org/10.3390/informatics12020050](https://doi.org/10.3390/informatics12020050)
15. Zhang, M., Lu, J., Cheng, L., Yang, X., Zhou, J., & Wan, M. (2026). Enhancing infant cry recognition using lightweight CNN with hybrid feature augmentation. *Biomedical Signal Processing and Control*, 121, 110367. [https://doi.org/10.1016/j.bspc.2026.110367](https://doi.org/10.1016/j.bspc.2026.110367)
16. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *International Conference on Learning Representations*. [https://openreview.net/forum?id=r1Ddp1-Rb](https://openreview.net/forum?id=r1Ddp1-Rb)
17. National Academies of Sciences, Engineering, and Medicine. (2022). *Machine Learning Evaluation*. In *Foundational Research Gaps and Future Directions for Digital Twins*. National Academies Press. [https://www.ncbi.nlm.nih.gov/books/NBK597473/](https://www.ncbi.nlm.nih.gov/books/NBK597473/)
18. Bates, S., Hastie, T., & Tibshirani, R. (2024). Cross-validation: what does it estimate and how well does it do it? *Journal of the American Statistical Association*. [https://doi.org/10.1080/01621459.2023.2197686](https://doi.org/10.1080/01621459.2023.2197686)
19. Qiao, X., Jiao, S., Li, H., Liu, G., Gao, X., & Li, Z. (2024). Infant cry classification using an efficient graph structure and attention-based model. *Kuwait Journal of Science*, 51(3), 100221. [https://doi.org/10.1016/j.kjs.2024.100221](https://doi.org/10.1016/j.kjs.2024.100221)
20. Zayed, Y., Hasasneh, A., & Tadj, C. (2023). Infant cry signal diagnostic system using deep learning and fused features. *Diagnostics*, 13(12), 2107. [https://doi.org/10.3390/diagnostics13122107](https://doi.org/10.3390/diagnostics13122107)
