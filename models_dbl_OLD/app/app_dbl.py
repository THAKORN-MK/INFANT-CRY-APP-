import tkinter as tk
from tkinter import filedialog
import threading
import numpy as np
import json
import librosa
import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ══════════════════════════════════════════
# ลงทะเบียน Custom Layer ก่อนโหลดโมเดล
# ══════════════════════════════════════════
@tf.keras.saving.register_keras_serializable()
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
        return tf.reduce_sum(x * tf.expand_dims(alpha, -1), axis=1)

# PATH
BASE_DIR      = 'D:/INFANT CRY'
BINARY_MODEL  = f'{BASE_DIR}/models_dbl/binary/best_model_binary_dbl.keras'
BINARY_NORM   = f'{BASE_DIR}/models_dbl/binary/norm_stats_binary_dbl.npy'
BINARY_LABELS = f'{BASE_DIR}/models_dbl/binary/labels_binary_dbl.json'
MAIN_MODEL    = f'{BASE_DIR}/models_dbl/main/best_model_main_dbl.keras'
MAIN_NORM     = f'{BASE_DIR}/models_dbl/main/norm_stats_main_dbl.npy'
MAIN_LABELS   = f'{BASE_DIR}/models_dbl/main/labels_main_dbl.json'

SR=22050; N_MFCC=40; N_MELS=64; N_CHROMA=12; MAX_LEN=128
LABEL_TH    = {'belly_pain':'ปวดท้อง','burping':'เรอ','discomfort':'ไม่สบาย','hungry':'หิว','tired':'เหนื่อย'}
LABEL_EMOJI = {'belly_pain':'😣','burping':'😮','discomfort':'😰','hungry':'🍼','tired':'😴'}
LABEL_COLOR = {'belly_pain':'#FF6B6B','burping':'#FFB347','discomfort':'#A78BFA','hungry':'#34D399','tired':'#60A5FA'}
LABEL_EN    = {'belly_pain':'Belly Pain','burping':'Burping','discomfort':'Discomfort','hungry':'Hungry','tired':'Tired'}
BG='#0F0F0F'; SURFACE='#1A1A1A'; SURFACE2='#242424'; BORDER='#2E2E2E'
TEXT='#F0F0F0'; TEXT_MUTED='#6B6B6B'; ACCENT='#E8E8E8'; RED='#FF6B6B'; GREEN='#34D399'

class BabyCryApp:
    def __init__(self, root):
        self.root=root; self.root.title('Baby Cry Detector — CNN+MFCC 2-Stage')
        self.root.configure(bg=BG); self.root.resizable(True,True)
        self.binary_model=None; self.binary_norm=None; self.binary_labels=None
        self.main_model=None; self.main_norm=None; self.main_labels=None
        self.selected_file=None
        self._build_ui(); self._load_models_async()

    def _build_ui(self):
        outer=tk.Frame(self.root,bg=BG); outer.pack(fill='both',expand=True)
        canvas=tk.Canvas(outer,bg=BG,highlightthickness=0)
        sb=tk.Scrollbar(outer,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right',fill='y'); canvas.pack(side='left',fill='both',expand=True)
        self.inner=tk.Frame(canvas,bg=BG)
        win_id=canvas.create_window((0,0),window=self.inner,anchor='nw')
        self.inner.bind('<Configure>',lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e: canvas.itemconfig(win_id,width=e.width))
        canvas.bind_all('<MouseWheel>',lambda e: canvas.yview_scroll(-1*(e.delta//120),'units'))
        self._build_content(self.inner)
        sw=self.root.winfo_screenwidth(); sh=self.root.winfo_screenheight()
        w,h=520,min(sh-80,960)
        self.root.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')

    def _build_content(self, p):
        pad=28
        hdr=tk.Frame(p,bg=BG); hdr.pack(fill='x',padx=pad,pady=(32,0))
        tk.Label(hdr,text='Baby Cry',font=('Helvetica',24,'bold'),bg=BG,fg=TEXT).pack(anchor='w')
        tk.Label(hdr,text='Emotion Detector  •  CNN+MFCC+Mel+Chroma+BiLSTM+Attention',font=('Helvetica',10),bg=BG,fg=TEXT_MUTED).pack(anchor='w')
        self._div(p,pad)
        bf=tk.Frame(p,bg=BG); bf.pack(fill='x',padx=pad)
        for title,sub,color in [('Stage 1','Binary CNN+MFCC\nตรวจเสียงเด็ก','#60A5FA'),('→','',BG),('Stage 2','CNN+MFCC\nวิเคราะห์อารมณ์','#34D399')]:
            if title=='→':
                tk.Label(bf,text='→',font=('Helvetica',18),bg=BG,fg=TEXT_MUTED).pack(side='left',padx=4)
            else:
                box=tk.Frame(bf,bg=SURFACE); box.pack(side='left',padx=4)
                tk.Label(box,text=title,font=('Helvetica',9,'bold'),bg=SURFACE,fg=color).pack(padx=12,pady=(8,2))
                tk.Label(box,text=sub,font=('Helvetica',9),bg=SURFACE,fg=TEXT_MUTED).pack(padx=12,pady=(0,8))
        self._div(p,pad)
        uz=tk.Frame(p,bg=SURFACE,cursor='hand2'); uz.pack(fill='x',padx=pad)
        uz.bind('<Button-1>',lambda e: self._pick_file())
        iu=tk.Frame(uz,bg=SURFACE); iu.pack(pady=24)
        self.upload_icon=tk.Label(iu,text='🎙️',font=('Helvetica',36),bg=SURFACE,cursor='hand2')
        self.upload_icon.pack(); self.upload_icon.bind('<Button-1>',lambda e: self._pick_file())
        self.upload_text=tk.Label(iu,text='คลิกเพื่อเลือกไฟล์เสียง .wav  →  วิเคราะห์ทันที',font=('Helvetica',12),bg=SURFACE,fg=TEXT_MUTED,cursor='hand2')
        self.upload_text.pack(pady=(6,2)); self.upload_text.bind('<Button-1>',lambda e: self._pick_file())
        self.file_lbl=tk.Label(iu,text='',font=('Helvetica',9),bg=SURFACE,fg=TEXT_MUTED,wraplength=440)
        self.file_lbl.pack(); self.file_lbl.bind('<Button-1>',lambda e: self._pick_file())
        self.status_var=tk.StringVar(value='กำลังโหลดโมเดล...')
        tk.Label(p,textvariable=self.status_var,font=('Helvetica',10),bg=BG,fg=TEXT_MUTED).pack(pady=10)
        self._div(p,pad)
        tk.Label(p,text='Stage 1 — ตรวจสอบเสียงเด็ก (Binary CNN+MFCC)',font=('Helvetica',11),bg=BG,fg=TEXT_MUTED).pack(anchor='w',padx=pad)
        s1=tk.Frame(p,bg=SURFACE); s1.pack(fill='x',padx=pad,pady=(8,0))
        self.s1_icon=tk.Label(s1,text='—',font=('Helvetica',22),bg=SURFACE,fg=TEXT_MUTED)
        self.s1_icon.pack(side='left',padx=(16,8),pady=16)
        s1r=tk.Frame(s1,bg=SURFACE); s1r.pack(side='left',pady=16)
        self.s1_result=tk.Label(s1r,text='รอการวิเคราะห์',font=('Helvetica',14,'bold'),bg=SURFACE,fg=TEXT_MUTED); self.s1_result.pack(anchor='w')
        self.s1_score=tk.Label(s1r,text='',font=('Helvetica',10),bg=SURFACE,fg=TEXT_MUTED); self.s1_score.pack(anchor='w')
        self._div(p,pad)
        tk.Label(p,text='Stage 2 — วิเคราะห์อารมณ์ (CNN+MFCC+Mel+Chroma)',font=('Helvetica',11),bg=BG,fg=TEXT_MUTED).pack(anchor='w',padx=pad)
        card=tk.Frame(p,bg=SURFACE); card.pack(fill='x',padx=pad,pady=(8,0))
        tr=tk.Frame(card,bg=SURFACE); tr.pack(fill='x',padx=20,pady=(18,4))
        self.emoji_lbl=tk.Label(tr,text='',font=('Helvetica',28),bg=SURFACE); self.emoji_lbl.pack(side='left')
        self.result_lbl=tk.Label(tr,text='—',font=('Helvetica',26,'bold'),bg=SURFACE,fg=TEXT_MUTED); self.result_lbl.pack(side='left',padx=(8,0))
        self.conf_lbl=tk.Label(card,text='',font=('Helvetica',11),bg=SURFACE,fg=TEXT_MUTED); self.conf_lbl.pack(anchor='w',padx=20,pady=(0,16))
        self._div(p,pad)
        tk.Label(p,text='ความน่าจะเป็นทุก class',font=('Helvetica',11),bg=BG,fg=TEXT_MUTED).pack(anchor='w',padx=pad)
        self.bars_frame=tk.Frame(p,bg=BG); self.bars_frame.pack(fill='x',padx=pad,pady=(8,0))
        self.bar_widgets={}
        for lbl in ['belly_pain','burping','discomfort','hungry','tired']:
            row=tk.Frame(self.bars_frame,bg=BG); row.pack(fill='x',pady=5)
            name=tk.Label(row,text=f"{LABEL_EMOJI[lbl]} {LABEL_TH[lbl]}",font=('Helvetica',11),bg=BG,fg=TEXT_MUTED,width=14,anchor='w'); name.pack(side='left')
            bar_bg=tk.Frame(row,bg=SURFACE2,height=10); bar_bg.pack(side='left',fill='x',expand=True,padx=(4,8)); bar_bg.pack_propagate(False)
            bar_fill=tk.Frame(bar_bg,bg=BORDER,height=10,width=0); bar_fill.place(x=0,y=0,relheight=1)
            pct=tk.Label(row,text='—',font=('Helvetica',10),bg=BG,fg=TEXT_MUTED,width=6,anchor='e'); pct.pack(side='right')
            self.bar_widgets[lbl]=(bar_bg,bar_fill,pct,name)
        self._div(p,pad)
        tk.Label(p,text='กราฟวงกลม',font=('Helvetica',11),bg=BG,fg=TEXT_MUTED).pack(anchor='w',padx=pad)
        cf=tk.Frame(p,bg=BG); cf.pack(fill='x',padx=pad,pady=(8,24))
        fig,self.ax=plt.subplots(figsize=(4.4,5.5),facecolor=BG)
        self.ax.set_facecolor(BG); self.ax.text(0,0,'Waiting...',ha='center',va='center',color=TEXT_MUTED,fontsize=11); self.ax.axis('off')
        self.chart_canvas=FigureCanvasTkAgg(fig,master=cf)
        self.chart_canvas.get_tk_widget().pack(fill='x'); self.chart_canvas.draw()
        tk.Frame(p,bg=BG,height=20).pack()

    def _div(self,parent,pad):
        tk.Frame(parent,bg=BORDER,height=1).pack(fill='x',padx=pad,pady=16)

    def _extract_features(self, audio, sr=SR, max_len=MAX_LEN):
        mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=N_MFCC,n_fft=2048,hop_length=512)
        delta=librosa.feature.delta(mfcc); delta2=librosa.feature.delta(mfcc,order=2)
        mel=librosa.power_to_db(librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=N_MELS,n_fft=2048,hop_length=512),ref=np.max)
        chroma=librosa.feature.chroma_stft(y=audio,sr=sr,n_chroma=N_CHROMA,n_fft=2048,hop_length=512)
        combined=np.vstack([mfcc,delta,delta2,mel,chroma])
        if combined.shape[1]<max_len: combined=np.pad(combined,((0,0),(0,max_len-combined.shape[1])))
        else: combined=combined[:,:max_len]
        return combined[...,np.newaxis].astype(np.float32)

    def _load_models_async(self):
        def load():
            try:
                self.status_var.set('โหลด Binary Model (Stage 1)...')
                self.binary_model=tf.keras.models.load_model(
                    BINARY_MODEL,
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
                self.binary_norm=np.load(BINARY_NORM)
                self.binary_labels=json.load(open(BINARY_LABELS))
                self.status_var.set('โหลด Main Model (Stage 2)...')
                self.main_model=tf.keras.models.load_model(
                    MAIN_MODEL,
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
                self.main_norm=np.load(MAIN_NORM)
                self.main_labels=json.load(open(MAIN_LABELS))
                self.status_var.set('✅ พร้อมใช้งาน — คลิกเลือกไฟล์เพื่อเริ่ม')
            except Exception as e:
                self.status_var.set(f'❌ โหลดไม่สำเร็จ: {e}')
        threading.Thread(target=load,daemon=True).start()

    def _pick_file(self):
        if self.main_model is None:
            self.status_var.set('⏳ รอโหลดโมเดลก่อน...'); return
        path=filedialog.askopenfilename(title='เลือกไฟล์เสียงเด็กร้อง',filetypes=[('WAV files','*.wav'),('All files','*.*')])
        if path:
            self.selected_file=path
            self.upload_icon.config(text='⏳'); self.upload_text.config(text='ไฟล์ที่เลือก:',fg=TEXT)
            self.file_lbl.config(text=os.path.basename(path),fg=ACCENT)
            self.s1_icon.config(text='—',fg=TEXT_MUTED); self.s1_result.config(text='กำลังตรวจสอบ...',fg=TEXT_MUTED); self.s1_score.config(text='')
            self.emoji_lbl.config(text=''); self.result_lbl.config(text='—',fg=TEXT_MUTED); self.conf_lbl.config(text='')
            self.status_var.set('กำลังวิเคราะห์...')
            threading.Thread(target=self._analyze,daemon=True).start()

    def _analyze(self):
        try:
            audio,_=librosa.load(self.selected_file,sr=SR,mono=True)
            audio,_=librosa.effects.trim(audio,top_db=20)
            audio=librosa.util.normalize(audio).astype(np.float32)
            feat=self._extract_features(audio)

            # Stage 1: Binary
            bm,bs=self.binary_norm[0],self.binary_norm[1]
            feat_b=(feat-bm)/bs; feat_b=feat_b[np.newaxis,...]
            b_probs=self.binary_model.predict(feat_b,verbose=0)[0]
            baby_score=float(b_probs[1]); is_baby=baby_score>=0.5
            self.root.after(0,lambda: self._show_s1(is_baby,baby_score))
            if not is_baby:
                self.root.after(0,self._show_not_baby); return

            # Stage 2: Main
            mm,ms=self.main_norm[0],self.main_norm[1]
            feat_m=(feat-mm)/ms; feat_m=feat_m[np.newaxis,...]
            probs=self.main_model.predict(feat_m,verbose=0)[0]
            top_idx=int(np.argmax(probs)); top_lbl=self.main_labels[top_idx]; top_pct=float(probs[top_idx])*100
            self.root.after(0,lambda: self._show_s2(top_lbl,top_pct,probs))
        except Exception as e:
            self.root.after(0,lambda: self.status_var.set(f'❌ Error: {e}'))
            self.root.after(0,lambda: self.upload_icon.config(text='🎙️'))

    def _show_s1(self,is_baby,score):
        if is_baby:
            self.s1_icon.config(text='✅',fg=GREEN); self.s1_result.config(text='ตรวจพบเสียงร้องเด็ก',fg=GREEN)
        else:
            self.s1_icon.config(text='❌',fg=RED); self.s1_result.config(text='ไม่ใช่เสียงร้องเด็ก',fg=RED)
        self.s1_score.config(text=f'Baby score: {score*100:.1f}%  (threshold: 50%)')

    def _show_not_baby(self):
        self.upload_icon.config(text='❌'); self.emoji_lbl.config(text='❌')
        self.result_lbl.config(text='ไม่ใช่เสียงเด็ก',fg=RED)
        self.conf_lbl.config(text='ไม่สามารถวิเคราะห์อารมณ์ได้',fg=TEXT_MUTED)
        self.status_var.set('❌ ไม่ใช่เสียงร้องเด็ก — ลองไฟล์ใหม่')
        self.ax.clear(); self.ax.set_facecolor(BG)
        self.ax.text(0,0,'Not a baby cry',ha='center',va='center',color=RED,fontsize=14,fontweight='bold')
        self.ax.axis('off'); self.chart_canvas.draw()

    def _show_s2(self,top_lbl,top_pct,probs):
        color=LABEL_COLOR.get(top_lbl,ACCENT)
        self.upload_icon.config(text='✅'); self.emoji_lbl.config(text=LABEL_EMOJI[top_lbl])
        self.result_lbl.config(text=LABEL_TH[top_lbl],fg=color)
        self.conf_lbl.config(text=f'ความมั่นใจ  {top_pct:.1f}%',fg=TEXT_MUTED)
        self.status_var.set('✅ วิเคราะห์เสร็จแล้ว — คลิกเพื่อเลือกไฟล์ใหม่')
        self.root.update_idletasks()
        for i,lbl in enumerate(self.main_labels):
            bg,fill,pct_lbl,name_lbl=self.bar_widgets[lbl]
            p=float(probs[i]); bar_w=bg.winfo_width(); fill_w=max(4,int(bar_w*p)); is_top=(lbl==top_lbl)
            fill.config(bg=LABEL_COLOR[lbl]); fill.place(x=0,y=0,relheight=1,width=fill_w)
            pct_lbl.config(text=f'{p*100:.1f}%',fg=TEXT if is_top else TEXT_MUTED,font=('Helvetica',10,'bold') if is_top else ('Helvetica',10))
            name_lbl.config(fg=TEXT if is_top else TEXT_MUTED,font=('Helvetica',11,'bold') if is_top else ('Helvetica',11))
        self.ax.clear(); self.ax.set_facecolor(BG)
        sizes=[float(probs[self.main_labels.index(l)]) for l in LABEL_TH]
        colors=[LABEL_COLOR[l] for l in LABEL_TH]; explode=[0.07 if l==top_lbl else 0 for l in LABEL_TH]
        wedges,_,ats=self.ax.pie(sizes,colors=colors,explode=explode,autopct=lambda p:f'{p:.1f}%' if p>3 else '',startangle=140,pctdistance=0.6,textprops={'color':TEXT,'fontsize':8})
        for at in ats: at.set_fontsize(7); at.set_color('#0F0F0F')
        self.ax.legend(wedges,[f"{LABEL_EN[l]}  {probs[self.main_labels.index(l)]*100:.1f}%" for l in LABEL_TH],
                       loc='upper center',bbox_to_anchor=(0.5,-0.02),fontsize=9,frameon=False,labelcolor=TEXT,ncol=1,handlelength=1.5,handleheight=1.0)
        self.ax.set_title('Probability Distribution',color=TEXT_MUTED,fontsize=10,pad=10)
        self.chart_canvas.draw()

if __name__=='__main__':
    root=tk.Tk(); BabyCryApp(root); root.mainloop()