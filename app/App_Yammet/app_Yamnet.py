import tkinter as tk
from tkinter import filedialog
import threading
import numpy as np
import json
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, 'best_model_yamnet.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'labels_yamnet.json')
NORM_PATH   = os.path.join(BASE_DIR, 'norm_stats_yamnet.npy')

LABEL_TH = {
    'belly_pain': 'ปวดท้อง',
    'burping':    'เรอ',
    'discomfort': 'ไม่สบาย',
    'hungry':     'หิว',
    'tired':      'เหนื่อย',
}
LABEL_EMOJI = {
    'belly_pain': '😣',
    'burping':    '😮',
    'discomfort': '😰',
    'hungry':     '🍼',
    'tired':      '😴',
}
LABEL_COLOR = {
    'belly_pain': '#FF6B6B',
    'burping':    '#FFB347',
    'discomfort': '#A78BFA',
    'hungry':     '#34D399',
    'tired':      '#60A5FA',
}
LABEL_EN = {
    'belly_pain': 'Belly Pain',
    'burping':    'Burping',
    'discomfort': 'Discomfort',
    'hungry':     'Hungry',
    'tired':      'Tired',
}

BG         = '#0F0F0F'
SURFACE    = '#1A1A1A'
SURFACE2   = '#242424'
BORDER     = '#2E2E2E'
TEXT       = '#F0F0F0'
TEXT_MUTED = '#6B6B6B'
ACCENT     = '#E8E8E8'


class BabyCryApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Baby Cry Detector')
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.yamnet = None
        self.model  = None
        self.labels = None
        self.mean   = None
        self.std    = None
        self.selected_file = None

        self._build_ui()
        self._load_models_async()

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill='both', expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self.inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=self.inner, anchor='nw')

        self.inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all('<MouseWheel>', lambda e: canvas.yview_scroll(-1*(e.delta//120), 'units'))

        self._build_content(self.inner)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = 500, min(sh - 80, 900)
        self.root.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')

    def _build_content(self, p):
        pad = 28

        # Header
        hdr = tk.Frame(p, bg=BG)
        hdr.pack(fill='x', padx=pad, pady=(32, 0))
        tk.Label(hdr, text='Baby Cry', font=('Helvetica', 24, 'bold'), bg=BG, fg=TEXT).pack(anchor='w')
        tk.Label(hdr, text='Emotion Detector  •  YAMNet ', font=('Helvetica', 11), bg=BG, fg=TEXT_MUTED).pack(anchor='w')

        self._div(p, pad)

        # Upload zone — คลิกแล้ววิเคราะห์ทันที
        uz = tk.Frame(p, bg=SURFACE, cursor='hand2')
        uz.pack(fill='x', padx=pad)
        uz.bind('<Button-1>', lambda e: self._pick_file())

        iu = tk.Frame(uz, bg=SURFACE)
        iu.pack(pady=24)

        self.upload_icon = tk.Label(iu, text='🎙️', font=('Helvetica', 36), bg=SURFACE, cursor='hand2')
        self.upload_icon.pack()
        self.upload_icon.bind('<Button-1>', lambda e: self._pick_file())

        self.upload_text = tk.Label(iu, text='คลิกเพื่อเลือกไฟล์  →  วิเคราะห์ทันที',
                                    font=('Helvetica', 12), bg=SURFACE, fg=TEXT_MUTED, cursor='hand2')
        self.upload_text.pack(pady=(6, 2))
        self.upload_text.bind('<Button-1>', lambda e: self._pick_file())

        self.file_lbl = tk.Label(iu, text='', font=('Helvetica', 9),
                                 bg=SURFACE, fg=TEXT_MUTED, wraplength=420)
        self.file_lbl.pack()
        self.file_lbl.bind('<Button-1>', lambda e: self._pick_file())

        # Status
        self.status_var = tk.StringVar(value='กำลังโหลดโมเดล...')
        tk.Label(p, textvariable=self.status_var, font=('Helvetica', 10),
                 bg=BG, fg=TEXT_MUTED).pack(pady=12)

        self._div(p, pad)

        # Result card
        tk.Label(p, text='ผลการวิเคราะห์', font=('Helvetica', 11), bg=BG, fg=TEXT_MUTED).pack(anchor='w', padx=pad)
        card = tk.Frame(p, bg=SURFACE)
        card.pack(fill='x', padx=pad, pady=(8, 0))

        top_row = tk.Frame(card, bg=SURFACE)
        top_row.pack(fill='x', padx=20, pady=(18, 4))
        self.emoji_lbl = tk.Label(top_row, text='', font=('Helvetica', 28), bg=SURFACE)
        self.emoji_lbl.pack(side='left')
        self.result_lbl = tk.Label(top_row, text='—', font=('Helvetica', 26, 'bold'), bg=SURFACE, fg=TEXT_MUTED)
        self.result_lbl.pack(side='left', padx=(8, 0))

        self.conf_lbl = tk.Label(card, text='', font=('Helvetica', 11), bg=SURFACE, fg=TEXT_MUTED)
        self.conf_lbl.pack(anchor='w', padx=20, pady=(0, 16))

        self._div(p, pad)

        # Bars — ทุก class มีสี
        tk.Label(p, text='ความน่าจะเป็นทุก class', font=('Helvetica', 11), bg=BG, fg=TEXT_MUTED).pack(anchor='w', padx=pad)
        self.bars_frame = tk.Frame(p, bg=BG)
        self.bars_frame.pack(fill='x', padx=pad, pady=(8, 0))

        self.bar_widgets = {}
        for lbl in ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']:
            row = tk.Frame(self.bars_frame, bg=BG)
            row.pack(fill='x', pady=5)

            name = tk.Label(row, text=f"{LABEL_EMOJI[lbl]} {LABEL_TH[lbl]}",
                            font=('Helvetica', 11), bg=BG, fg=TEXT_MUTED, width=14, anchor='w')
            name.pack(side='left')

            bar_bg = tk.Frame(row, bg=SURFACE2, height=10)
            bar_bg.pack(side='left', fill='x', expand=True, padx=(4, 8))
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=BORDER, height=10, width=0)
            bar_fill.place(x=0, y=0, relheight=1)

            pct = tk.Label(row, text='—', font=('Helvetica', 10), bg=BG, fg=TEXT_MUTED, width=6, anchor='e')
            pct.pack(side='right')

            self.bar_widgets[lbl] = (bar_bg, bar_fill, pct, name)

        self._div(p, pad)

        # Pie chart
        tk.Label(p, text='กราฟวงกลม', font=('Helvetica', 11), bg=BG, fg=TEXT_MUTED).pack(anchor='w', padx=pad)

        chart_frame = tk.Frame(p, bg=BG)
        chart_frame.pack(fill='x', padx=pad, pady=(8, 24))

        fig, self.ax = plt.subplots(figsize=(4.4, 5.5), facecolor=BG)
        self.ax.set_facecolor(BG)
        self.ax.text(0, 0, 'Waiting...', ha='center', va='center', color=TEXT_MUTED, fontsize=11)
        self.ax.axis('off')

        self.chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill='x')
        self.chart_canvas.draw()

        tk.Frame(p, bg=BG, height=20).pack()

    def _div(self, parent, pad):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=pad, pady=16)

    def _load_models_async(self):
        def load():
            try:
                self.status_var.set('กำลังโหลด YAMNet...')
                self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
                self.status_var.set('กำลังโหลดโมเดล CNN...')
                self.model  = tf.keras.models.load_model(MODEL_PATH)
                self.labels = json.load(open(LABELS_PATH))
                norm = np.load(NORM_PATH)
                self.mean, self.std = norm[0], norm[1]
                self.status_var.set('✅ พร้อมใช้งาน — คลิกเลือกไฟล์เพื่อเริ่ม')
            except Exception as e:
                self.status_var.set(f'❌ โหลดไม่สำเร็จ: {e}')
        threading.Thread(target=load, daemon=True).start()

    def _pick_file(self):
        if self.model is None:
            self.status_var.set('⏳ รอโหลดโมเดลก่อน...')
            return
        path = filedialog.askopenfilename(
            title='เลือกไฟล์เสียงเด็กร้อง',
            filetypes=[('WAV files', '*.wav'), ('All files', '*.*')]
        )
        if path:
            self.selected_file = path
            self.upload_icon.config(text='⏳')
            self.upload_text.config(text='ไฟล์ที่เลือก:', fg=TEXT)
            self.file_lbl.config(text=os.path.basename(path), fg=ACCENT)
            self.status_var.set('กำลังวิเคราะห์...')
            self._analyze()  # วิเคราะห์ทันที ไม่ต้องกดปุ่ม

    def _analyze(self):
        def run():
            try:
                audio, _ = librosa.load(self.selected_file, sr=16000, mono=True)
                audio, _ = librosa.effects.trim(audio, top_db=20)
                audio    = librosa.util.normalize(audio).astype(np.float32)
                _, emb, _ = self.yamnet(audio)
                emb = np.mean(emb.numpy(), axis=0)
                emb = (emb - self.mean) / self.std
                emb = emb[np.newaxis, ...]
                probs   = self.model.predict(emb, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                top_lbl = self.labels[top_idx]
                top_pct = float(probs[top_idx]) * 100
                self.root.after(0, lambda: self._show_result(top_lbl, top_pct, probs))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f'❌ Error: {e}'))
                self.root.after(0, lambda: self.upload_icon.config(text='🎙️'))

        threading.Thread(target=run, daemon=True).start()

    def _show_result(self, top_lbl, top_pct, probs):
        color = LABEL_COLOR.get(top_lbl, ACCENT)
        self.emoji_lbl.config(text=LABEL_EMOJI[top_lbl])
        self.result_lbl.config(text=LABEL_TH[top_lbl], fg=color)
        self.conf_lbl.config(text=f'ความมั่นใจ  {top_pct:.1f}%', fg=TEXT_MUTED)
        self.upload_icon.config(text='✅')
        self.status_var.set('✅ วิเคราะห์เสร็จแล้ว — คลิกเพื่อเลือกไฟล์ใหม่')

        # Bars — ทุก class มีสีของตัวเอง
        self.root.update_idletasks()
        for i, lbl in enumerate(self.labels):
            bg, fill, pct_lbl, name_lbl = self.bar_widgets[lbl]
            p = float(probs[i])
            bar_w  = bg.winfo_width()
            fill_w = max(4, int(bar_w * p))
            is_top = (lbl == top_lbl)

            # ทุก class มีสี แต่ top จะสว่างกว่า
            fill.config(bg=LABEL_COLOR[lbl])
            fill.place(x=0, y=0, relheight=1, width=fill_w)
            pct_lbl.config(
                text=f'{p*100:.1f}%',
                fg=TEXT if is_top else TEXT_MUTED,
                font=('Helvetica', 10, 'bold') if is_top else ('Helvetica', 10)
            )
            name_lbl.config(
                fg=TEXT if is_top else TEXT_MUTED,
                font=('Helvetica', 11, 'bold') if is_top else ('Helvetica', 11)
            )

        # Pie chart
        self.ax.clear()
        self.ax.set_facecolor(BG)
        sizes   = [float(probs[self.labels.index(l)]) for l in LABEL_TH]
        colors  = [LABEL_COLOR[l] for l in LABEL_TH]
        explode = [0.07 if l == top_lbl else 0 for l in LABEL_TH]

        wedges, texts, autotexts = self.ax.pie(
            sizes, colors=colors,
            explode=explode,
            autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
            startangle=140,
            pctdistance=0.6,
            textprops={'color': TEXT, 'fontsize': 8}
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color('#0F0F0F')

        legend_labels = [
            f"{LABEL_EN[l]}  {probs[self.labels.index(l)]*100:.1f}%"
            for l in LABEL_TH
        ]
        self.ax.legend(
            wedges, legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
            frameon=False,
            labelcolor=TEXT,
            ncol=1,
            handlelength=1.5,
            handleheight=1.0,
        )

        self.ax.set_title('Probability Distribution', color=TEXT_MUTED, fontsize=10, pad=10)
        self.chart_canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    BabyCryApp(root)
    root.mainloop()