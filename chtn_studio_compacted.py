import os, tempfile, threading, tkinter as tk, librosa, numpy as np, sounddevice as sd
from tkinter import filedialog, messagebox, ttk

S_RATE, B_SIZE, M_FRQ, MX_FRQ, F_VER, MX_OSC = 44100, 1024, 20, 20000, 262, 512

def r_canales(t): return t - int(t*0.15), int(t*0.15)
def q_vol(v): return np.clip(np.rint(np.power(v, 0.7)*15), 0, 15).astype(np.uint8)
def dq_vol(v): return np.power(v.astype(np.float32)/15.0, 1.428)

def pack_nb(a):
    s, p = a.shape, a.flatten()
    if len(p)%2!=0: p = np.append(p, np.uint8(0))
    return (p[0::2]<<4)|(p[1::2]&0x0F), np.array(s, dtype=np.uint32)

def unpack_nb(b, s):
    t = np.prod(s)
    p = np.zeros(t+(t%2), dtype=np.uint8)
    p[0::2], p[1::2] = (b>>4)&0x0F, b&0x0F
    return p[:t].reshape(s)

def p_psico(f):
    f = np.clip(f, 10, 22000)
    return (1.0 + 3.8*np.exp(-((np.log10(f)-3.15)**2)/0.45) + 1.5*np.exp(-((np.log10(f)-3.65)**2)/0.15)).astype(np.float32)

class ChtnStudioApp:
    def __init__(self, root):
        self.root = root
        root.title("CHTN Studio v26.2 Mini")
        root.geometry("1020x850")
        
        self.stream, self.reproduciendo, self.seeking = None, False, False
        self.b_tf, self.b_tv, self.b_tp, self.b_nv = None, None, None, None
        self.tot_f, self.fps, self.f_act, self.s_cnt = 0, 44100.0/512.0, 0, 0
        self.c_tot, self.c_ton, self.c_rui = 256, 218, 38
        
        self.phases = np.zeros(MX_OSC, dtype=np.float32)
        self.n_lk = 44100*4
        self.n_buf = np.random.normal(0.0, 0.18, size=self.n_lk).astype(np.float32)
        self.n_off = np.random.randint(0, self.n_lk-B_SIZE, size=MX_OSC)
        
        self.v_lock = threading.Lock()
        self.v_ch_buf = np.zeros((B_SIZE//16, 64), dtype=np.float32)
        self.v_inst_v, self.v_inst_f = [], []

        self.sv_ch, self.sv_gain, self.sv_vis = tk.IntVar(value=256), tk.DoubleVar(value=1.5), tk.BooleanVar(value=True)
        self.sv_mod, self.sv_auto = tk.StringVar(value="Espectrómetro de Barras"), tk.BooleanVar(value=True)
        self.progress, self.s_mp3, self.s_chtn, self.l_chtn = tk.DoubleVar(value=0), tk.StringVar(value="Ninguno"), tk.StringVar(value="Ninguno"), tk.StringVar(value="No cargado")
        self.lbl_t_t, self.lbl_t_r, self.lbl_t_tot = tk.StringVar(value="00:00"), tk.StringVar(value="00:00"), tk.StringVar(value="Duración: --:--")

        self._ui()
        root.after(100, self._loop_vis)

    def _ui(self):
        c = ttk.Frame(self.root, padding=10)
        c.pack(fill="both", expand=True)
        c.columnconfigure(0, weight=1)
        
        f1 = ttk.LabelFrame(c, text=" Codificación ")
        f1.grid(row=0, column=0, sticky="ew", pady=4)
        f1.columnconfigure(1, weight=1)
        
        ttk.Button(f1, text="Buscar Audio", command=self.sel_mp3).grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        ttk.Label(f1, textvariable=self.s_mp3).grid(row=0, column=1, padx=5, sticky="w")
        ttk.Button(f1, text="Destino .chtn", command=self.sel_out).grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        ttk.Label(f1, textvariable=self.s_chtn).grid(row=1, column=1, padx=5, sticky="w")
        
        tk.Scale(f1, from_=32, to=512, orient="horizontal", variable=self.sv_ch, highlightthickness=0).grid(row=2, column=0, columnspan=2, sticky="ew", padx=5)
        
        opc = ttk.Frame(f1)
        opc.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        opc.columnconfigure(1, weight=1)
        ttk.Checkbutton(opc, text="Auto-cargar", variable=self.sv_auto).grid(row=0, column=0)
        ttk.Progressbar(opc, variable=self.progress, maximum=100).grid(row=0, column=1, padx=5, sticky="ew")
        
        self.btn_conv = ttk.Button(f1, text="Compilar .chtn", command=self.convertir, state="disabled")
        self.btn_conv.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=4)

        f2 = ttk.LabelFrame(c, text=" Reproductor HQ ")
        f2.grid(row=1, column=0, sticky="ew", pady=4)
        f2.columnconfigure(1, weight=1)
        
        ttk.Label(f2, textvariable=self.l_chtn).grid(row=0, column=0, padx=5, sticky="w")
        ttk.Label(f2, textvariable=self.lbl_t_tot).grid(row=0, column=2, padx=5, sticky="e")
        
        ttk.Label(f2, textvariable=self.lbl_t_t, font=("Consolas", 10, "bold")).grid(row=1, column=0, padx=5)
        self.t_slider = tk.Scale(f2, from_=0, to=100, orient="horizontal", showvalue=False, highlightthickness=0, command=self._seek)
        self.t_slider.grid(row=1, column=1, sticky="ew")
        self.t_slider.bind("<ButtonPress-1>", lambda e: setattr(self, 'seeking', True))
        self.t_slider.bind("<ButtonRelease-1>", self._release)
        ttk.Label(f2, textvariable=self.lbl_t_r, font=("Consolas", 10, "bold")).grid(row=1, column=2, padx=5)
        
        g_fr = ttk.Frame(f2)
        g_fr.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5)
        g_fr.columnconfigure(1, weight=1)
        ttk.Label(g_fr, text="Ganancia:").grid(row=0, column=0, padx=2)
        ttk.Scale(g_fr, from_=0.0, to=4.0, orient="horizontal", variable=self.sv_gain).grid(row=0, column=1, sticky="ew")
        
        b_fr = ttk.Frame(f2)
        b_fr.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=4)
        ttk.Button(b_fr, text="Abrir .chtn", command=self.load_chtn).grid(row=0, column=0, padx=2)
        self.btn_p = ttk.Button(b_fr, text="Play", command=self.play, state="disabled")
        self.btn_p.grid(row=0, column=1, padx=2)
        self.btn_s = ttk.Button(b_fr, text="Stop", command=self.stop, state="disabled")
        self.btn_s.grid(row=0, column=2, padx=2)
        
        self.cb_v = ttk.Combobox(b_fr, textvariable=self.sv_mod, values=["Espectrómetro de Barras", "Matriz de Canales (LED)", "Multi-Osciloscopio (Scroll)", "Monitor Vectorial (Lissajous)"], state="readonly", width=22)
        self.cb_v.grid(row=0, column=3, padx=10)
        self.cb_v.bind("<<ComboboxSelected>>", lambda e: self._up_scroll())
        ttk.Checkbutton(b_fr, text="Vis", variable=self.sv_vis, command=self._tg_vis).grid(row=0, column=4)

        self.f_vis = ttk.LabelFrame(c, text=" Analizador Espectral ")
        self.f_vis.grid(row=2, column=0, sticky="nsew")
        c.rowconfigure(2, weight=1)
        self.f_vis.columnconfigure(0, weight=1)
        self.f_vis.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.f_vis, bg="#030609", highlightthickness=0)
        self.sb = ttk.Scrollbar(self.f_vis, orient="vertical", command=self.canvas.yview)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.sb.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.sb.grid_remove()

    def _up_scroll(self):
        if self.sv_mod.get() == "Multi-Osciloscopio (Scroll)": self.sb.grid()
        else: self.canvas.configure(scrollregion=(0,0,0,0)); self.sb.grid_remove()

    def sel_mp3(self):
        r = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a")])
        if r:
            self.r_mp3 = r; self.s_mp3.set(os.path.basename(r))
            self.r_out = os.path.splitext(r)[0] + ".chtn"; self.s_chtn.set(os.path.basename(self.r_out))
            self.btn_conv.config(state="normal")

    def sel_out(self):
        r = filedialog.asksaveasfilename(defaultextension=".chtn", filetypes=[("CHTN", "*.chtn")])
        if r: self.r_out = r; self.s_chtn.set(os.path.basename(r)); self.btn_conv.config(state="normal")

    def _seek(self, val):
        if self.seeking and self.tot_f > 0: self._clock(int(float(val)/100.0 * self.tot_f))

    def _release(self, e):
        if self.tot_f > 0:
            f = int(float(self.t_slider.get())/100.0 * self.tot_f)
            with self.v_lock: self.f_act = f; self.s_cnt = int((f/self.fps)*S_RATE)
            self.seeking = False; self._clock(f)

    def _clock(self, f):
        if self.tot_f <= 0: return
        t, a = int(self.tot_f/self.fps), int(f/self.fps)
        self.lbl_t_t.set(f"{a//60:02d}:{a%60:02d}")
        self.lbl_t_r.set(f"-{(t-a)//60:02d}:{(t-a)%60:02d}")
        self.lbl_t_tot.set(f"Duración: {t//60:02d}:{t%60:02d}")

    def convertir(self):
        self.btn_conv.config(state="disabled")
        self.progress.set(5)
        threading.Thread(target=self._proc_conv, daemon=True).start()

    def _proc_conv(self):
        try:
            ch_cfg = self.sv_ch.get()
            ct, cn = r_canales(ch_cfg)
            y, sr = librosa.load(self.r_mp3, sr=S_RATE, mono=False)
            if y.ndim == 1: y = np.vstack([y, y])
            
            self.progress.set(20)
            sl = np.abs(librosa.stft(y[0], n_fft=2048, hop_length=512)).astype(np.float32)
            sr_m = np.abs(librosa.stft(y[1], n_fft=2048, hop_length=512)).astype(np.float32)
            
            freqs = librosa.fft_frequencies(sr=S_RATE, n_fft=2048).astype(np.float32)
            m = (freqs >= M_FRQ) & (freqs <= MX_FRQ)
            f_u, n_f = freqs[m], sl.shape[1]
            sl, sr_m = sl[m], sr_m[m]
            
            st_tot = sl + sr_m
            mx = float(np.percentile(st_tot, 99.6)) or 1.0
            
            t_fr = np.zeros((n_f, ct), dtype=np.uint16)
            t_vo = np.zeros((n_f, ct), dtype=np.float32)
            t_pa = np.zeros((n_f, ct), dtype=np.float32)
            n_vo = np.zeros((n_f, cn), dtype=np.float32) if cn > 0 else np.array([[]])
            
            f_oido = p_psico(f_u)
            for f_idx in range(n_f):
                mag_p = st_tot[:, f_idx] * f_oido
                if len(mag_p) >= ct:
                    pks = np.argpartition(mag_p, -ct)[-ct:]
                    pks = pks[np.argsort(f_u[pks])]
                    for o_idx, r_idx in enumerate(pks):
                        al, ar = sl[r_idx, f_idx], sr_m[r_idx, f_idx]
                        amp = min(1.0, np.power((al+ar)/mx, 0.78))
                        if amp > 0.012:
                            t_fr[f_idx, o_idx] = int(f_u[r_idx])
                            t_vo[f_idx, o_idx] = amp
                            t_pa[f_idx, o_idx] = np.clip(ar/(al+ar), 0.0, 1.0) if (al+ar)>0.0005 else 0.5
                if cn > 0:
                    b_ruido = np.geomspace(M_FRQ, MX_FRQ, cn + 1)
                    for rc in range(cn):
                        sub_p = np.flatnonzero((f_u >= b_ruido[rc]) & (f_u < b_ruido[rc+1]))
                        if len(sub_p) > 0:
                            an = min(0.95, np.power(np.mean(st_tot[sub_p, f_idx])/mx, 0.72)*1.6)
                            if an > 0.02: n_vo[f_idx, rc] = an
                if f_idx % 2000 == 0: self.progress.set(25 + int(f_idx*55/n_f))

            self.progress.set(85)
            vp_t, vs_t = pack_nb(q_vol(t_vo))
            vp_p, vs_p = pack_nb(np.clip(np.rint(t_pa*15), 0, 15).astype(np.uint8))
            vp_n, vs_n = pack_nb(q_vol(n_vo) if cn > 0 else np.array([], dtype=np.uint8))
            
            tmp = tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(self.r_out) or ".", suffix=".tmp")
            with tmp as f:
                np.savez_compressed(f, format_version=np.array([F_VER], dtype=np.uint8), sample_rate=np.array([S_RATE], dtype=np.uint32),
                                    block_size=np.array([B_SIZE], dtype=np.uint16), cfg_total_oscillators=np.array([ch_cfg], dtype=np.uint16),
                                    cfg_tonal_channels=np.array([ct], dtype=np.uint16), cfg_noise_channels=np.array([cn], dtype=np.uint16),
                                    total_frames=np.array([n_f], dtype=np.uint32), t_freqs_start=t_fr[0].copy(),
                                    t_freqs_deltas=np.diff(t_fr, axis=0).astype(np.int16), t_vols_packed=vp_t, t_vols_shape=vs_t,
                                    t_pans_packed=vp_p, t_pans_shape=vs_p, n_vols_packed=vp_n, n_vols_shape=vs_n)
            os.replace(tmp.name, self.r_out)
            self.progress.set(100)
            if self.sv_auto.get(): self.root.after(0, lambda: self._load_path(self.r_out))
        except Exception as e: messagebox.showerror("Error", str(e))
        finally: self.root.after(0, lambda: self.btn_conv.config(state="normal"))

    def load_chtn(self):
        r = filedialog.askopenfilename(filetypes=[("CHTN", "*.chtn")])
        if r: self._load_path(r)

    def _load_path(self, path):
        if self.reproduciendo: self.stop()
        try:
            with np.load(path) as d:
                self.c_tot = int(d.get("cfg_total_oscillators", [256])[0])
                self.c_ton = int(d.get("cfg_tonal_channels", [218])[0])
                self.c_rui = int(d.get("cfg_noise_channels", [38])[0])
                self.tot_f = int(d.get("total_frames", [0])[0])
                self.b_tf = np.vstack([d["t_freqs_start"], d["t_freqs_start"] + np.cumsum(d["t_freqs_deltas"], axis=0)]).astype(np.float32)
                self.b_tv = dq_vol(unpack_nb(d["t_vols_packed"], d["t_vols_shape"]))
                self.b_tp = unpack_nb(d["t_pans_packed"], d["t_pans_shape"]).astype(np.float32)/15.0
                self.b_nv = dq_vol(unpack_nb(d["n_vols_packed"], d["n_vols_shape"])) if self.c_rui>0 and "n_vols_packed" in d else np.zeros((self.tot_f, self.c_rui), dtype=np.float32)
        except Exception as e: messagebox.showerror("Error", str(e)); return
        self.phases = np.zeros(self.c_ton, dtype=np.float32)
        self.s_cnt, self.f_act = 0, 0
        self.l_chtn.set(f"{os.path.basename(path)} ({self.c_tot} Oscs)")
        self.btn_p.config(state="normal")
        self._res_v_buf()
        self._clock(0)
        self._up_scroll()

    def audio_cb(self, out, frames, time_info, status):
        if not self.reproduciendo or self.seeking: out.fill(0); return
        idx = int((self.s_cnt/44100.0)*self.fps)
        if idx >= self.tot_f-2: self.reproduciendo = False; out.fill(0); return
        self.f_act = idx

        f0, f1 = self.b_tf[idx], self.b_tf[idx+1]
        v0, v1 = self.b_tv[idx], self.b_tv[idx+1]
        p0, p1 = self.b_tp[idx], self.b_tp[idx+1]
        
        mix = np.zeros((frames, 2), dtype=np.float32)
        gain = float(self.sv_gain.get())
        f_sc = 2.8 / (np.sqrt(self.c_ton) + 6.0)
        
        tm = (np.arange(frames, dtype=np.float32)/float(frames))[:, None]
        fr_i = (1.0-tm)*f0 + tm*f1
        vo_i = (1.0-tm)*v0 + tm*v1
        pa_i = (1.0-tm)*p0 + tm*p1
        
        f_b = self.phases + np.cumsum((2.0*np.pi*fr_i)/44100.0, axis=0)
        self.phases = (f_b[-1]%(2.0*np.pi)).astype(np.float32)
        f_mod = f_b % (2.0*np.pi)
        
        o_fin = np.zeros_like(f_mod)
        mb, mm, ma = (f0<280), (f0>=280)&(f0<1800), (f0>=1800)
        o_fin[:, mb] = ((f_mod[:, mb]/np.pi)-1.0)*0.25
        o_fin[:, mm] = ((2.0/np.pi)*np.arcsin(np.sin(f_mod[:, mm])))*0.75
        o_fin[:, ma] = np.sin(f_mod[:, ma])*0.95 + np.sin(f_mod[:, ma]*0.5)*(vo_i[:, ma]*0.08)
        
        w_ch = o_fin * vo_i * (f_sc*gain)
        pa_ang = pa_i * (np.pi/2.0)
        mix[:, 0] = np.sum(w_ch*np.cos(pa_ang), axis=1)
        mix[:, 1] = np.sum(w_ch*np.sin(pa_ang), axis=1)
        
        if self.c_rui > 0 and self.b_nv.shape[1] > 0:
            vn0, vn1 = self.b_nv[idx], self.b_nv[idx+1]
            vr_i = (1.0-tm)*vn0 + tm*vn1
            ptrs = (self.n_off[:self.c_rui] + self.s_cnt) % (self.n_lk - frames)
            r_mat = self.n_buf[ptrs + np.arange(frames)[:, None]] * (vr_i * (0.12/(np.sqrt(self.c_rui)+4.0)) * gain)
            p_rl = 0.40 + 0.20*(np.arange(self.c_rui)%3)/2.0
            mix[:, 0] += np.sum(r_mat*p_rl, axis=1)
            mix[:, 1] += np.sum(r_mat*(1.0-p_rl), axis=1)
            
        mix = np.tanh(mix*0.98)
        out[:] = mix
        if self.sv_vis.get() and self.s_cnt%(B_SIZE*4)==0:
            if self.v_lock.acquire(blocking=False):
                try: self.v_inst_v, self.v_inst_f, self.v_ch_buf = list(vo_i[0]), list(fr_i[0]), w_ch[::16, :64].copy()
                finally: self.v_lock.release()
        self.s_cnt += frames

    def play(self):
        if self.reproduciendo or self.b_tf is None: return
        self.reproduciendo, self.phases = True, np.zeros(self.c_ton, dtype=np.float32)
        self._res_v_buf()
        self.stream = sd.OutputStream(samplerate=S_RATE, channels=2, callback=self.audio_cb, blocksize=B_SIZE)
        self.stream.start()
        self.btn_p.config(state="disabled"); self.btn_s.config(state="normal")

    def stop(self):
        self.reproduciendo = False
        if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
        self.btn_p.config(state="normal" if self.b_tf is not None else "disabled"); self.btn_s.config(state="disabled")

    def _res_v_buf(self):
        self.v_inst_v, self.v_inst_f = [0.0]*MX_OSC, [0.0]*MX_OSC
        self.v_ch_buf = np.zeros((B_SIZE//16, 64), dtype=np.float32)

    def _tg_vis(self):
        if self.sv_vis.get(): self.f_vis.grid(); self._up_scroll()
        else: self.f_vis.grid_remove()

    def _loop_vis(self):
        self._draw_vis()
        self.root.after(35, self._loop_vis)

    def _draw_vis(self):
        if not self.sv_vis.get(): return
        w, h, md = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()), self.sv_mod.get()
        if self.reproduciendo and self.tot_f > 0 and not self.seeking:
            self.t_slider.set((self.f_act/self.tot_f)*100.0); self._clock(self.f_act)
        if self.v_lock.acquire(blocking=False):
            try: vl, fl, mat = list(self.v_inst_v), list(self.v_inst_f), self.v_ch_buf.copy()
            finally: self.v_lock.release()
        else: return
        self.canvas.delete("all")
        if md == "Matriz de Canales (LED)":
            col, fls = 16, max(1, int(np.ceil(self.c_tot/16)))
            wx, wy = (w-60)/col, (h-50)/fls
            for i in range(min(len(vl), self.c_tot)):
                f, c = i//col, i%col
                x, y = 30+c*wx+3, 25+f*wy+3
                colr = ("#00ffff" if i<self.c_ton else "#ff851b") if (vl[i] if i<len(vl) else 0)>0.02 else "#0b121e"
                self.canvas.create_rectangle(x, y, x+wx-6, y+wy-6, fill=colr, outline="#111a28")
        elif md == "Espectrómetro de Barras":
            n_b, v_b = 64, np.zeros(64, dtype=np.float32)
            for i, f in enumerate(fl[:self.c_ton]):
                v = vl[i] if i<len(vl) else 0.0
                if f>M_FRQ and v>0:
                    ib = int(np.clip((np.log10(f)-np.log10(M_FRQ))/(np.log10(MX_FRQ)-np.log10(M_FRQ))*n_b, 0, n_b-1))
                    v_b[ib] = max(v_b[ib], v)
            ax = (w-40)/n_b
            for b in range(n_b):
                hb = min(h-50, v_b[b]*(h-70)*2.2)
                self.canvas.create_rectangle(20+b*ax+1, h-30-hb, 20+b*ax+ax, h-30, fill="#39ff14", outline="")
        elif md == "Multi-Osciloscopio (Scroll)":
            c_v = min(32, self.c_tot)
            self.canvas.configure(scrollregion=(0,0,w,max(h, c_v*55)))
            for c_idx in range(c_v):
                yc = c_idx*55 + 27.5
                hz = fl[c_idx] if c_idx<len(fl) else 0
                lbl = f"OSC_{c_idx+1:03d} ({int(hz)} Hz)" if c_idx<self.c_ton else f"PERC_{c_idx-self.c_ton+1}"
                self.canvas.create_rectangle(2, c_idx*55, w-4, (c_idx+1)*55, fill="#05090f", outline="#101824")
                self.canvas.create_text(10, yc, text=lbl, anchor="w", fill="#9eb2c0", font=("Consolas", 8))
                if mat is not None and c_idx<mat.shape[1]:
                    p, pts_l = mat[:, c_idx], []
                    for i in range(len(p)):
                        pts_l.extend((110+i*(w-135)/(len(p)-1), yc - float(np.clip(p[i]*3.0, -1.0, 1.0))*23.1))
                    if len(pts_l)>=4: self.canvas.create_line(*pts_l, fill="#00f0ff" if c_idx<self.c_ton else "#ff851b", width=1)
        elif md == "Monitor Vectorial (Lissajous)":
            cx, cy, r = w/2, h/2, min(w, h)*0.4
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#111c28")
            if len(vl)>4:
                pts = []
                for i in range(0, len(vl)-1, max(1, len(vl)//64)*2):
                    if vl[i]>0.01 or vl[i+1]>0.01:
                        pts.append((cx + vl[i]*np.sin(i*0.1+self.s_cnt*0.002)*r*2.2, cy - vl[i+1]*np.cos(i*0.2+self.s_cnt*0.002)*r*2.2))
                for idx, pt in enumerate(pts[:-1]): self.canvas.create_line(pt[0], pt[1], pts[idx+1][0], pts[idx+1][1], fill="#00ffff")

if __name__ == "__main__":
    r = tk.Tk(); app = ChtnStudioApp(r); r.mainloop()