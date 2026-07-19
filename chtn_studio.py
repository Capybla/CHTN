import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import numpy as np
import sounddevice as sd

# --- CONFIGURACIÓN DEL MOTOR DE AUDIO HQ CRISTALINO v26.2 ---
SAMPLE_RATE = 44100      
BLOCK_SIZE = 1024        
MIN_FREQ = 20            
MAX_FREQ = 20000         
FORMAT_VERSION = 262     
MAX_ACTIVE_OSCILLATORS = 512

def calcular_reparto_canales_dinamico(total_canales):
    canales_ruido = int(total_canales * 0.15)  
    canales_tonales = total_canales - canales_ruido
    return canales_tonales, canales_ruido

def cuantizar_volumen_4bit(volumen):
    return np.clip(np.rint(np.power(volumen, 0.70) * 15), 0, 15).astype(np.uint8)

def decuantizar_volumen_4bit(volumen_4bit):
    return np.power(volumen_4bit.astype(np.float32) / 15.0, 1.428)

def empaquetar_nibbles(array_4bit):
    shape_original = array_4bit.shape
    plano = array_4bit.flatten()
    if len(plano) % 2 != 0:
        plano = np.append(plano, np.uint8(0))
    bytes_empaquetados = (plano[0::2] << 4) | (plano[1::2] & 0x0F)
    return bytes_empaquetados, np.array(shape_original, dtype=np.uint32)

def desempaquetar_nibbles(bytes_empaquetados, shape_original):
    total_elementos = np.prod(shape_original)
    plano = np.zeros(total_elementos + (total_elementos % 2), dtype=np.uint8)
    plano[0::2] = (bytes_empaquetados >> 4) & 0x0F
    plano[1::2] = bytes_empaquetados & 0x0F
    return plano[:total_elementos].reshape(shape_original)

def peso_psicoacustico(frecuencias):
    f = np.clip(frecuencias, 10, 22000)
    peso = 1.0 + (3.8 * np.exp(-((np.log10(f) - 3.15)**2) / 0.45))
    peso += 1.5 * np.exp(-((np.log10(f) - 3.65)**2) / 0.15)
    return peso.astype(np.float32)

class ChtnStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHTN Studio - Cristal HQ Stereo Edition v26.2")
        self.root.geometry("1020x960") 
        self.root.minsize(880, 850)

        self.stream = None
        self.reproduciendo = False
        self.user_is_seeking = False
        
        self.bin_tonal_freqs = None
        self.bin_tonal_vols = None
        self.bin_tonal_pans = None  
        self.bin_noise_vols = None
        
        self.total_frames_cancion = 0
        self.fps_analisis = 44100.0 / 512.0  
        self.frame_actual = 0
        
        self.total_canales_actual = 256
        self.canales_tonales_actual = 218
        self.canales_ruido_actual = 38
        
        self.phases = np.zeros(MAX_ACTIVE_OSCILLATORS, dtype=np.float32)
        self.sample_counter = 0
        
        self.NOISE_LOOKUP_SIZE = 44100 * 4  
        self.global_noise_buffer = np.random.normal(0.0, 0.18, size=self.NOISE_LOOKUP_SIZE).astype(np.float32)
        self.noise_offsets = np.random.randint(0, self.NOISE_LOOKUP_SIZE - BLOCK_SIZE, size=MAX_ACTIVE_OSCILLATORS)
        
        self.visual_lock = threading.Lock()
        self.visual_channel_buffers = np.zeros((BLOCK_SIZE // 16, 64), dtype=np.float32)
        self.visual_instant_vols = []
        self.visual_instant_freqs = []

        self.num_canales_slider_var = tk.IntVar(value=256) 
        self.master_tonal_gain_var = tk.DoubleVar(value=1.5)  
        self.visualizer_enabled = tk.BooleanVar(value=True)
        self.visualizer_mode_var = tk.StringVar(value="Espectrómetro de Barras")
        self.auto_load_after_convert = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0)
        self.selected_mp3_path = tk.StringVar(value="Ningún archivo seleccionado")
        self.output_chtn_path = tk.StringVar(value="Ningún destino .chtn elegido")
        self.loaded_chtn_path = tk.StringVar(value="Ningún contenedor .chtn cargado")
        
        self.ruta_directorio_lbl = tk.StringVar(value="Carpeta Destino: No especificada")
        self.tiempo_transcurrido_lbl = tk.StringVar(value="00:00")
        self.tiempo_restante_lbl = tk.StringVar(value="00:00")
        self.tiempo_total_lbl = tk.StringVar(value="Duración Total: --:--")

        self._crear_ui()
        self._actualizar_info_canales_encoder()
        self.root.after(100, self._programar_visualizador)

    def _crear_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        contenedor = ttk.Frame(self.root, padding=12)
        contenedor.pack(fill="both", expand=True)
        contenedor.columnconfigure(0, weight=1)
        contenedor.rowconfigure(4, weight=1)

        ttk.Label(contenedor, text="CHTN Studio - Cristal HQ Stereo v26.2", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(contenedor, text="Fidelidad acústica mejorada por interpolación suavizada y balance tridimensional.").grid(row=1, column=0, sticky="w", pady=(1, 10))

        self._crear_panel_conversion(contenedor)
        self._crear_panel_reproductor(contenedor)
        self._crear_panel_visualizador(contenedor)

        status_frame = ttk.Frame(contenedor)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        status_frame.columnconfigure(0, weight=1)

        self.lbl_estado = ttk.Label(status_frame, text="Listo.", relief="sunken", anchor="w")
        self.lbl_estado.grid(row=0, column=0, sticky="ew")
        
        lbl_dir = ttk.Label(status_frame, textvariable=self.ruta_directorio_lbl, font=("Segoe UI", 8, "italic"), foreground="#666666")
        lbl_dir.grid(row=0, column=1, padx=(10, 0), sticky="e")

    def _crear_panel_conversion(self, padre):
        frame = ttk.LabelFrame(padre, text=" Codificación Rápida de Densidad Espacial ")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="1. Buscar Audio", command=self.seleccionar_mp3).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        ttk.Label(frame, textvariable=self.selected_mp3_path).grid(row=0, column=1, padx=(0, 10), pady=(8, 4), sticky="ew")

        ttk.Button(frame, text="2. Destino .chtn", command=self.seleccionar_salida).grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        ttk.Label(frame, textvariable=self.output_chtn_path).grid(row=1, column=1, padx=(0, 10), pady=4, sticky="ew")

        canales_frame = ttk.Frame(frame)
        canales_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        canales_frame.columnconfigure(1, weight=1)
        
        ttk.Label(canales_frame, text="Osciladores Totales (Recomendado 256-384 para HQ):", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=(0, 8))
        self.slider_canales = tk.Scale(
            canales_frame, from_=32, to=512, orient="horizontal", showvalue=True,
            variable=self.num_canales_slider_var, highlightthickness=0, bg="#d9d9d9", troughcolor="#e6e6e6"
        )
        self.slider_canales.grid(row=0, column=1, sticky="ew")
        self.num_canales_slider_var.trace_add("write", self._actualizar_info_canales_encoder)

        self.lbl_resolucion = ttk.Label(frame, text="", font=("Segoe UI", 9, "italic"), foreground="#0275d8")
        self.lbl_resolucion.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="w")

        opciones = ttk.Frame(frame)
        opciones.grid(row=4, column=0, columnspan=2, padx=10, pady=(2, 6), sticky="ew")
        opciones.columnconfigure(1, weight=1)

        ttk.Checkbutton(opciones, text="Auto-cargar al finalizar", variable=self.auto_load_after_convert).grid(row=0, column=0, sticky="w")
        ttk.Progressbar(opciones, variable=self.progress_var, maximum=100).grid(row=0, column=1, padx=(12, 0), sticky="ew")

        self.btn_convertir = ttk.Button(frame, text="Compilar Contenedor de Alta Fidelidad .chtn", command=self.convertir_mp3, state="disabled")
        self.btn_convertir.grid(row=5, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="ew")

    def _actualizar_info_canales_encoder(self, *args):
        try: ch = self.num_canales_slider_var.get()
        except: ch = 256
        t_ch, n_ch = calcular_reparto_canales_dinamico(ch)
        self.lbl_resolucion.config(text=f"Imagen Estéreo Premium: {t_ch} armónicos nítidos + {n_ch} percusivos limpios.")

    def _crear_panel_reproductor(self, padre):
        frame = ttk.LabelFrame(padre, text=" Sintetizador Aditivo de Alta Fidelidad Cristalina ")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)

        top_info = ttk.Frame(frame)
        top_info.grid(row=0, column=0, padx=10, pady=(8, 2), sticky="ew")
        top_info.columnconfigure(0, weight=1)
        ttk.Label(top_info, textvariable=self.loaded_chtn_path).grid(row=0, column=0, sticky="w")
        ttk.Label(top_info, textvariable=self.tiempo_total_lbl, font=("Segoe UI", 9, "italic")).grid(row=0, column=1, sticky="e")

        timeline_frame = ttk.Frame(frame)
        timeline_frame.grid(row=1, column=0, padx=10, pady=2, sticky="ew")
        timeline_frame.columnconfigure(1, weight=1)

        self.lbl_transcurrido = ttk.Label(timeline_frame, textvariable=self.tiempo_transcurrido_lbl, font=("Consolas", 10, "bold"), width=6, anchor="center")
        self.lbl_transcurrido.grid(row=0, column=0, padx=(0, 6))

        self.timeline_slider = tk.Scale(
            timeline_frame, from_=0, to=100, orient="horizontal", showvalue=False,
            highlightthickness=0, bg="#d9d9d9", activebackground="#00f0ff", troughcolor="#e6e6e6",
            command=self._on_timeline_seek
        )
        self.timeline_slider.grid(row=0, column=1, sticky="ew")
        self.timeline_slider.bind("<ButtonPress-1>", lambda e: setattr(self, 'user_is_seeking', True))
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_timeline_release)

        self.lbl_restante = ttk.Label(timeline_frame, textvariable=self.tiempo_restante_lbl, font=("Consolas", 10, "bold"), width=7, anchor="center")
        self.lbl_restante.grid(row=0, column=2, padx=(6, 0))

        master_vol_frame = ttk.Frame(frame)
        master_vol_frame.grid(row=2, column=0, padx=10, pady=2, sticky="ew")
        master_vol_frame.columnconfigure(1, weight=1)
        ttk.Label(master_vol_frame, text="Presencia Acústica / Ganancia:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=(0, 8))
        
        self.slider_master_vol = ttk.Scale(
            master_vol_frame, from_=0.0, to=4.0, orient="horizontal", variable=self.master_tonal_gain_var
        )
        self.slider_master_vol.grid(row=0, column=1, sticky="ew")
        
        lbl_gain_num = ttk.Label(master_vol_frame, text="1.5x", width=6)
        lbl_gain_num.grid(row=0, column=2, padx=(6, 0))
        self.master_tonal_gain_var.trace_add("write", lambda *args: lbl_gain_num.config(text=f"{self.master_tonal_gain_var.get():.1f}x"))

        botones = ttk.Frame(frame)
        botones.grid(row=3, column=0, padx=10, pady=(2, 8), sticky="ew")
        botones.columnconfigure(3, weight=1)

        self.btn_cargar = ttk.Button(botones, text="Abrir .chtn HQ", command=self.cargar_chtn)
        self.btn_cargar.grid(row=0, column=0, padx=(0, 6))

        self.btn_play = ttk.Button(botones, text="Play", command=self.iniciar_audio, state="disabled")
        self.btn_play.grid(row=0, column=1, padx=6)

        self.btn_stop = ttk.Button(botones, text="Stop", command=self.detener_audio, state="disabled")
        self.btn_stop.grid(row=0, column=2, padx=6)

        ttk.Label(botones, text="Modo Visual:").grid(row=0, column=4, padx=(12, 4), sticky="e")
        self.combo_vis = ttk.Combobox(
            botones, textvariable=self.visualizer_mode_var, 
            values=["Espectrómetro de Barras", "Matriz de Canales (LED)", "Multi-Osciloscopio (Scroll)", "Monitor Vectorial (Lissajous)"], 
            state="readonly", width=25
        )
        self.combo_vis.grid(row=0, column=5, padx=(0, 10), sticky="e")
        self.combo_vis.bind("<<ComboboxSelected>>", self._on_visualizer_change)

        ttk.Checkbutton(botones, text="Activo", variable=self.visualizer_enabled, command=self._toggle_visualizador).grid(row=0, column=6, sticky="e")

    def _crear_panel_visualizador(self, padre):
        self.frame_visual = ttk.LabelFrame(padre, text=" Analizador Real-Time de Espectro Denso ")
        self.frame_visual.grid(row=4, column=0, sticky="nsew")
        self.frame_visual.columnconfigure(0, weight=1)
        self.frame_visual.rowconfigure(0, weight=1)

        self.container_scroll = ttk.Frame(self.frame_visual)
        self.container_scroll.grid(row=0, column=0, sticky="nsew")
        self.container_scroll.columnconfigure(0, weight=1)
        self.container_scroll.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.container_scroll, bg="#030609", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.container_scroll, orient="vertical", command=self.canvas.yview)
        
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

    def _on_visualizer_change(self, event):
        modo = self.visualizer_mode_var.get()
        if modo != "Multi-Osciloscopio (Scroll)":
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            self.v_scrollbar.grid_remove()
        else:
            self.v_scrollbar.grid()

    def seleccionar_mp3(self):
        ruta_mp3 = filedialog.askopenfilename(filetypes=[("Audios HQ", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("Todos", "*.*")])
        if not ruta_mp3: return
        self.ruta_mp3 = ruta_mp3
        self.selected_mp3_path.set(os.path.basename(ruta_mp3))
        if not getattr(self, "ruta_salida", None):
            self.ruta_salida = os.path.splitext(ruta_mp3)[0] + ".chtn"
            self.output_chtn_path.set(os.path.basename(self.ruta_salida))
            self.ruta_directorio_lbl.set(f"Carpeta Destino: {os.path.dirname(self.ruta_salida)}")
        self._actualizar_acciones()

    def seleccionar_salida(self):
        ruta_salida = filedialog.asksaveasfilename(defaultextension=".chtn", filetypes=[("Contenedor CHTN HQ", "*.chtn")])
        if not ruta_salida: return
        if not ruta_salida.lower().endswith(".chtn"): ruta_salida += ".chtn"
        self.ruta_salida = ruta_salida
        self.output_chtn_path.set(os.path.basename(ruta_salida))
        self.ruta_directorio_lbl.set(f"Carpeta Destino: {os.path.dirname(ruta_salida)}")
        self._actualizar_acciones()

    def _actualizar_acciones(self):
        puede = bool(getattr(self, "ruta_mp3", None) and getattr(self, "ruta_salida", None))
        self.btn_convertir.config(state="normal" if puede else "disabled")

    def _set_estado(self, texto):
        self.root.after(0, lambda: self.lbl_estado.config(text=texto))

    def _set_progreso(self, valor):
        self.root.after(0, lambda: self.progress_var.set(valor))

    def _on_timeline_seek(self, val):
        if self.user_is_seeking and self.total_frames_cancion > 0:
            porcentaje = float(val) / 100.0
            frame_temp = int(porcentaje * self.total_frames_cancion)
            self._calcular_reloj_tiempos(frame_temp)

    def _on_timeline_release(self, event):
        if self.total_frames_cancion > 0:
            porcentaje = float(self.timeline_slider.get()) / 100.0
            nuevo_frame = int(porcentaje * self.total_frames_cancion)
            with self.visual_lock:
                self.frame_actual = nuevo_frame
                tiempo_segundos = nuevo_frame / self.fps_analisis
                self.sample_counter = int(tiempo_segundos * SAMPLE_RATE)
            self.user_is_seeking = False
            self._calcular_reloj_tiempos(nuevo_frame)

    def _calcular_reloj_tiempos(self, f_pos):
        if self.total_frames_cancion <= 0: return
        seg_totales = int(self.total_frames_cancion / self.fps_analisis)
        seg_actuales = int(f_pos / self.fps_analisis)
        seg_restantes = max(0, seg_totales - seg_actuales)

        m_act, s_act = divmod(seg_actuales, 60)
        m_rest, s_rest = divmod(seg_restantes, 60)
        m_tot, s_tot = divmod(seg_totales, 60)

        self.tiempo_transcurrido_lbl.set(f"{m_act:02d}:{s_act:02d}")
        self.tiempo_restante_lbl.set(f"-{m_rest:02d}:{s_rest:02d}")
        self.tiempo_total_lbl.set(f"Duración Total: {m_tot:02d}:{s_tot:02d}")

    def convertir_mp3(self):
        ruta_mp3 = getattr(self, "ruta_mp3", None)
        ruta_salida = getattr(self, "ruta_salida", None)
        if not ruta_mp3 or not ruta_salida: return

        total_canales_config = self.num_canales_slider_var.get()
        canales_tonales, canales_ruido = calcular_reparto_canales_dinamico(total_canales_config)

        self.btn_convertir.config(state="disabled")
        self._set_progreso(5)
        self._set_estado("Cargando flujo estéreo original...")

        def hilo_analisis():
            archivo_temporal = None
            try:
                y, sr = librosa.load(ruta_mp3, sr=SAMPLE_RATE, mono=False)
                if y.ndim == 1:
                    y = np.vstack([y, y])
                    
                self._set_progreso(20)
                self._set_estado("Calculando Fast-STFT de alta resolución espectral...")
                
                hop_length = 512  
                n_fft = 2048     
                
                stft_l = np.abs(librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
                stft_r = np.abs(librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
                
                frecuencias = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
                mascara_util = (frecuencias >= MIN_FREQ) & (frecuencias <= MAX_FREQ)
                
                frecuencias_util = frecuencias[mascara_util]
                num_frames = stft_l.shape[1]
                
                stft_util_l = stft_l[mascara_util]
                stft_util_r = stft_r[mascara_util]
                
                stft_total = stft_util_l + stft_util_r
                max_stft = float(np.percentile(stft_total, 99.6)) or 1.0

                tonal_freqs_mat = np.zeros((num_frames, canales_tonales), dtype=np.uint16)
                tonal_vols_float = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                tonal_pans_float = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                noise_vols_float = np.zeros((num_frames, canales_ruido), dtype=np.float32) if canales_ruido > 0 else np.array([[]])

                factores_oido = peso_psicoacustico(frecuencias_util)
                
                self._set_estado("Aislando armónicos puros y paneo espacial continuo...")
                
                UMBRAL_PUERTA_SILENCIO = 0.012

                for frame_idx in range(num_frames):
                    mag_comb = stft_total[:, frame_idx]
                    mag_ponderada = mag_comb * factores_oido
                    
                    if len(mag_ponderada) >= canales_tonales:
                        idx_picos = np.argpartition(mag_ponderada, -canales_tonales)[-canales_tonales:]
                        idx_picos = idx_picos[np.argsort(frecuencias_util[idx_picos])]
                        
                        for idx_osc, r_idx in enumerate(idx_picos):
                            amp_l = stft_util_l[r_idx, frame_idx]
                            amp_r = stft_util_r[r_idx, frame_idx]
                            
                            amp_total = min(1.0, np.power((amp_l + amp_r) / max_stft, 0.78))
                            
                            # CORREGIDO AQUÍ DEFINITIVAMENTE: UMBRAL CON M
                            if amp_total > UMBRAL_PUERTA_SILENCIO:
                                tonal_freqs_mat[frame_idx, idx_osc] = int(frecuencias_util[r_idx])
                                tonal_vols_float[frame_idx, idx_osc] = amp_total
                                
                                sum_amps = amp_l + amp_r
                                if sum_amps > 0.0005:
                                    tonal_pans_float[frame_idx, idx_osc] = np.clip(amp_r / sum_amps, 0.0, 1.0)
                                else:
                                    tonal_pans_float[frame_idx, idx_osc] = 0.5

                    if canales_ruido > 0:
                        bordes_ruido = np.geomspace(MIN_FREQ, MAX_FREQ, canales_ruido + 1)
                        for r_c in range(canales_ruido):
                            idx_subbanda_p = np.flatnonzero((frecuencias_util >= bordes_ruido[r_c]) & (frecuencias_util < bordes_ruido[r_c+1]))
                            if len(idx_subbanda_p) > 0:
                                amp_noise = min(0.95, np.power(np.mean(mag_comb[idx_subbanda_p]) / max_stft, 0.72) * 1.6)
                                if amp_noise > 0.02:
                                    noise_vols_float[frame_idx, r_c] = amp_noise

                    if frame_idx % 2000 == 0:
                        self._set_progreso(25 + int(frame_idx * 55 / num_frames))

                self._set_progreso(85)
                self._set_estado("Guardando datos espectrales HQ...")

                freqs_primer_frame = tonal_freqs_mat[0].copy()
                freqs_deltas = np.diff(tonal_freqs_mat, axis=0).astype(np.int16)

                t_vols_pack, t_vols_shape = empaquetar_nibbles(cuantizar_volumen_4bit(tonal_vols_float))
                t_pans_pack, t_pans_shape = empaquetar_nibbles(np.clip(np.rint(tonal_pans_float * 15), 0, 15).astype(np.uint8))
                
                if canales_ruido > 0:
                    n_vols_pack, n_vols_shape = empaquetar_nibbles(cuantizar_volumen_4bit(noise_vols_float))
                else:
                    n_vols_pack, n_vols_shape = empaquetar_nibbles(np.array([], dtype=np.uint8))

                archivo_temporal = tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(ruta_salida) or ".", suffix=".tmp")

                with archivo_temporal as f:
                    np.savez_compressed(
                        f,
                        format_version=np.array([FORMAT_VERSION], dtype=np.uint8),
                        sample_rate=np.array([SAMPLE_RATE], dtype=np.uint32),
                        block_size=np.array([BLOCK_SIZE], dtype=np.uint16),
                        cfg_total_oscillators=np.array([total_canales_config], dtype=np.uint16),
                        cfg_tonal_channels=np.array([canales_tonales], dtype=np.uint16),
                        cfg_noise_channels=np.array([canales_ruido], dtype=np.uint16),
                        total_frames=np.array([num_frames], dtype=np.uint32),
                        
                        t_freqs_start=freqs_primer_frame,
                        t_freqs_deltas=freqs_deltas,
                        t_vols_packed=t_vols_pack,
                        t_vols_shape=t_vols_shape,
                        t_pans_packed=t_pans_pack,  
                        t_pans_shape=t_pans_shape,
                        n_vols_packed=n_vols_pack,
                        n_vols_shape=n_vols_shape
                    )

                os.replace(archivo_temporal.name, ruta_salida)
                self._set_progreso(100)
                
                t_ch = os.path.getsize(ruta_salida) / 1024
                self._set_estado(f"¡Completado! Contenedor de alta fidelidad compilado en {t_ch:.1f} KB.")
                self.root.after(0, lambda: messagebox.showinfo("CHTN HQ v26.2", f"¡Conversión Premium finalizada!\nTamaño óptimo: {t_ch:.1f} KB."))

                if self.auto_load_after_convert.get():
                    self.root.after(0, lambda: self.cargar_chtn_desde_ruta(ruta_salida))
            except Exception as e:
                if archivo_temporal and os.path.exists(archivo_temporal.name):
                    try: os.unlink(archivo_temporal.name)
                    except OSError: pass
                self._set_estado(f"Error: {str(e)}")
            finally:
                self.root.after(0, self._actualizar_acciones)

        threading.Thread(target=hilo_analisis, daemon=True).start()

    def cargar_chtn(self):
        ruta = filedialog.askopenfilename(filetypes=[("Contenedor CHTN HQ", "*.chtn")])
        if ruta: self.cargar_chtn_desde_ruta(ruta)

    def cargar_chtn_desde_ruta(self, ruta_chtn):
        if self.reproduciendo: self.detener_audio()
        try:
            with np.load(ruta_chtn) as data:
                v_total = data.get("cfg_total_oscillators", [256])
                v_tonal = data.get("cfg_tonal_channels", [218])
                v_noise = data.get("cfg_noise_channels", [38])
                v_frames = data.get("total_frames", [0])

                self.total_canales_actual = int(v_total[0])
                self.canales_tonales_actual = int(v_tonal[0])
                self.canales_ruido_actual = int(v_noise[0])
                self.total_frames_cancion = int(v_frames[0])
                
                freqs_start = data["t_freqs_start"]
                freqs_deltas = data["t_freqs_deltas"]
                
                self.bin_tonal_freqs = np.vstack([freqs_start, freqs_start + np.cumsum(freqs_deltas, axis=0)]).astype(np.float32)
                
                t_vols_4bit = desempaquetar_nibbles(data["t_vols_packed"], data["t_vols_shape"])
                self.bin_tonal_vols = decuantizar_volumen_4bit(t_vols_4bit)
                
                t_pans_4bit = desempaquetar_nibbles(data["t_pans_packed"], data["t_pans_shape"])
                self.bin_tonal_pans = t_pans_4bit.astype(np.float32) / 15.0
                
                if self.canales_ruido_actual > 0 and "n_vols_packed" in data:
                    n_vols_4bit = desempaquetar_nibbles(data["n_vols_packed"], data["n_vols_shape"])
                    self.bin_noise_vols = decuantizar_volumen_4bit(n_vols_4bit)
                else:
                    self.bin_noise_vols = np.zeros((self.total_frames_cancion, self.canales_ruido_actual), dtype=np.float32)

        except Exception as e:
            messagebox.showerror("Error v26.2", f"Error al abrir contenedor premium: {e}")
            return

        self.phases = np.zeros(self.canales_tonales_actual, dtype=np.float32)
        self.sample_counter = 0
        self.frame_actual = 0
        self.loaded_chtn_path.set(f"{os.path.basename(ruta_chtn)} ({self.total_canales_actual} Oscs HQ)")
        
        self.btn_play.config(state="normal")
        self._reset_visual_buffers()
        self._calcular_reloj_tiempos(0)
        self._on_visualizer_change(None)
        self.lbl_estado.config(text=f"Mapa analógico lineal inicializado correctamente.")

    def audio_callback(self, outdata, frames, time_info, status):
        if not self.reproduciendo or self.user_is_seeking:
            outdata.fill(0)
            return

        idx_frame = int((self.sample_counter / 44100.0) * self.fps_analisis)

        if idx_frame >= self.total_frames_cancion - 2:
            self.reproduciendo = False
            outdata.fill(0)
            return

        self.frame_actual = idx_frame
        
        f_0, f_1 = self.bin_tonal_freqs[idx_frame], self.bin_tonal_freqs[idx_frame + 1]
        v_0, v_1 = self.bin_tonal_vols[idx_frame], self.bin_tonal_vols[idx_frame + 1]
        p_0, p_1 = self.bin_tonal_pans[idx_frame], self.bin_tonal_pans[idx_frame + 1]

        mix_buffer_stereo = np.zeros((frames, 2), dtype=np.float32)
        user_gain = float(self.master_tonal_gain_var.get())
        
        factor_escalado_seguro = 2.8 / (np.sqrt(self.canales_tonales_actual) + 6.0)

        t_linear = np.arange(frames, dtype=np.float32) / float(frames)
        t_matrix = t_linear[:, None]

        freqs_interp = (1.0 - t_matrix) * f_0 + t_matrix * f_1
        vols_interp = (1.0 - t_matrix) * v_0 + t_matrix * v_1
        pans_interp = (1.0 - t_matrix) * p_0 + t_matrix * p_1

        fase_delta_acum = (2.0 * np.pi * freqs_interp) / 44100.0
        fases_muestras_bloque = self.phases + np.cumsum(fase_delta_acum, axis=0)
        self.phases = (fases_muestras_bloque[-1] % (2.0 * np.pi)).astype(np.float32)

        fase_mod = fases_muestras_bloque % (2.0 * np.pi)

        # SÍNTESIS ACÚSTICA AVANZADA (Evita sonido áspero)
        onda_final = np.zeros_like(fase_mod)
        mascara_bajos = (f_0 < 280)
        mascara_medios = (f_0 >= 280) & (f_0 < 1800)
        mascara_altos = (f_0 >= 1800)

        onda_final[:, mascara_bajos] = ((fase_mod[:, mascara_bajos] / np.pi) - 1.0) * 0.25
        onda_final[:, mascara_medios] = ((2.0 / np.pi) * np.arcsin(np.sin(fase_mod[:, mascara_medios]))) * 0.75
        onda_final[:, mascara_altos] = np.sin(fase_mod[:, mascara_altos]) * 0.95
        onda_final[:, mascara_altos] += np.sin(fase_mod[:, mascara_altos] * 0.5) * (vols_interp[:, mascara_altos] * 0.08)

        peso_canales = onda_final * vols_interp * (factor_escalado_seguro * user_gain)

        pan_angulo = pans_interp * (np.pi / 2.0)
        pan_izq_vector = np.cos(pan_angulo)
        pan_der_vector = np.sin(pan_angulo)

        mix_buffer_stereo[:, 0] = np.sum(peso_canales * pan_izq_vector, axis=1)
        mix_buffer_stereo[:, 1] = np.sum(peso_canales * pan_der_vector, axis=1)

        if self.canales_ruido_actual > 0 and self.bin_noise_vols.shape[1] > 0:
            vn_0 = self.bin_noise_vols[idx_frame]
            vn_1 = self.bin_noise_vols[idx_frame + 1]
            factor_ruido_seguro = 0.12 / (np.sqrt(self.canales_ruido_actual) + 4.0)
            v_ruido_interp = (1.0 - t_matrix) * vn_0 + t_matrix * vn_1
            
            start_ptrs = (self.noise_offsets[:self.canales_ruido_actual] + self.sample_counter) % (self.NOISE_LOOKUP_SIZE - frames)
            ruido_idx = start_ptrs + np.arange(frames)[:, None]
            
            buf_ruido_matriz = self.global_noise_buffer[ruido_idx] * (v_ruido_interp * factor_ruido_seguro * user_gain)
            
            indices_ruido = np.arange(self.canales_ruido_actual)
            pan_ruido_l = 0.40 + 0.20 * (indices_ruido % 3) / 2.0
            
            mix_buffer_stereo[:, 0] += np.sum(buf_ruido_matriz * pan_ruido_l, axis=1)
            mix_buffer_stereo[:, 1] += np.sum(buf_ruido_matriz * (1.0 - pan_ruido_l), axis=1)

        mix_buffer_stereo = np.tanh(mix_buffer_stereo * 0.98)
        outdata[:] = mix_buffer_stereo

        if self.visualizer_enabled.get() and self.sample_counter % (BLOCK_SIZE * 4) == 0:
            if self.visual_lock.acquire(blocking=False):
                try:
                    self.visual_instant_vols = list(vols_interp[0])
                    self.visual_instant_freqs = list(freqs_interp[0])
                    self.visual_channel_buffers = peso_canales[::16, :64].copy()
                finally:
                    self.visual_lock.release()

        self.sample_counter += frames

    def iniciar_audio(self):
        if self.reproduciendo or self.bin_tonal_freqs is None: return
        self.reproduciendo = True
        self.phases.fill(0.0)
        self._reset_visual_buffers()
        
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=self.audio_callback, blocksize=BLOCK_SIZE)
        self.stream.start()
        
        self.btn_play.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_cargar.config(state="disabled")
        self.lbl_estado.config(text="Reproduciendo flujo estéreo continuo de alta fidelidad.")

    def detener_audio(self):
        self.reproduciendo = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.btn_play.config(state="normal" if self.bin_tonal_freqs is not None else "disabled")
        self.btn_stop.config(state="disabled")
        self.btn_cargar.config(state="normal")
        self.lbl_estado.config(text="Audio detenido.")

    def _reset_visual_buffers(self):
        self.visual_instant_vols = [0.0] * MAX_ACTIVE_OSCILLATORS
        self.visual_instant_freqs = [0.0] * MAX_ACTIVE_OSCILLATORS
        self.visual_channel_buffers = np.zeros((BLOCK_SIZE // 16, 64), dtype=np.float32)

    def _toggle_visualizador(self):
        if self.visualizer_enabled.get():  
            self.frame_visual.grid()
            self._on_visualizer_change(None)
        else:                             
            self.frame_visual.grid_remove()

    def _programar_visualizador(self):
        self._dibujar_visualizador()
        self.root.after(35, self._programar_visualizador)

    def _dibujar_visualizador(self):
        if not self.visualizer_enabled.get(): return
        ancho = max(1, self.canvas.winfo_width())
        alto = max(1, self.canvas.winfo_height())
        modo = self.visualizer_mode_var.get()
        
        if self.reproduciendo and self.total_frames_cancion > 0 and not self.user_is_seeking:
            porcentaje_actual = (self.frame_actual / self.total_frames_cancion) * 100.0
            self.timeline_slider.set(porcentaje_actual)
            self._calcular_reloj_tiempos(self.frame_actual)
        
        if self.visual_lock.acquire(blocking=False):
            try:
                vols = list(self.visual_instant_vols)
                freqs = list(self.visual_instant_freqs)
                matriz_visual = self.visual_channel_buffers.copy() if isinstance(self.visual_channel_buffers, np.ndarray) else None
            finally:
                self.visual_lock.release()
        else:
            return

        self.canvas.delete("all")

        if modo == "Matriz de Canales (LED)":
            columnas = 16
            filas = max(1, int(np.ceil(self.total_canales_actual / columnas)))
            espacio_x = (ancho - 60) / columnas
            espacio_y = (alto - 50) / filas
            for idx in range(min(len(vols), self.total_canales_actual)):
                f_idx, c_idx = idx // columnas, idx % columnas
                x0 = 30 + c_idx * espacio_x + 3
                y0 = 25 + f_idx * espacio_y + 3
                v = vols[idx] if idx < len(vols) else 0.0
                color = ("#00ffff" if idx < self.canales_tonales_actual else "#ff851b") if v > 0.02 else "#0b121e"
                self.canvas.create_rectangle(x0, y0, x0 + espacio_x - 6, y0 + espacio_y - 6, fill=color, outline="#111a28")

        elif modo == "Espectrómetro de Barras":
            num_barras = 64  
            ancho_barra = (ancho - 40) / num_barras
            valores_barras = np.zeros(num_barras, dtype=np.float32)
            for idx, f in enumerate(freqs[:self.canales_tonales_actual]):
                v = vols[idx] if idx < len(vols) else 0.0
                if f > MIN_FREQ and v > 0:
                    idx_b = int(np.clip((np.log10(f) - np.log10(MIN_FREQ)) / (np.log10(MAX_FREQ) - np.log10(MIN_FREQ)) * num_barras, 0, num_barras - 1))
                    valores_barras[idx_b] = max(valores_barras[idx_b], v)
            for b in range(num_barras):
                h_barra = min(alto - 50, valores_barras[b] * (alto - 70) * 2.2)
                x0 = 20 + b * ancho_barra + 1
                self.canvas.create_rectangle(x0, alto - 30 - h_barra, x0 + ancho_barra - 1, alto - 30, fill="#39ff14", outline="")
            self.canvas.create_line(16, alto - 30, ancho - 16, alto - 30, fill="#1c2d3d", width=2)

        elif modo == "Multi-Osciloscopio (Scroll)":
            canales_visibles = min(32, self.total_canales_actual)
            alto_pista, margen_izq = 55, 110
            ancho_util = ancho - margen_izq - 25
            alto_total_virtual = max(alto, canales_visibles * alto_pista)
            self.canvas.configure(scrollregion=(0, 0, ancho, alto_total_virtual))
            for ch_idx in range(canales_visibles):
                y_centro = ch_idx * alto_pista + (alto_pista / 2)
                es_tonal = ch_idx < self.canales_tonales_actual
                color = "#00f0ff" if es_tonal else "#ff851b"
                hz = freqs[ch_idx] if ch_idx < len(freqs) else 0
                etiqueta = f"OSC_{ch_idx+1:03d}\n({int(hz)} Hz)" if es_tonal else f"PERC_{ch_idx-self.canales_tonales_actual+1}"
                self.canvas.create_rectangle(2, ch_idx * alto_pista, ancho - 4, (ch_idx + 1) * alto_pista, fill="#05090f", outline="#101824")
                self.canvas.create_text(10, y_centro, text=etiqueta, anchor="w", fill="#9eb2c0", font=("Consolas", 8, "bold"))
                if matriz_visual is not None and ch_idx < matriz_visual.shape[1]:
                    pts = matriz_visual[:, ch_idx]
                    num_puntos = len(pts)
                    puntos_linea = []
                    for i in range(num_puntos):
                        x = margen_izq + i * ancho_util / (num_puntos - 1)
                        y = y_centro - float(np.clip(pts[i] * 3.0, -1.0, 1.0)) * (alto_pista * 0.42)
                        puntos_linea.extend((x, y))
                    if len(puntos_linea) >= 4: self.canvas.create_line(*puntos_linea, fill=color, width=1)

        elif modo == "Monitor Vectorial (Lissajous)":
            cx, cy = ancho / 2, alto / 2
            radio_pantalla = min(ancho, alto) * 0.40
            self.canvas.create_oval(cx - radio_pantalla, cy - radio_pantalla, cx + radio_pantalla, cy + radio_pantalla, outline="#111c28", width=1)
            if len(vols) > 4:
                puntos_fase = []
                paso = max(1, len(vols) // 64) 
                for i in range(0, len(vols) - 1, paso * 2):
                    if vols[i] > 0.01 or vols[i+1] > 0.01:
                        v_x = vols[i] * np.sin(i * 0.1 + self.sample_counter * 0.002)
                        v_y = vols[i+1] * np.cos(i * 0.2 + self.sample_counter * 0.002)
                        puntos_fase.append((cx + v_x * radio_pantalla * 2.2, cy - v_y * radio_pantalla * 2.2))
                for idx, pt in enumerate(puntos_fase[:-1]):
                    self.canvas.create_line(pt[0], pt[1], puntos_fase[idx+1][0], puntos_fase[idx+1][1], fill="#00ffff", width=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChtnStudioApp(root)
    root.mainloop()