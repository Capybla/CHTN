import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import numpy as np
import sounddevice as sd

# --- MOTOR ULTRA-COMPRESIÓN CHTN v24.2 ESTÉREO FIJO ---
SAMPLE_RATE = 44100      
BLOCK_SIZE = 1024        
MIN_FREQ = 20            
MAX_FREQ = 20000         
FORMAT_VERSION = 242     # Versión estéreo indexado de bajo peso
MAX_ACTIVE_OSCILLATORS = 512

def calcular_reparto_canales_dinamico(total_canales):
    canales_ruido = int(total_canales * 0.12)
    canales_tonales = total_canales - canales_ruido
    return canales_tonales, canales_ruido

def cuantizar_volumen_4bit(volumen):
    return np.clip(np.rint(np.power(volumen, 0.75) * 15), 0, 15).astype(np.uint8)

def decuantizar_volumen_4bit(volumen_4bit):
    return np.power(volumen_4bit.astype(np.float32) / 15.0, 1.333)

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
    peso = 1.0 + (4.5 * np.exp(-((np.log10(f) - 3.15)**2) / 0.40))
    return peso.astype(np.float32)

class ChtnStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHTN Studio - 250KB Stereo Nano v24.2")
        self.root.geometry("1020x960") 

        self.stream = None
        self.reproduciendo = False
        self.user_is_seeking = False
        
        self.bin_tonal_freqs = None
        self.bin_tonal_vols = None
        self.bin_noise_vols = None
        
        self.total_frames_cancion = 0
        self.fps_analisis = 44100.0 / 512.0  
        self.frame_actual = 0
        
        self.total_canales_actual = 256
        self.canales_tonales_actual = 226
        self.canales_ruido_actual = 30
        
        self.phases = np.zeros(MAX_ACTIVE_OSCILLATORS, dtype=np.float32)
        self.sample_counter = 0
        
        self.NOISE_LOOKUP_SIZE = 44100 * 4  
        self.global_noise_buffer = np.random.normal(0.0, 0.22, size=self.NOISE_LOOKUP_SIZE).astype(np.float32)
        self.noise_offsets = np.random.randint(0, self.NOISE_LOOKUP_SIZE - BLOCK_SIZE, size=MAX_ACTIVE_OSCILLATORS)
        
        self.visual_lock = threading.Lock()
        self.visual_channel_buffers = np.zeros((BLOCK_SIZE // 16, 64), dtype=np.float32)
        self.visual_instant_vols = []
        self.visual_instant_freqs = []

        self.num_canales_slider_var = tk.IntVar(value=256) 
        self.master_tonal_gain_var = tk.DoubleVar(value=1.6)  
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

        ttk.Label(contenedor, text="CHTN Studio - Nano Stereo v24.2", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(contenedor, text="Compresión estricta ~250KB con enrutamiento de panorama indexado.").grid(row=1, column=0, sticky="w", pady=(1, 10))

        self._crear_panel_conversion(contenedor)
        self._crear_panel_reproductor(contenedor)
        self._crear_panel_visualizador(contenedor)

        status_frame = ttk.Frame(contenedor)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        status_frame.columnconfigure(0, weight=1)
        self.lbl_estado = ttk.Label(status_frame, text="Listo.", relief="sunken", anchor="w")
        self.lbl_estado.grid(row=0, column=0, sticky="ew")

    def _crear_panel_conversion(self, padre):
        frame = ttk.LabelFrame(padre, text=" Codificación Nano-Espectral Estéreo ")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="1. Buscar Audio", command=self.seleccionar_mp3).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        ttk.Label(frame, textvariable=self.selected_mp3_path).grid(row=0, column=1, padx=(0, 10), pady=(8, 4), sticky="ew")

        ttk.Button(frame, text="2. Destino .chtn", command=self.seleccionar_salida).grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        ttk.Label(frame, textvariable=self.output_chtn_path).grid(row=1, column=1, padx=(0, 10), pady=4, sticky="ew")

        canales_frame = ttk.Frame(frame)
        canales_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        canales_frame.columnconfigure(1, weight=1)
        
        ttk.Label(canales_frame, text="Osciladores Totales (Recomendado 256):", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=(0, 8))
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

        self.btn_convertir = ttk.Button(frame, text="Compilar Contenedor Ultra-Ligero (~250KB)", command=self.convertir_mp3, state="disabled")
        self.btn_convertir.grid(row=5, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="ew")

    def _actualizar_info_canales_encoder(self, *args):
        try: ch = self.num_canales_slider_var.get()
        except: ch = 256
        t_ch, n_ch = calcular_reparto_canales_dinamico(ch)
        self.lbl_resolucion.config(text=f"Enrutamiento Estéreo Indexado Fijo: {t_ch} armónicos + {n_ch} percusivos.")

    def _crear_panel_reproductor(self, padre):
        frame = ttk.LabelFrame(padre, text=" Sintetizador Aditivo Estéreo Cristalino ")
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
        ttk.Label(master_vol_frame, text="Ganancia de Presencia:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=(0, 8))
        
        self.slider_master_vol = ttk.Scale(
            master_vol_frame, from_=0.0, to=4.0, orient="horizontal", variable=self.master_tonal_gain_var
        )
        self.slider_master_vol.grid(row=0, column=1, sticky="ew")

        botones = ttk.Frame(frame)
        botones.grid(row=3, column=0, padx=10, pady=(2, 8), sticky="ew")
        botones.columnconfigure(3, weight=1)

        self.btn_cargar = ttk.Button(botones, text="Abrir .chtn Nano", command=self.cargar_chtn)
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
        if self.visualizer_mode_var.get() != "Multi-Osciloscopio (Scroll)":
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            self.v_scrollbar.grid_remove()
        else:
            self.v_scrollbar.grid()

    def seleccionar_mp3(self):
        ruta_mp3 = filedialog.askopenfilename(filetypes=[("Audios HQ", "*.mp3 *.wav *.flac *.ogg *.m4a")])
        if not ruta_mp3: return
        self.ruta_mp3 = ruta_mp3
        self.selected_mp3_path.set(os.path.basename(ruta_mp3))
        self.ruta_salida = os.path.splitext(ruta_mp3)[0] + ".chtn"
        self.output_chtn_path.set(os.path.basename(self.ruta_salida))
        self._actualizar_acciones()

    def seleccionar_salida(self):
        ruta_salida = filedialog.asksaveasfilename(defaultextension=".chtn", filetypes=[("Contenedor CHTN Nano", "*.chtn")])
        if not ruta_salida: return
        self.ruta_salida = ruta_salida
        self.output_chtn_path.set(os.path.basename(ruta_salida))
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
            self._calcular_reloj_tiempos(int((float(val) / 100.0) * self.total_frames_cancion))

    def _on_timeline_release(self, event):
        if self.total_frames_cancion > 0:
            nuevo_frame = int((float(self.timeline_slider.get()) / 100.0) * self.total_frames_cancion)
            with self.visual_lock:
                self.frame_actual = nuevo_frame
                self.sample_counter = int((nuevo_frame / self.fps_analisis) * SAMPLE_RATE)
            self.user_is_seeking = False

    def _calcular_reloj_tiempos(self, f_pos):
        if self.total_frames_cancion <= 0: return
        seg_totales = int(self.total_frames_cancion / self.fps_analisis)
        seg_actuales = int(f_pos / self.fps_analisis)
        m_act, s_act = divmod(seg_actuales, 60)
        m_tot, s_tot = divmod(seg_totales, 60)
        self.tiempo_transcurrido_lbl.set(f"{m_act:02d}:{s_act:02d}")
        self.tiempo_restante_lbl.set(f"-{max(0, seg_totales - seg_actuales)//60:02d}:{max(0, seg_totales - seg_actuales)%60:02d}")
        self.tiempo_total_lbl.set(f"Duración: {m_tot:02d}:{s_tot:02d}")

    def convertir_mp3(self):
        ruta_mp3, ruta_salida = getattr(self, "ruta_mp3", None), getattr(self, "ruta_salida", None)
        if not ruta_mp3 or not ruta_salida: return

        total_canales_config = self.num_canales_slider_var.get()
        canales_tonales, canales_ruido = calcular_reparto_canales_dinamico(total_canales_config)

        self.btn_convertir.config(state="disabled")
        self._set_progreso(5)
        self._set_estado("Abriendo flujo estéreo original...")

        def hilo_analisis():
            archivo_temporal = None
            try:
                y, sr = librosa.load(ruta_mp3, sr=SAMPLE_RATE, mono=False)
                if y.ndim == 1: y = np.vstack([y, y])
                    
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
                max_stft = float(np.percentile(stft_total, 99.4)) or 1.0

                tonal_freqs_mat = np.zeros((num_frames, canales_tonales), dtype=np.uint16)
                tonal_vols_float = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                noise_vols_float = np.zeros((num_frames, canales_ruido), dtype=np.float32)

                factores_oido = peso_psicoacustico(frecuencias_util)
                
                self._set_estado("Buscando polos con balance indexado fijo...")
                
                # Para evitar duplicar datos, dividimos la búsqueda de canales de forma estricta:
                # Osciladores PARES escuchan preferentemente el canal IZQUIERDO.
                # Osciladores IMPARES escuchan preferentemente el canal DERECHO.
                for frame_idx in range(num_frames):
                    mag_l = stft_util_l[:, frame_idx] * factores_oido
                    mag_r = stft_util_r[:, frame_idx] * factores_oido
                    
                    # Extraemos picos combinados ordenados rígidamente por frecuencia (Anti-Trémolo)
                    mag_comb = stft_total[:, frame_idx] * factores_oido
                    if len(mag_comb) >= canales_tonales:
                        idx_picos = np.argpartition(mag_comb, -canales_tonales)[-canales_tonales:]
                        idx_picos = idx_picos[np.argsort(frecuencias_util[idx_picos])]
                        
                        for idx_osc, r_idx in enumerate(idx_picos):
                            amp_l = mag_l[r_idx] / (factores_oido[r_idx] * max_stft)
                            amp_r = mag_r[r_idx] / (factores_oido[r_idx] * max_stft)
                            
                            # Enrutamos inteligentemente al canal par/impar para dar sensación estéreo real 
                            # sin guardar una matriz de paneos extra.
                            if idx_osc % 2 == 0:  # Lado Izquierdo
                                amp_final = min(1.0, amp_l * 1.3)
                            else:                 # Lado Derecho
                                amp_final = min(1.0, amp_r * 1.3)
                                
                            if amp_final > 0.025:
                                tonal_freqs_mat[frame_idx, idx_osc] = int(frecuencias_util[r_idx])
                                tonal_vols_float[frame_idx, idx_osc] = amp_final

                    if canales_ruido > 0:
                        bordes_ruido = np.geomspace(MIN_FREQ, MAX_FREQ, canales_ruido + 1)
                        for r_c in range(canales_ruido):
                            idx_sb = np.flatnonzero((frecuencias_util >= bordes_ruido[r_c]) & (frecuencias_util < bordes_ruido[r_c+1]))
                            if len(idx_sb) > 0:
                                noise_vols_float[frame_idx, r_c] = min(0.95, np.power(np.mean(stft_total[idx_sb, frame_idx]) / max_stft, 0.75) * 1.5)

                    if frame_idx % 2000 == 0:
                        self._set_progreso(25 + int(frame_idx * 55 / num_frames))

                self._set_progreso(85)
                self._set_estado("Comprimiendo espacio físico del contenedor (Objetivo: 250KB)...")

                freqs_primer_frame = tonal_freqs_mat[0].copy()
                freqs_deltas = np.diff(tonal_freqs_mat, axis=0).astype(np.int16)

                # Volvemos única y estrictamente al empaquetamiento de Nibbles de 4 bits para Volúmenes y Ruido.
                t_vols_pack, t_vols_shape = empaquetar_nibbles(cuantizar_volumen_4bit(tonal_vols_float))
                n_vols_pack, n_vols_shape = empaquetar_nibbles(cuantizar_volumen_4bit(noise_vols_float))

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
                        n_vols_packed=n_vols_pack,
                        n_vols_shape=n_vols_shape
                    )

                os.replace(archivo_temporal.name, ruta_salida)
                self._set_progreso(100)
                
                t_ch = os.path.getsize(ruta_salida) / 1024
                self._set_estado(f"¡Completado! Peso Real: {t_ch:.1f} KB.")
                
                if self.auto_load_after_convert.get():
                    self.root.after(0, lambda: self.cargar_chtn_desde_ruta(ruta_salida))
            except Exception as e:
                self._set_estado(f"Error: {str(e)}")
            finally:
                self.root.after(0, self._actualizar_acciones)

        threading.Thread(target=hilo_analisis, daemon=True).start()

    def cargar_chtn(self):
        ruta = filedialog.askopenfilename(filetypes=[("Contenedor CHTN Nano", "*.chtn")])
        if ruta: self.cargar_chtn_desde_ruta(ruta)

    def cargar_chtn_desde_ruta(self, ruta_chtn):
        if self.reproduciendo: self.detener_audio()
        try:
            with np.load(ruta_chtn) as data:
                self.total_canales_actual = int(data["cfg_total_oscillators"][0])
                self.canales_tonales_actual = int(data["cfg_tonal_channels"][0])
                self.canales_ruido_actual = int(data["cfg_noise_channels"][0])
                self.total_frames_cancion = int(data["total_frames"][0])
                
                freqs_start = data["t_freqs_start"]
                freqs_deltas = data["t_freqs_deltas"]
                self.bin_tonal_freqs = np.vstack([freqs_start, freqs_start + np.cumsum(freqs_deltas, axis=0)]).astype(np.float32)
                
                self.bin_tonal_vols = decuantizar_volumen_4bit(desempaquetar_nibbles(data["t_vols_packed"], data["t_vols_shape"]))
                self.bin_noise_vols = decuantizar_volumen_4bit(desempaquetar_nibbles(data["n_vols_packed"], data["n_vols_shape"]))

        except Exception as e:
            messagebox.showerror("Error", f"Error de lectura: {e}")
            return

        self.phases = np.zeros(self.canales_tonales_actual, dtype=np.float32)
        self.sample_counter = 0
        self.frame_actual = 0
        self.loaded_chtn_path.set(f"{os.path.basename(ruta_chtn)} ({os.path.getsize(ruta_chtn)//1024} KB)")
        self.btn_play.config(state="normal")
        self._reset_visual_buffers()
        self._calcular_reloj_tiempos(0)

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

        mix_buffer_stereo = np.zeros((frames, 2), dtype=np.float32)
        user_gain = float(self.master_tonal_gain_var.get())
        factor_escalado_seguro = 2.4 / (np.sqrt(self.canales_tonales_actual) + 5.0)

        t_matrix = (np.arange(frames, dtype=np.float32) / float(frames))[:, None]

        freqs_interp = (1.0 - t_matrix) * f_0 + t_matrix * f_1
        vols_interp = (1.0 - t_matrix) * v_0 + t_matrix * v_1

        fase_delta_acum = (2.0 * np.pi * freqs_interp) / 44100.0
        fases_muestras_bloque = self.phases + np.cumsum(fase_delta_acum, axis=0)
        self.phases = (fases_muestras_bloque[-1] % (2.0 * np.pi)).astype(np.float32)

        fase_mod = fases_muestras_bloque % (2.0 * np.pi)

        onda_final = np.zeros_like(fase_mod)
        mascara_sierra = (f_0 < 350)
        mascara_triangulo = (f_0 >= 350) & (f_0 < 2500)
        mascara_seno = (f_0 >= 2500)

        onda_final[:, mascara_sierra] = ((fase_mod[:, mascara_sierra] / np.pi) - 1.0) * 0.22
        onda_final[:, mascara_triangulo] = ((2.0 / np.pi) * np.arcsin(np.sin(fase_mod[:, mascara_triangulo]))) * 0.65
        onda_final[:, mascara_seno] = np.sin(fase_mod[:, mascara_seno]) * 1.0

        peso_canales = onda_final * vols_interp * (factor_escalado_seguro * user_gain)

        # --- REPARTO ESTÉREO INDEXADO FIJO (MÁXIMA COMPRESIÓN) ---
        # Los índices pares se atenúan un poco a la derecha, los impares a la izquierda.
        indices_globales = np.arange(self.canales_tonales_actual)
        pan_izq_vector = np.where(indices_globales % 2 == 0, 0.85, 0.15).astype(np.float32)
        pan_der_vector = 1.0 - pan_izq_vector

        mix_buffer_stereo[:, 0] = np.dot(peso_canales, pan_izq_vector)
        mix_buffer_stereo[:, 1] = np.dot(peso_canales, pan_der_vector)

        if self.canales_ruido_actual > 0:
            vn_0, vn_1 = self.bin_noise_vols[idx_frame], self.bin_noise_vols[idx_frame + 1]
            v_ruido_interp = (1.0 - t_matrix) * vn_0 + t_matrix * vn_1
            start_ptrs = (self.noise_offsets[:self.canales_ruido_actual] + self.sample_counter) % (self.NOISE_LOOKUP_SIZE - frames)
            ruido_idx = start_ptrs + np.arange(frames)[:, None]
            buf_ruido_matriz = self.global_noise_buffer[ruido_idx] * (v_ruido_interp * (0.08 / (np.sqrt(self.canales_ruido_actual) + 3.0)) * user_gain)
            
            # Paneado entrelazado del ruido percusivo
            indices_r = np.arange(self.canales_ruido_actual)
            pan_r_l = np.where(indices_r % 2 == 0, 0.70, 0.30).astype(np.float32)
            mix_buffer_stereo[:, 0] += np.dot(buf_ruido_matriz, pan_r_l)
            mix_buffer_stereo[:, 1] += np.dot(buf_ruido_matriz, 1.0 - pan_r_l)

        outdata[:] = np.tanh(mix_buffer_stereo * 0.99)

        if self.visualizer_enabled.get() and self.sample_counter % (BLOCK_SIZE * 4) == 0:
            if self.visual_lock.acquire(blocking=False):
                try:
                    self.visual_instant_vols = list(vols_interp[0])
                    self.visual_instant_freqs = list(freqs_interp[0])
                    self.visual_channel_buffers = peso_canales[::16, :64].copy()
                finally: self.visual_lock.release()

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

    def detener_audio(self):
        self.reproduciendo = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.btn_play.config(state="normal" if self.bin_tonal_freqs is not None else "disabled")
        self.btn_stop.config(state="disabled")
        self.btn_cargar.config(state="normal")

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
        ancho, alto = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        modo = self.visualizer_mode_var.get()
        
        if self.reproduciendo and self.total_frames_cancion > 0 and not self.user_is_seeking:
            self.timeline_slider.set((self.frame_actual / self.total_frames_cancion) * 100.0)
            self._calcular_reloj_tiempos(self.frame_actual)
        
        if self.visual_lock.acquire(blocking=False):
            try:
                vols = list(self.visual_instant_vols)
                freqs = list(self.visual_instant_freqs)
                matriz_visual = self.visual_channel_buffers.copy() if isinstance(self.visual_channel_buffers, np.ndarray) else None
            finally: self.visual_lock.release()
        else: return

        self.canvas.delete("all")

        if modo == "Matriz de Canales (LED)":
            columnas = 16
            filas = max(1, int(np.ceil(self.total_canales_actual / columnas)))
            espacio_x, espacio_y = (ancho - 60) / columnas, (alto - 50) / filas
            for idx in range(min(len(vols), self.total_canales_actual)):
                f_idx, c_idx = idx // columnas, idx % columnas
                x0, y0 = 30 + c_idx * espacio_x + 3, 25 + f_idx * espacio_y + 3
                v = vols[idx] if idx < len(vols) else 0.0
                color = ("#00ffff" if idx % 2 == 0 else "#ff00ff") if v > 0.02 else "#0b121e"
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

        elif modo == "Multi-Osciloscopio (Scroll)":
            canales_visibles = min(32, self.total_canales_actual)
            alto_pista, margen_izq = 55, 110
            ancho_util = ancho - margen_izq - 25
            alto_total_virtual = max(alto, canales_visibles * alto_pista)
            self.canvas.configure(scrollregion=(0, 0, ancho, alto_total_virtual))
            for ch_idx in range(canales_visibles):
                y_centro = ch_idx * alto_pista + (alto_pista / 2)
                self.canvas.create_rectangle(2, ch_idx * alto_pista, ancho - 4, (ch_idx + 1) * alto_pista, fill="#05090f", outline="#101824")
                self.canvas.create_text(10, y_centro, text=f"OSC_{ch_idx+1:03d}", anchor="w", fill="#9eb2c0", font=("Consolas", 8, "bold"))
                if matriz_visual is not None and ch_idx < matriz_visual.shape[1]:
                    pts = matriz_visual[:, ch_idx]
                    puntos_linea = []
                    for i in range(len(pts)):
                        x = margen_izq + i * ancho_util / (len(pts) - 1)
                        y = y_centro - float(np.clip(pts[i] * 3.0, -1.0, 1.0)) * (alto_pista * 0.42)
                        puntos_linea.extend((x, y))
                    if len(puntos_linea) >= 4: self.canvas.create_line(*puntos_linea, fill="#00f0ff", width=1)

        elif modo == "Monitor Vectorial (Lissajous)":
            cx, cy = ancho / 2, alto / 2
            radio_pantalla = min(ancho, alto) * 0.40
            self.canvas.create_oval(cx - radio_pantalla, cy - radio_pantalla, cx + radio_pantalla, cy + radio_pantalla, outline="#111c28", width=1)
            if len(vols) > 4:
                puntos_fase = []
                for i in range(0, len(vols) - 1, 4):
                    if vols[i] > 0.01:
                        puntos_fase.append((cx + vols[i] * np.sin(i*0.1 + self.sample_counter*0.002) * radio_pantalla * 2, cy - vols[i+1] * np.cos(i*0.2 + self.sample_counter*0.002) * radio_pantalla * 2))
                for idx, pt in enumerate(puntos_fase[:-1]):
                    self.canvas.create_line(pt[0], pt[1], puntos_fase[idx+1][0], puntos_fase[idx+1][1], fill="#00ffff", width=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChtnStudioApp(root)
    root.mainloop()