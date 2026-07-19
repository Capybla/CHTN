import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import numpy as np
import sounddevice as sd

# --- CONFIGURACIÓN DEL MOTOR PROCEDURAL RLE / SPARSE EVENT v15 ---
SAMPLE_RATE = 48000      
MIN_CHANNELS = 8
MAX_CHANNELS = 8192      
BLOCK_SIZE = 2048        
MIN_FREQ = 20            
MAX_FREQ = 20000         
FORMAT_VERSION = 15      # Formato por Diccionario de Eventos Activos + Modulación Orgánica

def calcular_reparto_canales(total_canales):
    canales_ruido = min(64, max(2, (total_canales // 16) * 2))
    canales_tonales = total_canales - canales_ruido
    return canales_tonales, canales_ruido

def cuantizar_volumen(volumen):
    return np.clip(np.rint(volumen * 255), 0, 255).astype(np.uint8)

def decuantizar_volumen(volumen):
    return volumen.astype(np.float32) / 255.0

def crear_bandas_logaritmicas(num_bandas):
    return np.geomspace(MIN_FREQ, MAX_FREQ, num_bandas + 1)

def suavizar_matriz(matriz, ataque=0.82, caida=0.86):
    if len(matriz) < 2: return matriz
    suavizada = matriz.copy()
    for idx in range(1, len(suavizada)):
        coef = np.where(suavizada[idx] > suavizada[idx - 1], ataque, caida)
        suavizada[idx] = suavizada[idx] * (1.0 - coef) + suavizada[idx - 1] * coef
    return suavizada

class ChtnStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHTN Studio - Event-Driven High Fidelity Engine")
        self.root.geometry("1020x920") 
        self.root.minsize(880, 800)

        self.stream = None
        self.reproduciendo = False
        self.user_is_seeking = False
        
        # Nuevas estructuras basadas en Eventos Indexados (Sparse Coords)
        self.tonal_events_dict = {}  # frame -> array de (canal, freq, vol)
        self.noise_events_dict = {}  # frame -> array de (canal, vol)
        
        self.total_frames_cancion = 0
        self.fps_analisis = 48000.0 / 384.0  
        self.frame_actual = 0
        
        self.total_canales_actual = 8
        self.canales_tonales_actual, self.canales_ruido_actual = calcular_reparto_canales(8)
        
        # Estado de síntesis optimizado numéricamente
        self.phases = np.zeros(MAX_CHANNELS, dtype=np.float32)
        self.sample_positions = np.arange(BLOCK_SIZE, dtype=np.float32)
        self.sample_counter = 0
        
        # Generador de ruido estático global blindado
        self.NOISE_LOOKUP_SIZE = 48000 * 2  
        self.global_noise_buffer = np.random.choice([-1.0, 1.0], size=self.NOISE_LOOKUP_SIZE).astype(np.float32)
        self.noise_offsets = np.random.randint(0, self.NOISE_LOOKUP_SIZE - BLOCK_SIZE, size=64)
        
        # Jitter Orgánico para quitar el efecto "robótico" o "mp3 malo"
        self.phase_jitter = np.random.uniform(-0.02, 0.02, size=MAX_CHANNELS).astype(np.float32)
        
        self.visual_lock = threading.Lock()
        self.visual_channel_buffers = np.zeros((BLOCK_SIZE // 16, 64), dtype=np.float32)
        self.visual_instant_vols = []
        self.visual_instant_freqs = []

        # Variables de control UI
        self.channel_count_var = tk.IntVar(value=128)
        self.master_tonal_gain_var = tk.DoubleVar(value=1.3)  
        self.visualizer_enabled = tk.BooleanVar(value=True)
        self.visualizer_mode_var = tk.StringVar(value="Multi-Osciloscopio (Scroll)")
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
        self._actualizar_resolucion()
        self._programar_visualizador()

    def _crear_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        contenedor = ttk.Frame(self.root, padding=12)
        contenedor.pack(fill="both", expand=True)
        contenedor.columnconfigure(0, weight=1)
        contenedor.rowconfigure(5, weight=1)

        ttk.Label(contenedor, text="CHTN Studio - Event-Driven High Fidelity Engine", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(contenedor, text="Compresión de Estructura RLE Pura por Eventos Activos / Jitter Orgánico").grid(row=1, column=0, sticky="w", pady=(1, 10))

        self._crear_panel_conversion(contenedor)
        self._crear_panel_reproductor(contenedor)
        self._crear_panel_visualizador(contenedor)

        status_frame = ttk.Frame(contenedor)
        status_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        status_frame.columnconfigure(0, weight=1)

        self.lbl_estado = ttk.Label(status_frame, text="Listo.", relief="sunken", anchor="w")
        self.lbl_estado.grid(row=0, column=0, sticky="ew")
        
        lbl_dir = ttk.Label(status_frame, textvariable=self.ruta_directorio_lbl, font=("Segoe UI", 8, "italic"), foreground="#666666")
        lbl_dir.grid(row=0, column=1, padx=(10, 0), sticky="e")

    def _crear_panel_conversion(self, padre):
        frame = ttk.LabelFrame(padre, text=" Análisis Espectral y Compresión RLE de Eventos ")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="1. Buscar Audio", command=self.seleccionar_mp3).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        ttk.Label(frame, textvariable=self.selected_mp3_path).grid(row=0, column=1, padx=(0, 10), pady=(8, 4), sticky="ew")

        ttk.Button(frame, text="2. Destino .chtn", command=self.seleccionar_salida).grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        ttk.Label(frame, textvariable=self.output_chtn_path).grid(row=1, column=1, padx=(0, 10), pady=4, sticky="ew")

        ttk.Label(frame, text="3. Escalar Canales (Hasta 8192)").grid(row=2, column=0, padx=10, pady=4, sticky="w")
        
        self.slider_canales = ttk.Scale(
            frame, from_=MIN_CHANNELS, to=MAX_CHANNELS, orient="horizontal", 
            variable=self.channel_count_var, command=self._on_slider_move
        )
        self.slider_canales.grid(row=2, column=1, padx=(0, 10), pady=4, sticky="ew")

        self.lbl_resolucion = ttk.Label(frame)
        self.lbl_resolucion.grid(row=3, column=1, padx=(0, 10), pady=(0, 6), sticky="w")

        opciones = ttk.Frame(frame)
        opciones.grid(row=4, column=0, columnspan=2, padx=10, pady=(2, 6), sticky="ew")
        opciones.columnconfigure(1, weight=1)

        ttk.Checkbutton(opciones, text="Auto-cargar al finalizar", variable=self.auto_load_after_convert).grid(row=0, column=0, sticky="w")
        ttk.Progressbar(opciones, variable=self.progress_var, maximum=100).grid(row=0, column=1, padx=(12, 0), sticky="ew")

        self.btn_convertir = ttk.Button(frame, text="Iniciar Extracción RLE de Alta Fidelidad (48 kHz)", command=self.convertir_mp3, state="disabled")
        self.btn_convertir.grid(row=5, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="ew")

    def _crear_panel_reproductor(self, padre):
        frame = ttk.LabelFrame(padre, text=" Motor de Síntesis y Monitor de Control ")
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
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
        ttk.Label(master_vol_frame, text="Volumen Maestro:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=(0, 8))
        
        self.slider_master_vol = ttk.Scale(
            master_vol_frame, from_=0.0, to=4.0, orient="horizontal", variable=self.master_tonal_gain_var
        )
        self.slider_master_vol.grid(row=0, column=1, sticky="ew")
        
        lbl_gain_num = ttk.Label(master_vol_frame, text="1.3x", width=6)
        lbl_gain_num.grid(row=0, column=2, padx=(6, 0))
        self.master_tonal_gain_var.trace_add("write", lambda *args: lbl_gain_num.config(text=f"{self.master_tonal_gain_var.get():.1f}x"))

        botones = ttk.Frame(frame)
        botones.grid(row=3, column=0, padx=10, pady=(2, 8), sticky="ew")
        botones.columnconfigure(3, weight=1)

        self.btn_cargar = ttk.Button(botones, text="Abrir .chtn", command=self.cargar_chtn)
        self.btn_cargar.grid(row=0, column=0, padx=(0, 6))

        self.btn_play = ttk.Button(botones, text="Play", command=self.iniciar_audio, state="disabled")
        self.btn_play.grid(row=0, column=1, padx=6)

        self.btn_stop = ttk.Button(botones, text="Stop", command=self.detener_audio, state="disabled")
        self.btn_stop.grid(row=0, column=2, padx=6)

        ttk.Label(botones, text="Visualización:").grid(row=0, column=4, padx=(12, 4), sticky="e")
        self.combo_vis = ttk.Combobox(
            botones, textvariable=self.visualizer_mode_var, 
            values=["Multi-Osciloscopio (Scroll)", "Espectrómetro de Barras", "Matriz de Canales (LED)", "Monitor Vectorial (Lissajous)"], 
            state="readonly", width=25
        )
        self.combo_vis.grid(row=0, column=5, padx=(0, 10), sticky="e")
        self.combo_vis.bind("<<ComboboxSelected>>", self._on_visualizer_change)

        ttk.Checkbutton(botones, text="Activo", variable=self.visualizer_enabled, command=self._toggle_visualizador).grid(row=0, column=6, sticky="e")

    def _crear_panel_visualizador(self, padre):
        self.frame_visual = ttk.LabelFrame(padre, text=" Centro de Análisis Gráfico Multimodo (Fidelidad 48kHz) ")
        self.frame_visual.grid(row=5, column=0, sticky="nsew")
        self.frame_visual.columnconfigure(0, weight=1)
        self.frame_visual.rowconfigure(0, weight=1)

        self.container_scroll = ttk.Frame(self.frame_visual)
        self.container_scroll.grid(row=0, column=0, sticky="nsew")
        self.container_scroll.columnconfigure(0, weight=1)
        self.container_scroll.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.container_scroll, bg="#04070a", highlightthickness=0)
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

    def _on_slider_move(self, value):
        val_actual = int(float(value))
        bloque_mas_cercano = max(MIN_CHANNELS, min(MAX_CHANNELS, round(val_actual / 8) * 8))
        self.channel_count_var.set(bloque_mas_cercano)
        self._actualizar_resolucion()

    def seleccionar_mp3(self):
        ruta_mp3 = filedialog.askopenfilename(filetypes=[("Audios", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("Todos", "*.*")])
        if not ruta_mp3: return
        self.ruta_mp3 = ruta_mp3
        self.selected_mp3_path.set(os.path.basename(ruta_mp3))
        if not getattr(self, "ruta_salida", None):
            self.ruta_salida = os.path.splitext(ruta_mp3)[0] + ".chtn"
            self.output_chtn_path.set(os.path.basename(self.ruta_salida))
            self.ruta_directorio_lbl.set(f"Carpeta Destino: {os.path.dirname(self.ruta_salida)}")
        self._actualizar_acciones()

    def seleccionar_salida(self):
        ruta_salida = filedialog.asksaveasfilename(defaultextension=".chtn", filetypes=[("Contenedor CHTN", "*.chtn")])
        if not ruta_salida: return
        if not ruta_salida.lower().endswith(".chtn"): ruta_salida += ".chtn"
        self.ruta_salida = ruta_salida
        self.output_chtn_path.set(os.path.basename(ruta_salida))
        self.ruta_directorio_lbl.set(f"Carpeta Destino: {os.path.dirname(ruta_salida)}")
        self._actualizar_acciones()

    def _actualizar_acciones(self):
        puede = bool(getattr(self, "ruta_mp3", None) and getattr(self, "ruta_salida", None))
        self.btn_convertir.config(state="normal" if puede else "disabled")

    def _actualizar_resolucion(self):
        total = self.channel_count_var.get()
        t, n = calcular_reparto_canales(total)
        self.lbl_resolucion.config(text=f"{total} Canales Activos -> {t} Osciladores Tonales + {n} Canales de Ruido")

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

    # =========================================================================
    # EXTRACTOR DE DICCIONARIO DE EVENTOS ACTIVOS (AHORRO ABSOLUTO RLE)
    # =========================================================================
    def convertir_mp3(self):
        ruta_mp3 = getattr(self, "ruta_mp3", None)
        ruta_salida = getattr(self, "ruta_salida", None)
        if not ruta_mp3 or not ruta_salida: return

        total_canales = self.channel_count_var.get()
        canales_tonales, canales_ruido = calcular_reparto_canales(total_canales)

        self.btn_convertir.config(state="disabled")
        self._set_progreso(5)
        self._set_estado("Decodificando audio original de alto rango a 48kHz...")

        def hilo_analisis():
            archivo_temporal = None
            try:
                y, sr = librosa.load(ruta_mp3, sr=SAMPLE_RATE)
                self._set_progreso(15)
                self._set_estado("Ejecutando mapeo espectral aditivo...")
                
                hop_length = 384
                n_fft = 8192 if total_canales > 1024 else 4096 
                stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
                frecuencias = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
                
                mascara_util = (frecuencias >= MIN_FREQ) & (frecuencias <= MAX_FREQ)
                stft_util = stft[mascara_util]
                frecuencias_util = frecuencias[mascara_util]
                num_frames = stft.shape[1]
                max_stft = float(np.percentile(stft_util, 99.8)) or 1.0

                tonal_freqs_full = np.zeros((num_frames, canales_tonales), dtype=np.uint16)
                tonal_vols_float = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                noise_vols_float = np.zeros((num_frames, canales_ruido), dtype=np.float32)

                bordes_ruido = crear_bandas_logaritmicas(canales_ruido)
                bandas_ruido = [np.flatnonzero((frecuencias_util >= bordes_ruido[idx]) & (frecuencias_util < bordes_ruido[idx + 1])) for idx in range(canales_ruido)]

                piso_ruido_global = float(np.median(stft_util))
                umbral_tonal = max(piso_ruido_global * 2.2, max_stft * 0.012) 

                maximos_globales = np.zeros_like(stft_util, dtype=bool)
                maximos_globales[1:-1, :] = (stft_util[1:-1, :] > stft_util[:-2, :]) & (stft_util[1:-1, :] >= stft_util[2:, :])

                for frame_idx in range(num_frames):
                    magnitudes_frame = stft_util[:, frame_idx]
                    candidatos = np.flatnonzero(maximos_globales[:, frame_idx])
                    
                    if len(candidatos) == 0: candidatos = np.arange(len(magnitudes_frame))

                    picos_a_extraer = min(canales_tonales, len(candidatos))
                    if picos_a_extraer > 0:
                        indices_locales = np.argpartition(magnitudes_frame[candidatos], -picos_a_extraer)[-picos_a_extraer:]
                        indices_picos = candidatos[indices_locales]
                        indices_picos = indices_picos[np.argsort(magnitudes_frame[indices_picos])[::-1]]
                    else:
                        indices_picos = np.array([], dtype=int)

                    indices_picos = indices_picos[magnitudes_frame[indices_picos] >= umbral_tonal][:canales_tonales]
                    num_picos_final = len(indices_picos)

                    if num_picos_final > 0:
                        vols_extraidos = np.power(np.clip(magnitudes_frame[indices_picos] / max_stft, 0.0, 1.0), 0.82)
                        freqs_extraidas = np.rint(frecuencias_util[indices_picos]).astype(np.uint16)
                        
                        destinos = (np.arange(num_picos_final) * 13) % canales_tonales
                        for p_idx, dest in enumerate(destinos):
                            destino = dest
                            while tonal_freqs_full[frame_idx, destino] != 0:
                                destino = (destino + 1) % canales_tonales
                            
                            if vols_extraidos[p_idx] > 0.035: 
                                tonal_freqs_full[frame_idx, destino] = freqs_extraidas[p_idx]
                                tonal_vols_float[frame_idx, destino] = vols_extraidos[p_idx]

                    residual = magnitudes_frame.copy()
                    if num_picos_final > 0:
                        for p in indices_picos:
                            residual[max(0, p - 2):min(len(residual), p + 3)] *= 0.005

                    for r in range(canales_ruido):
                        indices_b = bandas_ruido[r]
                        if len(indices_b) > 0:
                            energia_b = np.percentile(residual[indices_b], 30) 
                            noise_vols_float[frame_idx, r] = min(0.85, np.power(energia_b / max_stft, 0.72) * 1.3)

                    if frame_idx % 2000 == 0:
                        self._set_progreso(25 + int(frame_idx * 50 / num_frames))

                self._set_progreso(75)
                self._set_estado("Aplicando suavizado dinámico de armónicos...")
                
                tonal_vols_full = cuantizar_volumen(suavizar_matriz(tonal_vols_float, ataque=0.82, caida=0.86))
                noise_vols_full = cuantizar_volumen(suavizar_matriz(noise_vols_float, ataque=0.60, caida=0.88))

                self._set_estado("Escribiendo Diccionario RLE Indexado sin ceros redundantes...")
                
                # --- AQUÍ CONVERTIMOS LA MATRIZ DENSA A COORDENADAS SPARSE DE EVENTOS ---
                # Buscamos únicamente dónde los volúmenes son mayores a cero
                t_frames, t_channels = np.where(tonal_vols_full > 0)
                n_frames, n_channels = np.where(noise_vols_full > 6) # Umbral de ruido útil

                tonal_events_array = np.zeros(len(t_frames), dtype=[('f', 'u4'), ('c', 'u2'), ('fr', 'u2'), ('v', 'u1')])
                tonal_events_array['f'] = t_frames
                tonal_events_array['c'] = t_channels
                tonal_events_array['fr'] = tonal_freqs_full[t_frames, t_channels]
                tonal_events_array['v'] = tonal_vols_full[t_frames, t_channels]

                noise_events_array = np.zeros(len(n_frames), dtype=[('f', 'u4'), ('c', 'u1'), ('v', 'u1')])
                noise_events_array['f'] = n_frames
                noise_events_array['c'] = n_channels
                noise_events_array['v'] = noise_vols_full[n_frames, n_channels]

                archivo_temporal = tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(ruta_salida) or ".", suffix=".tmp")

                with archivo_temporal as f:
                    np.savez_compressed(
                        f,
                        format_version=np.array([FORMAT_VERSION], dtype=np.uint8),
                        sample_rate=np.array([SAMPLE_RATE], dtype=np.uint32),
                        block_size=np.array([BLOCK_SIZE], dtype=np.uint16),
                        channels=np.array([total_canales], dtype=np.uint16),
                        tonal_channels=np.array([canales_tonales], dtype=np.uint16),
                        noise_channels=np.array([canales_ruido], dtype=np.uint16),
                        total_frames=np.array([num_frames], dtype=np.uint32),
                        tonal_events=tonal_events_array,
                        noise_events=noise_events_array
                    )

                os.replace(archivo_temporal.name, ruta_salida)
                self._set_progreso(100)
                self._set_estado("¡Archivo CHTN RLE ultra-reducido creado con éxito!")
                
                self.root.after(0, lambda: messagebox.showinfo("CHTN Reducido", "El archivo se ha guardado optimizando el espacio al máximo."))

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
        ruta = filedialog.askopenfilename(filetypes=[("Contenedor CHTN", "*.chtn")])
        if ruta: self.cargar_chtn_desde_ruta(ruta)

    def cargar_chtn_desde_ruta(self, ruta_chtn):
        if self.reproduciendo: self.detener_audio()
        try:
            with np.load(ruta_chtn) as data:
                self.total_canales_actual = int(data["channels"][0])
                self.canales_tonales_actual = int(data["tonal_channels"][0])
                self.canales_ruido_actual = int(data["noise_channels"][0])
                self.total_frames_cancion = int(data["total_frames"][0])
                
                # Reconstruimos los diccionarios veloces de eventos para el callback
                self.tonal_events_dict = {f: [] for f in range(self.total_frames_cancion)}
                self.noise_events_dict = {f: [] for f in range(self.total_frames_cancion)}
                
                t_evs = data["tonal_events"]
                for ev in t_evs:
                    self.tonal_events_dict[ev['f']].append((ev['c'], ev['fr'], decuantizar_volumen(np.array([ev['v']]))[0]))
                
                n_evs = data["noise_events"]
                for ev in n_evs:
                    self.noise_events_dict[ev['f']].append((ev['c'], decuantizar_volumen(np.array([ev['v']]))[0]))

        except Exception as e:
            messagebox.showerror("Error", f"Error al abrir contenedor RLE: {e}")
            return

        self.phases = np.zeros(self.canales_tonales_actual, dtype=np.float32)
        self.sample_counter = 0
        self.frame_actual = 0
        self.loaded_chtn_path.set(os.path.basename(ruta_chtn))
        
        self.btn_play.config(state="normal")
        self._reset_visual_buffers()
        self._calcular_reloj_tiempos(0)
        self._on_visualizer_change(None)
        self.lbl_estado.config(text="Contenedor RLE cargado. Alta definición de fase habilitada.")

    # =========================================================================
    # CALLBACK PROCEDURAL VECTORIAL AVANZADO (CON MICRO-JITTER ARMÓNICO)
    # =========================================================================
    def audio_callback(self, outdata, frames, time_info, status):
        if not self.reproduciendo or self.user_is_seeking:
            outdata.fill(0)
            return

        tiempo_actual_segundos = self.sample_counter / 48000.0
        frame_exacto = tiempo_actual_segundos * self.fps_analisis
        idx_frame = int(frame_exacto)

        if idx_frame >= self.total_frames_cancion - 1:
            self.reproduciendo = False
            outdata.fill(0)
            return

        self.frame_actual = idx_frame
        alfa = frame_exacto - idx_frame
        
        # Inicializamos vectores limpios para el bloque actual
        freqs = np.zeros(self.canales_tonales_actual, dtype=np.float32)
        vols = np.zeros(self.canales_tonales_actual, dtype=np.float32)
        vols_noise = np.zeros(self.canales_ruido_actual, dtype=np.float32)

        # Llenamos desde el diccionario de eventos sparse (Extremadamente veloz)
        evs_tonales_actual = self.tonal_events_dict.get(idx_frame, [])
        evs_tonales_siguiente = self.tonal_events_dict.get(idx_frame + 1, [])
        
        for c, fr, v in evs_tonales_actual:
            freqs[c] = fr
            vols[c] = (1.0 - alfa) * v
        for c, fr, v in evs_tonales_siguiente:
            if freqs[c] == 0: freqs[c] = fr
            vols[c] += alfa * v

        evs_ruido_actual = self.noise_events_dict.get(idx_frame, [])
        evs_ruido_siguiente = self.noise_events_dict.get(idx_frame + 1, [])
        for c, v in evs_ruido_actual: vols_noise[c] = (1.0 - alfa) * v
        for c, v in evs_ruido_siguiente: vols_noise[c] += alfa * v

        mix_buffer_stereo = np.zeros((frames, 2), dtype=np.float32)
        user_gain = float(self.master_tonal_gain_var.get())
        factor_escalado_seguro = 1.45 / (np.sqrt(self.canales_tonales_actual) + 2.0)

        visual_activo = self.visualizer_enabled.get()
        modo_vis = self.visualizer_mode_var.get()
        buf_visual_lote = np.zeros((frames, 64), dtype=np.float32) if (visual_activo and modo_vis == "Multi-Osciloscopio (Scroll)") else None

        # --- SÍNTESIS CON MODULACIÓN DE JITTER FRACTAL ANTI-ROBOTICIDAD ---
        indices_activos = np.flatnonzero((freqs > 0) & (vols > 0.004))
        
        if len(indices_activos) > 0:
            f_activas = freqs[indices_activos]
            v_activas = vols[indices_activos]
            
            # El secreto de la fidelidad: Le inyectamos pequeñas desviaciones controladas a la fase base
            # para imitar los armónicos imperfectos de instrumentos reales analógicos.
            jitter_bloque = self.phase_jitter[indices_activos] * (v_activas * 0.015)
            
            t_matrix = (2.0 * np.pi * self.sample_positions[:, None] * f_activas) / 48000.0
            t_matrix += self.phases[indices_activos] + jitter_bloque

            self.phases[indices_activos] = (self.phases[indices_activos] + (2.0 * np.pi * f_activas * frames / 48000.0)) % (2.0 * np.pi)
            tipos = indices_activos % 6
            ondas_calculadas = np.zeros((frames, len(indices_activos)), dtype=np.float32)
            
            if np.any(tipos == 0): ondas_calculadas[:, tipos == 0] = np.sin(t_matrix[:, tipos == 0])
            if np.any(tipos == 1): ondas_calculadas[:, tipos == 1] = np.where((t_matrix[:, tipos == 1] % (2 * np.pi)) < (2 * np.pi * 0.38), 1.0, -1.0) * 0.34
            if np.any(tipos == 2): ondas_calculadas[:, tipos == 2] = np.where((t_matrix[:, tipos == 2] % (2 * np.pi)) < np.pi, 1.0, -1.0) * 0.32
            if np.any(tipos == 3): ondas_calculadas[:, tipos == 3] = (2.0 / np.pi) * np.arcsin(np.sin(t_matrix[:, tipos == 3])) * 0.82
            if np.any(tipos == 4): ondas_calculadas[:, tipos == 4] = ((2.0 / np.pi) * np.arccos(np.cos(t_matrix[:, tipos == 4] + np.pi/4)) - 1.0) * 0.78
            if np.any(tipos == 5): ondas_calculadas[:, tipos == 5] = ((2.0 / np.pi) * ((t_matrix[:, tipos == 5] + np.pi) % (2 * np.pi) - np.pi)) * 0.22

            peso_canales = ondas_calculadas * v_activas * factor_escalado_seguro * user_gain

            pan_izq_vector = np.where(indices_activos % 2 == 0, 0.74, 0.26).astype(np.float32)
            pan_der_vector = 1.0 - pan_izq_vector
            
            mix_buffer_stereo[:, 0] = np.dot(peso_canales, pan_izq_vector)
            mix_buffer_stereo[:, 1] = np.dot(peso_canales, pan_der_vector)

            if buf_visual_lote is not None:
                for idx_en_sublista, ch_global in enumerate(indices_activos):
                    if ch_global < 64:
                        buf_visual_lote[:, ch_global] = peso_canales[:, idx_en_sublista]

        # --- MOTOR DE RUIDO GLOBAL POR DESPLAZAMIENTO ESTÁTICO ---
        if self.canales_ruido_actual > 0:
            factor_ruido_seguro = 0.045 / (np.sqrt(self.canales_ruido_actual) + 4.0)
            
            for r in range(self.canales_ruido_actual):
                v_ruido = vols_noise[r]
                if v_ruido > 0.025:
                    start_ptr = (self.noise_offsets[r] + self.sample_counter) % (self.NOISE_LOOKUP_SIZE - frames)
                    buf_r_final = self.global_noise_buffer[start_ptr : start_ptr + frames] * (v_ruido * factor_ruido_seguro)
                    
                    pan_l = 0.75 if r % 2 == 0 else 0.25
                    mix_buffer_stereo[:, 0] += buf_r_final * pan_l
                    mix_buffer_stereo[:, 1] += buf_r_final * (1.0 - pan_l)

                    ch_global_ruido = self.canales_tonales_actual + r
                    if buf_visual_lote is not None and ch_global_ruido < 64:
                        buf_visual_lote[:, ch_global_ruido] = buf_r_final

        # Limitador contra saturación en partes extremas
        limite_pico = np.max(np.abs(mix_buffer_stereo))
        if limite_pico > 0.96:
            mix_buffer_stereo /= (limite_pico + 0.001)

        outdata[:] = np.tanh(mix_buffer_stereo * 0.98)

        if visual_activo and self.sample_counter % (BLOCK_SIZE * 2) == 0:
            with self.visual_lock:
                indices_ordenados = np.argsort(freqs)[::-1]
                self.visual_instant_vols = list(vols[indices_ordenados]) + list(vols_noise)
                self.visual_instant_freqs = list(freqs[indices_ordenados]) + [0] * self.canales_ruido_actual
                if buf_visual_lote is not None:
                    self.visual_channel_buffers = buf_visual_lote[::16, :].copy()

        self.sample_counter += frames

    def iniciar_audio(self):
        if self.reproduciendo or not self.tonal_events_dict: return
        self.reproduciendo = True
        self.phases.fill(0.0)
        self._reset_visual_buffers()
        
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=self.audio_callback, blocksize=BLOCK_SIZE)
        self.stream.start()
        
        self.btn_play.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_cargar.config(state="disabled")
        self.lbl_estado.config(text="Reproduciendo. Desviación analógica activa.")

    def detener_audio(self):
        self.reproduciendo = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.btn_play.config(state="normal" if self.tonal_events_dict else "disabled")
        self.btn_stop.config(state="disabled")
        self.btn_cargar.config(state="normal")
        self.lbl_estado.config(text="Audio detenido.")

    def _reset_visual_buffers(self):
        with self.visual_lock:
            self.visual_instant_vols = [0.0] * (self.canales_tonales_actual + self.canales_ruido_actual)
            self.visual_instant_freqs = [0.0] * (self.canales_tonales_actual + self.canales_ruido_actual)
            self.visual_channel_buffers = np.zeros((BLOCK_SIZE // 16, 64), dtype=np.float32)

    def _toggle_visualizador(self):
        if self.visualizer_enabled.get():  
            self.frame_visual.grid()
            self._on_visualizer_change(None)
        else:                             
            self.frame_visual.grid_remove()

    def _programar_visualizador(self):
        self._dibujar_visualizador()
        self.root.after(50, self._programar_visualizador)

    def _dibujar_visualizador(self):
        if not self.visualizer_enabled.get(): return
        ancho = max(1, self.canvas.winfo_width())
        alto = max(1, self.canvas.winfo_height())
        modo = self.visualizer_mode_var.get()
        
        if self.reproduciendo and self.total_frames_cancion > 0 and not self.user_is_seeking:
            porcentaje_actual = (self.frame_actual / self.total_frames_cancion) * 100.0
            self.timeline_slider.set(porcentaje_actual)
            self._calcular_reloj_tiempos(self.frame_actual)
        
        with self.visual_lock:
            vols = list(self.visual_instant_vols)
            freqs = list(self.visual_instant_freqs)
            matriz_visual = self.visual_channel_buffers.copy() if isinstance(self.visual_channel_buffers, np.ndarray) else None

        self.canvas.delete("all")
        colores_ondas = ["#00f0ff", "#00ff66", "#ff00ff", "#ffff00", "#ffaa00", "#ff0055"]

        if modo == "Multi-Osciloscopio (Scroll)":
            canales_visibles = min(64, self.total_canales_actual)
            alto_pista = 65  
            margen_izq = 95
            ancho_util = ancho - margen_izq - 25
            alto_total_virtual = canales_visibles * alto_pista
            
            self.canvas.configure(scrollregion=(0, 0, ancho, alto_total_virtual))
            
            try:
                v_top = self.canvas.canvasy(0)
                v_bottom = v_top + alto
            except:
                v_top, v_bottom = 0, alto_total_virtual

            for ch_idx in range(canales_visibles):
                y_centro = ch_idx * alto_pista + (alto_pista / 2)
                
                if (y_centro + alto_pista) < v_top or (y_centro - alto_pista) > v_bottom:
                    continue
                
                es_tonal = ch_idx < self.canales_tonales_actual
                if es_tonal:
                    tipo_onda = ch_idx % 6
                    nombre = ["SINE", "SQ35", "SQR", "TRI", "ATRI", "SAW"][tipo_onda]
                    color = colores_ondas[tipo_onda]
                    hz = freqs[ch_idx] if ch_idx < len(freqs) else 0
                    etiqueta = f"{nombre} {ch_idx+1:02d}\n({int(hz)} Hz)"
                else:
                    color = "#ff851b"
                    etiqueta = f"D-NOISE {ch_idx - self.canales_tonales_actual + 1:02d}"

                self.canvas.create_rectangle(2, ch_idx * alto_pista, ancho - 4, (ch_idx + 1) * alto_pista, fill="#070c12", outline="#111a24")
                self.canvas.create_text(10, y_centro, text=etiqueta, anchor="w", fill="#a2b7c4", font=("Consolas", 8, "bold"))
                self.canvas.create_line(margen_izq, y_centro, ancho - 15, y_centro, fill="#1c2836", dash=(4, 4))

                if matriz_visual is not None and ch_idx < matriz_visual.shape[1]:
                    pts = matriz_visual[:, ch_idx]
                    puntos_linea = []
                    num_puntos = len(pts)
                    for i in range(num_puntos):
                        x = margen_izq + i * ancho_util / (num_puntos - 1)
                        y = y_centro - float(np.clip(pts[i] * 2.5, -1.0, 1.0)) * (alto_pista * 0.42)
                        puntos_linea.extend((x, y))
                    if len(puntos_linea) >= 4:
                        self.canvas.create_line(*puntos_linea, fill=color, width=1)

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
                x1 = x0 + ancho_barra - 1
                y0 = alto - 30 - h_barra
                y1 = alto - 30
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#00f0ff", outline="")
            self.canvas.create_line(16, alto - 30, ancho - 16, alto - 30, fill="#2c3e50", width=2)

        elif modo == "Matriz de Canales (LED)":
            columnas, filas = 16, 16
            espacio_x = (ancho - 60) / columnas
            espacio_y = (alto - 50) / filas
            for idx in range(min(len(vols), columnas * filas)):
                f_idx, c_idx = idx // columnas, idx % columnas
                x0 = 30 + c_idx * espacio_x + 4
                y0 = 25 + f_idx * espacio_y + 4
                self.canvas.create_rectangle(x0, y0, x0 + espacio_x - 8, y0 + espacio_y - 8, fill=colores_ondas[idx % 6] if vols[idx] > 0.04 else "#0f151f", outline="#16202c")

        elif modo == "Monitor Vectorial (Lissajous)":
            cx, cy = ancho / 2, alto / 2
            radio_pantalla = min(ancho, alto) * 0.40
            if len(vols) > 4:
                puntos_fase = []
                for i in range(0, min(len(vols) - 1, 256), 2):
                    if vols[i] > 0.01:
                        puntos_fase.append((cx + vols[i] * np.sin(i * 0.1) * radio_pantalla * 1.8, cy - vols[i+1] * np.cos(i * 0.2) * radio_pantalla * 1.8))
                for idx, pt in enumerate(puntos_fase[:-1]):
                    self.canvas.create_line(pt[0], pt[1], puntos_fase[idx+1][0], puntos_fase[idx+1][1], fill="#00ff66", width=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChtnStudioApp(root)
    root.mainloop()