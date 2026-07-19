import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import numpy as np
import sounddevice as sd

# --- CONFIGURACIÓN DEL MOTOR DE ALTA FIDELIDAD (CHTN NEWERSTUDIO) ---
SAMPLE_RATE = 44100      # Frecuencia estándar de calidad CD / MP3
MIN_CHANNELS = 8
MAX_CHANNELS = 1024       # 64 canales de senos puros dan una gran resolución armónica
BLOCK_SIZE = 512         # Bloque más pequeño para mayor definición temporal en transitorios
MIN_FREQ = 20            # Rango completo del oído humano
MAX_FREQ = 20000
FORMAT_VERSION = 10      # Nueva especificación de alta fidelidad

def calcular_reparto_moderno(total_canales):
    # En la versión HQ, maximizamos los armónicos: 7 tonales + 1 de ruido avanzado por bloque de 8
    bloques = max(1, total_canales // 8)
    canales_tonales = bloques * 7
    canales_ruido = total_canales - canales_tonales
    return canales_tonales, canales_ruido

def cuantizar_volumen(volumen):
    return np.clip(np.rint(volumen * 255), 0, 255).astype(np.uint8)

def decuantizar_volumen(volumen):
    return volumen.astype(np.float32) / 255.0

def crear_bandas_logaritmicas(num_bandas):
    return np.geomspace(MIN_FREQ, MAX_FREQ, num_bandas + 1)

def suavizar_matriz_hq(matriz, ataque=0.40, caida=0.45):
    # Suavizado mucho más rápido para preservar la pegada (transitorios) del MP3 original
    if len(matriz) < 2: return matriz
    suavizada = matriz.copy()
    for idx in range(1, len(suavizada)):
        coef = np.where(suavizada[idx] > suavizada[idx - 1], ataque, caida)
        suavizada[idx] = suavizada[idx] * (1.0 - coef) + suavizada[idx - 1] * coef
    return suavizada

class ChtnNewerStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHTN NewerStudio - High Fidelity Spectral Engine")
        self.root.geometry("940x840")
        self.root.minsize(840, 720)

        self.stream = None
        self.reproduciendo = False
        
        # Matrices espectrales HQ
        self.tonal_freqs_mat = None
        self.tonal_vols_mat = None
        self.noise_vols_mat = None
        self.total_frames_cancion = 0
        self.fps_analisis = 44100.0 / 256.0  # Hop length optimizado para 44.1kHz
        
        self.total_canales_actual = 8
        self.canales_tonales_actual, self.canales_ruido_actual = calcular_reparto_moderno(8)
        
        # Osciladores sinusoidales puros de fase continua (HQ)
        self.phases = np.zeros(MAX_CHANNELS, dtype=np.float32)
        self.sample_positions = np.arange(BLOCK_SIZE, dtype=np.float32)
        self.sample_counter = 0
        
        # Intercambio de datos para monitorización
        self.visual_lock = threading.Lock()
        self.visual_channel_buffers = []
        self.visual_instant_vols = []
        self.visual_instant_freqs = []

        # UI
        self.channel_count_var = tk.IntVar(value=8)
        self.visualizer_enabled = tk.BooleanVar(value=True)
        self.visualizer_mode_var = tk.StringVar(value="Espectrómetro HQ")
        self.auto_load_after_convert = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0)
        self.selected_mp3_path = tk.StringVar(value="Ningún archivo seleccionado")
        self.output_chtn_path = tk.StringVar(value="Ningún destino .chtn elegido")
        self.loaded_chtn_path = tk.StringVar(value="Ningún contenedor .chtn cargado")

        self._crear_ui()
        self._actualizar_resolucion()
        self._programar_visualizador()

    def _crear_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        contenedor = ttk.Frame(self.root, padding=14)
        contenedor.pack(fill="both", expand=True)
        contenedor.columnconfigure(0, weight=1)
        contenedor.rowconfigure(5, weight=1)

        ttk.Label(contenedor, text="CHTN NewerStudio — HQ Reconstruction", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(contenedor, text="Síntesis aditiva polifónica de alta fidelidad | Audio continuo 44.1 kHz").grid(row=1, column=0, sticky="w", pady=(2, 12))

        self._crear_panel_conversion(contenedor)
        self._crear_panel_reproductor(contenedor)
        self._crear_panel_visualizador(contenedor)

        self.lbl_estado = ttk.Label(contenedor, text="Listo.", relief="sunken", anchor="w")
        self.lbl_estado.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _crear_panel_conversion(self, padre):
        frame = ttk.LabelFrame(padre, text=" Remuestreo y Extracción Espectral de Alta Fidelidad ")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="1. Buscar MP3/WAV", command=self.seleccionar_mp3).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        ttk.Label(frame, textvariable=self.selected_mp3_path).grid(row=0, column=1, padx=(0, 10), pady=(10, 6), sticky="ew")

        ttk.Button(frame, text="2. Guardar CHTN HQ", command=self.seleccionar_salida).grid(row=1, column=0, padx=10, pady=6, sticky="ew")
        ttk.Label(frame, textvariable=self.output_chtn_path).grid(row=1, column=1, padx=(0, 10), pady=6, sticky="ew")

        ttk.Label(frame, text="3. Escalar Canales (Múltiplos de 8)").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        
        self.slider_canales = ttk.Scale(
            frame, from_=MIN_CHANNELS, to=MAX_CHANNELS, orient="horizontal", 
            variable=self.channel_count_var, command=self._on_slider_move
        )
        self.slider_canales.grid(row=2, column=1, padx=(0, 10), pady=6, sticky="ew")

        self.lbl_resolucion = ttk.Label(frame)
        self.lbl_resolucion.grid(row=3, column=1, padx=(0, 10), pady=(0, 8), sticky="w")

        opciones = ttk.Frame(frame)
        opciones.grid(row=4, column=0, columnspan=2, padx=10, pady=(4, 8), sticky="ew")
        opciones.columnconfigure(1, weight=1)

        ttk.Checkbutton(opciones, text="Auto-cargar al finalizar", variable=self.auto_load_after_convert).grid(row=0, column=0, sticky="w")
        ttk.Progressbar(opciones, variable=self.progress_var, maximum=100).grid(row=0, column=1, padx=(12, 0), sticky="ew")

        self.btn_convertir = ttk.Button(frame, text="Iniciar Análisis de Super-Resolución", command=self.convertir_audio, state="disabled")
        self.btn_convertir.grid(row=5, column=0, columnspan=2, padx=10, pady=(4, 10), sticky="ew")

    def _crear_panel_reproductor(self, padre):
        frame = ttk.LabelFrame(padre, text=" Motor de Reconstrucción de Audio en Tiempo Real ")
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, textvariable=self.loaded_chtn_path).grid(row=0, column=0, padx=10, pady=(10, 4), sticky="ew")

        botones = ttk.Frame(frame)
        botones.grid(row=1, column=0, padx=10, pady=(4, 10), sticky="ew")
        botones.columnconfigure(3, weight=1)

        self.btn_cargar = ttk.Button(botones, text="Abrir .chtn HQ", command=self.cargar_chtn)
        self.btn_cargar.grid(row=0, column=0, padx=(0, 6))

        self.btn_play = ttk.Button(botones, text="Play HQ", command=self.iniciar_audio, state="disabled")
        self.btn_play.grid(row=0, column=1, padx=6)

        self.btn_stop = ttk.Button(botones, text="Stop", command=self.detener_audio, state="disabled")
        self.btn_stop.grid(row=0, column=2, padx=6)

        ttk.Label(botones, text="Monitor:").grid(row=0, column=4, padx=(12, 4), sticky="e")
        self.combo_vis = ttk.Combobox(
            botones, textvariable=self.visualizer_mode_var, 
            values=["Espectrómetro HQ", "Monitor de Osciladores", "Matriz Lumínica"], 
            state="readonly", width=22
        )
        self.combo_vis.grid(row=0, column=5, padx=(0, 10), sticky="e")

        ttk.Checkbutton(botones, text="Activo", variable=self.visualizer_enabled, command=self._toggle_visualizador).grid(row=0, column=6, sticky="e")

    def _crear_panel_visualizador(self, padre):
        self.frame_visual = ttk.LabelFrame(padre, text=" Monitor Espectral de Precisión ")
        self.frame_visual.grid(row=5, column=0, sticky="nsew")
        self.frame_visual.columnconfigure(0, weight=1)
        self.frame_visual.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.frame_visual, height=360, bg="#020406", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _on_slider_move(self, value):
        val_actual = int(float(value))
        bloque_mas_cercano = max(MIN_CHANNELS, min(MAX_CHANNELS, round(val_actual / 8) * 8))
        self.channel_count_var.set(bloque_mas_cercano)
        self._actualizar_resolucion()

    def seleccionar_mp3(self):
        ruta_mp3 = filedialog.askopenfilename(filetypes=[("Audios", "*.mp3 *.wav *.flac *.ogg"), ("Todos", "*.*")])
        if not ruta_mp3: return
        self.ruta_mp3 = ruta_mp3
        self.selected_mp3_path.set(os.path.basename(ruta_mp3))
        if not getattr(self, "ruta_salida", None):
            self.ruta_salida = os.path.splitext(ruta_mp3)[0] + ".chtn"
            self.output_chtn_path.set(os.path.basename(self.ruta_salida))
        self._actualizar_acciones()

    def seleccionar_salida(self):
        ruta_salida = filedialog.asksaveasfilename(defaultextension=".chtn", filetypes=[("Contenedor CHTN HQ", "*.chtn")])
        if not ruta_salida: return
        if not ruta_salida.lower().endswith(".chtn"): ruta_salida += ".chtn"
        self.ruta_salida = ruta_salida
        self.output_chtn_path.set(os.path.basename(ruta_salida))
        self._actualizar_acciones()

    def _actualizar_acciones(self):
        puede = bool(getattr(self, "ruta_mp3", None) and getattr(self, "ruta_salida", None))
        self.btn_convertir.config(state="normal" if puede else "disabled")

    def _actualizar_resolucion(self):
        total = self.channel_count_var.get()
        t, n = calcular_reparto_moderno(total)
        self.lbl_resolucion.config(text=f"{total} Canales HQ -> {t} Senos Reconstructores + {n} Buses de Ruido Filtrado")

    def _set_estado(self, texto):
        self.root.after(0, lambda: self.lbl_estado.config(text=texto))

    def _set_progreso(self, valor):
        self.root.after(0, lambda: self.progress_var.set(valor))

    # =========================================================================
    # 🔥 EXTRACTOR DE SUPER-RESOLUCIÓN EN PARALELO
    # =========================================================================
    def convertir_audio(self):
        ruta_mp3 = getattr(self, "ruta_mp3", None)
        ruta_salida = getattr(self, "ruta_salida", None)
        if not ruta_mp3 or not ruta_salida: return

        total_canales = self.channel_count_var.get()
        canales_tonales, canales_ruido = calcular_reparto_moderno(total_canales)

        self.btn_convertir.config(state="disabled")
        self._set_progreso(5)
        self._set_estado("Ejecutando STFT de alta definición...")

        def hilo_analisis():
            archivo_temporal = None
            try:
                # Cargamos a 44.1kHz reales para no perder agudos del MP3
                y, sr = librosa.load(ruta_mp3, sr=SAMPLE_RATE)
                hop_length = 256
                n_fft = 2048
                stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
                frecuencias = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
                
                mascara_util = (frecuencias >= MIN_FREQ) & (frecuencias <= MAX_FREQ)
                stft_util = stft[mascara_util]
                frecuencias_util = frecuencias[mascara_util]
                num_frames = stft.shape[1]
                max_stft = float(np.percentile(stft_util, 99.8)) or 1.0

                tonal_freqs = np.zeros((num_frames, canales_tonales), dtype=np.uint16)
                tonal_vols_float = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                noise_vols_float = np.zeros((num_frames, canales_ruido), dtype=np.float32)

                bordes_ruido = crear_bandas_logaritmicas(canales_ruido)
                bandas_ruido = [np.flatnonzero((frecuencias_util >= bordes_ruido[idx]) & (frecuencias_util < bordes_ruido[idx + 1])) for idx in range(canales_ruido)]

                piso_ruido_global = float(np.percentile(stft_util, 15))
                umbral_tonal = max(piso_ruido_global * 1.8, max_stft * 0.003)

                for frame_idx in range(num_frames):
                    magnitudes_frame = stft_util[:, frame_idx]
                    
                    maximos_locales = np.zeros_like(magnitudes_frame, dtype=bool)
                    maximos_locales[1:-1] = (magnitudes_frame[1:-1] > magnitudes_frame[:-2]) & (magnitudes_frame[1:-1] >= magnitudes_frame[2:])
                    candidatos = np.flatnonzero(maximos_locales)
                    
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
                        vols_extraidos = np.power(np.clip(magnitudes_frame[indices_picos] / max_stft, 0.0, 1.0), 0.65)
                        freqs_extraidas = np.rint(frecuencias_util[indices_picos]).astype(np.uint16)
                        
                        # Distribución entrelazada fina polifónica para no saturar las primeras bandas
                        for p_idx in range(num_picos_final):
                            destino = (p_idx * 7) % canales_tonales
                            while tonal_freqs[frame_idx, destino] != 0:
                                destino = (destino + 1) % canales_tonales
                                
                            tonal_freqs[frame_idx, destino] = freqs_extraidas[p_idx]
                            tonal_vols_float[frame_idx, destino] = vols_extraidos[p_idx]

                    # Extracción del residuo ambiental (para platillos y aire de MP3)
                    residual = magnitudes_frame.copy()
                    if num_picos_final > 0:
                        for p in indices_picos: 
                            residual[max(0, p - 2):min(len(residual), p + 3)] *= 0.01

                    for r in range(canales_ruido):
                        indices_b = bandas_ruido[r]
                        if len(indices_b) > 0:
                            energia_b = np.mean(residual[indices_b]) 
                            noise_vols_float[frame_idx, r] = min(1.0, np.power(energia_b / max_stft, 0.65) * 2.5)

                    if frame_idx % 500 == 0:
                        self._set_progreso(30 + int(frame_idx * 60 / num_frames))

                self._set_progreso(95)
                # Suavizado HQ rápido para mantener ritmos rápidos y transitorios limpios
                tonal_vols = cuantizar_volumen(suavizar_matriz_hq(tonal_vols_float, ataque=0.45, caida=0.50))
                noise_vols = cuantizar_volumen(suavizar_matriz_hq(noise_vols_float, ataque=0.35, caida=0.60))

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
                        tonal_freqs=tonal_freqs,
                        tonal_vols=tonal_vols,
                        noise_vols=noise_vols,
                    )

                os.replace(archivo_temporal.name, ruta_salida)
                self._set_progreso(100)
                self._set_estado("Contenedor HQ CHTN creado con éxito.")

                if self.auto_load_after_convert.get():
                    self.root.after(0, lambda: self.cargar_chtn_desde_ruta(ruta_salida))
            except Exception as e:
                if archivo_temporal and os.path.exists(archivo_temporal.name):
                    try: os.unlink(archivo_temporal.name)
                    except OSError: pass
                self._set_estado("Error en análisis de alta fidelidad.")
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
                self.tonal_freqs_mat = data["tonal_freqs"]
                self.tonal_vols_mat = decuantizar_volumen(data["tonal_vols"])
                self.noise_vols_mat = decuantizar_volumen(data["noise_vols"])
                self.total_canales_actual = int(data["channels"][0])
                self.canales_tonales_actual = int(data["tonal_channels"][0])
                self.canales_ruido_actual = int(data["noise_channels"][0])
                self.total_frames_cancion = self.tonal_freqs_mat.shape[0]
        except Exception as e:
            messagebox.showerror("Error", f"Error de lectura: {e}")
            return

        self.phases = np.zeros(self.canales_tonales_actual, dtype=np.float32)
        self.sample_counter = 0
        self.loaded_chtn_path.set(os.path.basename(ruta_chtn))
        self.btn_play.config(state="normal")
        self._reset_visual_buffers()
        self.lbl_estado.config(text=f"Cargado HQ: {self.total_canales_actual} canales puros a {SAMPLE_RATE}Hz.")

    # =========================================================================
    # 🔥 SINTETIZADOR ADITIVO DE ALTA FIDELIDAD (SIN RESTRICCIONES RETRO)
    # =========================================================================
    def audio_callback(self, outdata, frames, time_info, status):
        if not self.reproduciendo:
            outdata.fill(0)
            return

        # Sincronización milimétrica basada en reloj de la tarjeta de sonido
        tiempo_actual_segundos = self.sample_counter / float(SAMPLE_RATE)
        frame_exacto = tiempo_actual_segundos * self.fps_analisis
        idx_frame = int(np.floor(frame_exacto))

        if idx_frame >= self.total_frames_cancion - 1:
            self.reproduciendo = False
            outdata.fill(0)
            return

        alfa = frame_exacto - idx_frame
        freqs = self.tonal_freqs_mat[idx_frame].astype(np.float32)
        vols = (1.0 - alfa) * self.tonal_vols_mat[idx_frame] + alfa * self.tonal_vols_mat[idx_frame + 1]
        vols_noise = (1.0 - alfa) * self.noise_vols_mat[idx_frame] + alfa * self.noise_vols_mat[idx_frame + 1]

        visual_activo = self.visualizer_enabled.get()
        v_buffers = [None] * self.total_canales_actual

        mix_buffer = np.zeros(frames, dtype=np.float32)

        # Síntesis Aditiva Espectral: reconstruimos mediante ondas sinusoidales puras continuas
        # Al sumar múltiples senos alineados, se recrea cualquier sonido real de un MP3 sin distorsión retro
        for i in range(self.canales_tonales_actual):
            f_comp = freqs[i]
            v_comp = vols[i]
            
            if f_comp > 20 and v_comp > 0:
                # Generamos delta de fase exacto para evitar chasquidos entre muestras
                fase_bloque = (2.0 * np.pi * f_comp * self.sample_positions[:frames]) / SAMPLE_RATE
                fases_totales = self.phases[i] + fase_bloque
                self.phases[i] = (self.phases[i] + (2.0 * np.pi * f_comp * frames / SAMPLE_RATE)) % (2.0 * np.pi)

                # Reconstrucción pura de Fourier Inversa por canal
                buf_canal = np.sin(fases_totales) * v_comp * (1.6 / self.total_canales_actual)
                mix_buffer += buf_canal
                if visual_activo: v_buffers[i] = buf_canal
            else:
                if visual_activo: v_buffers[i] = np.zeros(frames, dtype=np.float32)

        # Ruido de Confort Ambiental (Acondiciona las frecuencias altas para emular la naturalidad del MP3)
        if self.canales_ruido_actual > 0:
            # Ruido Gaussiano suave en lugar de saltos binarios digitales agresivos
            ruido_base = np.random.normal(0.0, 0.35, size=(self.canales_ruido_actual, frames)).astype(np.float32)
            for r in range(self.canales_ruido_actual):
                v_ruido = vols_noise[r]
                buf_r = ruido_base[r] * v_ruido * (0.3 / self.canales_ruido_actual)
                mix_buffer += buf_r
                if visual_activo: v_buffers[self.canales_tonales_actual + r] = buf_r

        # Limitador suave para evitar clipping y distorsión digital
        mix_buffer = np.clip(mix_buffer, -1.0, 1.0)
        outdata[:] = mix_buffer.reshape(-1, 1)

        if visual_activo:
            with self.visual_lock:
                self.visual_channel_buffers = [b.copy() if b is not None else np.zeros(frames) for b in v_buffers]
                self.visual_instant_vols = list(vols) + list(vols_noise)
                self.visual_instant_freqs = list(freqs) + [0] * self.canales_ruido_actual

        self.sample_counter += frames

    def iniciar_audio(self):
        if self.reproduciendo or self.tonal_freqs_mat is None: return
        self.reproduciendo = True
        self.sample_counter = 0
        self.phases.fill(0.0)
        self._reset_visual_buffers()
        
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, callback=self.audio_callback, blocksize=BLOCK_SIZE)
        self.stream.start()
        
        self.btn_play.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_cargar.config(state="disabled")
        self.lbl_estado.config(text="Reproducción HQ en curso a 44.1 kHz reales.")

    def detener_audio(self):
        self.reproduciendo = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.btn_play.config(state="normal" if self.tonal_freqs_mat is not None else "disabled")
        self.btn_stop.config(state="disabled")
        self.btn_cargar.config(state="normal")
        self.lbl_estado.config(text="Reproducción finalizada.")

    def _reset_visual_buffers(self):
        with self.visual_lock:
            self.visual_channel_buffers = [np.zeros(BLOCK_SIZE) for _ in range(self.total_canales_actual)]
            self.visual_instant_vols = [0.0] * self.total_canales_actual
            self.visual_instant_freqs = [0.0] * self.total_canales_actual

    def _toggle_visualizador(self):
        if self.visualizer_enabled.get():  self.frame_visual.grid()
        else:                             self.frame_visual.grid_remove()

    def _programar_visualizador(self):
        self._dibujar_visualizador()
        self.root.after(35, self._programar_visualizador)

    # =========================================================================
    # 🔥 PANTALLAS DE VISUALIZACIÓN ANALÍTICA HQ
    # =========================================================================
    def _dibujar_visualizador(self):
        if not self.visualizer_enabled.get(): return
        ancho = max(1, self.canvas.winfo_width())
        alto = max(1, self.canvas.winfo_height())
        modo = self.visualizer_mode_var.get()
        
        with self.visual_lock:
            if not self.visual_channel_buffers or len(self.visual_channel_buffers) != self.total_canales_actual: return
            buffers = [b.copy() for b in self.visual_channel_buffers]
            vols = list(self.visual_instant_vols)
            freqs = list(self.visual_instant_freqs)

        self.canvas.delete("all")

        # ---------------------------------------------------------------------
        # MODO 1: ESPECTRÓMETRO HQ (ECUALIZADOR LOGARÍTMICO CONTINUO)
        # ---------------------------------------------------------------------
        if modo == "Espectrómetro HQ":
            num_barras = 32
            ancho_barra = (ancho - 50) / num_barras
            valores_barras = np.zeros(num_barras, dtype=np.float32)
            conteos = np.zeros(num_barras, dtype=np.float32)
            
            for idx, f in enumerate(freqs[:self.canales_tonales_actual]):
                v = vols[idx] if idx < len(vols) else 0.0
                if f > MIN_FREQ and v > 0:
                    idx_b = int(np.clip((np.log10(f) - np.log10(MIN_FREQ)) / (np.log10(MAX_FREQ) - np.log10(MIN_FREQ)) * num_barras, 0, num_barras - 1))
                    valores_barras[idx_b] += v
                    conteos[idx_b] += 1.0

            for b in range(num_barras):
                val = valores_barras[b] / max(1.0, conteos[b])
                h_barra = min(alto - 40, val * (alto - 50) * 2.2)
                
                x0 = 25 + b * ancho_barra + 1
                x1 = x0 + ancho_barra - 1
                y0 = alto - 25 - h_barra
                y1 = alto - 25
                
                # Degradado estético de azul a cian neón para un look moderno de estudio
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#00bfff" if b < 22 else "#00ffaa", outline="")
            
            self.canvas.create_line(20, alto - 25, ancho - 20, alto - 25, fill="#1e2936")

        # ---------------------------------------------------------------------
        # MODO 2: MONITOR DE OSCILADORES (ONDA PURA)
        # ---------------------------------------------------------------------
        elif modo == "Monitor de Osciladores":
            canales_a_mostrar = min(12, self.total_canales_actual)
            banda = alto / canales_a_mostrar
            margen_x = 80

            for idx in range(canales_a_mostrar):
                buf_canal = buffers[idx]
                y_centro = idx * banda + banda / 2
                
                lbl = f"SINE {idx+1:02d}" if idx < self.canales_tonales_actual else f"AMBIENT {idx - self.canales_tonales_actual + 1:02d}"
                color = "#00ffcc" if idx < self.canales_tonales_actual else "#ff7f50"
                
                self.canvas.create_text(10, y_centro, text=lbl, anchor="w", fill="#6b859e", font=("Consolas", 8, "bold"))
                self.canvas.create_line(margen_x, y_centro, ancho - 10, y_centro, fill="#0f1722")

                if len(buf_canal) < 2: continue
                escala = banda * 0.45
                puntos = []
                # Paso adaptativo para dibujar ondas continuas HD fluidamente
                paso = max(1, len(buf_canal) // 128)
                for i in range(0, len(buf_canal), paso):
                    x = margen_x + i * (ancho - margen_x - 20) / (len(buf_canal) - 1)
                    y = y_centro - float(np.clip(buf_canal[i], -1.0, 1.0)) * escala
                    puntos.extend((x, y))
                
                if len(puntos) >= 4:
                    self.canvas.create_line(*puntos, fill=color, width=1)

        # ---------------------------------------------------------------------
        # MODO 3: MATRIZ LUMÍNICA (LED HQ)
        # ---------------------------------------------------------------------
        elif modo == "Matriz Lumínica":
            columnas = 8
            filas = int(np.ceil(self.total_canales_actual / columnas))
            espacio_x = (ancho - 60) / columnas
            espacio_y = (alto - 50) / filas
            
            for idx in range(self.total_canales_actual):
                f_idx = idx // columnas
                c_idx = idx % columnas
                v = vols[idx] if idx < len(vols) else 0.0
                
                x0 = 30 + c_idx * espacio_x + 4
                y0 = 25 + f_idx * espacio_y + 4
                x1 = x0 + espacio_x - 8
                y1 = y0 + espacio_y - 8
                
                if v > 0.02:
                    # El color responde en brillo real a los dBs del MP3
                    color_led = "#00ffff" if idx < self.canales_tonales_actual else "#ffaa44"
                    color_txt = "#ffffff"
                else:
                    color_led = "#0b111a"
                    color_txt = "#344454"
                    
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color_led, outline="#141f2e")
                self.canvas.create_text((x0+x1)/2, (y0+y1)/2, text=f"CH {idx+1:02d}", fill=color_txt, font=("Consolas", 8))

if __name__ == "__main__":
    root = tk.Tk()
    app = ChtnNewerStudioApp(root)
    root.mainloop()