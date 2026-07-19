import os
import tempfile
import threading
import queue
import time
import subprocess
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import numpy as np
import sounddevice as sd

# =========================================================================
# CONFIGURACIÓN GLOBAL - MOTOR ESPECTRAL ADITIVO CHTN v31
# =========================================================================
SAMPLE_RATE = 44100      
BLOCK_SIZE = 1024        
MIN_FREQ = 20.0          
MAX_FREQ = 20000.0       
FORMAT_VERSION = 310     # v31: Tracker Sinusoidal con Cuantización Mu-Law
MAX_ACTIVE_OSCILLATORS = 1024
MAX_NOISE_BANDS = 64

# =========================================================================
# FUNCIONES MATEMÁTICAS SEGURAS (CERO RUNTIME WARNINGS)
# =========================================================================
def safe_log10(x):
    return np.log10(np.clip(x, 1e-10, None))

def safe_power(x, y):
    return np.power(np.clip(x, 0.0, 1.0), y)

def safe_divide(num, den):
    return num / np.where(den == 0.0, 1e-10, den)

def calcular_reparto_canales_dinamico(total_canales):
    canales_ruido = min(MAX_NOISE_BANDS, int(total_canales * 0.15))
    canales_tonales = total_canales - canales_ruido
    return canales_tonales, canales_ruido

# =========================================================================
# CUANTIZADORES PERCEPTUALES Y MU-LAW (v31)
# =========================================================================
def cuantizar_volumen_mulaw(v):
    """Compresión logarítmica Mu-law estándar para conservar reverbs y ecos débiles."""
    mu = 255.0
    v_clip = np.clip(v, 0.0, 1.0)
    val = np.log(1.0 + mu * v_clip) / np.log(1.0 + mu)
    return np.clip(np.rint(val * 255.0), 0, 255).astype(np.uint8)

def decuantizar_volumen_mulaw(v_quant):
    """Expansión Mu-law de vuelta a amplitud lineal."""
    mu = 255.0
    norm = v_quant.astype(np.float32) / 255.0
    return (np.power(1.0 + mu, norm) - 1.0) / mu

def cuantizar_frecuencia_perceptual(f):
    if f < 200.0: return np.round(f / 2.0) * 2.0
    elif f > 6000.0: return np.round(f / 4.0) * 4.0
    else: return np.round(f)

# Retro-compatibilidad de volumen
def cuantizar_volumen_8bit(volumen): return np.clip(np.rint(volumen * 255.0), 0, 255).astype(np.uint8)
def decuantizar_volumen_8bit(volumen_8bit): return volumen_8bit.astype(np.float32) / 255.0
def decuantizar_volumen_4bit_v242(volumen_4bit): return safe_power(volumen_4bit.astype(np.float32) / 15.0, 1.333)
def decuantizar_volumen_4bit_v262(volumen_4bit): return safe_power(volumen_4bit.astype(np.float32) / 15.0, 1.428)
def decuantizar_volumen_logaritmico(v_quant):
    if v_quant == 0: return 0.0
    v_db = ((v_quant - 1.0) / 254.0) * 60.0 - 60.0
    return np.power(10.0, v_db / 20.0)

def desempaquetar_nibbles(bytes_empaquetados, shape_original):
    total_elementos = np.prod(shape_original)
    plano = np.zeros(total_elementos + (total_elementos % 2), dtype=np.uint8)
    plano[0::2] = (bytes_empaquetados >> 4) & 0x0F
    plano[1::2] = bytes_empaquetados & 0x0F
    return plano[:total_elementos].reshape(shape_original)

def interpolacion_parabolica(log_mag, bin_idx):
    if bin_idx <= 0 or bin_idx >= len(log_mag) - 1:
        return float(bin_idx), float(log_mag[bin_idx])
    a, b, c = log_mag[bin_idx - 1], log_mag[bin_idx], log_mag[bin_idx + 1]
    denominador = a - 2.0 * b + c
    if abs(denominador) < 1e-7:
        return float(bin_idx), float(b)
    p = 0.5 * (a - c) / denominador
    return float(bin_idx + p), float(b - 0.25 * (a - c) * p)

def peso_psicoacustico(frecuencias):
    f = np.clip(frecuencias, 10.0, 22000.0)
    log_f = safe_log10(f)
    return (1.0 + (1.2 * np.exp(-((log_f - 3.15)**2) / 0.6))).astype(np.float32)

# =========================================================================
# DECODIFICADOR Y EXTRACTOR UNIVERSAL DE MULTIMEDIA (FFMPEG PIPES)
# =========================================================================
def abrir_audio_universal(ruta, target_sr=SAMPLE_RATE):
    """
    Interroga y decodifica flujos de audio y vídeo de forma universal usando
    pipes y flujos directos desde FFmpeg sin crear archivos temporales.
    Acepta cualquier formato multimedia y preserva todos los canales (Estéreo, 5.1, 7.1).
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError(
            "FFmpeg no está instalado o no se encuentra en el PATH del sistema.\n"
            "Por favor, instala FFmpeg para poder codificar audio y vídeo universalmente."
        )

    cmd_probe = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels", "-of", "csv=p=0", ruta
    ]
    try:
        res = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        canales = int(res.stdout.strip())
    except Exception:
        canales = 2  

    cmd_decode = [
        "ffmpeg", "-v", "error", "-i", ruta,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(target_sr), "-"
    ]
    
    proceso = subprocess.Popen(cmd_decode, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_data, _ = proceso.communicate()

    if proceso.returncode != 0 or len(raw_data) == 0:
        raise ValueError("FFmpeg no pudo extraer o decodificar flujo de audio del contenedor multimedia especificado.")

    data_pcm = np.frombuffer(raw_data, dtype=np.float32)
    data_pcm = data_pcm.reshape(-1, canales).T

    return data_pcm, target_sr

# =========================================================================
# CLASE PRINCIPAL: CHTN STUDIO V31
# =========================================================================
class ChtnStudioApp:
    def __init__(self, root):
        self.root = root
        
        # ---------------------------------------------------------
        # 0. SISTEMA DE INTERNACIONALIZACIÓN (i18n) & AJUSTES
        # ---------------------------------------------------------
        self.current_lang = "en"
        self.translations = {}
        self.lang_flags = {
            "en": "🇬🇧",
            "es": "🇪🇸",
            "fr": "🇫🇷",
            "ja": "🇯🇵",
            "de": "🇩🇪",
            "it": "🇮🇹",
            "ru": "🇷🇺"
        }
        
        # Diccionarios nativos de respaldo para garantizar un arranque inmune a caídas
        self.english_fallback_data = {
            "_language": "English",
            "_code": "en",
            "_version": 31,
            "app_title": "CHTN Studio v31 Pro - Advanced Spectral DAW",
            "encoder_title": " SPECTRUM MASTERING & ENCODING STATION (SINUSOIDAL TRACKER) ",
            "btn_load_pcm": "1. Load Track (Audio or Video)",
            "btn_save_chtn": "2. Destination .CHTN",
            "lbl_poles": "Total Oscillators:",
            "lbl_res_msg": "Tracker: {t} pure harmonics + {n} noise banks.",
            "chk_autoload": "Auto-load upon conversion",
            "btn_convert": "Compile CHTN v31 (Mu-law + Zero-Padding)",
            "decoder_title": " ADDITIVE SYNTHESIS ENGINE (MULTITHREAD DAW) ",
            "lbl_gain": "Gain:",
            "lbl_smooth": "Temporal Smoothing:",
            "lbl_stereo": "Stereo Width:",
            "btn_load_chtn": "Open .CHTN",
            "btn_play": "PLAY ⏸",
            "btn_stop": "STOP 🛑",
            "diag_title": " DIAGNOSTICS & SYSTEM TELEMETRY PANEL ",
            "lbl_vis_selector": "Laboratory Measurement Instrument:",
            "chk_vis": "Enable Spectrum Visualizer Render",
            "chk_debug": "Print DSP Thread Trace in Terminal",
            "lbl_sys_stats": "SYSTEM TELEMETRY (REAL-TIME READINGS):",
            "stats_wait": "Awaiting audio stream initialization...",
            "status_ready": "Ready.",
            "no_file_selected": "No file selected",
            "no_dest_selected": "No destination .chtn chosen",
            "no_container_loaded": "No container loaded",
            "status_convert_ffmpeg": "Opening and decoding stream using FFmpeg...",
            "status_convert_stft": "Running STFT with Blackman-Harris and Zero-Padding (Extreme Resolution)...",
            "status_convert_tracking": "Active Sinusoidal Tracking: Pitch trajectory matching...",
            "status_convert_mulaw": "Mu-Law Quantization & Delta Event Compression...",
            "status_convert_write": "Writing compressed CHTN container...",
            "convert_done_msg": "Completed! Final Size: {sz:.1f} KB.",
            "convert_success_title": "CHTN Studio v31",
            "convert_success_msg": "High-Fidelity encoding completed!\nOptimal file size: {sz:.1f} KB.",
            "err_ffmpeg_title": "FFmpeg Required",
            "err_media_title": "Multimedia Error",
            "err_media_msg": "Could not process file: {ex}",
            "err_read_title": "Reading Error",
            "status_decoding": "Decoding container under spec CHTN v{version}...",
            "status_dec_done": "Spectral CHTN container successfully loaded to RAM.",
            "status_play_daw": "DSP DAW Engine active and playing.",
            "status_play_stop": "DSP Engine stopped.",
            "elapsed_duration": "Duration: {h:02d}:{m:02d}",
            "total_duration_lbl": "Total Duration: --:--",
            "stats_wait_play": "Start playback to monitor real-time DSP data...",
            "vis_fft_log": "Logarithmic FFT Analyzer",
            "vis_fft_bar": "FFT Bar Spectrometer",
            "vis_spectrogram": "2D Sonogram (Spectrogram)",
            "vis_waterfall": "3D Spectral Waterfall",
            "vis_lissajous": "Vector Monitor (Lissajous)",
            "vis_correlation": "Correlation Goniometer",
            "vis_oscilloscope_dual": "Dual L/R Oscilloscope",
            "vis_oscilloscope_scroll": "Multi-Osciloscopio (Scroll)",
            "vis_vu_meter": "Analog VU Meter",
            "vis_peak_digital": "RMS / Digital Peak Meter",
            "vis_bark": "Bark Psychoacoustic Scale",
            "vis_activity": "Oscillator Activity Tracker",
            "menu_file": "File",
            "menu_settings": "Settings",
            "menu_language": "Language",
            "menu_exit": "Exit",
            "stat_engine": "CHTN DECODER ENGINE v31",
            "stat_version": "Format Version : {ver}",
            "stat_sr": "WAV SampleRate : {sr} Hz",
            "stat_tonal": "Tonal Channels : {ton}",
            "stat_noise": "Noise Channels : {noi}",
            "stat_file_sz": "CHTN File Size : {sz:.2f} KB",
            "stat_reduction": "Reduction Ratio: {red:.2f} %",
            "stat_active_rt": "Active Oscs RT : {oscs}",
            "stat_dsp_load": "DSP Thread Load: {load:.1f} %",
            "stat_dsp_fps": "DSP Frame Rate : {fps:.1f} FPS",
            "stat_block_ms": "Block Render   : {ms:.2f} ms",
            "stat_stage_title": "DSP STAGE PROFILE (ms):",
            "stat_stage_prep": " - Prep Frame  : {val:.3f}",
            "stat_stage_syn": " - Additive Syn: {val:.3f}",
            "stat_stage_mix": " - Mix & Limit : {val:.3f}",
            "stat_stage_vis": " - Visual Data : {val:.3f}"
        }

        self.spanish_data = {
            "_language": "Español",
            "_code": "es",
            "_version": 31,
            "app_title": "CHTN Studio v31 Pro - Advanced Spectral DAW",
            "encoder_title": " UNIDAD DE MASTERING Y ENCODER (TRACKING SINUSOIDAL) ",
            "btn_load_pcm": "1. Cargar Pista (Audio o Vídeo)",
            "btn_save_chtn": "2. Destino .CHTN",
            "lbl_poles": "Osciladores Totales:",
            "lbl_res_msg": "Tracker: {t} armónicos puros + {n} bancos de ruido.",
            "chk_autoload": "Auto-cargar al finalizar",
            "btn_convert": "Compilar CHTN v31 (Mu-law + Zero-Padding)",
            "decoder_title": " MOTOR DE SÍNTESIS ADITIVA (DAW MULTIHILO) ",
            "lbl_gain": "Ganancia:",
            "lbl_smooth": "Suavizado Temporal:",
            "lbl_stereo": "Ancho Estéreo:",
            "btn_load_chtn": "Abrir .CHTN",
            "btn_play": "PLAY ⏸",
            "btn_stop": "STOP 🛑",
            "diag_title": " PANEL DE CONTROL DE DIAGNÓSTICO Y TELEMETRÍA ",
            "lbl_vis_selector": "Instrumento de Medición de Laboratorio:",
            "chk_vis": "Habilitar Visualizador de Espectro",
            "chk_debug": "Imprimir Trace de Hilo DSP en Consola",
            "lbl_sys_stats": "TELEMETRÍA DEL SISTEMA (LECTURAS EN TIEMPO REAL):",
            "stats_wait": "Esperando inicialización del flujo de audio...",
            "status_ready": "Listo.",
            "no_file_selected": "Ningún archivo seleccionado",
            "no_dest_selected": "Ningún destino .chtn elegido",
            "no_container_loaded": "Ningún contenedor cargado",
            "status_convert_ffmpeg": "Abriendo y decodificando flujo mediante FFmpeg...",
            "status_convert_stft": "Ejecutando STFT con Blackman-Harris y Zero-Padding (Extrema Resolución)...",
            "status_convert_tracking": "Active Sinusoidal Tracking: Emparejamiento de trayectorias de pitch...",
            "status_convert_mulaw": "Cuantización Mu-Law y Compresión Delta de Eventos...",
            "status_convert_write": "Escribiendo contenedor comprimido CHTN...",
            "convert_done_msg": "¡Completado! Peso final: {sz:.1f} KB.",
            "convert_success_title": "CHTN Studio v31",
            "convert_success_msg": "¡Codificación finalizada de Alta Fidelidad!\nTamaño final óptimo: {sz:.1f} KB.",
            "err_ffmpeg_title": "FFmpeg Requerido",
            "err_media_title": "Error multimedia",
            "err_media_msg": "No se pudo procesar el archivo: {ex}",
            "err_read_title": "Error de lectura",
            "status_decoding": "Decodificando contenedor bajo especificación CHTN v{version}...",
            "status_dec_done": "Contenedor Espectral CHTN cargado correctamente en RAM.",
            "status_play_daw": "Motor DSP DAW activo y reproduciendo.",
            "status_play_stop": "Motor DSP detenido.",
            "elapsed_duration": "Duración: {h:02d}:{m:02d}",
            "total_duration_lbl": "Duración Total: --:--",
            "stats_wait_play": "Inicie la reproducción para monitorizar datos del DSP...",
            "vis_fft_log": "Analizador FFT Logarítmico",
            "vis_fft_bar": "Espectrómetro de Barras (FFT)",
            "vis_spectrogram": "Sonograma 2D (Espectrograma)",
            "vis_waterfall": "Cascada Espectral 3D",
            "vis_lissajous": "Monitor Vectorial (Lissajous)",
            "vis_correlation": "Goniometro de Correlación",
            "vis_oscilloscope_dual": "Osciloscopio Dual L/R",
            "vis_oscilloscope_scroll": "Multi-Osciloscopio (Scroll)",
            "vis_vu_meter": "Vúmetro Analógico (VU Meter)",
            "vis_peak_digital": "Medidor RMS / Peak Digital",
            "vis_bark": "Escala Psicoacústica Bark",
            "vis_activity": "Actividad de Osciladores",
            "menu_file": "Archivo",
            "menu_settings": "Configuración",
            "menu_language": "Idioma",
            "menu_exit": "Salir",
            "stat_engine": "MOTOR DE DECODIFICACIÓN CHTN v31",
            "stat_version": "Versión del formato: {ver}",
            "stat_sr": "Tasa de Muestreo WAV: {sr} Hz",
            "stat_tonal": "Canales Tonales   : {ton}",
            "stat_noise": "Canales de Ruido  : {noi}",
            "stat_file_sz": "Tamaño del CHTN   : {sz:.2f} KB",
            "stat_reduction": "Tasa de Compresión: {red:.2f} %",
            "stat_active_rt": "Osciladores Activos: {oscs}",
            "stat_dsp_load": "Carga del hilo DSP : {load:.1f} %",
            "stat_dsp_fps": "Tasa de frames DSP : {fps:.1f} FPS",
            "stat_block_ms": "Renderizado Bloque : {ms:.2f} ms",
            "stat_stage_title": "TIEMPOS DE ETAPAS DSP (ms):",
            "stat_stage_prep": " - Prep. Frame    : {val:.3f}",
            "stat_stage_syn": " - Síntesis Adit. : {val:.3f}",
            "stat_stage_mix": " - Mezcla y Limit.: {val:.3f}",
            "stat_stage_vis": " - Datos Visores  : {val:.3f}"
        }

        self._setup_i18n()

        # Mapeo inmutable para los visualizadores espectrales y de fase
        self.visualizadores_mapping = [
            ("vis_fft_log", "Analizador FFT Logarítmico"),
            ("vis_fft_bar", "Espectrómetro de Barras (FFT)"),
            ("vis_spectrogram", "Sonograma 2D (Espectrograma)"),
            ("vis_waterfall", "Cascada Espectral 3D"),
            ("vis_lissajous", "Monitor Vectorial (Lissajous)"),
            ("vis_correlation", "Goniometro de Correlación"),
            ("vis_oscilloscope_dual", "Osciloscopio Dual L/R"),
            ("vis_oscilloscope_scroll", "Multi-Osciloscopio (Scroll)"),
            ("vis_vu_meter", "Vúmetro Analógico (VU Meter)"),
            ("vis_peak_digital", "Medidor RMS / Peak Digital"),
            ("vis_bark", "Escala Psicoacústica Bark"),
            ("vis_activity", "Actividad de Osciladores")
        ]

        # ---------------------------------------------------------
        # 2. INICIALIZACIÓN PREVENTIVA DE TODAS LAS VARIABLES (Evita AttributeError)
        # ---------------------------------------------------------
        self.block_render_ms = 0.0
        self.prep_frame_ms = 0.0
        self.update_osc_ms = 0.0
        self.additive_ms = 0.0
        self.noise_ms = 0.0
        self.mix_ms = 0.0
        self.limiter_ms = 0.0
        self.vis_ms = 0.0
        self.dsp_load = 0.0
        self.dsp_fps = 0.0
        self.cpu_usage_pct = 0.0
        self.active_oscs_realtime = 0
        self.detected_version = "N/A"
        
        self.prof_prep = 0.0
        self.prof_synth = 0.0
        self.prof_mix = 0.0
        self.prof_vis = 0.0
        
        # ---------------------------------------------------------
        # 3. ESTADO DE AUDIO Y DOBLE BUFFER POR ANILLO CONTIGUO (RING BUFFER)
        # ---------------------------------------------------------
        self.stream = None
        self.reproduciendo = False
        self.user_is_seeking = False
        self.v_lock = threading.RLock()
        self.visual_lock = threading.Lock()
        
        self.audio_queue = queue.Queue(maxsize=32)
        self.dsp_thread = None
        
        # Ring Buffer pre-asignado contiguo para evitar reservas dinámicas de memoria
        self.RING_BUFFER_SIZE = BLOCK_SIZE * 64  # ~1.5 segundos de colchón temporal
        self.ring_buffer = np.zeros((self.RING_BUFFER_SIZE, 2), dtype=np.float32)
        self.ring_write_ptr = 0
        self.ring_read_ptr = 0
        self.ring_available = 0
        
        self.bin_tonal_freqs = None
        self.bin_tonal_vols = None
        self.bin_tonal_pans = None  
        self.bin_noise_vols = None
        
        self.total_frames_cancion = 0
        self.fps_analisis = SAMPLE_RATE / 512.0  
        self.frame_actual = 0
        self.sample_rate_archivo = SAMPLE_RATE
        self.sample_rate_efectivo = SAMPLE_RATE
        self.playback_start_time = time.perf_counter()
        self.original_wav_size = 0.0
        
        self.total_canales_actual = 256
        self.canales_tonales_actual = 218
        self.canales_ruido_actual = 38
        
        self.phases = np.zeros(MAX_ACTIVE_OSCILLATORS, dtype=np.float32)
        
        # Desacoplamiento de Relojes de Hardware y Software:
        self.sample_counter = 0     # Contador de muestras producidas
        self.samples_played = 0     # Contador de muestras consumidas en DAC
        
        self.t_linear = np.arange(BLOCK_SIZE, dtype=np.float32) / float(BLOCK_SIZE)
        self.t_matrix = self.t_linear[:, None]
        self.t_smooth_mat = 3.0 * (self.t_matrix ** 2) - 2.0 * (self.t_matrix ** 3)
        
        # ---------------------------------------------------------
        # 4. BANCOS DE RUIDO ESPECTRAL DEDICADOS
        # ---------------------------------------------------------
        self.NOISE_LOOKUP_SIZE = SAMPLE_RATE * 3  
        self.noise_bands_lookup = np.zeros((MAX_NOISE_BANDS, self.NOISE_LOOKUP_SIZE), dtype=np.float32)
        self._inicializar_bancos_ruido()
        self.noise_offsets = np.random.randint(0, self.NOISE_LOOKUP_SIZE - BLOCK_SIZE, size=MAX_NOISE_BANDS)
        
        self.visual_buffer_l = np.zeros(BLOCK_SIZE, dtype=np.float32)
        self.visual_buffer_r = np.zeros(BLOCK_SIZE, dtype=np.float32)
        self.visual_instant_vols = np.zeros(MAX_ACTIVE_OSCILLATORS, dtype=np.float32)
        self.visual_instant_freqs = np.zeros(MAX_ACTIVE_OSCILLATORS, dtype=np.float32)
        self.visual_waterfall_history = []

        self.num_canales_slider_var = tk.IntVar(value=384) 
        self.master_tonal_gain_var = tk.DoubleVar(value=1.0)  
        self.stereo_width_var = tk.DoubleVar(value=1.0)       
        self.temporal_smooth_var = tk.DoubleVar(value=0.45)   
        
        self.visualizer_enabled = tk.BooleanVar(value=True)
        self.dsp_debug_enabled = tk.BooleanVar(value=True)  
        self.visualizer_mode_var = tk.StringVar()  # Seteado dinámicamente en actualizar_idioma_ui
        self.auto_load_after_convert = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0)
        
        self.selected_mp3_path = tk.StringVar()
        self.output_chtn_path = tk.StringVar()
        self.loaded_chtn_path = tk.StringVar()
        self.tiempo_transcurrido_lbl = tk.StringVar(value="00:00")
        self.tiempo_restante_lbl = tk.StringVar(value="00:00")
        self.tiempo_total_lbl = tk.StringVar()

        self._configurar_estetica_retro()
        self._crear_ui()
        self.actualizar_idioma_ui()
        self._actualizar_info_canales_encoder()
        self.root.after(100, self._programar_visualizador)

    # =========================================================================
    # GESTOR DE IDIOMAS v31 (i18n & SETTINGS PERSISTENTES)
    # =========================================================================
    def _setup_i18n(self):
        """Crea el directorio 'languages' y escribe los archivos de traducción si no existen."""
        languages_dir = "languages"
        if not os.path.exists(languages_dir):
            try:
                os.makedirs(languages_dir)
            except Exception:
                pass
        
        # Volcado inicial transparente de diccionarios por defecto
        for lang_code, data in [("en", self.english_fallback_data), ("es", self.spanish_data)]:
            filepath = os.path.join(languages_dir, f"{lang_code}.json")
            if not os.path.exists(filepath):
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                except Exception:
                    pass

        # Cargar y escanear todos los JSON presentes en el directorio
        self.translations = {}
        try:
            for file in os.listdir(languages_dir):
                if file.endswith(".json"):
                    lang_key = file[:-5]
                    filepath = os.path.join(languages_dir, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        loaded_data = json.load(f)
                        
                        # Fase 5: Validación profunda del archivo cargado (Mismos campos, misma versión)
                        missing_keys = []
                        for k in self.english_fallback_data.keys():
                            if k not in loaded_data:
                                missing_keys.append(k)
                        
                        # Fase 2: Control de versión de idiomas
                        file_ver = loaded_data.get("_version", 0)
                        if file_ver < 31:
                            print(f"[LANG] {file} is outdated (v{file_ver} < v31). Using English fallback for missing keys.")
                        
                        if missing_keys:
                            print(f"[LANG] Missing {len(missing_keys)} keys in {file}. Using English fallback.")
                            # Inyección directa de fallbacks ingleses para evitar excepciones silenciosas
                            for k in missing_keys:
                                loaded_data[k] = self.english_fallback_data[k]

                        self.translations[lang_key] = loaded_data
        except Exception as e:
            print(f"[LANG] Error reading translation files: {e}")
        
        # En caso de fallas de lectura críticas de disco, inyectamos los fallbacks en memoria
        if "en" not in self.translations: self.translations["en"] = self.english_fallback_data
        if "es" not in self.translations: self.translations["es"] = self.spanish_data

        # Carga de Settings Persistentes
        self.current_lang = "en"
        settings_path = "settings.json"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    self.current_lang = settings.get("language", "en")
            except Exception:
                pass
        
        # Si el idioma preferido no está cargado, caemos a inglés
        if self.current_lang not in self.translations:
            self.current_lang = "en"

    def tr(self, key, **kwargs):
        """Función de traducción universal con fallback integrado al diccionario inglés."""
        val = self.translations.get(self.current_lang, {}).get(key, self.translations.get("en", {}).get(key, key))
        if kwargs:
            try:
                return val.format(**kwargs)
            except Exception:
                return val
        return val

    def change_language(self, lang_code):
        """Cambia el idioma activo de forma síncrona, guarda en settings.json y refresca la UI."""
        if lang_code in self.translations:
            self.current_lang = lang_code
            self.actualizar_idioma_ui()
            
            # Persistencia de ajustes en disco
            try:
                with open("settings.json", "w", encoding="utf-8") as f:
                    json.dump({"language": lang_code}, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"[SETTINGS] Could not write config file: {e}")

    def _crear_menus(self):
        """Reconstruye y enlaza la barra de menús usando la API oficial y limpia de Tkinter."""
        self.root.config(menu="")
        
        self.menubar = tk.Menu(self.root)
        
        # Menú File
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label=self.tr("btn_load_chtn"), command=self.cargar_chtn)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=self.tr("menu_exit"), command=self.detener_audio)
        self.menubar.add_cascade(menu=self.file_menu, label=self.tr("menu_file"))
        
        # Menú Settings
        self.settings_menu = tk.Menu(self.menubar, tearoff=0)
        self.lang_menu = tk.Menu(self.settings_menu, tearoff=0)
        
        # Clasificar y ordenar los idiomas
        sorted_languages = []
        for code, data in self.translations.items():
            lang_name = data.get("_language", code.upper())
            sorted_languages.append((lang_name, code))
        sorted_languages.sort(key=lambda x: x[0])

        for lang_name, code in sorted_languages:
            flag = self.lang_flags.get(code, "🌐")
            menu_label = f"{flag} {lang_name}"
            # Se fija 'code' por default param en lambda para evitar clónicos de closures
            self.lang_menu.add_command(label=menu_label, command=lambda c=code: self.change_language(c))
            
        self.settings_menu.add_cascade(menu=self.lang_menu, label=self.tr("menu_language"))
        self.menubar.add_cascade(menu=self.settings_menu, label=self.tr("menu_settings"))
        
        self.root.config(menu=self.menubar)

    def actualizar_idioma_ui(self):
        """Regenera y traduce de forma dinámica absolutamente todos los controles y etiquetas de la GUI."""
        # Comprobación de seguridad para evitar llamadas prematuras al cambiar idioma
        if not hasattr(self, 'panel_encoder_frame') or not hasattr(self, 'panel_decoder_frame'):
            return

        self.root.title(self.tr("app_title"))
        
        # 1. Marcos principales (LabelFrames)
        self.panel_encoder_frame.config(text=self.tr("encoder_title"))
        self.panel_decoder_frame.config(text=self.tr("decoder_title"))
        self.panel_diagnosticos_frame.config(text=self.tr("diag_title"))
        
        # 2. Etiquetas de sliders y campos fijos
        if hasattr(self, 'lbl_poles_title'): self.lbl_poles_title.config(text=self.tr("lbl_poles"))
        if hasattr(self, 'lbl_gain_title'): self.lbl_gain_title.config(text=self.tr("lbl_gain"))
        if hasattr(self, 'lbl_stereo_title'): self.lbl_stereo_title.config(text=self.tr("lbl_stereo"))
        if hasattr(self, 'lbl_smooth_title'): self.lbl_smooth_title.config(text=self.tr("lbl_smooth"))
        if hasattr(self, 'lbl_vis_selector'): self.lbl_vis_selector.config(text=self.tr("lbl_vis_selector"))
        if hasattr(self, 'lbl_sys_stats_title'): self.lbl_sys_stats_title.config(text=self.tr("lbl_sys_stats"))
        
        # 3. Botones y selectores
        if hasattr(self, 'btn_load_multimedia'): self.btn_load_multimedia.config(text=self.tr("btn_load_pcm"))
        if hasattr(self, 'btn_save_chtn_dest'): self.btn_save_chtn_dest.config(text=self.tr("btn_save_chtn"))
        if hasattr(self, 'btn_convertir'): self.btn_convertir.config(text=self.tr("btn_convert"))
        if hasattr(self, 'btn_cargar'): self.btn_cargar.config(text=self.tr("btn_load_chtn"))
        if hasattr(self, 'btn_play'): self.btn_play.config(text=self.tr("btn_play"))
        if hasattr(self, 'btn_stop'): self.btn_stop.config(text=self.tr("btn_stop"))
        
        # 4. Checkboxes
        if hasattr(self, 'chk_autoload'): self.chk_autoload.config(text=self.tr("chk_autoload"))
        if hasattr(self, 'chk_vis'): self.chk_vis.config(text=self.tr("chk_vis"))
        if hasattr(self, 'chk_debug'): self.chk_debug.config(text=self.tr("chk_debug"))
        
        # 5. Reconstrucción segura de menús (API limpia)
        self._crear_menus()

        # 6. Mapeo y traducción del combobox de visualizadores
        if hasattr(self, 'combo_vis'):
            vis_traducidos = []
            for key_vis, default_val in self.visualizadores_mapping:
                vis_traducidos.append(self.tr(key_vis))
                
            current_idx = self.combo_vis.current()
            self.combo_vis['values'] = vis_traducidos
            
            if current_idx >= 0:
                self.combo_vis.current(current_idx)
            else:
                self.combo_vis.current(0)
            
        # 7. Actualización de StringVars reactivas
        placeholders_mp3 = ["Ningún archivo seleccionado", "No file selected", self.translations.get("en", {}).get("no_file_selected"), self.translations.get("es", {}).get("no_file_selected")]
        if self.selected_mp3_path.get() in placeholders_mp3 or self.selected_mp3_path.get() == "":
            self.selected_mp3_path.set(self.tr("no_file_selected"))
            
        placeholders_chtn = ["Ningún destino .chtn elegido", "No destination .chtn chosen", self.translations.get("en", {}).get("no_dest_selected"), self.translations.get("es", {}).get("no_dest_selected")]
        if self.output_chtn_path.get() in placeholders_chtn or self.output_chtn_path.get() == "":
            self.output_chtn_path.set(self.tr("no_dest_selected"))
            
        placeholders_loaded = ["Ningún contenedor cargado", "No container loaded", self.translations.get("en", {}).get("no_container_loaded"), self.translations.get("es", {}).get("no_container_loaded")]
        if self.loaded_chtn_path.get() in placeholders_loaded or self.loaded_chtn_path.get() == "":
            self.loaded_chtn_path.set(self.tr("no_container_loaded"))
            
        self._actualizar_info_canales_encoder()
        self._calcular_reloj_tiempos(self.frame_actual)
        
        # 9. Estado inicial estático si no está reproduciendo
        if not self.reproduciendo:
            self.display_stats.config(state="normal")
            self.display_stats.delete("1.0", tk.END)
            self.display_stats.insert("1.0", self.tr("stats_wait_play"))
            self.display_stats.config(state="disabled")

        # Cambiar el título del LabelFrame del visualizador dinámicamente según el modo seleccionado
        if hasattr(self, 'combo_vis'):
            modo_actual = self.visualizer_mode_var.get()
            self.frame_visual.config(text=" " + modo_actual.upper() + " ")

    def _inicializar_bancos_ruido(self):
        """Genera bancos de ruido filtrados por banda perfectos mediante IFFT para máximo realismo percusivo."""
        white_noise = np.random.normal(0.0, 1.0, self.NOISE_LOOKUP_SIZE).astype(np.float32)
        fft_noise = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(self.NOISE_LOOKUP_SIZE, 1.0 / SAMPLE_RATE)
        
        borders = np.geomspace(MIN_FREQ, MAX_FREQ, MAX_NOISE_BANDS + 1)
        for i in range(MAX_NOISE_BANDS):
            mask = (freqs >= borders[i]) & (freqs < borders[i+1])
            fft_band = fft_noise * mask
            band_time = np.fft.irfft(fft_band, n=self.NOISE_LOOKUP_SIZE)
            max_v = np.max(np.abs(band_time)) + 1e-8
            tilt = np.clip(1.2 - (i / MAX_NOISE_BANDS), 0.5, 1.2)
            self.noise_bands_lookup[i] = (band_time / max_v) * 0.15 * tilt

    def _configurar_estetica_retro(self):
        self.bg_back = "#d4d0c8"      
        self.bg_panel = "#c0c0c0"     
        self.bg_disp = "#050505"      
        self.fg_text = "#000000"      
        self.fg_cyan = "#00ffff"      
        self.fg_amber = "#ffaa00"     
        self.fg_green = "#39ff14"     

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.bg_back, foreground=self.fg_text, font=("Consolas", 9))
        style.configure("TLabelframe", background=self.bg_panel, relief="sunken", borderwidth=2)
        style.configure("TLabelframe.Label", background=self.bg_panel, foreground="#0000aa", font=("Consolas", 10, "bold"))
        style.configure("TButton", background=self.bg_back, foreground=self.fg_text, relief="raised", borderwidth=3, padding=3)
        style.map("TButton", background=[("active", "#e4e0d8")], foreground=[("active", "#000000")])
        style.configure("TProgressbar", thickness=12, troughcolor=self.bg_disp, background=self.fg_green)

    def _crear_ui(self):
        self.root.configure(bg=self.bg_back)
        
        main_container = ttk.Frame(self.root, padding=8)
        main_container.pack(fill="both", expand=True)
        main_container.columnconfigure(0, weight=3)
        main_container.columnconfigure(1, weight=1) 
        main_container.rowconfigure(0, weight=1)

        left_side = ttk.Frame(main_container)
        left_side.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_side.columnconfigure(0, weight=1)
        left_side.rowconfigure(2, weight=1) 

        # Panel de Conversión (Encoder)
        self.panel_encoder_frame = ttk.LabelFrame(left_side, text="")
        self.panel_encoder_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.panel_encoder_frame.columnconfigure(1, weight=1)

        self.btn_load_multimedia = ttk.Button(self.panel_encoder_frame, text="", command=self.seleccionar_mp3)
        self.btn_load_multimedia.grid(row=0, column=0, padx=8, pady=4, sticky="ew")
        lbl_sel = ttk.Label(self.panel_encoder_frame, textvariable=self.selected_mp3_path, background=self.bg_disp, foreground=self.fg_amber, relief="sunken", anchor="w", padding=4)
        lbl_sel.grid(row=0, column=1, padx=8, pady=4, sticky="ew")

        self.btn_save_chtn_dest = ttk.Button(self.panel_encoder_frame, text="", command=self.seleccionar_salida)
        self.btn_save_chtn_dest.grid(row=1, column=0, padx=8, pady=4, sticky="ew")
        lbl_out = ttk.Label(self.panel_encoder_frame, textvariable=self.output_chtn_path, background=self.bg_disp, foreground=self.fg_amber, relief="sunken", anchor="w", padding=4)
        lbl_out.grid(row=1, column=1, padx=8, pady=4, sticky="ew")

        scale_frame = ttk.Frame(self.panel_encoder_frame)
        scale_frame.grid(row=2, column=0, columnspan=2, padx=8, pady=6, sticky="ew")
        scale_frame.columnconfigure(1, weight=1)
        self.lbl_poles_title = ttk.Label(scale_frame, text="", font=("Consolas", 9, "bold"))
        self.lbl_poles_title.grid(row=0, column=0, padx=(0, 8))
        self.slider_canales = tk.Scale(scale_frame, from_=32, to=1024, orient="horizontal", showvalue=True, variable=self.num_canales_slider_var, highlightthickness=0, bg=self.bg_panel, fg="#0000aa")
        self.slider_canales.grid(row=0, column=1, sticky="ew")
        self.num_canales_slider_var.trace_add("write", self._actualizar_info_canales_encoder)

        self.lbl_resolucion = ttk.Label(self.panel_encoder_frame, text="", font=("Consolas", 8, "italic"), foreground="#0000aa")
        self.lbl_resolucion.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="w")

        opcs = ttk.Frame(self.panel_encoder_frame)
        opcs.grid(row=4, column=0, columnspan=2, padx=8, pady=4, sticky="ew")
        opcs.columnconfigure(1, weight=1)
        self.chk_autoload = ttk.Checkbutton(opcs, text="", variable=self.auto_load_after_convert)
        self.chk_autoload.grid(row=0, column=0, sticky="w")
        ttk.Progressbar(opcs, variable=self.progress_var, maximum=100).grid(row=0, column=1, padx=(12, 0), sticky="ew")

        self.btn_convertir = ttk.Button(self.panel_encoder_frame, text="", command=self.convertir_mp3, state="disabled")
        self.btn_convertir.grid(row=5, column=0, columnspan=2, padx=8, pady=6, sticky="ew")

        # Panel de Reproducción (Decoder / Synth)
        self.panel_decoder_frame = ttk.LabelFrame(left_side, text="")
        self.panel_decoder_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.panel_decoder_frame.columnconfigure(0, weight=1)

        t_info = ttk.Frame(self.panel_decoder_frame)
        t_info.grid(row=0, column=0, padx=8, pady=(4, 2), sticky="ew")
        t_info.columnconfigure(0, weight=1)
        ttk.Label(t_info, textvariable=self.loaded_chtn_path, font=("Consolas", 10, "bold"), foreground="#0000aa").grid(row=0, column=0, sticky="w")
        self.lbl_duracion_total = ttk.Label(t_info, textvariable=self.tiempo_total_lbl, font=("Consolas", 9, "italic"))
        self.lbl_duracion_total.grid(row=0, column=1, sticky="e")

        tl_frame = ttk.Frame(self.panel_decoder_frame)
        tl_frame.grid(row=1, column=0, padx=8, pady=2, sticky="ew")
        tl_frame.columnconfigure(1, weight=1)
        self.lbl_transcurrido = ttk.Label(tl_frame, textvariable=self.tiempo_transcurrido_lbl, font=("Consolas", 10, "bold"), width=6, anchor="center", background=self.bg_disp, foreground=self.fg_green, relief="sunken")
        self.lbl_transcurrido.grid(row=0, column=0, padx=(0, 6))
        self.timeline_slider = tk.Scale(tl_frame, from_=0, to=100, orient="horizontal", showvalue=False, highlightthickness=0, bg=self.bg_panel, activebackground=self.fg_green, troughcolor="#888", command=self._on_timeline_seek)
        self.timeline_slider.grid(row=0, column=1, sticky="ew")
        self.timeline_slider.bind("<ButtonPress-1>", lambda e: setattr(self, 'user_is_seeking', True))
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_timeline_release)
        self.lbl_restante = ttk.Label(tl_frame, textvariable=self.tiempo_restante_lbl, font=("Consolas", 10, "bold"), width=7, anchor="center", background=self.bg_disp, foreground=self.fg_amber, relief="sunken")
        self.lbl_restante.grid(row=0, column=2, padx=(6, 0))

        ctrls = ttk.Frame(self.panel_decoder_frame)
        ctrls.grid(row=2, column=0, padx=8, pady=4, sticky="ew")
        ctrls.columnconfigure(1, weight=1); ctrls.columnconfigure(3, weight=1)
        self.lbl_gain_title = ttk.Label(ctrls, text="", font=("Consolas", 9, "bold"))
        self.lbl_gain_title.grid(row=0, column=0, padx=(0, 4))
        self.slider_master_vol = ttk.Scale(ctrls, from_=0.0, to=4.0, orient="horizontal", variable=self.master_tonal_gain_var)
        self.slider_master_vol.grid(row=0, column=1, padx=(0, 15), sticky="ew")
        self.lbl_stereo_title = ttk.Label(ctrls, text="", font=("Consolas", 9, "bold"))
        self.lbl_stereo_title.grid(row=0, column=2, padx=(0, 4))
        self.slider_stereo = ttk.Scale(ctrls, from_=0.0, to=2.0, orient="horizontal", variable=self.stereo_width_var)
        self.slider_stereo.grid(row=0, column=3, padx=(0, 15), sticky="ew")

        dsp_controls2 = ttk.Frame(self.panel_decoder_frame)
        dsp_controls2.grid(row=3, column=0, padx=8, pady=4, sticky="ew")
        dsp_controls2.columnconfigure(1, weight=1); dsp_controls2.columnconfigure(3, weight=1)
        self.lbl_smooth_title = ttk.Label(dsp_controls2, text="", font=("Consolas", 9, "bold"))
        self.lbl_smooth_title.grid(row=0, column=0, padx=(0, 4))
        self.slider_smooth = ttk.Scale(dsp_controls2, from_=0.1, to=0.95, orient="horizontal", variable=self.temporal_smooth_var)
        self.slider_smooth.grid(row=0, column=1, padx=(0, 15), sticky="ew")

        btns = ttk.Frame(self.panel_decoder_frame)
        btns.grid(row=4, column=0, padx=8, pady=(4, 6), sticky="ew")
        self.btn_cargar = ttk.Button(btns, text="", command=self.cargar_chtn)
        self.btn_cargar.grid(row=0, column=0, padx=(0, 4))
        self.btn_play = ttk.Button(btns, text="", command=self.iniciar_audio, state="disabled")
        self.btn_play.grid(row=0, column=1, padx=4)
        self.btn_stop = ttk.Button(btns, text="", command=self.detener_audio, state="disabled")
        self.btn_stop.grid(row=0, column=2, padx=4)

        # Panel de Visualizador Espectral (Centro)
        self.frame_visual = ttk.LabelFrame(left_side, text="")
        self.frame_visual.grid(row=2, column=0, sticky="nsew", pady=4)
        self.frame_visual.columnconfigure(0, weight=1)
        self.frame_visual.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.frame_visual, height=220, bg=self.bg_disp, highlightthickness=1, highlightbackground="#555")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        right_side = ttk.Frame(main_container)
        right_side.grid(row=0, column=1, sticky="nsew")
        
        # Panel de Diagnósticos (Derecha)
        self.panel_diagnosticos_frame = ttk.LabelFrame(right_side, text="")
        self.panel_diagnosticos_frame.pack(fill="both", expand=True)

        self.lbl_vis_selector = ttk.Label(self.panel_diagnosticos_frame, text="", font=("Consolas", 9, "bold"))
        self.lbl_vis_selector.pack(anchor="w", padx=8, pady=(8, 2))
        self.combo_vis = ttk.Combobox(self.panel_diagnosticos_frame, textvariable=self.visualizer_mode_var, values=[], state="readonly")
        self.combo_vis.pack(fill="x", padx=8, pady=2)
        self.combo_vis.bind("<<ComboboxSelected>>", self._on_visualizer_selected_change)

        self.chk_vis = ttk.Checkbutton(self.panel_diagnosticos_frame, text="", variable=self.visualizer_enabled, command=self._toggle_visualizador)
        self.chk_vis.pack(anchor="w", padx=8, pady=2)
        self.chk_debug = ttk.Checkbutton(self.panel_diagnosticos_frame, text="", variable=self.dsp_debug_enabled)
        self.chk_debug.pack(anchor="w", padx=8, pady=2)

        self.lbl_sys_stats_title = ttk.Label(self.panel_diagnosticos_frame, text="", font=("Consolas", 9, "bold"), foreground="#0000aa")
        self.lbl_sys_stats_title.pack(anchor="w", padx=8, pady=(12, 4))

        self.display_stats = tk.Text(self.panel_diagnosticos_frame, bg=self.bg_disp, fg=self.fg_amber, font=("Consolas", 9), height=18, width=32, relief="sunken", bd=2)
        self.display_stats.pack(fill="both", expand=True, padx=8, pady=4)
        self.display_stats.insert("1.0", self.tr("stats_wait"))
        self.display_stats.config(state="disabled")

    # =========================================================================
    # ACCIONES E INTERFAZ
    # =========================================================================
    def seleccionar_mp3(self):
        formatos_soportados = [
            ("Flujos Multimedia Soportados", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.ac3 *.eac3 *.thd *.dts *.opus *.aiff *.wma *.mp4 *.mkv *.mov *.avi *.webm *.mts *.m2ts *.ts *.mpeg *.mpg"),
            ("Todos los archivos", "*.*")
        ]
        r = filedialog.askopenfilename(filetypes=formatos_soportados)
        if r:
            self.ruta_mp3 = r; self.selected_mp3_path.set(os.path.basename(r))
            self.ruta_salida = os.path.splitext(r)[0] + ".chtn"
            self.output_chtn_path.set(os.path.basename(self.ruta_salida))
            self.btn_convertir.config(state="normal")

    def seleccionar_salida(self):
        r = filedialog.asksaveasfilename(defaultextension=".chtn", filetypes=[("CHTN", "*.chtn")])
        if r:
            self.ruta_salida = r; self.output_chtn_path.set(os.path.basename(r))
            self.btn_convertir.config(state="normal")

    def _on_timeline_seek(self, val):
        if self.user_is_seeking and self.total_frames_cancion > 0:
            self._calcular_reloj_tiempos(int(float(val) / 100.0 * self.total_frames_cancion))

    def _on_timeline_release(self, event):
        if self.total_frames_cancion > 0:
            nf = int(float(self.timeline_slider.get()) / 100.0 * self.total_frames_cancion)
            with self.v_lock:
                self.frame_actual = nf
                # Alinear de forma estricta los contadores tanto del hilo productor (DSP) como de consumo (Played)
                nueva_pos_muestras = int((nf / self.fps_analisis) * self.sample_rate_archivo)
                self.sample_counter = nueva_pos_muestras
                self.samples_played = nueva_pos_muestras
                self.phases.fill(0.0)
                # Reiniciar los punteros del Ring Buffer para vaciar muestras antiguas
                self.ring_write_ptr = 0
                self.ring_read_ptr = 0
                self.ring_available = 0
                self.ring_buffer.fill(0.0)
            
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.user_is_seeking = False
            self._calcular_reloj_tiempos(nf)

    def _on_visualizer_selected_change(self, event):
        """Callback al cambiar el visualizador desde la lista del Combobox."""
        self.canvas.configure(scrollregion=(0, 0, 0, 0))
        modo_actual = self.visualizer_mode_var.get()
        self.frame_visual.config(text=" " + modo_actual.upper() + " ")

    def _actualizar_info_canales_encoder(self, *args):
        try: ch = self.num_canales_slider_var.get()
        except Exception: ch = 256
        t, n = calcular_reparto_canales_dinamico(ch)
        self.lbl_resolucion.config(text=self.tr("lbl_res_msg", t=t, n=n))

    def _calcular_reloj_tiempos(self, f_pos):
        if self.total_frames_cancion <= 0: return
        t_tot = int(self.total_frames_cancion / self.fps_analisis)
        t_act = int(f_pos / self.fps_analisis)
        self.tiempo_transcurrido_lbl.set(f"{t_act//60:02d}:{t_act%60:02d}")
        self.tiempo_restante_lbl.set(f"-{max(0, t_tot-t_act)//60:02d}:{max(0, t_tot-t_act)%60:02d}")
        self.tiempo_total_lbl.set(self.tr("elapsed_duration", h=t_tot//60, m=t_tot%60))

    # =========================================================================
    # ENCODER v31: TRACKING SINUSOIDAL + ZERO PADDING + MU-LAW QUANTIZATION
    # =========================================================================
    def convertir_mp3(self):
        ruta_mp3 = getattr(self, "ruta_mp3", None)
        ruta_salida = getattr(self, "ruta_salida", None)
        if not ruta_mp3 or not ruta_salida: return

        total_canales_config = self.num_canales_slider_var.get()
        canales_tonales, canales_ruido = calcular_reparto_canales_dinamico(total_canales_config)
        self.btn_convertir.config(state="disabled")
        
        def hilo_analisis():
            try:
                self._set_estado(self.tr("status_convert_ffmpeg"))
                
                try:
                    # Con esta llamada universal ya no nos preocupamos del codec
                    # de entrada (MP3, WAV, FLAC, AC3, MKV, DTS, etc.)
                    # El resultado es un ndarray float32 contiguo mapeado directamente.
                    y, sr = abrir_audio_universal(ruta_mp3, target_sr=SAMPLE_RATE)
                except RuntimeError as re:
                    self._set_estado(self.tr("err_read_title"))
                    self.root.after(0, lambda: messagebox.showerror(self.tr("err_ffmpeg_title"), str(re)))
                    return
                except Exception as ex:
                    self._set_estado(self.tr("err_read_title"))
                    self.root.after(0, lambda: messagebox.showerror(self.tr("err_media_title"), self.tr("err_media_msg", ex=ex)))
                    return

                if y.ndim == 1:
                    y = np.vstack([y, y])
                # Si es estéreo o multicanal (5.1 o 7.1), nos quedamos con las dos primeras pistas para síntesis L/R
                # El ndarray completo se mantendrá cargado en memoria de forma nativa para futuras ampliaciones espaciales.
                self.original_wav_size = float(y.nbytes) / (1024.0 * 1024.0)

                self._set_progreso(10)
                self._set_estado(self.tr("status_convert_stft"))
                
                hop_length = 512  
                win_length = 2048
                n_fft = 8192     
                
                window = librosa.filters.get_window('blackmanharris', win_length)
                stft_l = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
                stft_r = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
                
                mag_l, mag_r = np.abs(stft_l), np.abs(stft_r)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
                
                mascara_util = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
                freqs_util = freqs[mascara_util]
                num_frames = mag_l.shape[1]
                
                mag_util_l, mag_util_r = mag_l[mascara_util], mag_r[mascara_util]
                stft_tot = mag_util_l + mag_util_r
                max_stft = float(np.percentile(stft_tot, 99.8)) or 1.0

                tonal_freqs_mat = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                tonal_vols_mat = np.zeros((num_frames, canales_tonales), dtype=np.float32)
                tonal_pans_mat = np.full((num_frames, canales_tonales), 0.5, dtype=np.float32)
                noise_vols_mat = np.zeros((num_frames, canales_ruido), dtype=np.float32)

                factores_oido = peso_psicoacustico(freqs_util)
                umbral_abs = max_stft * 0.0001 

                self._set_estado(self.tr("status_convert_tracking"))
                
                active_tracks_f = np.zeros(canales_tonales, dtype=np.float32)
                active_tracks_v = np.zeros(canales_tonales, dtype=np.float32)
                
                for f_idx in range(num_frames):
                    mag_comb = stft_tot[:, f_idx] * factores_oido
                    picos_idx = np.flatnonzero((mag_comb[1:-1] > mag_comb[:-2]) & (mag_comb[1:-1] >= mag_comb[2:])) + 1
                    picos_idx = picos_idx[mag_comb[picos_idx] > umbral_abs]
                    
                    cand_f, cand_v, cand_p = [], [], []
                    if len(picos_idx) > 0:
                        mag_log = safe_log10(np.clip(mag_comb, 1e-10, None))
                        mask_offset = np.flatnonzero(mascara_util)[0]
                        
                        for p in picos_idx:
                            b_ref, _ = interpolacion_parabolica(mag_log, p)
                            bin_corrected = b_ref + mask_offset
                            f_ref = bin_corrected * (sr / n_fft)
                            if MIN_FREQ <= f_ref <= MAX_FREQ:
                                v_ref = min(1.0, float(stft_tot[p, f_idx] / max_stft))
                                p_ref = mag_util_r[p, f_idx] / (stft_tot[p, f_idx] + 1e-6)
                                cand_f.append(f_ref); cand_v.append(v_ref); cand_p.append(p_ref)
                                
                    cands_assigned = np.zeros(len(cand_f), dtype=bool)
                    oscs_assigned = np.zeros(canales_tonales, dtype=bool)
                    
                    for osc in range(canales_tonales):
                        if active_tracks_v[osc] > 0:
                            best_cand, min_dist = -1, 9999.0
                            for c_i, cf in enumerate(cand_f):
                                if not cands_assigned[c_i]:
                                    dist_cents = abs(1200.0 * np.log2(cf / (active_tracks_f[osc] + 1e-5)))
                                    if dist_cents < 80.0 and dist_cents < min_dist: 
                                        min_dist = dist_cents
                                        best_cand = c_i
                            
                            if best_cand != -1:
                                tonal_freqs_mat[f_idx, osc] = cand_f[best_cand]
                                tonal_vols_mat[f_idx, osc] = cand_v[best_cand]
                                tonal_pans_mat[f_idx, osc] = cand_p[best_cand]
                                active_tracks_f[osc] = cand_f[best_cand]
                                active_tracks_v[osc] = cand_v[best_cand]
                                cands_assigned[best_cand] = True
                                oscs_assigned[osc] = True

                    cands_left = [i for i in range(len(cand_f)) if not cands_assigned[i]]
                    cands_left.sort(key=lambda i: cand_v[i], reverse=True)
                    
                    for c_i in cands_left:
                        libres = np.flatnonzero(~oscs_assigned)
                        if len(libres) == 0: break
                        osc = libres[0] 
                        
                        if f_idx > 0 and tonal_vols_mat[f_idx-1, osc] == 0:
                            tonal_freqs_mat[f_idx-1, osc] = cand_f[c_i]
                            
                        tonal_freqs_mat[f_idx, osc] = cand_f[c_i]
                        tonal_vols_mat[f_idx, osc] = cand_v[c_i]
                        tonal_pans_mat[f_idx, osc] = cand_p[c_i]
                        active_tracks_f[osc] = cand_f[c_i]
                        active_tracks_v[osc] = cand_v[c_i]
                        oscs_assigned[osc] = True

                    for osc in range(canales_tonales):
                        if active_tracks_v[osc] > 0 and not oscs_assigned[osc]:
                            tonal_freqs_mat[f_idx, osc] = active_tracks_f[osc] 
                            tonal_vols_mat[f_idx, osc] = 0.0                   
                            active_tracks_v[osc] = 0.0

                    if canales_ruido > 0:
                        b_ruido = np.geomspace(MIN_FREQ, MAX_FREQ, canales_ruido + 1)
                        for rc in range(canales_ruido):
                            sb = np.flatnonzero((freqs_util >= b_ruido[rc]) & (freqs_util < b_ruido[rc+1]))
                            if len(sb) > 0:
                                amp = min(0.95, float(np.mean(stft_tot[sb, f_idx]) / max_stft))
                                if amp > 0.005: noise_vols_mat[f_idx, rc] = amp

                    if f_idx % 1000 == 0: self._set_progreso(15 + int(f_idx * 60 / num_frames))

                self._set_estado(self.tr("status_convert_mulaw"))
                
                v31_t_f, v31_t_c, v31_t_fr, v31_t_v, v31_t_p = [], [], [], [], []
                p_f, p_v, p_p = np.zeros(canales_tonales), np.zeros(canales_tonales), np.zeros(canales_tonales)
                
                for f_idx in range(num_frames):
                    for c_idx in range(canales_tonales):
                        fv, vv, pv = tonal_freqs_mat[f_idx, c_idx], tonal_vols_mat[f_idx, c_idx], tonal_pans_mat[f_idx, c_idx]
                        
                        vq = cuantizar_volumen_mulaw(vv)
                        fq = cuantizar_frecuencia_perceptual(fv)
                        pq = np.clip(np.rint(pv * 255), 0, 255).astype(np.uint8)
                        
                        emitir = False
                        if (vq > 0) != (p_v[c_idx] > 0): emitir = True
                        elif vq > 0:
                            if abs(fq - p_f[c_idx]) > 2.0 or abs(int(vq) - int(p_v[c_idx])) > 4 or abs(int(pq) - int(p_p[c_idx])) > 15:
                                emitir = True
                        
                        if emitir or f_idx == 0:
                            v31_t_f.append(f_idx); v31_t_c.append(c_idx)
                            v31_t_fr.append(fq); v31_t_v.append(vq); v31_t_p.append(pq)
                            p_f[c_idx], p_v[c_idx], p_p[c_idx] = fq, vq, pq

                v31_n_f, v31_n_c, v31_n_v = [], [], []
                p_nv = np.zeros(canales_ruido)
                for f_idx in range(num_frames):
                    for c_idx in range(canales_ruido):
                        nv = noise_vols_mat[f_idx, c_idx]
                        nvq = cuantizar_volumen_mulaw(nv)
                        if (nvq > 0) != (p_nv[c_idx] > 0) or (nvq > 0 and abs(int(nvq) - int(p_nv[c_idx])) > 5):
                            v31_n_f.append(f_idx); v31_n_c.append(c_idx); v31_n_v.append(nvq)
                            p_nv[c_idx] = nvq

                self._set_progreso(95)
                self._set_estado(self.tr("status_convert_write"))

                tmp = tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(ruta_salida) or ".", suffix=".tmp")
                with tmp as f:
                    np.savez_compressed(
                        f,
                        format_version=np.array([FORMAT_VERSION], dtype=np.uint16),
                        sample_rate=np.array([SAMPLE_RATE], dtype=np.uint32),
                        block_size=np.array([BLOCK_SIZE], dtype=np.uint16),
                        cfg_total_oscillators=np.array([total_canales_config], dtype=np.uint16),
                        cfg_tonal_channels=np.array([canales_tonales], dtype=np.uint16),
                        cfg_noise_channels=np.array([canales_ruido], dtype=np.uint16),
                        total_frames=np.array([num_frames], dtype=np.uint32),
                        t_ev_f=np.array(v31_t_f, dtype=np.uint32), t_ev_c=np.array(v31_t_c, dtype=np.uint16),
                        t_ev_fr=np.array(v31_t_fr, dtype=np.uint16), t_ev_v=np.array(v31_t_v, dtype=np.uint8), t_ev_p=np.array(v31_t_p, dtype=np.uint8),
                        n_ev_f=np.array(v31_n_f, dtype=np.uint32), n_ev_c=np.array(v31_n_c, dtype=np.uint8), n_ev_v=np.array(v31_n_v, dtype=np.uint8)
                    )
                os.replace(tmp.name, ruta_salida)
                self._set_progreso(100)
                sz = os.path.getsize(ruta_salida) / 1024.0
                self._set_estado(self.tr("convert_done_msg", sz=sz))
                self.root.after(0, lambda: messagebox.showinfo(self.tr("convert_success_title"), self.tr("convert_success_msg", sz=sz)))
                if self.auto_load_after_convert.get(): self.root.after(0, lambda: self.cargar_chtn_desde_ruta(ruta_salida))
            except Exception as e:
                self._set_estado(self.tr("err_read_title")); messagebox.showerror(self.tr("err_media_title"), str(e))
            finally:
                self.root.after(0, self._actualizar_acciones)
        threading.Thread(target=hilo_analisis, daemon=True).start()

    # =========================================================================
    # DECODIFICADOR UNIVERSAL CHTN (v10 -> v31)
    # =========================================================================
    def cargar_chtn(self):
        r = filedialog.askopenfilename(filetypes=[("Contenedores CHTN", "*.chtn")])
        if r: self.cargar_chtn_desde_ruta(r)

    def cargar_chtn_desde_ruta(self, ruta_chtn):
        if self.reproduciendo: self.detener_audio()
        file_size_kb = os.path.getsize(ruta_chtn) / 1024.0
        
        try:
            with np.load(ruta_chtn, allow_pickle=True) as data:
                version = 17
                if "format_version" in data: version = int(data["format_version"][0])
                elif "t_ev_f" in data: version = 310
                elif "t_event_frames" in data: version = 300
                elif "tonal_events" in data: version = 15
                elif "tonal_freqs" in data: version = 10

                self.detected_version = f"v{version//10}.{version%10}" if version >= 10 else f"v{version}"
                self.sample_rate_archivo = int(data.get("sample_rate", [44100])[0])
                self.fps_analisis = self.sample_rate_archivo / 512.0
                
                c_tot = int(data.get("cfg_total_oscillators", data.get("channels", [256]))[0])
                c_ton = int(data.get("cfg_tonal_channels", data.get("tonal_channels", [218]))[0])
                c_noi = int(data.get("cfg_noise_channels", data.get("noise_channels", [38]))[0])
                tot_f = int(data["total_frames"][0])
                
                self.total_canales_actual, self.canales_tonales_actual, self.canales_ruido_actual, self.total_frames_cancion = c_tot, c_ton, c_noi, tot_f
                
                bf = np.zeros((tot_f, c_ton), dtype=np.float32)
                bv = np.zeros((tot_f, c_ton), dtype=np.float32)
                bp = np.full((tot_f, c_ton), 0.5, dtype=np.float32)
                bn = np.zeros((tot_f, c_noi), dtype=np.float32)

                # --- LECTOR v31 (Mu-law Delta Sparse Tracker) ---
                if version == 310:
                    tf, tc, tfr, tv, tp = data["t_ev_f"], data["t_ev_c"], data["t_ev_fr"], data["t_ev_v"], data["t_ev_p"]
                    last_f, last_v, last_p = np.zeros(c_ton), np.zeros(c_ton), np.full(c_ton, 0.5)
                    
                    ptr = 0
                    total_evs = len(tf)
                    for f_idx in range(tot_f):
                        while ptr < total_evs and tf[ptr] == f_idx:
                            c = tc[ptr]
                            if c < c_ton:
                                last_f[c] = float(tfr[ptr])
                                last_v[c] = decuantizar_volumen_mulaw(tv[ptr])
                                last_p[c] = float(tp[ptr]) / 255.0
                            ptr += 1
                        bf[f_idx, :] = last_f.copy()
                        bv[f_idx, :] = last_v.copy()
                        bp[f_idx, :] = last_p.copy()
                        
                    if "n_ev_f" in data:
                        nf, nc, nv = data["n_ev_f"], data["n_ev_c"], data["n_ev_v"]
                        last_nv = np.zeros(c_noi)
                        ptr = 0
                        total_nev = len(nf)
                        for f_idx in range(tot_f):
                            while ptr < total_nev and nf[ptr] == f_idx:
                                c = nc[ptr]
                                if c < c_noi: last_nv[c] = decuantizar_volumen_mulaw(nv[ptr])
                                ptr += 1
                            bn[f_idx, :] = last_nv.copy()

                # --- LECTORES HEREDADOS (v10 -> v30) ---
                elif version == 300: 
                    tf, tc, tfr, tv, tp = data["t_event_frames"], data["t_event_channels"], data["t_event_freqs"], data["t_event_vols"], data["t_event_pans"]
                    last_f, last_v, last_p = np.zeros(c_ton), np.zeros(c_ton), np.full(c_ton, 0.5)
                    ptr = 0
                    for f_idx in range(tot_f):
                        while ptr < len(tf) and tf[ptr] == f_idx:
                            c = tc[ptr]
                            if c < c_ton:
                                last_f[c] = float(tfr[ptr])
                                last_v[c] = decuantizar_volumen_logaritmico(tv[ptr])
                                last_p[c] = float(tp[ptr]) / 255.0
                            ptr += 1
                        bf[f_idx, :] = last_f.copy(); bv[f_idx, :] = last_v.copy(); bp[f_idx, :] = last_p.copy()
                        
                    if "n_event_frames" in data:
                        nf, nc, nv = data["n_event_frames"], data["n_event_channels"], data["n_event_vols"]
                        last_nv = np.zeros(c_noi); ptr = 0
                        for f_idx in range(tot_f):
                            while ptr < len(nf) and nf[ptr] == f_idx:
                                c = nc[ptr]
                                if c < c_noi: last_nv[c] = decuantizar_volumen_logaritmico(nv[ptr])
                                ptr += 1
                            bn[f_idx, :] = last_nv.copy()
                else: 
                    if "t_freqs" in data: bf = data["t_freqs"].astype(np.float32)
                    if "t_vols" in data: bv = decuantizar_volumen_8bit(data["t_vols"])
                    if "n_vols" in data: bn = decuantizar_volumen_8bit(data["n_vols"])
                    bp = np.tile(np.where(np.arange(c_ton) % 2 == 0, 0.7, 0.3).astype(np.float32), (tot_f, 1))

                # --- AUTO-CORRECCIÓN HISTÓRICA DE FRECUENCIA v31 ---
                if version < 310:
                    n_fft_old = 8192 if version == 300 else 2048
                    if version == 17:
                        n_fft_old = 4096
                    elif version == 10:
                        n_fft_old = 2048
                    
                    freqs_old = np.fft.rfftfreq(n_fft_old, d=1.0/self.sample_rate_archivo)
                    mask_old = (freqs_old >= MIN_FREQ) & (freqs_old <= MAX_FREQ)
                    mask_offset_old = np.flatnonzero(mask_old)[0]
                    hz_offset = mask_offset_old * (self.sample_rate_archivo / n_fft_old)
                    
                    bf = np.where(bf > 0.0, bf + hz_offset, 0.0)

                self.bin_tonal_freqs, self.bin_tonal_vols, self.bin_tonal_pans, self.bin_noise_vols = bf, bv, bp, bn

        except Exception as e:
            messagebox.showerror(self.tr("err_read_title"), str(e)); return

        with self.v_lock:
            self.phases.fill(0.0)
            self.sample_counter = 0
            self.samples_played = 0
            self.frame_actual = 0
            
        self.loaded_chtn_path.set(f"{os.path.basename(ruta_chtn)} ({file_size_kb:.1f} KB)")
        self.btn_play.config(state="normal")
        self._reset_visual_buffers()
        self._calcular_reloj_tiempos(0)
        self.canvas.configure(scrollregion=(0, 0, 0, 0)) 
        self._actualizar_estadisticas_gui(file_size_kb)
        
        if self.dsp_debug_enabled.get():
            print("\n>> ARCHIVO CARGADO")
            print(f"Versión: {self.detected_version}")
            print(f"Osciladores: {c_ton} | Ruido: {c_noi}")

    def _actualizar_estadisticas_gui(self, file_size_kb):
        self.display_stats.config(state="normal")
        self.display_stats.delete("1.0", tk.END)
        
        real_elapsed = 0.0
        audio_elapsed = 0.0
        time_error = 0.0
        if self.reproduciendo:
            real_elapsed = time.perf_counter() - self.playback_start_time
            audio_elapsed = self.samples_played / self.sample_rate_efectivo
            time_error = real_elapsed - audio_elapsed

        reduction_percentage = 0.0
        if self.original_wav_size > 0:
            reduction_percentage = (1.0 - (file_size_kb / (self.original_wav_size * 1024.0))) * 100.0

        info = [
            self.tr("stat_engine"),
            "-"*24,
            self.tr("stat_version", ver=self.detected_version),
            self.tr("stat_sr", sr=self.sample_rate_archivo),
            self.tr("stat_tonal", ton=self.canales_tonales_actual),
            self.tr("stat_noise", noi=self.canales_ruido_actual),
            self.tr("stat_file_sz", sz=file_size_kb),
            self.tr("stat_reduction", red=reduction_percentage),
            "-"*24,
            self.tr("stat_active_rt", oscs=self.active_oscs_realtime),
            self.tr("stat_dsp_load", load=self.cpu_usage_pct),
            self.tr("stat_dsp_fps", fps=self.dsp_fps),
            self.tr("stat_block_ms", ms=self.block_render_ms),
            "-"*24,
            self.tr("stat_stage_title"),
            self.tr("stat_stage_prep", val=self.prof_prep),
            self.tr("stat_stage_syn", val=self.prof_synth),
            self.tr("stat_stage_mix", val=self.prof_mix),
            self.tr("stat_stage_vis", val=self.prof_vis),
        ]
        self.display_stats.insert("1.0", "\n".join(info))
        self.display_stats.config(state="disabled")

    # =========================================================================
    # SÍNTESIS DE ALTA FIDELIDAD Y MULTIHILO DAW (DESACOPLADO DE TIEMPO DE CPU)
    # =========================================================================
    def _dsp_synthesis_loop(self):
        """Generador Espectral. Genera frames de forma asíncrona sin influir en el reloj de hardware."""
        fs_ratio = self.sample_rate_efectivo
        factor_escalado = 1.8 / (np.sqrt(self.canales_tonales_actual) + 4.0)
        
        fps_start = time.perf_counter()
        blocks_rendered = 0
        last_print = time.perf_counter()
        
        while self.reproduciendo:
            with self.v_lock:
                space_available = self.RING_BUFFER_SIZE - self.ring_available
                
            if space_available < BLOCK_SIZE:
                time.sleep(0.005)
                continue
                
            t0 = time.perf_counter()
            with self.v_lock:
                idx = int((self.sample_counter / fs_ratio) * self.fps_analisis)
                sc_curr = self.sample_counter
            
            if idx >= self.total_frames_cancion - 2:
                with self.v_lock:
                    self.ring_available += BLOCK_SIZE
                break
                
            f0, f1 = self.bin_tonal_freqs[idx], self.bin_tonal_freqs[idx+1]
            v0, v1 = self.bin_tonal_vols[idx], self.bin_tonal_vols[idx+1]
            p0, p1 = self.bin_tonal_pans[idx], self.bin_tonal_pans[idx+1]
            
            u_gain, s_width = float(self.master_tonal_gain_var.get()), float(self.stereo_width_var.get())
            
            # 1. Prep
            v_i = (1.0 - self.t_smooth_mat) * v0 + self.t_smooth_mat * v1
            p_i = (1.0 - self.t_matrix) * p0 + self.t_matrix * p1
            f_i = (1.0 - self.t_matrix) * f0 + self.t_matrix * f1
            self.prof_prep = (time.perf_counter() - t0) * 1000.0
            
            # 2. Activos
            v_max = np.maximum(v0, v1)
            idx_act = np.flatnonzero(v_max > 1e-4)
            self.active_oscs_realtime = len(idx_act)
            
            # 3. Síntesis Fourier Pura (Sincronizada estrictamente a la fase del oscilador)
            t2 = time.perf_counter()
            block_stereo = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)
            
            if self.active_oscs_realtime > 0:
                f_act, v_act, p_act = f_i[:, idx_act], v_i[:, idx_act], p_i[:, idx_act]
                fase_delta = (2.0 * np.pi * f_act) / fs_ratio
                
                with self.v_lock:
                    fases = self.phases[idx_act] + np.cumsum(fase_delta, axis=0)
                    self.phases[idx_act] = (fases[-1] % (2.0 * np.pi)).astype(np.float32)
                    self.sample_counter += BLOCK_SIZE
                
                onda = np.sin(fases % (2.0 * np.pi))
                peso = onda * v_act * (factor_escalado * u_gain)
                
                pan_ang = (0.5 + (p_act - 0.5) * s_width) * (np.pi / 2.0)
                block_stereo[:, 0] = np.sum(peso * np.cos(pan_ang), axis=1)
                block_stereo[:, 1] = np.sum(peso * np.sin(pan_ang), axis=1)
            else:
                with self.v_lock: 
                    self.sample_counter += BLOCK_SIZE
            self.prof_synth = (time.perf_counter() - t2) * 1000.0
            
            # 4. Mezcla Ruido y Limitador
            t3 = time.perf_counter()
            if self.canales_ruido_actual > 0:
                n0, n1 = self.bin_noise_vols[idx], self.bin_noise_vols[idx+1]
                n_i = (1.0 - self.t_matrix) * n0 + self.t_matrix * n1
                for r in range(self.canales_ruido_actual):
                    v_r = n_i[:, r]
                    if np.max(v_r) < 1e-4: continue
                    ptr = (self.noise_offsets[r] + sc_curr) % (self.NOISE_LOOKUP_SIZE - BLOCK_SIZE)
                    buf = self.noise_bands_lookup[r, ptr : ptr + BLOCK_SIZE]
                    pan = 0.35 + 0.3 * (r % 2)
                    block_stereo[:, 0] += buf * v_r * pan * u_gain
                    block_stereo[:, 1] += buf * v_r * (1.0 - pan) * u_gain
            
            pico = np.max(np.abs(block_stereo))
            if pico > 0.98: block_stereo *= (0.98 / (pico + 1e-10))
            
            # Escribir el bloque de forma atómica en el Ring Buffer
            with self.v_lock:
                end_ptr = (self.ring_write_ptr + BLOCK_SIZE) % self.RING_BUFFER_SIZE
                if end_ptr > self.ring_write_ptr:
                    self.ring_buffer[self.ring_write_ptr:end_ptr] = block_stereo
                else:
                    part1 = self.RING_BUFFER_SIZE - self.ring_write_ptr
                    self.ring_buffer[self.ring_write_ptr:] = block_stereo[:part1]
                    self.ring_buffer[:end_ptr] = block_stereo[part1:]
                    
                self.ring_write_ptr = end_ptr
                self.ring_available += BLOCK_SIZE
                
            self.prof_mix = (time.perf_counter() - t3) * 1000.0
            
            # 5. Visualizador
            t4 = time.perf_counter()
            if self.visualizer_enabled.get() and sc_curr % (BLOCK_SIZE * 4) == 0:
                with self.visual_lock:
                    self.visual_instant_vols[:self.canales_tonales_actual] = v_i[0]
                    self.visual_instant_freqs[:self.canales_tonales_actual] = f_i[0]
                    self.visual_buffer_l[:] = block_stereo[:, 0]
                    self.visual_buffer_r[:] = block_stereo[:, 1]
            self.prof_vis = (time.perf_counter() - t4) * 1000.0
            
            # Rendimiento y Profiling de CPU
            tf = time.perf_counter()
            self.block_render_ms = (tf - t0) * 1000.0
            self.cpu_usage_pct = (self.block_render_ms / ((BLOCK_SIZE / fs_ratio) * 1000.0)) * 100.0
            blocks_rendered += 1
            
            if tf - fps_start >= 1.0:
                self.dsp_fps = blocks_rendered / (tf - fps_start)
                blocks_rendered, fps_start = 0, tf
                sz = os.path.getsize(getattr(self, "ruta_salida", "")) / 1024.0 if getattr(self, "ruta_salida", None) else 1.0
                self.root.after(0, lambda: self._actualizar_estadisticas_gui(sz))
                
            if self.dsp_debug_enabled.get() and tf - last_print > 1.0:
                real_elapsed = tf - self.playback_start_time
                audio_elapsed = self.samples_played / fs_ratio
                time_err = real_elapsed - audio_elapsed
                print(f"DSP Debug | SR Solicitado: {self.sample_rate_archivo} | SR Efectivo: {fs_ratio:.1f} | Bloque: {BLOCK_SIZE} | Render: {self.block_render_ms:.2f}ms | Err Temporal: {time_err:+.4f}s")
                last_print = tf

    def audio_callback(self, outdata, frames, time_info, status):
        """Callback de SoundDevice: Consumidor estricto guiado por el reloj de hardware."""
        if not self.reproduciendo: 
            outdata.fill(0)
            return
            
        try:
            # Consume exactamente BLOCK_SIZE del doble buffer precalculado
            with self.v_lock:
                if self.ring_available < frames:
                    outdata.fill(0)
                    return
                    
                end_ptr = (self.ring_read_ptr + frames) % self.RING_BUFFER_SIZE
                if end_ptr > self.ring_read_ptr:
                    outdata[:] = self.ring_buffer[self.ring_read_ptr:end_ptr]
                else:
                    part1 = self.RING_BUFFER_SIZE - self.ring_read_ptr
                    outdata[:part1] = self.ring_buffer[self.ring_read_ptr:]
                    outdata[part1:] = self.ring_buffer[:end_ptr]
                    
                self.ring_read_ptr = end_ptr
                self.ring_available -= frames
                self.samples_played += frames
            
            # Actualización del visualizador síncrono
            if self.visualizer_enabled.get() and self.samples_played % (BLOCK_SIZE * 4) == 0:
                with self.visual_lock:
                    fs_ratio = float(self.sample_rate_archivo) if self.sample_rate_archivo > 0 else SAMPLE_RATE
                    idx_played = int((self.samples_played / fs_ratio) * self.fps_analisis)
                    if idx_played < self.total_frames_cancion:
                        self.visual_instant_vols[:self.canales_tonales_actual] = self.bin_tonal_vols[idx_played]
                        self.visual_instant_freqs[:self.canales_tonales_actual] = self.bin_tonal_freqs[idx_played]
            
        except Exception:
            outdata.fill(0)

    # =========================================================================
    # REPRODUCTOR
    # =========================================================================
    def iniciar_audio(self):
        if self.reproduciendo or self.bin_tonal_freqs is None: return
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
            
        self.reproduciendo = True
        self.phases.fill(0.0)
        self._reset_visual_buffers()
        
        self.stream = sd.OutputStream(samplerate=self.sample_rate_archivo, channels=2, callback=self.audio_callback, blocksize=BLOCK_SIZE)
        self.stream.start()
        
        self.sample_rate_efectivo = self.stream.samplerate
        self.playback_start_time = time.perf_counter()
        
        with self.v_lock:
            self.ring_write_ptr = 0
            self.ring_read_ptr = 0
            self.ring_available = 0
            self.ring_buffer.fill(0.0)
            self.sample_counter = 0
            self.samples_played = 0
        
        self.dsp_thread = threading.Thread(target=self._dsp_synthesis_loop, daemon=True)
        self.dsp_thread.start()
        
        self.btn_play.config(state="disabled"); self.btn_stop.config(state="normal")
        self._set_estado(self.tr("status_play_daw"))

    def detener_audio(self):
        self.reproduciendo = False
        if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
        if self.dsp_thread: self.dsp_thread.join(timeout=1.0); self.dsp_thread = None
        self.btn_play.config(state="normal" if self.bin_tonal_freqs is not None else "disabled"); self.btn_stop.config(state="disabled")
        self._set_estado(self.tr("status_play_stop"))

    def _reset_visual_buffers(self):
        self.visual_buffer_l.fill(0.0); self.visual_buffer_r.fill(0.0)
        self.visual_instant_vols.fill(0.0); self.visual_instant_freqs.fill(0.0)
        self.visual_waterfall_history = []

    def _toggle_visualizador(self):
        if self.visualizer_enabled.get(): self.frame_visual.grid()
        else: self.frame_visual.grid_remove()

    def _programar_visualizador(self):
        self._dibujar_visualizador()
        self.root.after(35, self._programar_visualizador)

    def _dibujar_visualizador(self):
        if not self.visualizer_enabled.get(): return
        ancho, alto = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        modo_actual = self.visualizer_mode_var.get()
        
        # Mapear el nombre traducido del Combobox de vuelta a su Key inmutable de visualizador
        modo_key = "vis_fft_log"
        for key, default_translation in self.visualizadores_mapping:
            if self.tr(key) == modo_actual or default_translation == modo_actual:
                modo_key = key
                break
        
        # Actualización de barra de reproducción en base al reloj de muestras reproducidas (DAC)
        if self.reproduciendo and self.total_frames_cancion > 0 and not self.user_is_seeking:
            fs_ratio = self.sample_rate_efectivo
            with self.v_lock:
                self.frame_actual = int((self.samples_played / fs_ratio) * self.fps_analisis)
            self.timeline_slider.set((self.frame_actual / self.total_frames_cancion) * 100.0)
            self._calcular_reloj_tiempos(self.frame_actual)

        with self.visual_lock:
            buf_l, buf_r = self.visual_buffer_l.copy(), self.visual_buffer_r.copy()
            vols, freqs = self.visual_instant_vols.copy(), self.visual_instant_freqs.copy()
            
        self.canvas.delete("all")

        if modo_key == "vis_fft_bar":
            n_b = 64; ax = (ancho - 40) / n_b; v_b = np.zeros(n_b, dtype=np.float32)
            for i, f in enumerate(freqs[:self.canales_tonales_actual]):
                if f > MIN_FREQ and vols[i] > 0:
                    ib = int(np.clip((safe_log10(f)-safe_log10(MIN_FREQ))/(safe_log10(MAX_FREQ)-safe_log10(MIN_FREQ))*n_b, 0, n_b-1))
                    v_b[ib] = max(v_b[ib], vols[i])
            for b in range(n_b):
                h_b = min(alto - 50, v_b[b] * (alto - 70) * 2.5)
                color = self.fg_green if h_b < (alto*0.5) else (self.fg_amber if h_b < (alto*0.75) else "#ff3333")
                self.canvas.create_rectangle(20+b*ax+1, alto-30-h_b, 20+b*ax+ax-1, alto-30, fill=color, outline="")
            self.canvas.create_line(16, alto-30, ancho-16, alto-30, fill="#555", width=2)

        elif modo_key == "vis_fft_log":
            n_b = 128; ax = (ancho - 40) / n_b; v_b = np.zeros(n_b, dtype=np.float32)
            for i, f in enumerate(freqs[:self.canales_tonales_actual]):
                if f > MIN_FREQ and vols[i] > 0:
                    ib = int(np.clip((safe_log10(f)-safe_log10(MIN_FREQ))/(safe_log10(MAX_FREQ)-safe_log10(MIN_FREQ))*n_b, 0, n_b-1))
                    v_b[ib] = max(v_b[ib], vols[i])
            pts = []
            for b in range(n_b):
                pts.extend((20 + b*ax + ax/2, alto - 30 - min(alto-50, v_b[b]*(alto-70)*2.8)))
            if len(pts) >= 4: self.canvas.create_line(*pts, fill=self.fg_cyan, width=2)
            self.canvas.create_line(16, alto-30, ancho-16, alto-30, fill="#555", width=2)

        elif modo_key == "vis_oscilloscope_dual":
            a_o = (alto - 60) / 2.0
            yl, yr = 30 + a_o / 2.0, 30 + a_o * 1.5
            self.canvas.create_text(15, yl, text="L", fill=self.fg_amber); self.canvas.create_text(15, yr, text="R", fill=self.fg_amber)
            self.canvas.create_line(30, yl, ancho-20, yl, fill="#333"); self.canvas.create_line(30, yr, ancho-20, yr, fill="#333")
            pl, pr = [], []
            for i in range(0, BLOCK_SIZE, 8):
                x = 30 + i * (ancho-50) / BLOCK_SIZE
                pl.extend((x, yl - buf_l[i] * a_o * 0.45)); pr.extend((x, yr - buf_r[i] * a_o * 0.45))
            if len(pl) >= 4:
                self.canvas.create_line(*pl, fill=self.fg_green, width=1.5); self.canvas.create_line(*pr, fill=self.fg_cyan, width=1.5)

        elif modo_key == "vis_spectrogram":
            if not hasattr(self, 'spec_hist'): self.spec_hist = np.zeros((100, 64), dtype=np.float32)
            nr = np.zeros(64, dtype=np.float32)
            for i, f in enumerate(freqs[:self.canales_tonales_actual]):
                if f > MIN_FREQ and vols[i] > 0:
                    ib = int(np.clip((safe_log10(f)-safe_log10(MIN_FREQ))/(safe_log10(MAX_FREQ)-safe_log10(MIN_FREQ))*64, 0, 63))
                    nr[ib] = max(nr[ib], vols[i])
            self.spec_hist = np.roll(self.spec_hist, -1, axis=0); self.spec_hist[-1] = nr
            cw, ch = (ancho-40)/64, (alto-60)/100
            for r in range(0, 100, 2):
                for c in range(64):
                    if self.spec_hist[r, c] > 0.01:
                        bx = int(np.clip(self.spec_hist[r, c]*255, 0, 255))
                        self.canvas.create_rectangle(20+c*cw, 30+r*ch, 20+c*cw+cw, 30+r*ch+ch, fill=f"#00{bx:02x}{int(bx*0.8):02x}", outline="")

        elif modo_key == "vis_vu_meter":
            cx, cy, r = ancho/2.0, alto-20, min(ancho, alto)*0.70
            self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=45, extent=90, style="arc", outline=self.fg_text, width=3)
            rms = np.sqrt(np.mean(buf_l**2 + buf_r**2))
            ang = np.radians(np.clip(135 - rms*90, 45, 135))
            self.canvas.create_line(cx, cy, cx + np.cos(ang)*r*0.9, cy - np.sin(ang)*r*0.9, fill="#ff3333", width=2)
            self.canvas.create_text(cx, cy-30, text="VU METER", fill=self.fg_cyan, font=("Consolas", 10, "bold"))

        elif modo_key == "vis_peak_digital":
            xl, xr, hm = ancho/2.0-50, ancho/2.0+10, alto-100
            pl, pr, rl, rr = np.max(np.abs(buf_l)), np.max(np.abs(buf_r)), np.sqrt(np.mean(buf_l**2)), np.sqrt(np.mean(buf_r**2))
            self.canvas.create_rectangle(xl, 50, xl+40, 50+hm, fill="#111", outline="#555")
            self.canvas.create_rectangle(xl, 50+hm - rl*hm*2.5, xl+40, 50+hm, fill=self.fg_green, outline="")
            self.canvas.create_line(xl-5, 50+hm - pl*hm, xl+45, 50+hm - pl*hm, fill="#ff3333", width=3)
            self.canvas.create_rectangle(xr, 50, xr+40, 50+hm, fill="#111", outline="#555")
            self.canvas.create_rectangle(xr, 50+hm - rr*hm*2.5, xr+40, 50+hm, fill=self.fg_green, outline="")
            self.canvas.create_line(xr-5, 50+hm - pr*hm, xr+45, 50+hm - pr*hm, fill="#ff3333", width=3)
            self.canvas.create_text(xl+20, alto-35, text="L", fill=self.fg_cyan); self.canvas.create_text(xr+20, alto-35, text="R", fill=self.fg_cyan)

        elif modo_key == "vis_lissajous":
            cx, cy = ancho / 2.0, alto / 2.0
            r_pantalla = min(ancho, alto) * 0.40
            self.canvas.create_oval(cx - r_pantalla, cy - r_pantalla, cx + r_pantalla, cy + r_pantalla, outline="#444", width=1)
            self.canvas.create_line(cx - r_pantalla, cy, cx + r_pantalla, cy, fill="#333")
            self.canvas.create_line(cx, cy - r_pantalla, cx, cy + r_pantalla, fill="#333")
            paso = 4
            puntos_fase = []
            for i in range(0, BLOCK_SIZE, paso):
                val_x = buf_l[i] - buf_r[i]
                val_y = buf_l[i] + buf_r[i]
                x = cx + val_x * r_pantalla * 1.5
                y = cy - val_y * r_pantalla * 1.5
                puntos_fase.append((x, y))
            for idx, pt in enumerate(puntos_fase[:-1]):
                self.canvas.create_line(pt[0], pt[1], puntos_fase[idx+1][0], puntos_fase[idx+1][1], fill=self.fg_cyan, width=1)

        elif modo_key == "vis_correlation":
            cx, cy = ancho / 2.0, alto / 2.0
            r = min(ancho, alto) * 0.40
            self.canvas.create_line(cx - r, cy, cx + r, cy, fill="#444")
            self.canvas.create_text(cx - r - 20, cy, text="L", fill=self.fg_text)
            self.canvas.create_text(cx + r + 20, cy, text="R", fill=self.fg_text)
            num = np.sum(buf_l * buf_r)
            den = np.sqrt(np.sum(buf_l**2) * np.sum(buf_r**2))
            correlacion = num / den if den > 1e-6 else 0.0
            x_corr = cx + correlacion * r
            self.canvas.create_rectangle(cx, cy - 15, x_corr, cy + 15, fill=self.fg_green, outline="")
            self.canvas.create_line(cx, cy - 25, cx, cy + 25, fill="#ff3333", width=2)
            self.canvas.create_text(cx, cy - 40, text=f"Fase: {correlacion:+.2f}", fill=self.fg_cyan, font=("Consolas", 10, "bold"))

        elif modo_key == "vis_waterfall":
            current_fft = np.zeros(32, dtype=np.float32)
            for idx, f in enumerate(freqs[:self.canales_tonales_actual]):
                if f > MIN_FREQ and vols[idx] > 0:
                    idx_b = int(np.clip((safe_log10(f)-safe_log10(MIN_FREQ))/(safe_log10(MAX_FREQ)-safe_log10(MIN_FREQ))*32, 0, 31))
                    current_fft[idx_b] = max(current_fft[idx_b], vols[idx])
            self.visual_waterfall_history.append(current_fft)
            if len(self.visual_waterfall_history) > 20:
                self.visual_waterfall_history.pop(0)
            for step, fft_data in enumerate(self.visual_waterfall_history):
                points = []
                offset_x = (20 - step) * 2
                offset_y = step * 10
                for b in range(32):
                    val = fft_data[b]
                    x = offset_x + b * (ancho - 100) / 32.0
                    y = alto - 50 - offset_y - (val * 80.0)
                    points.extend((x, y))
                if len(points) >= 4:
                    color_val = int(step * 12)
                    self.canvas.create_line(*points, fill=f"#{color_val:02x}ff{color_val:02x}", width=1)

        elif modo_key == "vis_activity":
            col = 32
            fils = max(1, int(np.ceil(self.total_canales_actual / col)))
            cell_w = (ancho - 40) / col
            cell_h = (alto - 50) / fils
            for idx in range(min(len(vols), self.total_canales_actual)):
                f_idx = idx // col
                c_idx = idx % col
                v = vols[idx]
                color = self.fg_green if v > 0.01 else "#222"
                x0 = 20 + c_idx * cell_w + 1
                y0 = 25 + f_idx * cell_h + 1
                self.canvas.create_rectangle(x0, y0, x0 + cell_w - 2, y0 + cell_h - 2, fill=color, outline="")

        elif modo_key == "vis_bark":
            num_bandas_bark = 24  
            ancho_banda = (ancho - 40) / num_bandas_bark
            energia_barks = np.zeros(num_bandas_bark, dtype=np.float32)
            for idx, f in enumerate(freqs[:self.canales_tonales_actual]):
                if f > MIN_FREQ and vols[idx] > 0:
                    bk = escala_bark(f)
                    idx_bk = int(np.clip(bk, 0, num_bandas_bark - 1))
                    energia_barks[idx_bk] = max(energia_barks[idx_bk], vols[idx])
            for b in range(num_bandas_bark):
                h_b = min(alto - 50, energia_barks[b] * (alto - 70) * 2.3)
                x0 = 20 + b * ancho_banda + 2
                self.canvas.create_rectangle(x0, alto - 30 - h_b, x0 + ancho_banda - 2, alto - 30, fill=self.fg_amber, outline="")
            self.canvas.create_line(16, alto-30, ancho-16, alto-30, fill="#555", width=2)

        elif modo_key == "vis_oscilloscope_scroll":
            canales_visibles = min(16, self.total_canales_actual)
            alto_pista = (alto - 40) / canales_visibles
            margen_izq = 90
            for ch_idx in range(canales_visibles):
                y_centro = 20 + ch_idx * alto_pista + (alto_pista / 2)
                hz = freqs[ch_idx]
                etiqueta = f"OSC_{ch_idx+1:03d}\n{int(hz)}Hz"
                self.canvas.create_rectangle(2, 20 + ch_idx * alto_pista, ancho - 4, 20 + (ch_idx + 1) * alto_pista, fill="#161a1e", outline="#333")
                self.canvas.create_text(10, y_centro, text=etiqueta, anchor="w", fill=self.fg_amber, font=("Consolas", 7, "bold"))
                v_comp = vols[ch_idx]
                pts = []
                for i in range(0, 100, 5):
                    x = margen_izq + i * (ancho - margen_izq - 20) / 100.0
                    y = y_centro - np.sin(i * 0.4) * v_comp * (alto_pista * 0.4)
                    pts.extend((x, y))
                if len(pts) >= 4 and v_comp > 0.01:
                    self.canvas.create_line(*pts, fill=self.fg_green, width=1)

if __name__ == "__main__":
    r = tk.Tk(); app = ChtnStudioApp(r); r.mainloop()