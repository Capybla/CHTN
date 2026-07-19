@echo off
title CHTN NewerStudio - Bootloader
color 0b
cls

echo ===================================================
echo    CHTN NEWERSTUDIO - CONTROL DE ARRANQUE HQ
echo ===================================================
echo.

:: 1. Verificar si Python está instalado en el sistema
echo [+] Verificando entorno de ejecucion Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no se encuentra en las variables de entorno (PATH^).
    echo Por favor, instala Python y marca la casilla "Add Python to PATH".
    pause
    exit
)

:: 2. Comprobar e instalar dependencias críticas de audio y análisis
echo [+] Validando librerias de espectro y hardware...
python -c "import librosa, sounddevice, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Detectadas librerias faltantes. Iniciando instalacion automatica de soporte HQ...
    echo [+] Esto solo ocurrira la primera vez, por favor espera...
    python -m pip install --upgrade pip
    python -m pip install librosa sounddevice numpy
    echo [+] Dependencias instaladas con exito.
    cls
    echo ===================================================
    echo    CHTN NEWERSTUDIO - CONTROL DE ARRANQUE HQ
    echo ===================================================
    echo.
)

:: 3. Lanzar el estudio de super-resolución
echo [+] Entorno optimizado correctamente.
echo [+] Ejecutando chtn_newerstudio.py en tiempo real...
echo.

python chtn_newerstudio.py

:: 4. Cierre seguro del hilo de consola tras cerrar la UI
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] El programa se ha cerrado con un codigo de retorno inusual (%errorlevel%^).
    pause
)
exit