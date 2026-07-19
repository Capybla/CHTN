@echo off
setlocal
cd /d "%~dp0"

python chtn_studio_compacted.py
if errorlevel 1 (
    echo.
    echo CHTN Studio se cerro con un error.
    pause
)
