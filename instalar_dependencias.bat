@echo off
title Instalar Dependencias - ForwardGuidanceAI
echo ========================================================
echo   Instalando dependencias necesarias de Python...
echo ========================================================
cd /d "%~dp0"

REM Probar pip estándar
pip install -r requirements.txt
if %ERRORLEVEL% equ 0 goto ok

REM Probar python -m pip
python -m pip install -r requirements.txt
if %ERRORLEVEL% equ 0 goto ok

REM Probar py launcher
py -m pip install -r requirements.txt
if %ERRORLEVEL% equ 0 goto ok

echo.
echo [ERROR] No se pudo ejecutar pip. Asegurate de tener Python instalado.
pause
exit /b 1

:ok
echo.
echo ========================================================
echo   Instalacion completada con exito!
echo   Ya puedes hacer doble clic en 'iniciar_app.bat'.
echo ========================================================
pause
