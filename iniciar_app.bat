@echo off
title ForwardGuidanceAI - Servidor Local
echo ========================================================
echo   Iniciando ForwardGuidanceAI (Streamlit)...
echo ========================================================
cd /d "%~dp0"

REM 1. Probar comando python estándar
python -m streamlit run app.py
if %ERRORLEVEL% equ 0 goto fin

REM 2. Probar lanzador py de Windows
py -m streamlit run app.py
if %ERRORLEVEL% equ 0 goto fin

REM 3. Probar ruta Anaconda en perfil de usuario
if exist "%USERPROFILE%\anaconda3\python.exe" (
    "%USERPROFILE%\anaconda3\python.exe" -m streamlit run app.py
    goto fin
)

echo.
echo ========================================================
echo [ERROR] No se pudo iniciar Streamlit.
echo Si es la primera vez que abres el proyecto en este PC,
echo ejecuta primero 'instalar_dependencias.bat'.
echo ========================================================
pause

:fin
