@echo off
setlocal EnableExtensions
title AcademiaSD - LTX-2.3 NF4 Converter

rem Los scripts usan rutas relativas a la raiz del LoRAlab (venv, LTX23-Raw, LTX23-NF4_weights).
cd /d "%~dp0.."
call venv\Scripts\activate.bat

:menu
echo.
echo ====================================================
echo   ACADEMIASD - LTX-2.3 NF4 CONVERTER
echo ====================================================
echo 1. Descargar LTX-2.3 Raw / Download LTX-2.3 Raw (.\LTX23-Raw)
echo 2. Convertir Raw a NF4 / Convert Raw to NF4 (.\LTX23-NF4_weights)
echo 3. Salir / Exit
echo.
set "opcion="
set /p opcion="Elige una opcion / Choose an option (1-3): "
rem Sin respuesta (Enter o entrada cerrada) se sale: evita repetir el menu sin fin.
if not defined opcion goto fin

if "%opcion%"=="1" goto descarga
if "%opcion%"=="2" goto convertir
if "%opcion%"=="3" goto fin
goto menu

:descarga
python tools\4_descarga_LTX23.py
pause
goto menu

:convertir
python tools\5_conversor_LTX23_NF4.py
pause
goto menu

:fin
endlocal
