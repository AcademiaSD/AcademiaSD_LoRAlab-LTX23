@echo off
call venv\Scripts\activate.bat
echo ====================================================
echo   GESTOR DE ENTRENAMIENTO LORA	EN LOCAL LTX23
echo ====================================================

:menu
echo.
echo Selecciona una opcion:
echo 1. Pre-procesar Dataset (Convertir a Latentes)
echo 2. Iniciar / Continuar Entrenamiento LoRA
echo 3. Descargar LTX23 Raw
echo 4. Salir
echo 5. Convertir Raw a NF4
echo.

set /p opcion="Elige una opcion (1-4): "

if "%opcion%"=="1" goto preprocesar
if "%opcion%"=="2" goto entrenar
if "%opcion%"=="3" goto descarga
if "%opcion%"=="4" goto fin
if "%opcion%"=="5" goto convertir

:preprocesar
echo Ejecutando generacion de latentes...
python 1_pre_cache_krea2.py
pause
goto menu

:entrenar
echo Iniciando el entrenador (Si existe un punto de guardado, continuara)...
python 2_train_lora_krea2.py
pause
goto menu

:fix
echo Arreglando los keys del LoRA...
python 3_arreglar_lora.py
pause
goto menu

:descarga
echo Comenzando la descarga en Local...
python 4_descarga_LTX23.py
pause
goto menu

:convertir
echo Convertir Raw a NF4...
python 5_conversor_LTX23_NF4.py
pause
goto menu

:fin
echo Saliendo...