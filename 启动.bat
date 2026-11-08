@echo off
setlocal
chcp 65001 >nul
title Hunyuan3D-Omni 可控 3D 生成
cd /d "%~dp0"
set "ROOT=%~dp0"

if not exist "%ROOT%venv\python.exe" goto :no_python
if exist "%ROOT%weights\Hunyuan3D-Omni\model\model_fp16.safetensors" goto :weights_ok
if exist "%ROOT%weights\Hunyuan3D-Omni\model\pytorch_model.bin" goto :weights_ok
goto :no_weights
:weights_ok

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONUTF8=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HY3D_DINO_PATH=%ROOT%weights\dinov2-large"
set "U2NET_HOME=%ROOT%weights\rembg"
set "PATH=%ROOT%venv;%ROOT%venv\Library\bin;%ROOT%venv\Scripts;%PATH%"

echo ============================================================
echo   Hunyuan3D-Omni 可控 3D 生成  -  启动中，请稍候...
echo   启动后会自动打开浏览器：http://127.0.0.1:7860
echo   关闭本窗口即退出程序
echo ============================================================

"%ROOT%venv\python.exe" -u "%ROOT%app.py" %*
echo.
echo 程序已退出。
pause
goto :eof

:no_python
echo [错误] 没有找到 venv\python.exe，整合包不完整。
pause
goto :eof

:no_weights
echo [错误] 没有找到主模型权重 weights\Hunyuan3D-Omni\model\model_fp16.safetensors
pause
goto :eof
