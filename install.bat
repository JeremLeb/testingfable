@echo off
REM One-command setup for Windows. Double-click this file, or run it in a
REM terminal. Use "install.bat cpu" to force the CPU-only build.
REM Creates a local .venv, installs PyTorch + dependencies, checks your GPU.
setlocal
cd /d "%~dp0"

set CUDA_URL=https://download.pytorch.org/whl/cu124
set MODE=gpu
if /I "%1"=="cpu" set MODE=cpu

echo ==============================================
echo  Embodied Agent - installer (%MODE% build)
echo ==============================================

echo [1/4] Creating virtual environment (.venv) ...
py -3 -m venv .venv 2>nul || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

echo [2/4] Installing PyTorch ...
if "%MODE%"=="gpu" (
  echo       (CUDA build - large download, be patient^)
  pip install torch --index-url %CUDA_URL%
  if errorlevel 1 (
     echo       GPU build failed; falling back to the CPU build.
     pip install torch
  )
) else (
  pip install torch
)

echo [3/4] Installing the rest ^(numpy, matplotlib, pyyaml, imageio, pytest^) ...
pip install -r requirements.txt

echo [4/4] Checking your environment ...
python -m embodied_agent.doctor

echo.
echo Done. To watch the agent learn, double-click run.bat
pause
