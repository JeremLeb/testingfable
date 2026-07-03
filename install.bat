@echo off
REM One-command setup for Windows. Double-click this file, or run it in a
REM terminal. Use "install.bat cpu" to force the CPU-only build.
REM Creates a local .venv, installs PyTorch + dependencies, checks your GPU.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set CUDA_URL=https://download.pytorch.org/whl/cu124
set MODE=gpu
if /I "%1"=="cpu" set MODE=cpu

echo ==============================================
echo  Embodied Agent - installer (%MODE% build)
echo ==============================================

echo [1/4] Creating virtual environment (.venv) ...
REM Prefer a Python that HAS CUDA PyTorch wheels (3.12/3.11/3.13). The newest
REM Python (e.g. 3.14) often has no GPU wheels yet, which silently forces CPU.
set "PYCMD="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PYCMD ( py -%%V -c "print(1)" >nul 2>&1 && set "PYCMD=py -%%V" )
)
if not defined PYCMD ( py -3 -c "print(1)" >nul 2>&1 && set "PYCMD=py -3" )
if not defined PYCMD set "PYCMD=python"
echo       Using Python: !PYCMD!
!PYCMD! -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

echo [2/4] Installing PyTorch ...
if "%MODE%"=="gpu" (
  echo       (CUDA build - large download, be patient^)
  pip install torch --index-url %CUDA_URL%
  if errorlevel 1 (
     echo.
     echo       !! Could not install the CUDA build - there may be no GPU wheel
     echo       !! for this Python version. Falling back to CPU-only PyTorch.
     echo       !! For GPU support, install Python 3.12 and re-run this script.
     echo.
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
