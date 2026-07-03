@echo off
REM Open the live dashboard in your browser. Runs install.bat first if needed.
setlocal
cd /d "%~dp0"
if not exist .venv (
  echo First run - setting up ^(this happens only once^) ...
  call install.bat
)
call .venv\Scripts\activate.bat
python -m embodied_agent.gui %*
