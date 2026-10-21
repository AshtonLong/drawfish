@echo off
rem One-time setup: creates the Python environment in %LOCALAPPDATA%\Drawfish\venv
rem (outside OneDrive) with CUDA PyTorch. cu126 wheels still support Pascal GPUs (GTX 10xx).
rem Pass "stockfish" to also download Stockfish (only needed to train or benchmark).
setlocal
set "DF_HOME=%DRAWFISH_HOME%"
if "%DF_HOME%"=="" set "DF_HOME=%LOCALAPPDATA%\Drawfish"
if not exist "%DF_HOME%" mkdir "%DF_HOME%"
if not exist "%DF_HOME%\venv\Scripts\python.exe" (
    py -3.12 -m venv "%DF_HOME%\venv" || python -m venv "%DF_HOME%\venv" || goto :fail
)
set "PY=%DF_HOME%\venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip || goto :fail
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :fail
"%PY%" -m pip install -r "%~dp0requirements.txt" || goto :fail
if /I "%~1"=="stockfish" (
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $d='%DF_HOME%\stockfish_dl'; New-Item -ItemType Directory -Force $d | Out-Null; Invoke-WebRequest 'https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip' -OutFile \"$d\sf.zip\"; Expand-Archive -Force \"$d\sf.zip\" $d" || goto :fail
)
"%PY%" -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
echo.
echo Setup complete. Double-click "Play Drawfish.bat" to play.
pause
exit /b 0
:fail
echo Setup failed.
pause
exit /b 1
