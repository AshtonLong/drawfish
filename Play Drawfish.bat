@echo off
rem Launch the Drawfish GUI (play against the engine).
setlocal
set "DF_HOME=%DRAWFISH_HOME%"
if "%DF_HOME%"=="" set "DF_HOME=%LOCALAPPDATA%\Drawfish"
set "PYW=%DF_HOME%\venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
    echo Python environment not found at "%DF_HOME%\venv".
    echo Run setup.bat once first.
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0play.py" %*
