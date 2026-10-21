@echo off
rem Drawfish as a UCI engine. Point your chess GUI (Arena, Cute Chess, Banksia, ...) at this file.
setlocal
set "DF_HOME=%DRAWFISH_HOME%"
if "%DF_HOME%"=="" set "DF_HOME=%LOCALAPPDATA%\Drawfish"
"%DF_HOME%\venv\Scripts\python.exe" "%~dp0uci.py"
