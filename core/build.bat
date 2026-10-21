@echo off
rem Rebuilds the Rust search core into drawfish\drawfish_core.dll (run after changing core\src).
rem Needs Rust (https://rustup.rs); build files go to %LOCALAPPDATA%\Drawfish\cargo-target.
setlocal
set "DF_HOME=%DRAWFISH_HOME%"
if "%DF_HOME%"=="" set "DF_HOME=%LOCALAPPDATA%\Drawfish"
set "CARGO=cargo"
where cargo >nul 2>nul || set "CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"
"%CARGO%" build --release --manifest-path "%~dp0Cargo.toml" --target-dir "%DF_HOME%\cargo-target" || goto :fail
copy /y "%DF_HOME%\cargo-target\release\drawfish_core.dll" "%~dp0..\drawfish\drawfish_core.dll" >nul || goto :fail
echo Built drawfish\drawfish_core.dll
exit /b 0
:fail
echo Build failed. If the DLL is in use, close Drawfish first.
exit /b 1
