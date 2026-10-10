@echo off
rem ============================================================================
rem  Sentinel PdM - start the dashboard. Double-click this file.
rem
rem  The first time, it runs setup.bat for you (install + model training).
rem  Then it starts the dashboard on the first free port from 8501 to 8510 and
rem  opens it in your browser. Press Ctrl+C or close this window to stop it.
rem
rem  Env   SENTINEL_NONINTERACTIVE=1   no browser and no pause (CI, scripts)
rem  Exit  0 = dashboard stopped normally, 1 = it could not start or crashed
rem ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" 2>nul || pushd "%~dp0" || (echo [ERROR] Cannot open the project folder "%~dp0". & exit /b 1)

rem Pin the data\ and artifacts\ folders to this checkout (see setup.bat).
set "SENTINEL_PDM_HOME=%CD%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "VENV_PY=.venv\Scripts\python.exe"
set "INTERACTIVE=1"
if defined SENTINEL_NONINTERACTIVE if not "%SENTINEL_NONINTERACTIVE%"=="0" set "INTERACTIVE=0"
if /i "%CI%"=="true" set "INTERACTIVE=0"
title Sentinel PdM

rem ---------------------------------------------------------------------------
rem 1. Python environment. A quick import probe (no heavy imports) catches both a
rem    missing .venv and a half-finished install; setup.bat repairs either.
rem ---------------------------------------------------------------------------
set "READY=0"
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ('sentinel_pdm', 'streamlit', 'sklearn', 'pandas', 'plotly', 'pyarrow')) else 1)" >nul 2>&1 && set "READY=1"
)
if "%READY%"=="0" (
    echo Sentinel PdM is not installed yet - running setup.bat first...
    call "%CD%\setup.bat" --from-run
    if errorlevel 1 goto :setup_failed
)

rem ---------------------------------------------------------------------------
rem 2. Trained models. setup.bat normally creates them; this covers a deleted or
rem    interrupted artifacts folder.
rem ---------------------------------------------------------------------------
if exist "artifacts\metrics.json" if exist "artifacts\predictions.parquet" goto :find_port
echo.
echo No trained models found - simulating the plant data and training them now.
echo This takes about 2-3 minutes and is only needed once...
"%VENV_PY%" -m sentinel_pdm train
if errorlevel 1 goto :train_failed

rem ---------------------------------------------------------------------------
rem 3. Free port. A port counts as free when nothing answers on it AND it can be
rem    bound: the first test catches another dashboard already listening, the
rem    second catches ports Windows has reserved (Hyper-V / WSL exclusion ranges).
rem ---------------------------------------------------------------------------
:find_port
set "PORT="
for /l %%P in (8501,1,8510) do if not defined PORT (
    "%VENV_PY%" -c "import socket, sys; probe = socket.socket(); probe.settimeout(0.25); busy = probe.connect_ex(('127.0.0.1', %%P)) == 0; probe.close(); s = socket.socket(); s.bind(('0.0.0.0', %%P)); s.close(); sys.exit(1 if busy else 0)" >nul 2>&1 && set "PORT=%%P"
)
if not defined PORT goto :no_port
set "URL=http://localhost:%PORT%"

rem ---------------------------------------------------------------------------
rem 4. Start the dashboard in this window, and open the browser from a small
rem    background poller once the server reports healthy (opening it earlier
rem    shows a "connection refused" page). The poller shares this console, so
rem    Ctrl+C or closing the window stops it too.
rem ---------------------------------------------------------------------------
if "%INTERACTIVE%"=="1" (
    start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; [System.Net.WebRequest]::DefaultWebProxy = $null; for ($i = 0; $i -lt 240; $i++) { try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://127.0.0.1:%PORT%/_stcore/health').StatusCode -eq 200) { Start-Process '%URL%'; break } } catch {}; Start-Sleep -Milliseconds 500 }"
)
echo.
echo  ==================================================================
echo   Sentinel PdM is running at %URL%
if "%INTERACTIVE%"=="1" echo   Your browser opens by itself in a few seconds.
echo   Press Ctrl+C to stop (or just close this window).
echo  ==================================================================
echo.
rem Listening on localhost only keeps the dashboard private to this PC and avoids
rem the Windows Defender Firewall "allow access" prompt that listening on every
rem network interface triggers the first time.
"%VENV_PY%" -m streamlit run app\Home.py --server.address localhost --server.port %PORT% --server.headless true --browser.gatherUsageStats false
set "RC=%ERRORLEVEL%"
rem -1073741510 = 0xC000013A, the exit status of a process stopped with Ctrl+C.
if "%RC%"=="-1073741510" set "RC=0"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] The dashboard stopped with an error ^(exit code %RC%^) - see the messages above.
    goto :fail
)
echo.
echo Sentinel PdM stopped.
goto :finish

:setup_failed
echo.
echo [ERROR] Setup did not complete, so the dashboard cannot start yet.
echo         Fix the problem described above and double-click run.bat again.
goto :fail

:train_failed
echo.
echo [ERROR] Training the models failed - see the messages above.
goto :fail

:no_port
echo.
echo [ERROR] Ports 8501 to 8510 are all in use. Close other dashboards or Streamlit
echo         apps (or their command windows) and double-click run.bat again.
goto :fail

:fail
set "RC=1"

:finish
if "%INTERACTIVE%"=="1" (
    echo.
    pause
)
exit /b %RC%
