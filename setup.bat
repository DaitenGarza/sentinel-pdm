@echo off
rem ============================================================================
rem  Sentinel PdM - one-time setup for Windows. Double-click this file.
rem
rem    [1/4] find a 64-bit Python 3.10+ (offers to install Python 3.12 with winget)
rem    [2/4] create the virtual environment .venv and install the project into it
rem    [3/4] simulate the plant data and train every model (about 2-3 minutes)
rem    [4/4] real-data benchmark on the UCI AI4I 2020 dataset (needs internet)
rem
rem  Re-running is safe: a healthy .venv is reused, a broken one is rebuilt and
rem  the models are retrained from scratch.
rem
rem  Options  --from-run                  used by run.bat: keep existing results
rem                                       and do not pause at the end
rem  Env      SENTINEL_NONINTERACTIVE=1   no prompts and no pause (CI, scripts)
rem  Exit     0 = success, 1 = failure (the message above it says why)
rem ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" 2>nul || pushd "%~dp0" || (echo [ERROR] Cannot open the project folder "%~dp0". & exit /b 1)

rem Capture everything path-like BEFORE delayed expansion is switched on, so a "!"
rem in a folder name cannot be mangled. SENTINEL_PDM_HOME pins the pipeline's
rem data\ and artifacts\ folders to this checkout, whatever the caller's settings.
set "SENTINEL_PDM_HOME=%CD%"
set "PYTHONIOENCODING=utf-8"
rem Unbuffered: training progress streams live even when the output is piped to a log.
set "PYTHONUNBUFFERED=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_PY=.venv\Scripts\python.exe"
rem 64-bit is required: pyarrow (Parquet files) publishes no 32-bit Windows wheels.
set "CHECK=import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)"
rem Same import probe as run.bat: can this environment train and serve as it is?
set "PROBE=import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ('sentinel_pdm', 'streamlit', 'sklearn', 'pandas', 'plotly', 'pyarrow')) else 1)"
setlocal EnableDelayedExpansion

set "FROM_RUN=0"
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--from-run" (set "FROM_RUN=1" & shift & goto :parse_args)
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
if "%~1"=="/?" goto :usage
echo [ERROR] Unknown option "%~1". Run "setup.bat --help" for the options.
exit /b 2
:args_done

set "INTERACTIVE=1"
if defined SENTINEL_NONINTERACTIVE if not "!SENTINEL_NONINTERACTIVE!"=="0" set "INTERACTIVE=0"
if /i "!CI!"=="true" set "INTERACTIVE=0"

echo.
echo  ==================================================================
echo   Sentinel PdM setup - predictive maintenance for centrifugal pumps
echo   Folder: !CD!
echo  ==================================================================

rem ---------------------------------------------------------------------------
rem [1/4] Python. A healthy .venv from an earlier run is reused as is, so a
rem re-run needs no system Python at all.
rem ---------------------------------------------------------------------------
echo.
echo [1/4] Looking for Python 3.10 or newer (64-bit)...
set "VENV_STATE=missing"
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import pip; !CHECK!" >nul 2>&1
    if errorlevel 1 (set "VENV_STATE=broken") else (set "VENV_STATE=ok")
) else if exist ".venv\" (
    set "VENV_STATE=broken"
)
if "%VENV_STATE%"=="ok" (
    "%VENV_PY%" -c "import platform; print('      Found Python ' + platform.python_version() + ' in the existing virtual environment .venv')"
    echo.
    echo [2/4] Updating the packages in the existing virtual environment .venv ...
    goto :install
)

set "WINGET_TRIED=0"
:find_python
rem Candidates in order of preference. Each one must actually run and pass the
rem version check, which also rules out the Microsoft Store "python.exe" alias:
rem that stub only prints an install hint and exits with an error.
set "PY="
for %%C in ("py -3.12" "py -3.11" "py -3.10" "py -3" "python") do if not defined PY (
    %%~C -c "!CHECK!" >nul 2>&1 && set "PY=%%~C"
)
rem Standard install folders, for a Python that is installed but not on PATH
rem (for example right after the winget install below, in this same window).
if not defined PY for %%V in (312 311 310 313 314) do if not defined PY (
    for %%D in ("%LOCALAPPDATA%\Programs\Python\Python%%V" "%ProgramFiles%\Python%%V") do if not defined PY (
        if exist "%%~D\python.exe" ( "%%~D\python.exe" -c "!CHECK!" >nul 2>&1 && set "PY="%%~D\python.exe"" )
    )
)
if not defined PY goto :no_python
%PY% -c "import platform, sys; print('      Found Python ' + platform.python_version() + ' at ' + sys.executable)"

rem ---------------------------------------------------------------------------
rem [2/4] Virtual environment and packages.
rem ---------------------------------------------------------------------------
echo.
echo [2/4] Creating the virtual environment .venv and installing packages...
if "%VENV_STATE%"=="broken" (
    echo       The existing .venv is broken or incomplete - rebuilding it.
    rmdir /s /q ".venv" 2>nul
    if exist ".venv\" (
        echo.
        echo [ERROR] Could not delete the old .venv folder. Close any Sentinel PdM window
        echo         ^(run.bat^) or editor that is using it, then run setup.bat again.
        goto :fail
    )
)
%PY% -m venv .venv
if errorlevel 1 (
    echo.
    echo [ERROR] Python could not create the virtual environment in .venv - see above.
    goto :fail
)
if not exist "%VENV_PY%" (
    echo.
    echo [ERROR] The virtual environment was created but .venv\Scripts\python.exe is missing.
    goto :fail
)

:install
echo       Installing Sentinel PdM and its packages (the first time this downloads
echo       about 150 MB and takes a few minutes)...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 echo [WARN] Could not upgrade pip - continuing with the installed version.
"%VENV_PY%" -m pip install -e .
if not errorlevel 1 goto :train
rem Even an up-to-date install needs PyPI (pip builds the editable package in an
rem isolated environment). A reused .venv that already has everything can still
rem train and serve, so an offline re-run must not be blocked by the update.
if "%VENV_STATE%"=="ok" (
    "%VENV_PY%" -c "!PROBE!" >nul 2>&1 && (
        echo [WARN] Could not update the packages - continuing with the ones already installed.
        goto :train
    )
)
echo.
echo [ERROR] Installing the Python packages failed - the messages above say why.
echo         Usual causes: no internet connection, a company proxy or firewall, or
echo         antivirus software locking files. Fix that and run setup.bat again.
goto :fail

rem ---------------------------------------------------------------------------
rem [3/4] Simulate the plant data and train + evaluate every model.
rem ---------------------------------------------------------------------------
:train
echo.
if "%FROM_RUN%"=="1" if exist "artifacts\metrics.json" if exist "artifacts\predictions.parquet" (
    echo [3/4] Trained models found in artifacts - keeping them. Run setup.bat to retrain.
    goto :benchmark
)
echo [3/4] Simulating one year of data for 48 pumps and training every model
echo       (about 2-3 minutes - progress is shown below)...
"%VENV_PY%" -m sentinel_pdm train
if errorlevel 1 (
    echo.
    echo [ERROR] Training the models failed - see the messages above.
    goto :fail
)

rem ---------------------------------------------------------------------------
rem [4/4] Real-data benchmark. Optional: everything else works without it, so a
rem failure here (typically no internet) is reported but does not fail setup.
rem ---------------------------------------------------------------------------
:benchmark
echo.
if "%FROM_RUN%"=="1" if exist "artifacts\ai4i_metrics.json" (
    echo [4/4] Real-data benchmark results found - keeping them.
    goto :done
)
echo [4/4] Real-data benchmark on the UCI AI4I 2020 dataset (downloads about 0.5 MB)...
"%VENV_PY%" -m sentinel_pdm ai4i
if errorlevel 1 (
    echo.
    echo [NOTE] The real-data benchmark could not run - usually this just means there is
    echo        no internet connection. Everything else is ready. To retry it later, run:
    echo        .venv\Scripts\python.exe -m sentinel_pdm ai4i
)

:done
echo.
echo  ==================================================================
if "%FROM_RUN%"=="1" (
    echo   Setup complete - starting the dashboard...
) else (
    echo   Setup complete. To open the dashboard, double-click run.bat
)
echo  ==================================================================
set "RC=0"
goto :finish

rem ---------------------------------------------------------------------------
rem No usable Python: explain, and offer a winget install when someone is there
rem to answer. After a successful install the search runs once more.
rem ---------------------------------------------------------------------------
:no_python
echo.
if "%WINGET_TRIED%"=="1" (
    echo [ERROR] Python was installed, but this window cannot see it yet.
    echo         Close this window and double-click setup.bat again.
    goto :fail
)
echo [ERROR] Could not find a 64-bit Python 3.10 or newer on this PC.
rem Say what "python" on PATH actually is, since that is what the user will check.
set "FIRST_PY="
for /f "delims=" %%W in ('where python 2^>nul') do if not defined FIRST_PY set "FIRST_PY=%%W"
if defined FIRST_PY if not "!FIRST_PY:WindowsApps=!"=="!FIRST_PY!" (
    echo         The "python" command on this PC is only the Microsoft Store shortcut, not a real install.
) else (
    "!FIRST_PY!" -c "import platform, struct, sys; print('        The python command on this PC is Python ' + platform.python_version() + ', ' + str(struct.calcsize('P') * 8) + '-bit, at ' + sys.executable)" 2>nul
)
echo         Install Python 3.12 (64-bit) from https://www.python.org/downloads/
echo         and tick "Add python.exe to PATH" in the installer. Then run setup.bat again.
if "%INTERACTIVE%"=="0" goto :fail
where winget >nul 2>&1 || goto :fail
echo.
echo         Or let Windows install it for you now with winget (about 30 MB download).
choice /c YN /n /m "        Install Python 3.12 with winget now? [Y/N] "
if errorlevel 2 goto :fail
echo.
winget install -e --id Python.Python.3.12 --source winget
if errorlevel 1 (
    echo.
    echo [ERROR] winget could not install Python. Please install it from
    echo         https://www.python.org/downloads/ and run setup.bat again.
    goto :fail
)
set "WINGET_TRIED=1"
goto :find_python

:usage
echo Usage: setup.bat [--from-run]
echo.
echo   Installs Sentinel PdM into .venv, trains the models and runs the real-data
echo   benchmark. Double-click it once, then use run.bat to open the dashboard.
echo.
echo   --from-run                  keep existing results and do not pause (used by run.bat)
echo   SENTINEL_NONINTERACTIVE=1   environment variable: no prompts and no pause
exit /b 0

:fail
set "RC=1"
echo.
echo  Setup did not finish. Fix the problem described above and run setup.bat again.

:finish
if "%INTERACTIVE%"=="1" if "%FROM_RUN%"=="0" (
    echo.
    pause
)
exit /b %RC%
