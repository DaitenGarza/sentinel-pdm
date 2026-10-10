#!/usr/bin/env bash
# Sentinel PdM - one-time setup for macOS / Linux.   Usage: ./setup.sh [--from-run]
#
#   [1/4] find a 64-bit Python 3.10+
#   [2/4] create the virtual environment .venv and install the project into it
#   [3/4] simulate the plant data and train every model (about 2-3 minutes)
#   [4/4] real-data benchmark on the UCI AI4I 2020 dataset (needs internet)
#
# Re-running is safe: a healthy .venv is reused, a broken one is rebuilt and the
# models are retrained from scratch.
#
#   --from-run                  used by run.sh: keep existing results
#   SENTINEL_NONINTERACTIVE=1   accepted for parity with setup.bat (this script never prompts)
#   Exit code 0 = success, non-zero = failure (the message above it says why).
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Pin the pipeline's data/ and artifacts/ folders to this checkout.
export SENTINEL_PDM_HOME="$PWD"
export PYTHONIOENCODING=utf-8
# Unbuffered: training progress streams live even when the output is piped to a log.
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
VENV_PY=".venv/bin/python"
# 64-bit is required: pyarrow (Parquet files) publishes no 32-bit wheels.
CHECK='import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)'
# Same import probe as run.sh: can this environment train and serve as it is?
PROBE='import importlib.util as u, sys
sys.exit(0 if all(u.find_spec(m) for m in ("sentinel_pdm", "streamlit", "sklearn", "pandas", "plotly", "pyarrow")) else 1)'

FROM_RUN=0
for arg in "$@"; do
  case "$arg" in
    --from-run) FROM_RUN=1 ;;
    # BASH_SOURCE, not $0: after the cd above a relative $0 no longer points at this file.
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]##*/}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[ERROR] Unknown option '$arg'. Run ./setup.sh --help for the options." >&2; exit 2 ;;
  esac
done

fail() {
  echo >&2
  printf '[ERROR] %s\n' "$@" >&2
  echo "Setup did not finish. Fix the problem described above and run ./setup.sh again." >&2
  exit 1
}

echo
echo " =================================================================="
echo "  Sentinel PdM setup - predictive maintenance for centrifugal pumps"
echo "  Folder: $PWD"
echo " =================================================================="

# --- [1/4] Python. A healthy .venv from an earlier run is reused as is. ---------------
echo
echo "[1/4] Looking for Python 3.10 or newer (64-bit)..."
venv_state=missing
if [ -x "$VENV_PY" ]; then
  if "$VENV_PY" -c "import pip; $CHECK" >/dev/null 2>&1; then venv_state=ok; else venv_state=broken; fi
elif [ -e .venv ]; then
  venv_state=broken   # e.g. a half-created venv, or a Windows .venv copied over
fi

if [ "$venv_state" = ok ]; then
  "$VENV_PY" -c "import platform; print('      Found Python ' + platform.python_version() + ' in the existing virtual environment .venv')"
  echo
  echo "[2/4] Updating the packages in the existing virtual environment .venv ..."
else
  PY=""
  for cand in python3.12 python3.11 python3.10 python3.13 python3.14 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "$CHECK" >/dev/null 2>&1; then
      PY="$(command -v "$cand")"
      break
    fi
  done
  if [ -z "$PY" ]; then
    fail "Could not find a 64-bit Python 3.10 or newer." \
         "Install Python 3.12 from https://www.python.org/downloads/" \
         "  macOS:          brew install python@3.12" \
         "  Ubuntu/Debian:  sudo apt install python3 python3-venv  (22.04+ ships 3.10 or newer)" \
         "then run ./setup.sh again."
  fi
  "$PY" -c "import platform, sys; print('      Found Python ' + platform.python_version() + ' at ' + sys.executable)"

  # --- [2/4] Virtual environment and packages. ----------------------------------------
  echo
  echo "[2/4] Creating the virtual environment .venv and installing packages..."
  if [ "$venv_state" = broken ]; then
    echo "      The existing .venv is broken or incomplete - rebuilding it."
    rm -rf .venv || fail "Could not delete the old .venv folder. Stop any running dashboard and try again."
  fi
  "$PY" -m venv .venv \
    || fail "Python could not create the virtual environment in .venv (see above)." \
            "On Debian/Ubuntu install the venv module first: sudo apt install python3-venv"
  [ -x "$VENV_PY" ] || fail "The virtual environment was created but $VENV_PY is missing."
fi

echo "      Installing Sentinel PdM and its packages (the first time this downloads"
echo "      about 150 MB and takes a few minutes)..."
"$VENV_PY" -m pip install --upgrade pip || echo "[WARN] Could not upgrade pip - continuing with the installed version."
if ! "$VENV_PY" -m pip install -e .; then
  # Even an up-to-date install needs PyPI (pip builds the editable package in an isolated
  # environment). A reused .venv that already has everything can still train and serve, so
  # an offline re-run must not be blocked by the update.
  if [ "$venv_state" = ok ] && "$VENV_PY" -c "$PROBE" >/dev/null 2>&1; then
    echo "[WARN] Could not update the packages - continuing with the ones already installed."
  else
    fail "Installing the Python packages failed - the messages above say why." \
         "Usual causes: no internet connection or a company proxy / firewall."
  fi
fi

# --- [3/4] Simulate the plant data and train + evaluate every model. ---------------------
echo
if [ "$FROM_RUN" = 1 ] && [ -f artifacts/metrics.json ] && [ -f artifacts/predictions.parquet ]; then
  echo "[3/4] Trained models found in artifacts/ - keeping them. Run ./setup.sh to retrain."
else
  echo "[3/4] Simulating one year of data for 48 pumps and training every model"
  echo "      (about 2-3 minutes - progress is shown below)..."
  "$VENV_PY" -m sentinel_pdm train || fail "Training the models failed - see the messages above."
fi

# --- [4/4] Real-data benchmark. Optional, so a failure (typically offline) is only a note.
echo
if [ "$FROM_RUN" = 1 ] && [ -f artifacts/ai4i_metrics.json ]; then
  echo "[4/4] Real-data benchmark results found - keeping them."
else
  echo "[4/4] Real-data benchmark on the UCI AI4I 2020 dataset (downloads about 0.5 MB)..."
  if ! "$VENV_PY" -m sentinel_pdm ai4i; then
    echo
    echo "[NOTE] The real-data benchmark could not run - usually this just means there is"
    echo "       no internet connection. Everything else is ready. To retry it later, run:"
    echo "       .venv/bin/python -m sentinel_pdm ai4i"
  fi
fi

echo
echo " =================================================================="
if [ "$FROM_RUN" = 1 ]; then
  echo "  Setup complete - starting the dashboard..."
else
  echo "  Setup complete. To open the dashboard, run:  ./run.sh"
fi
echo " =================================================================="
