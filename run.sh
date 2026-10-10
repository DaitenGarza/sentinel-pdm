#!/usr/bin/env bash
# Sentinel PdM - start the dashboard on macOS / Linux.   Usage: ./run.sh
#
# The first time, it runs setup.sh for you (install + model training). Then it
# starts the dashboard on the first free port from 8501 to 8510 and opens it in
# your browser. Press Ctrl+C to stop it.
#
#   SENTINEL_NONINTERACTIVE=1   do not open a browser (CI, scripts)
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Pin the data/ and artifacts/ folders to this checkout (see setup.sh).
export SENTINEL_PDM_HOME="$PWD"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
VENV_PY=".venv/bin/python"
INTERACTIVE=1
if [ "${SENTINEL_NONINTERACTIVE:-0}" != 0 ] || [ "${CI:-}" = true ]; then INTERACTIVE=0; fi

case "${1:-}" in
  # BASH_SOURCE, not $0: after the cd above a relative $0 no longer points at this file.
  -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]##*/}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

# --- 1. Python environment: a quick import probe catches a missing .venv and a
#        half-finished install alike; setup.sh repairs either. --------------------------
probe='import importlib.util as u, sys
sys.exit(0 if all(u.find_spec(m) for m in ("sentinel_pdm", "streamlit", "sklearn", "pandas", "plotly", "pyarrow")) else 1)'
if ! { [ -x "$VENV_PY" ] && "$VENV_PY" -c "$probe" >/dev/null 2>&1; }; then
  echo "Sentinel PdM is not installed yet - running setup.sh first..."
  bash ./setup.sh --from-run || { echo "[ERROR] Setup did not complete, so the dashboard cannot start yet." >&2; exit 1; }
fi

# --- 2. Trained models (setup.sh normally creates them). -------------------------------
if [ ! -f artifacts/metrics.json ] || [ ! -f artifacts/predictions.parquet ]; then
  echo
  echo "No trained models found - simulating the plant data and training them now."
  echo "This takes about 2-3 minutes and is only needed once..."
  "$VENV_PY" -m sentinel_pdm train || { echo "[ERROR] Training the models failed - see the messages above." >&2; exit 1; }
fi

# --- 3. Free port: nothing answers on it AND it can be bound. --------------------------
PORT="$("$VENV_PY" - <<'PY'
import socket

for port in range(8501, 8511):
    with socket.socket() as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            continue  # another server is already listening here
    try:
        with socket.socket() as sock:
            # Like the web server itself, ignore sockets of a just-stopped run in TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
    except OSError:
        continue  # bound by another process or reserved
    print(port)
    break
PY
)"
if [ -z "$PORT" ]; then
  echo "[ERROR] Ports 8501 to 8510 are all in use. Stop other dashboards or Streamlit apps and try again." >&2
  exit 1
fi
URL="http://localhost:$PORT"

# --- 4. Open the browser once the server reports healthy (opening it earlier shows a
#        "connection refused" page). The poller is a single background process, so the
#        EXIT trap can stop it cleanly when the dashboard stops first.
POLLER=""
if [ "$INTERACTIVE" = 1 ]; then
  "$VENV_PY" - "$PORT" "$URL" <<'PY' &
import shutil
import subprocess
import sys
import time
import urllib.request

port, url = sys.argv[1], sys.argv[2]
health = f"http://127.0.0.1:{port}/_stcore/health"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy
for _ in range(240):
    try:
        with opener.open(health, timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        pass
    time.sleep(0.5)
else:
    sys.exit(1)
# macOS has "open"; on Linux "open" may be an unrelated tool (openvt), so use xdg-open.
launcher = "open" if sys.platform == "darwin" else "xdg-open"
if shutil.which(launcher):
    subprocess.Popen([launcher, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PY
  POLLER=$!
fi
cleanup() { if [ -n "$POLLER" ]; then kill "$POLLER" 2>/dev/null || true; fi; }
trap cleanup EXIT

echo
echo " =================================================================="
echo "  Sentinel PdM is running at $URL"
if [ "$INTERACTIVE" = 1 ]; then echo "  Your browser opens by itself in a few seconds."; fi
echo "  Press Ctrl+C to stop."
echo " =================================================================="
echo
# Listening on localhost only keeps the dashboard private to this machine (and avoids the
# macOS "accept incoming network connections?" firewall prompt).
"$VENV_PY" -m streamlit run app/Home.py --server.address localhost --server.port "$PORT" \
  --server.headless true --browser.gatherUsageStats false
