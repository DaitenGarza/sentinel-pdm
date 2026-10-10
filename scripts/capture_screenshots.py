"""Regenerate the README screenshots in docs/images/ from the running dashboard.

Streamlit renders over a websocket after the page loads, so a plain ``chrome --screenshot``
captures an empty skeleton. This script drives headless Chrome (or Edge) over the DevTools
protocol instead: navigate, wait until the script has finished running and every Plotly chart
has drawn, perform a few UI actions where needed, size the viewport to the page, then capture.

It starts its own dashboard on a spare port and stops it afterwards. Needs Google Chrome or
Microsoft Edge; ``websockets`` is already installed as a Streamlit dependency.

    python scripts/capture_screenshots.py              # all six images
    python scripts/capture_screenshots.py asset_detail # just one
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images"
APP_PORT = 8795
CDP_PORT = 9335
WIDTH = 1600

# (output name, page URL path, UI actions before the capture)
SHOTS: list[tuple[str, str, list[dict]]] = [
    ("fleet_overview", "", []),
    ("asset_detail", "Asset_Detail", [
        {"do": "pick", "index": 0, "text": "P-703B"},   # a pump with a live bearing alert
        {"do": "click_text", "text": "365 d"},
    ]),
    ("model_performance", "Model_Performance", []),
    ("economics", "Maintenance_Economics", []),
    ("live_monitor", "Live_Monitor", [
        {"do": "click_button", "text": "Play"}, {"do": "sleep", "s": 60}, {"do": "click_button", "text": "Pause"},
    ]),
    ("real_data", "Real_Data_Benchmark", []),
]

READY_JS = """(() => {
  const status = document.querySelector('[data-testid="stStatusWidget"]');
  const main = document.querySelector('[data-testid="stMain"]');
  return {
    running: !!(status && /running|connecting/i.test(status.innerText || '')),
    busy: document.querySelectorAll('[data-testid="stSkeleton"], [data-testid="stSpinner"]').length,
    plots: document.querySelectorAll('[data-testid="stPlotlyChart"]').length,
    drawn: document.querySelectorAll('[data-testid="stPlotlyChart"] .main-svg').length,
    text: main ? main.innerText.length : 0,
    height: main ? main.scrollHeight : 0,
  };
})()"""


def _browser() -> str:
    candidates = [
        os.environ.get("CHROME", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chrome") or "",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("Google Chrome or Microsoft Edge is required (set the CHROME environment variable to its path).")


class DevTools:
    """Just enough of the Chrome DevTools protocol: call a method, evaluate JavaScript."""

    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        my_id = self.n
        self.ws.send(json.dumps({"id": my_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=120))
            if msg.get("id") == my_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JavaScript error: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    def wait_ready(self, timeout: float = 120, settle: float = 2.5) -> dict:
        """Wait until Streamlit is idle, every chart has drawn and the page has stopped changing."""
        t0, stable_since, last = time.time(), None, None
        while time.time() - t0 < timeout:
            st = self.js(READY_JS)
            ok = not st["running"] and not st["busy"] and st["text"] > 200 and st["drawn"] >= st["plots"]
            key = (st["plots"], st["drawn"], st["height"], st["text"])
            if ok and key == last:
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= settle:
                    return st
            else:
                stable_since = None
            last = key
            time.sleep(0.5)
        raise TimeoutError(f"page did not settle: {last}")


def _act(dt: DevTools, act: dict) -> None:
    kind = act["do"]
    if kind == "sleep":
        time.sleep(act["s"])
    elif kind == "click_text":  # tabs, segmented-control options...
        dt.js(f"""(() => {{
            const el = [...document.querySelectorAll('button, [role=tab], p')]
                .find(e => (e.innerText || '').trim() === {json.dumps(act['text'])} && e.offsetParent !== null);
            if (el) {{ el.scrollIntoView({{block: 'center'}}); el.click(); }}
        }})()""")
    elif kind == "click_button":  # icon buttons carry the icon name in their text: match the label suffix
        dt.js(f"""(() => {{
            const b = [...document.querySelectorAll('button')]
                .find(e => (e.innerText || '').trim().endsWith({json.dumps(act['text'])}) && e.offsetParent !== null);
            if (b && !b.disabled) b.click();
        }})()""")
    elif kind == "pick":  # type into the n-th selectbox and accept the first match
        box = dt.js(f"""(() => {{
            const e = document.querySelectorAll('[data-testid="stSelectbox"] input[role=combobox]')[{act['index']}];
            e.scrollIntoView({{block: 'center'}});
            const r = e.getBoundingClientRect();
            return [r.x + r.width / 2, r.y + r.height / 2];
        }})()""")
        for t in ("mousePressed", "mouseReleased"):
            dt.call("Input.dispatchMouseEvent", type=t, x=box[0], y=box[1], button="left", clickCount=1)
        time.sleep(0.5)
        dt.call("Input.insertText", text=act["text"])
        time.sleep(0.8)
        for t in ("keyDown", "keyUp"):
            dt.call("Input.dispatchKeyEvent", type=t, key="Enter", code="Enter", windowsVirtualKeyCode=13,
                    **({"text": "\r"} if t == "keyDown" else {}))
        dt.js("document.activeElement && document.activeElement.blur()")
    if kind != "sleep":
        time.sleep(1.0)
        dt.wait_ready()


def capture(dt: DevTools, name: str, path: str, actions: list[dict]) -> Path:
    dt.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=1100, deviceScaleFactor=1, mobile=False)
    dt.call("Page.navigate", url=f"http://localhost:{APP_PORT}/{path}")
    time.sleep(1.5)
    dt.wait_ready()
    for act in actions:
        _act(dt, act)
    st = dt.wait_ready()
    height = int(st["height"]) + 40
    dt.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height, deviceScaleFactor=1, mobile=False)
    time.sleep(1.0)
    dt.wait_ready(settle=2.0)  # Plotly re-lays out after the resize
    dt.js("(() => { const m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollTo(0, 0); })()")
    time.sleep(0.5)
    png = base64.b64decode(dt.call("Page.captureScreenshot", format="png")["data"])
    out = OUT_DIR / f"{name}.png"
    out.write_bytes(png)
    print(f"  {out.relative_to(ROOT)}  ({WIDTH}x{height})")
    return out


def _wait_http(url: str, timeout: float, expect: str | None = None) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            body = urllib.request.urlopen(url, timeout=2).read().decode()
            if expect is None or body.strip() == expect:
                return body
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} did not answer within {timeout:.0f} s")


def main(names: list[str]) -> None:
    if not (ROOT / "artifacts" / "metrics.json").exists():
        sys.exit("Train the models first: python -m sentinel_pdm train")
    shots = [s for s in SHOTS if not names or s[0] in names]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py"), "--server.port", str(APP_PORT),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    profile = tempfile.mkdtemp(prefix="sentinel-shots-")
    browser = None
    try:
        _wait_http(f"http://localhost:{APP_PORT}/_stcore/health", 90, expect="ok")
        browser = subprocess.Popen(
            [_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={CDP_PORT}",
             f"--user-data-dir={profile}", f"--window-size={WIDTH},1100", "--force-device-scale-factor=1",
             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tabs = json.loads(_wait_http(f"http://127.0.0.1:{CDP_PORT}/json/list", 30))
        ws_url = next(t for t in tabs if t.get("type") == "page")["webSocketDebuggerUrl"]
        with connect(ws_url, max_size=2**28, open_timeout=30) as ws:
            dt = DevTools(ws)
            dt.call("Page.enable")
            dt.call("Runtime.enable")
            dt.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
            print(f"Capturing {len(shots)} screenshot(s):")
            for name, path, actions in shots:
                capture(dt, name, path, actions)
    finally:
        for proc in (browser, app):
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
