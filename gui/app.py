"""Silicon Node — the Windows counterpart of Silicon Optimizer for Mac.

Tray app + dashboard, ported from the Mac app's shape: sidebar navigation
(Dashboard / Chat / Models / Images / 3D / Settings), measured numbers
everywhere, warn-don't-refuse memory planning — over this box's stack
(ninfer CUDA LLM, TRELLIS.2 + LATO.2 3D in WSL, the swarm).

Run:    .venv\\Scripts\\pythonw.exe app.py
Build:  build.ps1  ->  dist\\SiliconNode\\SiliconNode.exe
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

from PySide6.QtCore import QDir, QLockFile, QObject, QSize, Qt, QThread, \
    QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QStackedWidget, QSystemTrayIcon,
    QVBoxLayout, QWidget,
)

APP_NAME = "Silicon Node"
VERSION = "0.2.0"
NODE = os.environ.get("SILICON_NODE_URL", "http://127.0.0.1:8790")
LLM_URL = os.environ.get("SILICON_NODE_LLM_URL", "http://127.0.0.1:8081")
TAILNET_URL = "http://100.118.191.121:8790"
LAN_URL = "http://192.168.4.23:8790"
HUB_URL = "https://memories.zamasu.dev/p/silicon-node"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
NINFER_MODELS_DIR = Path(r"F:\Windows Silicon Optimizer\ninfer-3090\dist"
                         r"\ninfer-rtx3090-windows-x64-0.6.0-rtx3090\models")
SWARM_JSON = Path(r"\\wsl$\SiliconNode\opt\silicon\swarm.json")

# ---------------------------------------------------------------------------
# Design tokens — one dark system, everything derives from these.
# ---------------------------------------------------------------------------

BG = "#0e1116"          # window ground
CARD = "#161a22"        # raised surface
CARD2 = "#1c212b"       # interactive surface
LINE = "#252b37"        # hairline
TEXT = "#e8ebf2"
MUTED = "#8b94a7"
ACCENT = "#39d98a"      # silicon green
AMBER = "#e8a13c"
BLUE = "#6c9bff"
RED = "#e5534b"

QSS = f"""
* {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; color: {TEXT};
     font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#sidebar {{ background: {CARD}; border-right: 1px solid {LINE}; }}
QListWidget#nav {{ background: transparent; border: none; outline: none;
                   font-size: 14px; }}
QListWidget#nav::item {{ padding: 10px 14px; margin: 2px 8px;
                         border-radius: 8px; color: {MUTED}; }}
QListWidget#nav::item:hover {{ background: {CARD2}; color: {TEXT}; }}
QListWidget#nav::item:selected {{ background: {CARD2}; color: {TEXT};
    border-left: 3px solid {ACCENT}; }}
QFrame.card {{ background: {CARD}; border: 1px solid {LINE};
               border-radius: 12px; }}
QLabel.h1 {{ font-size: 20px; font-weight: 600; }}
QLabel.h2 {{ font-size: 11px; font-weight: 600; color: {MUTED};
             letter-spacing: 1px; }}
QLabel.big {{ font-size: 26px; font-weight: 600; }}
QLabel.muted {{ color: {MUTED}; }}
QLabel.warn {{ color: {AMBER}; }}
QLabel.bad {{ color: {RED}; }}
QLabel.good {{ color: {ACCENT}; }}
QPushButton {{ background: {CARD2}; border: 1px solid {LINE};
    border-radius: 8px; padding: 7px 16px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton.primary {{ background: {ACCENT}; color: #08130c;
    font-weight: 600; border: none; }}
QPushButton.primary:hover {{ background: #5ce3a0; }}
QPushButton.danger {{ background: transparent; border-color: {RED};
    color: {RED}; }}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {{ background: {BG};
    border: 1px solid {LINE}; border-radius: 8px; padding: 7px 10px;
    selection-background-color: {ACCENT}; selection-color: #08130c; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background: {CARD2}; border: 1px solid
    {LINE}; selection-background-color: {LINE}; }}
QProgressBar {{ background: {BG}; border: 1px solid {LINE};
    border-radius: 7px; height: 14px; text-align: center;
    color: {MUTED}; font-size: 10px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QProgressBar.vram::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {ACCENT}, stop:1 {AMBER}); }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px;
    min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QFrame.bubble_user {{ background: {CARD2}; border: 1px solid {LINE};
    border-radius: 12px; }}
QFrame.bubble_ai {{ background: {CARD}; border: 1px solid {LINE};
    border-radius: 12px; }}
QFrame.hline {{ background: {LINE}; max-height: 1px; border: none; }}
"""


# ---------------------------------------------------------------------------
# HTTP + async plumbing
# ---------------------------------------------------------------------------

def _req(url: str, data: bytes | None = None, method: str = "GET",
         headers: dict | None = None, timeout: float = 8.0):
    h = {"User-Agent": "silicon-node-gui"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url, timeout=6.0, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return json.loads(_req(url, headers=h, timeout=timeout).decode())


def post_json(url, body, timeout=240.0, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return json.loads(_req(url, json.dumps(body or {}).encode(), "POST", h,
                           timeout).decode())


def post_multipart(url, field, path: Path, extra: dict, timeout=120.0):
    b = "----siliconnode"
    parts = []
    for k, v in extra.items():
        if v not in (None, ""):
            parts.append(f"--{b}\r\nContent-Disposition: form-data; "
                         f'name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name="
                 f'"{field}"; filename="{path.name}"\r\n'
                 "Content-Type: application/octet-stream\r\n\r\n".encode())
    parts += [path.read_bytes(), f"\r\n--{b}--\r\n".encode()]
    return json.loads(_req(url, b"".join(parts), "POST",
                           {"Content-Type":
                            f"multipart/form-data; boundary={b}"},
                           timeout).decode())


class Async(QObject):
    """Run fn() on a daemon thread; deliver (result, error) on the UI
    thread. Nothing in this app blocks the UI."""
    _done = Signal(object, object, object)

    def __init__(self):
        super().__init__()
        self._done.connect(lambda cb, res, err: cb(res, err))

    def run(self, fn, callback):
        def work():
            try:
                self._done.emit(callback, fn(), None)
            except Exception as exc:  # noqa: BLE001
                self._done.emit(callback, None, exc)
        threading.Thread(target=work, daemon=True).start()


ASYNC = Async()


def swarm_config() -> dict:
    try:
        return json.loads(SWARM_JSON.read_text())
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

class Trigger(QObject):
    fire = Signal()


class Poller(QObject):
    updated = Signal(dict)

    def __init__(self):
        super().__init__()
        self._jobs: list[str] = []
        self.downloads: dict[str, int] = {}   # filename -> expected bytes

    def track(self, job_id):
        if job_id not in self._jobs:
            self._jobs.append(job_id)

    def poll(self):
        s: dict = {"ok": False}
        try:
            s["health"] = get_json(f"{NODE}/health", 4)
            s["node"] = get_json(f"{NODE}/v1/node", 6)
            s["llm"] = get_json(f"{NODE}/v1/llm", 6)
            s["ok"] = True
        except Exception as exc:  # noqa: BLE001
            s["error"] = str(exc)
        s["jobs"] = []
        for jid in list(self._jobs[-12:]):
            try:
                s["jobs"].append(get_json(f"{NODE}/v1/jobs/{jid}", 5))
            except Exception:  # noqa: BLE001
                pass
        dl = {}
        for name, expected in self.downloads.items():
            p = NINFER_MODELS_DIR / name
            dl[name] = (p.stat().st_size if p.exists() else 0, expected)
        s["downloads"] = dl
        self.updated.emit(s)


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

def card() -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setProperty("class", "card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(18, 16, 18, 16)
    lay.setSpacing(10)
    return f, lay


def h2(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setProperty("class", "h2")
    return lbl


def label(text="", cls=None, wrap=False) -> QLabel:
    lbl = QLabel(text)
    if cls:
        lbl.setProperty("class", cls)
    lbl.setWordWrap(wrap)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


def stat_tile(title: str) -> tuple[QFrame, QLabel, QLabel]:
    f, lay = card()
    lay.addWidget(h2(title))
    big = label("—", "big")
    sub = label("", "muted")
    lay.addWidget(big)
    lay.addWidget(sub)
    return f, big, sub


def make_icon(color: str) -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(QColor("#10131a"))
    p.drawRoundedRect(12, 12, 40, 40, 8, 8)
    p.setBrush(QColor("#10131a"))
    for i in (20, 30, 40):
        p.drawRect(i, 4, 4, 8)
        p.drawRect(i, 52, 4, 8)
        p.drawRect(4, i, 8, 4)
        p.drawRect(52, i, 8, 4)
    p.end()
    return QIcon(pm)


ICONS = {"idle": ACCENT, "job": AMBER, "llm": BLUE, "down": RED}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)
        lay.addWidget(label("Dashboard", "h1"))

        row = QHBoxLayout()
        row.setSpacing(14)
        t1, self.big_head, self.sub_head = stat_tile("VRAM headroom")
        t2, self.big_util, self.sub_util = stat_tile("GPU")
        t3, self.big_queue, self.sub_queue = stat_tile("Job queue")
        t4, self.big_llm, self.sub_llm = stat_tile("Local LLM")
        for t in (t1, t2, t3, t4):
            row.addWidget(t, 1)
        lay.addLayout(row)

        vcard, vlay = card()
        vlay.addWidget(h2("VRAM — RTX 3090 Ti"))
        self.bar = QProgressBar()
        self.bar.setProperty("class", "vram")
        self.bar.setFormat("%v / %m MiB")
        vlay.addWidget(self.bar)
        self.lbl_svc = label("", "muted")
        vlay.addWidget(self.lbl_svc)
        lay.addWidget(vcard)

        ccard, clay = card()
        clay.addWidget(h2("Capabilities — measured, not promised"))
        self.caps_box = QVBoxLayout()
        clay.addLayout(self.caps_box)
        lay.addWidget(ccard)

        scard, slay = card()
        slay.addWidget(h2("Swarm"))
        self.lbl_peer = label("…", None, True)
        slay.addWidget(self.lbl_peer)
        lay.addWidget(scard)
        lay.addStretch(1)

    def apply(self, s: dict):
        if not s.get("ok"):
            self.lbl_svc.setText(f"service unreachable — {s.get('error','')[:90]}")
            self.big_head.setText("—")
            return
        met = s["node"].get("metrics", {})
        prof = s["node"].get("profile", {})
        self.big_head.setText(f"{met.get('headroom_gb','—')} GB")
        self.sub_head.setText("free on the card right now")
        self.big_util.setText(f"{met.get('gpu_util_pct','—')}%")
        self.sub_util.setText(f"driver {prof.get('driver','?')}")
        q = met.get("queue_depth", 0)
        self.big_queue.setText(str(q))
        self.sub_queue.setText("jobs waiting" if q else "idle")
        llm = s["llm"]
        if llm.get("running"):
            self.big_llm.setText(llm.get("model") or "running")
            self.sub_llm.setText(
                f"{llm.get('profile')} · "
                + ("healthy" if llm.get("healthy") else "starting…"))
        else:
            self.big_llm.setText("off")
            self.sub_llm.setText("3D jobs own the GPU")
        total, used = int(prof.get("vram_mb", 0)), int(met.get("vram_used_mb", 0))
        if total:
            self.bar.setMaximum(total)
            self.bar.setValue(used)
        h = s["health"]
        self.lbl_svc.setText(
            f"{h['server']['name']} v{h['server']['version']} · up "
            f"{h.get('uptime_s',0)} s · {NODE}")

        while self.caps_box.count():
            item = self.caps_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for c in s["node"].get("capabilities", []):
            line = QHBoxLayout()
            dot = label("●", "good" if c["ready"] else "bad")
            line.addWidget(dot)
            line.addWidget(label(c["id"]))
            meta = []
            if c.get("peak_vram_gb"):
                meta.append(f"{c['peak_vram_gb']} GB peak")
            if c.get("typical_seconds"):
                meta.append(f"~{round(c['typical_seconds'])} s")
            line.addStretch(1)
            line.addWidget(label(" · ".join(meta) or "unmeasured", "muted"))
            w = QWidget()
            w.setLayout(line)
            self.caps_box.addWidget(w)

        peers = s.get("node", {}).get("peers", [])
        if peers:
            self.lbl_peer.setText("   ".join(
                f"⬢ {p['name']} — {p['base_url']}" for p in peers))
        else:
            self.lbl_peer.setText("No peers registered.")


class ChatPage(QWidget):
    """Plain chat with the local model (the Mac's Chat tab, minus the
    agent harness for now)."""

    def __init__(self, poller: Poller):
        super().__init__()
        self.history: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)
        top = QHBoxLayout()
        top.addWidget(label("Chat", "h1"))
        top.addStretch(1)
        self.cmb_effort = QComboBox()
        self.cmb_effort.addItems(["low effort", "medium effort",
                                  "xhigh effort"])
        top.addWidget(self.cmb_effort)
        btn_clear = QPushButton("New chat")
        btn_clear.clicked.connect(self._clear)
        top.addWidget(btn_clear)
        lay.addLayout(top)
        self.lbl_model = label("", "muted")
        lay.addWidget(self.lbl_model)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        inner = QWidget()
        self.msgs = QVBoxLayout(inner)
        self.msgs.setSpacing(10)
        self.msgs.addStretch(1)
        self.scroll.setWidget(inner)
        lay.addWidget(self.scroll, 1)

        row = QHBoxLayout()
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Message the local model…")
        self.inp.returnPressed.connect(self._send)
        self.btn_send = QPushButton("Send")
        self.btn_send.setProperty("class", "primary")
        self.btn_send.clicked.connect(self._send)
        row.addWidget(self.inp, 1)
        row.addWidget(self.btn_send)
        lay.addLayout(row)

    def apply(self, s: dict):
        llm = s.get("llm", {})
        if llm.get("running") and llm.get("healthy"):
            self.lbl_model.setText(
                f"{llm.get('model')} · everything stays on this machine")
            self.inp.setEnabled(True)
        else:
            self.lbl_model.setText(
                "The model is not running — start it on the Models page.")
            self.inp.setEnabled(False)

    def _bubble(self, text: str, who: str):
        f = QFrame()
        f.setProperty("class", f"bubble_{who}")
        v = QVBoxLayout(f)
        v.setContentsMargins(14, 10, 14, 10)
        v.addWidget(label(("You" if who == "user" else "Qwen"), "h2"))
        body = label(text, None, True)
        v.addWidget(body)
        row = QHBoxLayout()
        if who == "user":
            row.addStretch(1)
            row.addWidget(f, 4)
        else:
            row.addWidget(f, 4)
            row.addStretch(1)
        w = QWidget()
        w.setLayout(row)
        self.msgs.insertWidget(self.msgs.count() - 1, w)
        QTimer.singleShot(50, lambda: self.scroll.verticalScrollBar()
                          .setValue(self.scroll.verticalScrollBar().maximum()))
        return body

    def _clear(self):
        self.history = []
        while self.msgs.count() > 1:
            item = self.msgs.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _send(self):
        text = self.inp.text().strip()
        if not text:
            return
        self.inp.clear()
        self._bubble(text, "user")
        self.history.append({"role": "user", "content": text})
        thinking = self._bubble("…", "ai")
        effort = ["low", "medium", "xhigh"][self.cmb_effort.currentIndex()]
        msgs = list(self.history)

        def call():
            return post_json(f"{LLM_URL}/v1/chat/completions", {
                "model": "local", "messages": msgs, "max_tokens": 2000,
                "reasoning_effort": effort}, timeout=300)

        def done(res, err):
            if err:
                thinking.setText(f"[{err}]")
                return
            reply = res["choices"][0]["message"]["content"].strip()
            thinking.setText(reply or "(empty reply)")
            self.history.append({"role": "assistant", "content": reply})

        ASYNC.run(call, done)


NINFER_CATALOG = [
    {"id": "qwen3.8-27b", "repo": "neroued/Qwen3.8-27B-NInfer",
     "file_gib": 16.96, "serve_gb": 20.5,
     "blurb": "The flagship. Reasoning modes, vision variant, 161 tok/s "
              "aggregate in c8."},
    {"id": "qwen3.6-27b", "repo": "neroued/Qwen3.6-27B-NInfer",
     "file_gib": 16.29, "serve_gb": 19.8,
     "blurb": "Previous generation, slightly smaller."},
    {"id": "qwen3.6-35b-a3b", "repo": "neroued/Qwen3.6-35B-A3B-NInfer",
     "file_gib": 20.84, "serve_gb": 24.2,
     "blurb": "MoE 35B. The big one."},
]


class ModelsPage(QWidget):
    """The Mac's Models tab, ported: catalog, install state, VRAM plan
    (warn-don't-refuse), load/unload."""

    def __init__(self, poller: Poller):
        super().__init__()
        self.poller = poller
        self.rows: dict[str, dict] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)
        lay.addWidget(label("Models", "h1"))
        lay.addWidget(label(
            "Language models run natively through ninfer-3090; the plan "
            "numbers are measured or derived from real file sizes, and a "
            "tight fit warns instead of refusing.", "muted", True))

        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems([
            "c1 — one request at a time, lowest latency",
            "c8 — up to 8 concurrent, highest throughput"])
        prow = QHBoxLayout()
        prow.addWidget(h2("Serving profile"))
        prow.addWidget(self.cmb_profile, 1)
        lay.addLayout(prow)

        for entry in NINFER_CATALOG:
            f, v = card()
            top = QHBoxLayout()
            top.addWidget(label(entry["id"]))
            fit = label("", "muted")
            top.addStretch(1)
            top.addWidget(fit)
            v.addLayout(top)
            v.addWidget(label(entry["blurb"], "muted", True))
            bar = QProgressBar()
            bar.hide()
            v.addWidget(bar)
            brow = QHBoxLayout()
            state = label("", "muted")
            brow.addWidget(state)
            brow.addStretch(1)
            btn = QPushButton("…")
            btn.setEnabled(False)
            brow.addWidget(btn)
            v.addLayout(brow)
            lay.addWidget(f)
            self.rows[entry["id"]] = {
                "entry": entry, "fit": fit, "state": state, "btn": btn,
                "bar": bar, "filename": None, "installed": False,
                "downloading": False}
            btn.clicked.connect(lambda _=None, e=entry: self._act(e))

        tcard, tlay = card()
        tlay.addWidget(h2("3D engines (in the SiliconNode WSL distro)"))
        tlay.addWidget(label(
            "TRELLIS.2-4B — image → dense textured mesh · installed · "
            "9.75 GB peak measured", None, True))
        tlay.addWidget(label(
            "LATO.2 — retopology to clean low-poly · installed · 5.9 GB "
            "peak, ~16 s warm", None, True))
        lay.addWidget(tcard)
        lay.addStretch(1)

    def _fit_text(self, serve_gb: float, headroom_gb) -> tuple[str, str]:
        # Warn-don't-refuse: state the number and what to expect.
        total_free = 24.0  # with LLM stopped the card frees up
        if serve_gb <= 21.0:
            return (f"fits — needs ~{serve_gb} GB of 24 GB", "good")
        return (f"tight — needs ~{serve_gb} GB; alongside the desktop "
                "expect shared-memory slowdown", "warn")

    def _act(self, entry: dict):
        row = self.rows[entry["id"]]
        if row["installed"]:
            if row.get("loaded"):
                ASYNC.run(lambda: post_json(f"{NODE}/v1/llm/stop", {}, 60),
                          lambda r, e: None)
            else:
                profile = "c1" if self.cmb_profile.currentIndex() == 0 else "c8"
                fname = row["filename"]
                row["btn"].setEnabled(False)
                row["state"].setText("starting — first tokens in ~1 min…")
                ASYNC.run(
                    lambda: post_json(f"{NODE}/v1/llm/start",
                                      {"profile": profile,
                                       "model_file": fname}, 300),
                    lambda r, e: row["state"].setText(
                        f"[{e}]" if e else ""))
        elif not row["downloading"]:
            self._download(entry)

    def _download(self, entry: dict):
        row = self.rows[entry["id"]]
        row["downloading"] = True
        row["state"].setText("resolving file…")

        def resolve():
            info = get_json(
                f"https://huggingface.co/api/models/{entry['repo']}", 15)
            names = [s["rfilename"] for s in info.get("siblings", [])
                     if s["rfilename"].endswith(".ninfer")]
            if not names:
                raise RuntimeError("no .ninfer file in the repo")
            return names[0]

        def got_name(name, err):
            if err:
                row["state"].setText(f"[{err}]")
                row["downloading"] = False
                return
            url = (f"https://huggingface.co/{entry['repo']}/resolve/main/"
                   f"{name}")
            NINFER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(
                ["curl.exe", "-L", "-C", "-", "--fail", "-s", "-o",
                 str(NINFER_MODELS_DIR / name), url],
                creationflags=subprocess.CREATE_NO_WINDOW)
            expected = int(entry["file_gib"] * (1 << 30))
            self.poller.downloads[name] = expected
            row["filename"] = name
            row["state"].setText("downloading…")

        ASYNC.run(resolve, got_name)

    def apply(self, s: dict):
        llm = s.get("llm", {})
        installed = set(llm.get("installed_models", []))
        active = llm.get("model") if llm.get("running") else None
        head = (s.get("node", {}).get("metrics", {})
                .get("headroom_gb", "?"))
        dl = s.get("downloads", {})
        for cap_id, row in self.rows.items():
            e = row["entry"]
            fname = row["filename"] or next(
                (n for n in installed
                 if n.replace(".ninfer", "").replace("_", ".")
                 .startswith(cap_id.replace("-", "."))), None)
            row["filename"] = fname
            row["installed"] = fname in installed if fname else False
            txt, cls = self._fit_text(e["serve_gb"], head)
            row["fit"].setText(txt)
            row["fit"].setProperty("class", cls)
            row["fit"].style().unpolish(row["fit"])
            row["fit"].style().polish(row["fit"])
            if fname and fname in dl:
                got, expected = dl[fname]
                if got >= expected * 0.999:
                    row["downloading"] = False
                    del self.poller.downloads[fname]
                    row["bar"].hide()
                else:
                    row["bar"].show()
                    row["bar"].setMaximum(100)
                    row["bar"].setValue(int(got / expected * 100))
                    row["state"].setText(
                        f"downloading — {got / (1<<30):.1f} / "
                        f"{expected / (1<<30):.1f} GiB")
            loaded = bool(active and fname and active ==
                          fname.replace(".ninfer", "").replace("_", "."))
            row["loaded"] = loaded
            row["btn"].setEnabled(not row["downloading"])
            if row["downloading"]:
                row["btn"].setText("Downloading…")
            elif loaded:
                row["btn"].setText("Unload")
                row["state"].setText("loaded and serving")
            elif row["installed"]:
                row["btn"].setText("Load")
                if not row["state"].text().startswith(("[", "starting")):
                    row["state"].setText(
                        f"installed · {e['file_gib']} GiB on disk")
            else:
                row["btn"].setText("Download")
                if not row["downloading"]:
                    row["state"].setText(f"{e['file_gib']} GiB download")


class ImagesPage(QWidget):
    """Image generation by swarm delegation: the Mac renders FLUX, this
    page submits and fetches over the shared-token registry."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)
        lay.addWidget(label("Images", "h1"))
        lay.addWidget(label(
            "This box has no image model installed yet — but the swarm "
            "does: generation is delegated to the Mac's FLUX (image-flux, "
            "advertised ready). Windows-local SDXL/FLUX is a planned "
            "capability.", "muted", True))
        f, v = card()
        self.inp = QPlainTextEdit()
        self.inp.setPlaceholderText("Describe the image…")
        self.inp.setFixedHeight(80)
        v.addWidget(self.inp)
        row = QHBoxLayout()
        self.btn = QPushButton("Generate on the Mac")
        self.btn.setProperty("class", "primary")
        self.btn.clicked.connect(self._go)
        self.lbl = label("", "muted", True)
        row.addWidget(self.btn)
        row.addWidget(self.lbl, 1)
        v.addLayout(row)
        lay.addWidget(f)
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setMinimumHeight(360)
        lay.addWidget(self.img, 1)

    def _go(self):
        prompt = self.inp.toPlainText().strip()
        if not prompt:
            return
        cfg = swarm_config()
        peers = cfg.get("peers", [])
        token = cfg.get("swarm_token")
        if not peers or not token:
            self.lbl.setText("No swarm registry on this machine.")
            return
        base = peers[0]["base_url"]
        self.btn.setEnabled(False)
        self.lbl.setText("submitting to the Mac…")

        def call():
            return post_json(f"{base}/v1/jobs",
                             {"capability": "image-flux",
                              "prompt": prompt}, 60, token=token)

        def done(res, err):
            self.btn.setEnabled(True)
            if err:
                self.lbl.setText(
                    "The Mac hasn't exposed image jobs to the swarm yet "
                    f"({str(err)[:80]}). Asked for on the hub — this "
                    "button goes live the moment it ships.")
                return
            self.lbl.setText(f"submitted: {res}")

        ASYNC.run(call, done)


class JobsPage(QWidget):
    def __init__(self, poller: Poller):
        super().__init__()
        self.poller = poller
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)
        lay.addWidget(label("3D", "h1"))

        f, v = card()
        v.addWidget(h2("New generation"))
        self.cmb = QComboBox()
        self.cmb.addItems(["Image → textured mesh + clean retopo "
                           "(TRELLIS.2 + LATO.2)",
                           "Existing mesh → clean retopo (LATO.2)"])
        v.addWidget(self.cmb)
        frow = QHBoxLayout()
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("input file…")
        btn_b = QPushButton("Browse…")
        btn_b.clicked.connect(self._browse)
        frow.addWidget(self.txt_file, 1)
        frow.addWidget(btn_b)
        v.addLayout(frow)
        crow = QHBoxLayout()
        crow.addWidget(label("Vertices", "muted"))
        self.spn = QSpinBox()
        self.spn.setRange(200, 5000)
        self.spn.setValue(2000)
        self.spn.setSingleStep(100)
        crow.addWidget(self.spn)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("seed (optional)")
        crow.addWidget(self.seed)
        crow.addStretch(1)
        self.btn_go = QPushButton("Generate")
        self.btn_go.setProperty("class", "primary")
        self.btn_go.clicked.connect(self._submit)
        crow.addWidget(self.btn_go)
        v.addLayout(crow)
        self.lbl_note = label("A running LLM is paused automatically while "
                              "the GPU renders.", "muted", True)
        v.addWidget(self.lbl_note)
        lay.addWidget(f)

        jf, jv = card()
        jv.addWidget(h2("Jobs — double-click a finished one to open"))
        self.jobs_box = QVBoxLayout()
        jv.addLayout(self.jobs_box)
        lay.addWidget(jf)
        lay.addStretch(1)
        self._job_rows: dict[str, dict] = {}

    def _browse(self):
        filt = ("Images (*.png *.jpg *.jpeg *.webp)"
                if self.cmb.currentIndex() == 0
                else "Meshes (*.glb *.obj *.ply *.gltf)")
        path, _ = QFileDialog.getOpenFileName(self, "Choose input", "", filt)
        if path:
            self.txt_file.setText(path)

    def _submit(self):
        path = Path(self.txt_file.text().strip('"'))
        if not path.is_file():
            self.lbl_note.setText("Pick an input file first.")
            return
        kind = self.cmb.currentIndex()
        ep = "/v1/image-to-mesh" if kind == 0 else "/v1/retopologize"
        field = "image" if kind == 0 else "mesh"
        extra = {"vert_num": str(self.spn.value()),
                 "seed": self.seed.text().strip()}
        self.btn_go.setEnabled(False)

        def done(res, err):
            self.btn_go.setEnabled(True)
            if err:
                self.lbl_note.setText(f"[{err}]")
                return
            self.poller.track(res["job_id"])

        ASYNC.run(lambda: post_multipart(f"{NODE}{ep}", field, path, extra),
                  done)

    def apply(self, s: dict):
        for j in s.get("jobs", []):
            jid = j["job_id"]
            if jid not in self._job_rows:
                f = QFrame()
                f.setProperty("class", "card")
                h = QHBoxLayout(f)
                h.setContentsMargins(12, 8, 12, 8)
                name = label(jid[-8:])
                st = label("", "muted")
                bar = QProgressBar()
                bar.setMaximum(100)
                bar.setFixedWidth(160)
                btn = QPushButton("Open")
                btn.hide()
                h.addWidget(name)
                h.addWidget(st, 1)
                h.addWidget(bar)
                h.addWidget(btn)
                self.jobs_box.insertWidget(0, f)
                self._job_rows[jid] = {"st": st, "bar": bar, "btn": btn,
                                       "urls": []}
                btn.clicked.connect(
                    lambda _=None, i=jid: self._open(i))
            row = self._job_rows[jid]
            status = j.get("status")
            stage = j.get("stage", "")
            if status == "done":
                row["st"].setText("done")
                row["bar"].setValue(100)
                row["urls"] = j.get("result_urls", [])
                row["btn"].show()
            elif status == "failed":
                row["st"].setText(f"failed — {j.get('error','')[:70]}")
            else:
                prog = j.get("progress")
                row["bar"].setValue(int((prog or 0) * 100))
                row["st"].setText(stage or "queued")

    def _open(self, jid: str):
        row = self._job_rows[jid]
        out = Path.home() / "Downloads" / "silicon-node"
        out.mkdir(parents=True, exist_ok=True)

        def fetch():
            for rel in row["urls"]:
                name = rel.rsplit("/", 1)[-1]
                (out / name).write_bytes(_req(f"{NODE}{rel}", timeout=120))
            return out

        ASYNC.run(fetch, lambda res, err: os.startfile(res) if res else None)


class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)
        lay.addWidget(label("Settings", "h1"))

        f, v = card()
        v.addWidget(h2("Endpoints"))
        for name, val in (("Service (local)", NODE),
                          ("Service (tailnet)", TAILNET_URL),
                          ("Service (LAN)", LAN_URL),
                          ("LLM OpenAI API", f"{LLM_URL}/v1"),
                          ("LLM tailnet", "http://100.118.191.121:8081/v1"),
                          ("Memories hub", HUB_URL)):
            row = QHBoxLayout()
            row.addWidget(label(name, "muted"))
            row.addStretch(1)
            row.addWidget(label(val))
            w = QWidget()
            w.setLayout(row)
            v.addWidget(w)
        lay.addWidget(f)

        f2, v2 = card()
        v2.addWidget(h2("Engines detected"))
        self.lbl_engines = label("…", None, True)
        v2.addWidget(self.lbl_engines)
        lay.addWidget(f2)

        f3, v3 = card()
        v3.addWidget(h2("Options"))
        self.chk = QCheckBox("Start Silicon Node when I sign in")
        self.chk.setChecked(self._autostart_enabled())
        self.chk.toggled.connect(self._set_autostart)
        v3.addWidget(self.chk)
        lay.addWidget(f3)
        lay.addWidget(label(
            f"{APP_NAME} v{VERSION} — Windows counterpart of Silicon "
            "Optimizer. Feature parity tracks the Mac app; Windows-native "
            "extras (ninfer CUDA serving, WSL 3D stack, swarm node) stay.",
            "muted", True))
        lay.addStretch(1)
        ASYNC.run(self._detect, lambda res, err: self.lbl_engines.setText(
            res or f"[{err}]"))

    @staticmethod
    def _detect():
        lines = []
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
            lines.append(f"CUDA — {out.stdout.strip()}")
        except Exception:  # noqa: BLE001
            lines.append("CUDA — nvidia-smi not found")
        exe = NINFER_MODELS_DIR.parent / "ninfer-serve.exe"
        lines.append(f"ninfer-3090 — {'present' if exe.exists() else 'missing'}"
                     f" · v0.6.0 · {NINFER_MODELS_DIR.parent}")
        try:
            out = subprocess.run(["wsl", "-d", "SiliconNode", "--", "echo",
                                  "ok"], capture_output=True, text=True,
                                 timeout=15,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
            lines.append("WSL SiliconNode — running (TRELLIS.2 + LATO.2, "
                         "conda lato2)" if "ok" in out.stdout
                         else "WSL SiliconNode — not responding")
        except Exception:  # noqa: BLE001
            lines.append("WSL SiliconNode — not responding")
        tok = "present" if swarm_config().get("swarm_token") else "absent"
        lines.append(f"swarm.json — {tok}")
        return "\n".join(lines)

    def _autostart_enabled(self) -> bool:
        try:
            import winreg  # noqa: PLC0415
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                winreg.QueryValueEx(k, APP_NAME)
            return True
        except OSError:
            return False

    def _set_autostart(self, on: bool):
        import winreg  # noqa: PLC0415
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                            winreg.KEY_SET_VALUE) as k:
            if on:
                exe = sys.executable
                target = (f'"{exe}" "{Path(__file__).resolve()}"'
                          if exe.lower().endswith(("python.exe",
                                                   "pythonw.exe"))
                          else f'"{exe}"')
                winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ, target)
            else:
                try:
                    winreg.DeleteValue(k, APP_NAME)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Main window + tray
# ---------------------------------------------------------------------------

NAV = [("Dashboard", "◱"), ("Chat", "◇"), ("Models", "▤"),
       ("Images", "▣"), ("3D", "⬢"), ("Settings", "⚙")]


class Main(QMainWindow):
    def __init__(self, poller: Poller):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1040, 720)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(190)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 18, 0, 12)
        brand = QHBoxLayout()
        brand.setContentsMargins(18, 0, 0, 0)
        logo = QLabel()
        logo.setPixmap(make_icon(ACCENT).pixmap(26, 26))
        brand.addWidget(logo)
        brand.addWidget(label(" Silicon Node"))
        brand.addStretch(1)
        bw = QWidget()
        bw.setLayout(brand)
        sv.addWidget(bw)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        for name, glyph in NAV:
            QListWidgetItem(f"{glyph}   {name}", self.nav)
        sv.addWidget(self.nav, 1)
        self.lbl_foot = label("", "muted")
        self.lbl_foot.setContentsMargins(18, 0, 0, 6)
        sv.addWidget(self.lbl_foot)
        h.addWidget(side)

        self.stack = QStackedWidget()
        self.page_dash = DashboardPage()
        self.page_chat = ChatPage(poller)
        self.page_models = ModelsPage(poller)
        self.page_images = ImagesPage()
        self.page_jobs = JobsPage(poller)
        self.page_settings = SettingsPage()
        for p in (self.page_dash, self.page_chat, self.page_models,
                  self.page_images, self.page_jobs, self.page_settings):
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(p)
            self.stack.addWidget(sc)
        h.addWidget(self.stack, 1)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self._dark_title_bar()

    def _dark_title_bar(self):
        try:
            hwnd = int(self.winId())
            v = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(v), 4)
        except Exception:  # noqa: BLE001
            pass

    def apply(self, s: dict):
        self.page_dash.apply(s)
        self.page_chat.apply(s)
        self.page_models.apply(s)
        self.page_jobs.apply(s)
        if s.get("ok"):
            met = s["node"].get("metrics", {})
            self.lbl_foot.setText(f"◉ {met.get('headroom_gb','?')} GB free")
        else:
            self.lbl_foot.setText("○ offline")


class App:
    def __init__(self):
        self.qt = QApplication(sys.argv)
        self.qt.setQuitOnLastWindowClosed(False)
        self.qt.setStyleSheet(QSS)
        f = QFont("Segoe UI")
        f.setPointSize(10)
        self.qt.setFont(f)

        self.poller = Poller()
        self.win = Main(self.poller)
        self.poller.updated.connect(self.win.apply)
        self.poller.updated.connect(self._tray_state)

        self.tray = QSystemTrayIcon(make_icon(ICONS["down"]))
        self.tray.setToolTip(APP_NAME)
        menu = QMenu()
        menu.setStyleSheet(QSS)
        a1 = QAction("Open Silicon Node")
        a1.triggered.connect(self._show)
        a2 = QAction("Open Memories hub")
        a2.triggered.connect(lambda: webbrowser.open(HUB_URL))
        a3 = QAction("Quit")
        a3.triggered.connect(self.qt.quit)
        menu.addAction(a1)
        menu.addAction(a2)
        menu.addSeparator()
        menu.addAction(a3)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self._show() if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()
        self._refs = (menu, a1, a2, a3)

        self._thread = QThread()
        self.poller.moveToThread(self._thread)
        self._thread.start()
        self.trigger = Trigger()
        self.trigger.fire.connect(self.poller.poll)
        self.timer = QTimer()
        self.timer.timeout.connect(self.trigger.fire.emit)
        self.timer.start(2500)
        self.trigger.fire.emit()

    def _show(self):
        self.win.show()
        self.win.raise_()
        self.win.activateWindow()

    def _tray_state(self, s: dict):
        if not s.get("ok"):
            key, tip = "down", "Silicon Node — service unreachable"
        elif any(j.get("status") == "running" for j in s.get("jobs", [])) \
                or s["health"].get("queue_depth", 0) > 0:
            key, tip = "job", "Silicon Node — GPU rendering"
        elif s["llm"].get("running"):
            met = s["node"].get("metrics", {})
            key = "llm"
            tip = (f"Silicon Node — {s['llm'].get('model')} loaded · "
                   f"{met.get('headroom_gb','?')} GB free")
        else:
            met = s["node"].get("metrics", {})
            key, tip = "idle", (f"Silicon Node — idle · "
                                f"{met.get('headroom_gb','?')} GB free")
        self.tray.setIcon(make_icon(ICONS[key]))
        self.tray.setToolTip(tip)

    def run(self):
        return self.qt.exec()


def _single_instance_lock():
    lock = QLockFile(QDir.tempPath() + "/silicon-node-gui.lock")
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        sys.exit(0)
    return lock


if __name__ == "__main__":
    _lock = _single_instance_lock()
    sys.exit(App().run())
