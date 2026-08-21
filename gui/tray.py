"""Silicon Node tray shell.

The GUI itself is the web dashboard the node serves at /ui (viewable from
any browser, any machine on the tailnet). This shell is just the Windows
presence: a status-colored tray icon and an "Open" that launches the
dashboard as a chromeless Edge app window.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import pystray
from PIL import Image, ImageDraw

APP_NAME = "Silicon Node"
NODE = "http://127.0.0.1:8790"
UI = f"{NODE}/ui"
HUB = "https://memories.zamasu.dev/p/silicon-node"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

GREEN, AMBER, BLUE, RED = ("#43d17c", "#e0a458", "#7ba6e8", "#e06456")


def glyph(color: str) -> Image.Image:
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([12, 12, 52, 52], radius=8, fill=color,
                        outline="#10131a", width=2)
    for i in (20, 30, 40):
        d.rectangle([i, 4, i + 4, 12], fill="#10131a")
        d.rectangle([i, 52, i + 4, 60], fill="#10131a")
        d.rectangle([4, i, 12, i + 4], fill="#10131a")
        d.rectangle([52, i, 60, i + 4], fill="#10131a")
    return im


def get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def open_dashboard(*_):
    for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        try:
            subprocess.Popen([exe, f"--app={UI}"])
            return
        except OSError:
            continue
    webbrowser.open(UI)


def autostart_on() -> bool:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, APP_NAME)
        return True
    except OSError:
        return False


def toggle_autostart(icon, item):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                        winreg.KEY_SET_VALUE) as k:
        if item.checked:
            try:
                winreg.DeleteValue(k, APP_NAME)
            except OSError:
                pass
        else:
            winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ,
                              f'"{sys.executable}"')


def watcher(icon: pystray.Icon) -> None:
    while True:
        try:
            health = get(f"{NODE}/health")
            node = get(f"{NODE}/v1/node")
            llm = get(f"{NODE}/v1/llm")
            met = node.get("metrics", {})
            free = met.get("headroom_gb", "?")
            if health.get("queue_depth", 0) > 0:
                color, tip = AMBER, "Silicon Node — GPU rendering"
            elif llm.get("running"):
                color = BLUE
                tip = (f"Silicon Node — {llm.get('model')} loaded · "
                       f"{free} GB free")
            else:
                color, tip = GREEN, f"Silicon Node — idle · {free} GB free"
        except Exception:  # noqa: BLE001
            color, tip = RED, "Silicon Node — service unreachable"
        icon.icon = glyph(color)
        icon.title = tip
        time.sleep(5)


def main() -> None:
    # Single instance via a named mutex.
    import ctypes
    ctypes.windll.kernel32.CreateMutexW(None, False, "SiliconNodeTray")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        sys.exit(0)

    icon = pystray.Icon(
        APP_NAME, glyph(RED), APP_NAME,
        menu=pystray.Menu(
            pystray.MenuItem("Open Silicon Node", open_dashboard,
                             default=True),
            pystray.MenuItem("Open Memories hub",
                             lambda *_: webbrowser.open(HUB)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Start when I sign in", toggle_autostart,
                             checked=lambda item: autostart_on()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda ic, _: ic.stop()),
        ))
    threading.Thread(target=watcher, args=(icon,), daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()
