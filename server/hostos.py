"""Where the engines live relative to this service (Linux port, no hub).

Two deployments share this codebase:

- **windows-wsl2**: the service runs inside a WSL2 distro. Chat engines
  that ship as native Windows CUDA binaries (ninfer, llama-server, dsh's
  node) run on the Windows side, spawned through WSL interop, and are
  reached with interop curl.exe — a WSL socket cannot reach the Windows
  loopback, but an interop process's 127.0.0.1 IS the Windows loopback.
- **linux**: everything is one OS. Engines spawn as ordinary children
  and are reached over plain sockets; none of the interop machinery
  exists or is needed.

Detection is automatic (WSL leaves "microsoft" in /proc/version and sets
WSL_DISTRO_NAME); SILICON_NODE_HOST_OS=linux|windows-wsl2 overrides it,
which is also how the Linux paths are integration-tested from inside the
WSL distro.
"""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("silicon-node.hostos")


def _detect_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


_OVERRIDE = os.environ.get("SILICON_NODE_HOST_OS", "").strip().lower()
if _OVERRIDE in ("linux", "windows-wsl2"):
    IS_WSL = _OVERRIDE == "windows-wsl2"
else:
    IS_WSL = _detect_wsl()
MODE = "windows-wsl2" if IS_WSL else "linux"

_WIN_CURL = "/mnt/c/Windows/System32/curl.exe"


def win_path(p: Path | str) -> str:
    """/mnt/<drive>/rest -> <DRIVE>:\rest — the shape interop-spawned
    Windows binaries need, since interop translates the executable path
    but passes arguments verbatim."""
    s = str(p)
    if not (len(s) > 6 and s.startswith("/mnt/") and s[6] == "/"):
        raise ValueError(f"not under a /mnt drive root: {s}")
    return s[5].upper() + ":" + s[6:].replace("/", "\\")


def proxy_ip() -> str:
    """The source IP that means "forwarded, not the real sender": on WSL
    every Windows-proxied connection (portproxy, tailscale serve, the
    tray) arrives from the NAT gateway; on plain Linux only loopback
    forwarders (e.g. tailscale serve dialing localhost) obscure the
    origin."""
    if not IS_WSL:
        return "127.0.0.1"
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            f = line.split()
            if f[1] == "00000000":
                h = f[2]
                return ".".join(str(int(h[i:i + 2], 16))
                                for i in (6, 4, 2, 0))
    except Exception:  # noqa: BLE001
        pass
    return "127.0.0.1"


def http_status(url: str, timeout: float = 4.0) -> str:
    """HTTP status code as a string ("" on any failure). On WSL this
    rides interop curl so it can reach engines on the Windows loopback;
    on Linux it is a plain request."""
    if IS_WSL:
        try:
            out = subprocess.run(
                [_WIN_CURL, "-s", "-m", str(max(1, int(timeout))), "-o",
                 "NUL", "-w", "%{http_code}", url],
                capture_output=True, text=True, timeout=timeout + 3)
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return str(r.status)
    except urllib.error.HTTPError as exc:
        return str(exc.code)
    except Exception:  # noqa: BLE001
        return ""


def http_get(url: str, timeout: float = 5.0) -> str:
    """Response body as text ("" on any failure); same reachability
    story as http_status."""
    if IS_WSL:
        try:
            out = subprocess.run(
                [_WIN_CURL, "-s", "-m", str(max(1, int(timeout))), url],
                capture_output=True, text=True, timeout=timeout + 5)
            return out.stdout
        except Exception:  # noqa: BLE001
            return ""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def kill_by_name(image: str) -> None:
    """Kill every instance of an engine by executable name — the
    single-instance guarantee. On WSL the target is a *Windows* process
    (terminating the interop proxy handle does not kill it; taskkill by
    image name is the only reliable teardown). On Linux pkill -x matches
    the exact process name and never touches lookalikes."""
    try:
        if IS_WSL:
            subprocess.run(
                ["/mnt/c/Windows/System32/taskkill.exe", "/IM",
                 f"{image}.exe", "/F"],
                capture_output=True, timeout=20)
        else:
            subprocess.run(["pkill", "-x", image],
                           capture_output=True, timeout=20)
    except Exception:  # noqa: BLE001
        log.exception("kill_by_name(%s) failed", image)


def chat_spool_dir() -> Path:
    """Where the chat proxy spools request bodies for the bridge curl.
    On WSL it must be a Windows-visible directory (interop curl reads a
    Windows path); on Linux any tmpfs will do."""
    if IS_WSL:
        win_tmp = os.environ.get("SILICON_NODE_WIN_TMP", r"C:\Windows\Temp")
        return Path("/mnt/" + win_tmp[0].lower()
                    + win_tmp[2:].replace("\\", "/")) / "silicon-chat"
    return Path("/tmp/silicon-chat")


def bridge_curl_argv(url: str, spool_file: Path,
                     connect_s: int = 10, total_s: int = 1800) -> list[str]:
    """argv for the streaming chat bridge (hub 132/135 guards included:
    --connect-timeout bounds the dial, -m caps the whole exchange)."""
    curl = _WIN_CURL if IS_WSL else "curl"
    data = "@" + (win_path(spool_file) if IS_WSL else str(spool_file))
    return [curl, "-s", "-N", "-X", "POST",
            "--connect-timeout", str(connect_s), "-m", str(total_s),
            url, "-H", "Content-Type: application/json",
            "--data-binary", data]
