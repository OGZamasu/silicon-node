"""DeepSeek Harness (dsh) manager — the agentic chat sidecar.

Port of the Mac's HarnessRuntime: same pinned package, same isolated
DSH_HOME, same provider mechanics (openai-completions provider pointed at
the locally served model, contextWindow advertised so the harness budgets
compaction against reality). Adapted for this node:

- dsh runs on native Windows (spawned through WSL interop) because the
  model listens on Windows loopback.
- The provider's model id is the *real* engine id — ninfer validates the
  model field where llama-server ignores it.
- Node.js is a private portable runtime beside the app (v22, above the
  harness's 20.12 floor), never the machine's PATH node.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from . import config
from .llm import LLM, PORT as LLM_PORT

log = logging.getLogger("silicon-node.harness")

PACKAGE_SPEC = "@deepseek-ai/dsh@0.1.0-rc.7"   # pinned, like the Mac
PROVIDER_ID = "silicon-local"
API_KEY_VAR = "SILICON_LOCAL_API_KEY"
WEB_PORT = int(os.environ.get("SILICON_NODE_DSH_PORT", "8090"))

from . import hostos

_RUNTIME = Path(os.environ.get(
    "SILICON_NODE_RUNTIME",
    "/mnt/f/Windows Silicon Optimizer/silicon-node/runtime"
    if hostos.IS_WSL
    else str(Path(__file__).resolve().parent.parent / "runtime")))
NODE_EXE = _RUNTIME / "node" / ("node.exe" if hostos.IS_WSL
                                else "bin/node")
DSH_PKG = _RUNTIME / "dsh" / "node_modules" / "@deepseek-ai" / "dsh"
DSH_HOME_WSL = _RUNTIME / "dsh-home"


def _engine_path(p: Path) -> str:
    """A path as the NODE PROCESS must see it — Windows form through
    interop on WSL, POSIX on Linux."""
    return hostos.win_path(p) if hostos.IS_WSL else str(p)


def _dsh_entry_win() -> str | None:
    """The package's bin script as the node process must see it (see
    _engine_path — interop translates the executable path but never the
    arguments)."""
    import json  # noqa: PLC0415
    pj = DSH_PKG / "package.json"
    if not pj.is_file():
        return None
    data = json.loads(pj.read_text())
    bin_field = data.get("bin")
    rel = bin_field if isinstance(bin_field, str) else \
        next(iter(bin_field.values()), None) if bin_field else None
    if not rel:
        return None
    if not (DSH_PKG / rel).is_file():
        return None
    if hostos.IS_WSL:
        return hostos.win_path(DSH_PKG) + "\\" + rel.replace("/", "\\")
    return str(DSH_PKG / rel)


def _settings_yaml(model_id: str, model_name: str, context: int) -> str:
    return f"""# Managed by Silicon Node: the '{PROVIDER_ID}' provider below is how the
# harness reaches the model this node serves locally.
llm-pi-ai:
  providers:
    {PROVIDER_ID}:
      apiKeyEnv: {API_KEY_VAR}
      api: openai-completions
      baseURL: http://127.0.0.1:{LLM_PORT}/v1
      models:
        - id: {model_id}
          name: "{model_name}"
          contextWindow: {context}
"""


class HarnessManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._logfile = _RUNTIME / "dsh.log"
        self._probe: tuple[float, bool] | None = None

    @property
    def installed(self) -> bool:
        return NODE_EXE.exists() and _dsh_entry_win() is not None

    def healthy(self, timeout: float = 4.0, max_age: float = 0.0) -> bool:
        """The port is the truth: an interop proxy handle can die while
        the Windows process lives on, so never gate this on self._proc.
        Cached (max_age) because each probe spawns a Windows process and
        the dashboard polls constantly — a fresh spawn under load misses
        small timeouts and flaps the UI."""
        now = time.time()
        if max_age and self._probe and now - self._probe[0] < max_age:
            return self._probe[1]
        ok = hostos.http_status(
            f"http://127.0.0.1:{WEB_PORT}/",
            timeout).startswith(("2", "3"))
        self._probe = (now, ok)
        return ok

    @property
    def running(self) -> bool:
        return self.healthy(3, max_age=5)

    def status(self) -> dict:
        alive = self.healthy(3, max_age=5)
        return {
            "installed": self.installed,
            "running": alive,
            "healthy": alive,
            "url_hint": f"http://<this-host>:{WEB_PORT}/",
            "uptime_s": round(time.time() - self._started_at)
            if alive and self._started_at else None,
            "needs_llm": not LLM.running,
        }

    def start(self, wait_healthy_s: float = 90.0) -> None:
        with self._lock:
            if self.running:
                return
            if not self.installed:
                raise RuntimeError(
                    "The harness is not installed yet (portable Node + the "
                    "dsh package under silicon-node/runtime).")
            entry = _dsh_entry_win()
            DSH_HOME_WSL.mkdir(parents=True, exist_ok=True)
            model_id = getattr(LLM, "model_id", "qwen3.8-27b") or "qwen3.8-27b"
            context = 65536 if LLM._profile == "c1" else 32768
            (DSH_HOME_WSL / "settings.yaml").write_text(
                _settings_yaml(model_id, f"{model_id} (this PC)", context))

            env = dict(os.environ)
            env["DSH_HOME"] = _engine_path(DSH_HOME_WSL)
            env[API_KEY_VAR] = "local"
            if hostos.IS_WSL:
                # Only WSLENV-listed variables cross the interop boundary.
                env["WSLENV"] = (env.get("WSLENV", "").rstrip(":") +
                                 f":DSH_HOME:{API_KEY_VAR}").lstrip(":")
            log.info("starting dsh web on :%d (model %s)…", WEB_PORT,
                     model_id)
            self._kill_instances()
            logfh = open(self._logfile, "ab")  # noqa: SIM115
            self._proc = subprocess.Popen(
                [str(NODE_EXE), entry, "web", "--port", str(WEB_PORT)],
                stdout=logfh, stderr=subprocess.STDOUT,
                cwd=str(DSH_HOME_WSL), env=env)
            logfh.close()
            self._started_at = time.time()
        deadline = time.time() + wait_healthy_s
        while time.time() < deadline:
            if self.healthy():
                log.info("dsh healthy on :%d", WEB_PORT)
                return
            # A dead proxy handle is not proof of failure (the Windows
            # child can outlive it) — only the port decides, at deadline.
            time.sleep(2)
        tail = ""
        try:
            tail = self._logfile.read_text(errors="replace")[-300:]
        except OSError:
            pass
        self.stop()
        raise RuntimeError(
            f"dsh did not come up within {wait_healthy_s:.0f}s. "
            f"Log tail: {tail}")

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                self._proc = None
            self._kill_instances()
            self._started_at = None

    @staticmethod
    def _kill_instances() -> None:
        """Kill only *our* node (the dsh entry in its command line) —
        never every node process, the user runs Node too."""
        try:
            if hostos.IS_WSL:
                script = (
                    "Get-CimInstance Win32_Process -Filter "
                    "\"Name='node.exe'\""
                    " | Where-Object { $_.CommandLine -match 'deepseek-ai' }"
                    " | ForEach-Object "
                    "{ Stop-Process -Id $_.ProcessId -Force }")
                subprocess.run(
                    ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/"
                     "powershell.exe", "-NoProfile", "-Command", script],
                    capture_output=True, timeout=30)
            else:
                subprocess.run(
                    ["pkill", "-f", "deepseek-ai/dsh"],
                    capture_output=True, timeout=20)
        except Exception:  # noqa: BLE001
            log.exception("harness kill failed")


HARNESS = HarnessManager()
