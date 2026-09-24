"""HyperQwen — a third chat engine: patched vLLM in Docker.

github.com/syv-ai/HyperQwen (Apache-2.0) is not a model: it is a pinned
vLLM 0.28.0 plus a patch series and a model-preparation pipeline, shipped
as a container that serves Qwen3.8-27B — the same checkpoint ninfer
already serves here — on one 24 GB card. Its claim on an RTX 3090 is
127 tok/s single-stream against the 35-60 we measure from ninfer, and
~1,035 tok/s aggregate at 64 concurrent, with 150k-262k context.

So this is an engine *choice*, not a new capability: the node already has
ninfer (native Windows, :8081) and llama.cpp (:8082), and this adds
:18020. It is off unless the owner turns it on, because it is the most
expensive of the three to have around — a 9.5 GB image, a ~20 GB one-time
requantization, and a container that wants the whole card.

Driven the same way as the other Windows-side engines: through interop.
The node lives in WSL, `docker.exe` and the container live on Windows, and
the served port is the Windows loopback that hostos' interop curl can
already reach. Nothing here needs Docker inside the distro.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("silicon-node.hyperqwen")

from . import hostos  # noqa: E402 (after the logger it configures)

PORT = int(os.environ.get("SILICON_NODE_HYPERQWEN_PORT", "18020"))
REPO = "https://github.com/syv-ai/HyperQwen"
# The checkout and the requantized weights both live under runtime/, which
# is gitignored — they are installed artifacts, not source.
CHECKOUT = Path(os.environ.get(
    "SILICON_NODE_HYPERQWEN_DIR",
    str(hostos.WIN_RUNTIME / "hyperqwen")
    if hostos.IS_WSL else "/opt/silicon/hyperqwen"))
DOCKER = os.environ.get(
    "SILICON_NODE_DOCKER",
    "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    if hostos.IS_WSL else "docker")
IMAGE = "ghcr.io/syv-ai/hyperqwen:latest"

MODES = ("single", "batch")
# The knobs from the project's .env.example that are worth exposing. Each
# maps to one line of the .env the container reads; anything not listed
# stays at the project's own default rather than being guessed at here.
SPECS = ("dflash2", "mtp")
CONTEXTS = ("fast", "long", "huge")

DEFAULTS: dict[str, object] = {
    # single = one or a few people chatting (127 tok/s single-stream);
    # batch = an API backend for many concurrent requests (~1,035 tok/s
    # aggregate at 64 concurrent). One GPU runs one mode at a time.
    "mode": "single",
    "spec": "dflash2",
    "context": "fast",       # fast ~64k, long ~150k, huge ~240k
    "prefix_cache": True,
    "dflash_tokens": 0,      # >0 turns on the quote/copy mode (try 15)
    "gpu_util": 0.93,
}


def _settings() -> dict:
    """The owner's choices, from the capability-settings store."""
    from .capsettings import CAPS  # noqa: PLC0415
    s = dict(DEFAULTS)
    s.update(CAPS.settings("hyperqwen") or {})
    return s


def enabled() -> bool:
    from .capsettings import CAPS  # noqa: PLC0415
    # Off unless the owner turns it on: it is the expensive engine.
    return bool((CAPS.settings("hyperqwen") or {}).get("enabled", False))


def _compose_argv(*args: str) -> list[str]:
    """docker compose, run from the checkout, as the ENGINE sees paths."""
    if hostos.IS_WSL:
        try:
            cwd = hostos.win_path(CHECKOUT)
        except ValueError:
            # docker.exe runs on Windows and cannot see the distro's own
            # filesystem, so the checkout has to live on a mounted drive.
            raise RuntimeError(
                f"The HyperQwen checkout is at {CHECKOUT}, which Windows "
                "cannot see. On this node it must sit under a mounted "
                "drive (/mnt/...) — set SILICON_NODE_HYPERQWEN_DIR to one."
            ) from None
    else:
        cwd = str(CHECKOUT)
    return [DOCKER, "compose", "--project-directory", cwd,
            "-f", f"{cwd}\\docker-compose.yml" if hostos.IS_WSL
            else f"{cwd}/docker-compose.yml", *args]


def _run(argv: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout)


class HyperQwenManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._probe: tuple[float, bool] | None = None
        self._started_at: float | None = None
        self.install_state: dict | None = None
        self.mode: str | None = None
        self.last_error: str | None = None

    # -- availability ------------------------------------------------------

    @staticmethod
    def docker_ready() -> tuple[bool, str]:
        """Docker Desktop has to be running; it is a user-launched app on
        Windows, so this is a normal state to report rather than an error."""
        exe = Path(DOCKER)
        if hostos.IS_WSL and not exe.exists():
            return False, "Docker Desktop is not installed on the host."
        try:
            out = _run([DOCKER, "info", "--format", "{{.ServerVersion}}"],
                       timeout=25)
        except Exception as exc:  # noqa: BLE001
            return False, f"docker did not answer: {type(exc).__name__}"
        if out.returncode != 0:
            return False, ("Docker Desktop is installed but not running — "
                           "start it, then load this engine again.")
        return True, out.stdout.strip()

    @property
    def checked_out(self) -> bool:
        return (CHECKOUT / "docker-compose.yml").is_file()

    def image_present(self) -> bool:
        try:
            out = _run([DOCKER, "image", "inspect", IMAGE], timeout=30)
            return out.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def prepared(self) -> bool:
        """The one-time model preparation downloads a ~19.5 GB W4A16
        checkpoint into ./models. Anything much smaller means it is still
        in flight or never ran."""
        models = CHECKOUT / "models"
        if not models.is_dir():
            return False
        total = 0
        for p in models.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
            if total > 15 * 1024**3:
                return True
        return False

    def installed(self) -> bool:
        return self.checked_out and self.image_present() and self.prepared()

    # -- install -----------------------------------------------------------

    def install_async(self) -> None:
        """Everything that must happen before the engine can serve, in
        one background job: the checkout, the 9.5 GB image, and the
        one-time model preparation (~19.5 GB of W4A16 weights). Model
        prep belongs here and not in start(): it is tens of minutes of
        downloading, and compose's `up` would otherwise sit blocked on
        the dependency with nothing to show for it."""
        if self.install_state and not self.install_state.get("error"):
            return
        self.install_state = {"stage": "starting", "error": None}

        def work() -> None:
            try:
                if not self.checked_out:
                    self.install_state = {"stage": "cloning HyperQwen",
                                          "error": None}
                    CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
                    if CHECKOUT.exists():
                        shutil.rmtree(CHECKOUT, ignore_errors=True)
                    out = _run(["git", "clone", "--depth", "1", REPO,
                                str(CHECKOUT)], timeout=600)
                    if out.returncode != 0:
                        raise RuntimeError(out.stderr[-200:] or "clone failed")
                self.write_env()
                if not self.image_present():
                    self.install_state = {
                        "stage": "pulling the image (9.5 GB)", "error": None}
                    out = _run(_compose_argv("--profile", "single", "pull"),
                               timeout=7200)
                    if out.returncode != 0 and not self.image_present():
                        raise RuntimeError(out.stderr[-200:] or "pull failed")
                if not self.prepared():
                    self.install_state = {
                        "stage": "preparing the model (~19.5 GB, once)",
                        "error": None}
                    # The project's own one-shot: downloads and lays out
                    # the W4A16 checkpoint under ./models.
                    out = _run(_compose_argv("run", "--rm", "prepare"),
                               timeout=14400)
                    if out.returncode != 0 and not self.prepared():
                        raise RuntimeError(
                            (out.stderr or out.stdout)[-200:]
                            or "model preparation failed")
                self.install_state = None
                log.info("HyperQwen installed and prepared")
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"[:200]
                self.install_state = {"stage": "failed", "error": msg}
                log.exception("HyperQwen install failed")
        threading.Thread(target=work, daemon=True,
                         name="hyperqwen-install").start()

    def write_env(self) -> dict:
        """Render the owner's settings into the .env the container reads.
        Returned for the status view — the API key is never included."""
        s = _settings()
        lines = [
            "# Written by silicon-node from the hyperqwen capability "
            "settings — edit them there, not here.",
            f"SPEC={s['spec']}",
            f"PREFIX_CACHE={1 if s['prefix_cache'] else 0}",
            # Docker Desktop on WSL2 aborts with a UVA error without this.
            "VLLM_WSL2_ENABLE_PIN_MEMORY=1",
            f"PORT={PORT}",
        ]
        if str(s["context"]) != "fast":
            lines.append(f"CTX={s['context']}")
        if int(s["dflash_tokens"] or 0) > 0:
            lines.append(f"DFLASH_TOKENS={int(s['dflash_tokens'])}")
        if float(s["gpu_util"]) != 0.93:
            lines.append(f"GPU_UTIL={s['gpu_util']}")
        # The port is bound on the host's loopback only; an API key would
        # be the project's own auth, which we do not need because nothing
        # but this node reaches the port. The node's own bearer still
        # guards /v1/chat/completions.
        (CHECKOUT / ".env").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")
        return s

    # -- health ------------------------------------------------------------

    def healthy(self, timeout: float = 4.0, max_age: float = 0.0) -> bool:
        now = time.time()
        if max_age and self._probe and now - self._probe[0] < max_age:
            return self._probe[1]
        ok = hostos.http_status(
            f"http://127.0.0.1:{PORT}/health", timeout).startswith("2")
        self._probe = (now, ok)
        return ok

    @property
    def running(self) -> bool:
        return self.healthy(3, max_age=5)

    def container_state(self) -> str | None:
        try:
            out = _run(_compose_argv("ps", "--format", "json"), timeout=30)
            if out.returncode != 0 or not out.stdout.strip():
                return None
            for line in out.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("State"):
                    return f"{row.get('Service')}: {row['State']}"
        except Exception:  # noqa: BLE001
            return None
        return None

    # -- control -----------------------------------------------------------

    def start(self, mode: str | None = None,
              wait_healthy_s: float = 2400.0) -> None:
        """Bring the container up. The first start also requantizes the
        model inside the container (~20 GB, once), which is why the
        default wait is generous."""
        with self._lock:
            if not enabled():
                raise RuntimeError(
                    "The HyperQwen engine is switched off on this node. "
                    "Turn it on in the Models page (or POST "
                    '/v1/capabilities/hyperqwen {"settings":'
                    '{"enabled":true}}) first.')
            ok, detail = self.docker_ready()
            if not ok:
                raise RuntimeError(detail)
            if not self.installed():
                self.install_async()
                raise RuntimeError(
                    "HyperQwen is installing — a 9.5 GB image and a "
                    "one-time ~19.5 GB model preparation. Watch the "
                    "Models page and load it again when it lands.")
            mode = (mode or str(_settings()["mode"])).lower()
            if mode not in MODES:
                raise ValueError(
                    f"mode must be one of {', '.join(MODES)} (got {mode!r}).")
            self.write_env()
            # One language engine per card, and this one wants all of it.
            from .llm import LLM  # noqa: PLC0415
            from .llamacpp import LLAMACPP  # noqa: PLC0415
            if LLM.running:
                LLM.stop()
            if LLAMACPP.running:
                LLAMACPP.stop()
            self._down_all()
            log.info("starting HyperQwen (%s mode) on :%d", mode, PORT)
            # --no-deps: preparation already ran during install, and
            # without this compose blocks `up` on the prepare service's
            # completion condition rather than returning.
            out = _run(_compose_argv("--profile", mode, "up", "-d",
                                     "--no-deps", mode),
                       timeout=900)
            if out.returncode != 0:
                self.last_error = (out.stderr or out.stdout)[-300:]
                raise RuntimeError(
                    f"docker compose up failed: {self.last_error}")
            self.mode = mode
            self._started_at = time.time()
            self._probe = None
            self.last_error = None
        deadline = time.time() + wait_healthy_s
        while time.time() < deadline:
            if self.healthy():
                log.info("HyperQwen healthy in %s mode", mode)
                return
            time.sleep(5)
        tail = self.logs(40)
        self.stop()
        raise RuntimeError(
            f"HyperQwen did not become healthy in {wait_healthy_s:.0f}s. "
            f"Log tail: {tail[-300:]}")

    def _down_all(self) -> None:
        for m in MODES:
            try:
                _run(_compose_argv("--profile", m, "down"), timeout=180)
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        with self._lock:
            if not self.checked_out:
                return
            log.info("stopping HyperQwen")
            self._down_all()
            self.mode = None
            self._started_at = None
            self._probe = None

    def logs(self, lines: int = 60) -> str:
        try:
            out = _run(_compose_argv("logs", "--tail", str(lines)),
                       timeout=60)
            return out.stdout or out.stderr or ""
        except Exception as exc:  # noqa: BLE001
            return f"(could not read logs: {type(exc).__name__})"

    # -- advertisement -----------------------------------------------------

    def status(self) -> dict:
        alive = self.healthy(3, max_age=5)
        docker_ok, docker_detail = self.docker_ready()
        return {
            "engine": "hyperqwen",
            "enabled": enabled(),
            "installed": self.checked_out and self.image_present(),
            "checked_out": self.checked_out,
            "install": self.install_state,
            "docker": {"ready": docker_ok, "detail": docker_detail},
            "running": alive,
            "mode": self.mode if alive else None,
            "port": PORT,
            "model": "Qwen3.8-27B (requantized by HyperQwen)",
            "settings": _settings(),
            "modes": list(MODES),
            "contexts": list(CONTEXTS),
            "specs": list(SPECS),
            "container": self.container_state(),
            "uptime_s": round(time.time() - self._started_at)
            if alive and self._started_at else None,
            "error": self.last_error,
            "note": ("Patched vLLM in Docker serving the same Qwen3.8-27B "
                     "as the ninfer lane. Mutually exclusive with the "
                     "other chat engines and with GPU jobs: it wants the "
                     "whole card."),
        }


HYPERQWEN = HyperQwenManager()
