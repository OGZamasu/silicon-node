"""llama.cpp engine — GGUF models from Hugging Face, served locally.

The Mac's default engine, ported: llama-server.exe (official win-cuda
build, fetched once) serving any downloaded GGUF on 127.0.0.1:8082 with
an OpenAI-compatible API. Managed like ninfer: spawned via interop,
port-truth health, single instance, GPU-exclusive with ninfer and the 3D
pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("silicon-node.llamacpp")

from . import hostos

PORT = int(os.environ.get("SILICON_NODE_GGUF_PORT", "8082"))
ENGINE_DIR = Path(os.environ.get(
    "SILICON_NODE_LLAMACPP_DIR",
    "/mnt/f/Windows Silicon Optimizer/silicon-node/runtime/llamacpp"
    if hostos.IS_WSL else "/opt/silicon/llamacpp"))
GGUF_DIR = Path(os.environ.get(
    "SILICON_NODE_GGUF_DIR",
    "/mnt/f/ai-model-cache/gguf"
    if hostos.IS_WSL else "/opt/silicon/models/gguf"))
_EXE = "llama-server.exe" if hostos.IS_WSL else "llama-server"


def _path_arg(p: Path) -> str:
    """A path as the ENGINE must see it (F:\... through interop on WSL,
    the POSIX path itself on Linux)."""
    return hostos.win_path(p) if hostos.IS_WSL else str(p)

# The Mac's "sharp" Qwen chat template (silicon-optimizer #9): a jinja
# replacement template handed to llama-server, so answers lead with the
# answer instead of preamble. Same source repo and validation as their
# SharpTemplate.swift; one copy serves every Qwen GGUF.
SHARP_REPO = "peculiar-ragdoll/Qwen-Sharp-Chat-Templates"
SHARP_FILE = "chat_template.jinja"
TEMPLATE_DIR = Path(os.environ.get(
    "SILICON_NODE_TEMPLATE_DIR",
    "/mnt/f/ai-model-cache/chat-templates"
    if hostos.IS_WSL else "/opt/silicon/chat-templates"))
SHARP_TEMPLATE = TEMPLATE_DIR / "qwen-sharp.jinja"


def sharp_suits(model_name: str) -> bool:
    """Port of the Mac's SharpTemplate.suits: the template was written
    for Qwen 3.5/3.6/3.8 — on anything else it is a quiet quality
    regression, so check the model rather than trusting a switch."""
    name = model_name.lower()
    if "qwen" not in name:
        return False
    versions = ["3.5", "3-5", "3_5", "35", "3.6", "3-6", "3_6", "36",
                "3.8", "3-8", "3_8", "38"]
    return any(f"qwen{v}" in name or f"qwen {v}" in name
               or f"qwen-{v}" in name for v in versions)


def download_sharp_template() -> None:
    url = f"https://huggingface.co/{SHARP_REPO}/resolve/main/{SHARP_FILE}"
    req = urllib.request.Request(url, headers={"User-Agent": "silicon-node"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8")
    # Same guard as the Mac: a non-template here would break every load
    # that used it, and the failure would look like a model problem.
    if "{%" not in text or len(text) < 200:
        raise RuntimeError("What came back was not a chat template.")
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SHARP_TEMPLATE.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(SHARP_TEMPLATE)
    log.info("sharp chat template downloaded (%d bytes)", len(text))


def ensure_sharp_template_async() -> None:
    """Fetch the template once at boot; chat still works on the stock
    template if the fetch fails, so this is best-effort."""
    if SHARP_TEMPLATE.exists():
        return

    def work() -> None:
        try:
            download_sharp_template()
        except Exception:  # noqa: BLE001
            log.exception("sharp template download failed")
    threading.Thread(target=work, daemon=True).start()


def _fetch_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={
        "User-Agent": "silicon-node"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class LlamaCppManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._probe: tuple[float, bool] | None = None
        self._logfile = ENGINE_DIR / "llama-server.log"
        self.model_file: str | None = None
        self.engine_install: dict | None = None  # progress while fetching

    # -- engine install ---------------------------------------------------

    @property
    def engine_installed(self) -> bool:
        return (ENGINE_DIR / _EXE).exists()

    def install_engine_async(self) -> None:
        if self.engine_installed or self.engine_install:
            return
        self.engine_install = {"stage": "resolving", "error": None}
        threading.Thread(target=self._install_engine, daemon=True).start()

    def _install_engine(self) -> None:
        try:
            rel = _fetch_json("https://api.github.com/repos/ggml-org/"
                              "llama.cpp/releases/latest")
            names = rel.get("assets", [])
            if hostos.IS_WSL:
                asset = next(
                    (a for a in names
                     if "win" in a["name"].lower()
                     and "cuda" in a["name"].lower()
                     and "x64" in a["name"].lower()
                     and a["name"].endswith(".zip")
                     and "cudart" not in a["name"].lower()), None)
                cudart = next(
                    (a for a in names
                     if "cudart" in a["name"].lower()
                     and a["name"].endswith(".zip")), None)
            else:
                def _lin(a):
                    n = a["name"].lower()
                    return (("ubuntu" in n or "linux" in n)
                            and "x64" in n and n.endswith(".zip"))
                # Prefer a CUDA build when the release carries one; the
                # plain ubuntu build still serves (CPU-only) rather than
                # failing the install outright.
                asset = (next((a for a in names
                               if _lin(a) and "cuda" in a["name"].lower()),
                              None)
                         or next((a for a in names if _lin(a)), None))
                cudart = None
            if not asset:
                raise RuntimeError(
                    "No usable build for this OS in the latest llama.cpp "
                    "release.")
            ENGINE_DIR.mkdir(parents=True, exist_ok=True)
            for i, a in enumerate([asset] + ([cudart] if cudart else [])):
                self.engine_install = {
                    "stage": f"downloading {a['name']}", "error": None}
                dest = ENGINE_DIR / a["name"]
                urllib.request.urlretrieve(a["browser_download_url"], dest)
                self.engine_install = {"stage": f"extracting {a['name']}",
                                       "error": None}
                import zipfile  # noqa: PLC0415
                with zipfile.ZipFile(dest) as z:
                    z.extractall(ENGINE_DIR)
                dest.unlink()
            # Some releases nest binaries under build/bin — flatten.
            if not self.engine_installed:
                for sub in ("build/bin", "bin"):
                    cand = ENGINE_DIR / sub
                    if (cand / _EXE).exists():
                        for f in cand.iterdir():
                            f.rename(ENGINE_DIR / f.name)
                        break
            if not self.engine_installed:
                raise RuntimeError(f"{_EXE} not found after extraction.")
            if not hostos.IS_WSL:
                # zipfile drops the exec bit.
                for f in ENGINE_DIR.iterdir():
                    if f.is_file() and (f.name.startswith("llama")
                                        or f.suffix == ".so"):
                        f.chmod(f.stat().st_mode | 0o755)
            self.engine_install = None
            log.info("llama.cpp engine installed (%s)", rel.get("tag_name"))
        except Exception as exc:  # noqa: BLE001
            log.exception("engine install failed")
            self.engine_install = {"stage": "failed", "error": str(exc)[:200]}

    # -- state ------------------------------------------------------------

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

    def installed_models(self) -> list[dict]:
        if not GGUF_DIR.is_dir():
            return []
        return [{"file": p.name,
                 "size_gb": round(p.stat().st_size / 1e9, 1)}
                for p in sorted(GGUF_DIR.glob("*.gguf"))]

    def status(self) -> dict:
        alive = self.healthy(3, max_age=5)
        return {
            "engine_installed": self.engine_installed,
            "engine_install": self.engine_install,
            "running": alive,
            "model": self.model_file if alive else None,
            "port": PORT,
            "models": self.installed_models(),
            "sharp_template": {
                "downloaded": SHARP_TEMPLATE.exists(),
                "active": bool(getattr(self, "sharp_active", False)
                               and alive),
            },
            "context_length": getattr(self, "context", None)
            if alive else None,
            "uptime_s": round(time.time() - self._started_at)
            if alive and self._started_at else None,
        }

    # -- control ----------------------------------------------------------

    @staticmethod
    def default_context(name: str) -> int:
        """Long context by default where it plainly fits: 128K of KV
        beside a small GGUF is cheap on a 24 GB card, while a big GGUF
        needs the VRAM for weights. Callers can always override."""
        try:
            size = (GGUF_DIR / name).stat().st_size
        except OSError:
            return 32768
        if size < 6_000_000_000:
            return 131072
        if size < 12_000_000_000:
            return 65536
        return 32768

    def start(self, model_file: str, context: int | None = None,
              wait_healthy_s: float = 240.0) -> None:
        with self._lock:
            if not self.engine_installed:
                self.install_engine_async()
                raise RuntimeError(
                    "The llama.cpp engine is downloading — watch the "
                    "Models page and try again when it lands.")
            name = Path(model_file).name
            if not (GGUF_DIR / name).exists():
                raise RuntimeError(f"No {name} in the model library.")
            if context is None:
                context = self.default_context(name)
            self.context = int(context)
            self._kill_instances()
            self._probe = None
            args = [str(ENGINE_DIR / _EXE),
                    "-m", _path_arg(GGUF_DIR / name),
                    "--host", "127.0.0.1", "--port", str(PORT),
                    "-ngl", "999", "-c", str(context), "--no-webui",
                    # Mirror the Mac's LlamaArguments: --jinja is what
                    # makes tool calls work without per-model cases.
                    "--jinja"]
            self.sharp_active = False
            if (sharp_suits(name) and SHARP_TEMPLATE.exists()
                    and os.environ.get("SILICON_NODE_SHARP_TEMPLATE",
                                       "1") != "0"):
                # As the engine sees it: F:\... through interop on
                # WSL (the exe cannot open /mnt/f), POSIX on Linux.
                args += ["--chat-template-file",
                         _path_arg(SHARP_TEMPLATE)]
                self.sharp_active = True
            env = dict(os.environ)
            if not hostos.IS_WSL:
                # The official Linux builds ship their .so files beside
                # the binary; make sure the loader finds them.
                env["LD_LIBRARY_PATH"] = (str(ENGINE_DIR) + ":"
                                          + env.get("LD_LIBRARY_PATH", ""))
            logfh = open(self._logfile, "ab")  # noqa: SIM115
            self._proc = subprocess.Popen(
                args, stdout=logfh, stderr=subprocess.STDOUT,
                cwd=str(ENGINE_DIR), env=env)
            logfh.close()
            self.model_file = name
            self._started_at = time.time()
        deadline = time.time() + wait_healthy_s
        while time.time() < deadline:
            if self.healthy():
                log.info("llama-server healthy with %s", name)
                return
            time.sleep(3)
        tail = ""
        try:
            tail = self._logfile.read_text(errors="replace")[-300:]
        except OSError:
            pass
        self.stop()
        raise RuntimeError(
            f"llama-server did not come up in {wait_healthy_s:.0f}s. "
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
            self._probe = None
            self._started_at = None
        time.sleep(2)

    @staticmethod
    def _kill_instances() -> None:
        hostos.kill_by_name("llama-server")


class GGUFDownloads:
    """HF GGUF downloads into the shared model library (F:\\ai-model-cache),
    resumable, progress by file size — same pattern as the ninfer manager."""

    def __init__(self) -> None:
        self.active: dict[str, dict] = {}

    def start(self, repo: str, filename: str) -> None:
        GGUF_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        name = Path(filename).name
        if name in self.active and not self.active[name].get("error"):
            return
        self.active[name] = {"got": 0, "total": 0, "error": None,
                             "repo": repo}
        threading.Thread(target=self._worker, args=(url, name),
                         daemon=True).start()

    def _worker(self, url: str, name: str) -> None:
        dest = GGUF_DIR / name
        try:
            have = dest.stat().st_size if dest.exists() else 0
            req = urllib.request.Request(url, headers={
                "User-Agent": "silicon-node"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get("Content-Length", 0))
                self.active[name]["total"] = total
                with open(dest, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        self.active[name]["got"] = dest.stat().st_size
            self.active[name]["got"] = dest.stat().st_size
        except Exception as exc:  # noqa: BLE001
            log.exception("gguf download failed")
            self.active[name]["error"] = str(exc)[:200]

    def progress(self) -> dict:
        out = {}
        for name, st in self.active.items():
            p = GGUF_DIR / name
            out[name] = {"got": p.stat().st_size if p.exists() else 0,
                         "total": st["total"], "error": st["error"]}
        return out


LLAMACPP = LlamaCppManager()
GGUF_DL = GGUFDownloads()
