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
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("silicon-node.llamacpp")

from . import hostos  # noqa: E402 (after the logger it configures)

PORT = int(os.environ.get("SILICON_NODE_GGUF_PORT", "8082"))
ENGINE_DIR = Path(os.environ.get(
    "SILICON_NODE_LLAMACPP_DIR",
    "/mnt/f/Windows Silicon Optimizer/silicon-node/runtime/llamacpp"
    if hostos.IS_WSL else "/opt/silicon/llamacpp"))
# PrismML's llama.cpp fork, for Ternary Bonsai 2's PTQ1_0/PQ2_0 packings —
# stock builds refuse those files outright. Same layout as the stock
# engine, one folder over, fetched from the fork's own releases.
PRISM_ENGINE_DIR = Path(os.environ.get(
    "SILICON_NODE_LLAMACPP_PRISM_DIR", str(ENGINE_DIR.parent / "llamacpp-prism")))
PRISM_REPO = "PrismML-Eng/llama.cpp"
GGUF_DIR = Path(os.environ.get(
    "SILICON_NODE_GGUF_DIR",
    "/mnt/f/ai-model-cache/gguf"
    if hostos.IS_WSL else "/opt/silicon/models/gguf"))
_EXE = "llama-server.exe" if hostos.IS_WSL else "llama-server"


def _path_arg(p: Path) -> str:
    r"""A path as the ENGINE must see it (F:\... through interop on WSL,
    the POSIX path itself on Linux)."""
    return hostos.win_path(p) if hostos.IS_WSL else str(p)


# The library as the dashboard names it. main.py imported this before it
# existed, which turned every GET /v1/models into a 500.
GGUF_DIR_WIN = _path_arg(GGUF_DIR)

_PRISM_QUANTS = re.compile(r"-(PTQ1_0|PQ2_0)\.gguf$", re.IGNORECASE)


def needs_prism(model_file: str) -> bool:
    """PrismML's ternary packings (Ternary Bonsai 2) load only on their fork."""
    return bool(_PRISM_QUANTS.search(Path(model_file).name))


def mmproj_for(model_file: str) -> Path | None:
    """The vision projector shipped beside a 27B Bonsai GGUF, if it was
    downloaded: `<stem>-mmproj-*.gguf`, Q8_0 preferred. None = text only."""
    stem = re.sub(r"-[A-Za-z0-9_]+\.gguf$", "", Path(model_file).name)
    candidates = sorted(GGUF_DIR.glob(f"{stem}-mmproj-*.gguf"))
    if not candidates:
        return None
    return next((c for c in candidates if "Q8_0" in c.name), candidates[0])


# One-click picks beside the Hugging Face search: models worth naming because
# the search can't tell you what they need (a companion file, a fork).
GGUF_PICKS = [
    {"id": "bonsai-2-27b", "name": "Bonsai 2 27B (PrismML, ternary)",
     "repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
     "file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
     "mmproj": "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
     "size_gb": 6.6,
     "note": "Qwen3.8 27B in 1.76-bit ternary weights: 98% of its benchmarks "
             "in 5.9 GB, vision and tool calling included. Needs PrismML's "
             "llama.cpp fork, fetched automatically with the file."},
]

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
        self.prism_engine_install: dict | None = None
        self.engine_flavor: str | None = None  # which engine is serving

    # -- engine install ---------------------------------------------------

    @staticmethod
    def engine_dir(flavor: str = "stock") -> Path:
        return PRISM_ENGINE_DIR if flavor == "prism" else ENGINE_DIR

    @property
    def engine_installed(self) -> bool:
        return (ENGINE_DIR / _EXE).exists()

    @property
    def prism_engine_installed(self) -> bool:
        return (PRISM_ENGINE_DIR / _EXE).exists()

    def engine_ready(self, flavor: str = "stock") -> bool:
        return (self.prism_engine_installed if flavor == "prism"
                else self.engine_installed)

    def _progress(self, flavor: str, state: dict | None) -> None:
        if flavor == "prism":
            self.prism_engine_install = state
        else:
            self.engine_install = state

    def install_engine_async(self, flavor: str = "stock") -> None:
        if flavor == "prism":
            if self.prism_engine_installed or self.prism_engine_install:
                return
        elif self.engine_installed or self.engine_install:
            return
        self._progress(flavor, {"stage": "resolving", "error": None})
        threading.Thread(target=self._install_engine, args=(flavor,),
                         daemon=True).start()

    @staticmethod
    def _matching_assets(assets: list[dict]) -> tuple[dict | None, dict | None]:
        """The engine build for this host and, on WSL, the CUDA runtime zip
        that goes with it. CUDA 12.4 first: it is the toolkit the WSL side
        was provisioned with, and every driver here runs it."""
        def _rank(a):
            return "cuda-12.4" not in a["name"].lower()
        if hostos.IS_WSL:
            cands = [a for a in assets
                     if "win" in a["name"].lower()
                     and "cuda" in a["name"].lower()
                     and "x64" in a["name"].lower()
                     and a["name"].endswith(".zip")
                     and "cudart" not in a["name"].lower()]
            asset = min(cands, key=_rank) if cands else None
            cudart = None
            if asset:
                token = next((t for t in ("cuda-12.4", "cuda-12.8", "cuda-13.3",
                                          "cuda-13.4")
                              if t in asset["name"].lower()), "cuda")
                cudart = next((a for a in assets
                               if "cudart" in a["name"].lower()
                               and token in a["name"].lower()
                               and a["name"].endswith(".zip")), None)
            return asset, cudart

        def _lin(a):
            n = a["name"].lower()
            return (("ubuntu" in n or "linux" in n) and "x64" in n
                    and (n.endswith(".zip") or n.endswith(".tar.gz")))
        # Prefer a CUDA build when the release carries one; the plain
        # ubuntu build still serves (CPU-only) rather than failing outright.
        cuda = [a for a in assets if _lin(a) and "cuda" in a["name"].lower()]
        asset = (min(cuda, key=_rank) if cuda
                 else next((a for a in assets if _lin(a)), None))
        return asset, None

    def _release_for(self, flavor: str) -> dict:
        """Stock: upstream's latest. Prism: the newest fork release that
        actually has this host's build attached — a fresh tag can sit for
        an hour with only some platforms uploaded."""
        if flavor != "prism":
            return _fetch_json("https://api.github.com/repos/ggml-org/"
                               "llama.cpp/releases/latest")
        releases = _fetch_json(
            f"https://api.github.com/repos/{PRISM_REPO}/releases?per_page=10")
        for rel in releases:
            if rel.get("draft") or rel.get("prerelease"):
                continue
            asset, _ = self._matching_assets(rel.get("assets", []))
            if asset and asset["name"].startswith("llama-prism-"):
                return rel
        raise RuntimeError(
            "No PrismML llama.cpp release carries a build for this host yet.")

    @staticmethod
    def _extract(archive: Path, target: Path) -> None:
        if archive.name.endswith(".tar.gz"):
            import tarfile  # noqa: PLC0415
            with tarfile.open(archive) as t:
                try:
                    t.extractall(target, filter="data")
                except TypeError:  # Python < 3.12 without the backport
                    t.extractall(target)
        else:
            import zipfile  # noqa: PLC0415
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)

    def _install_engine(self, flavor: str = "stock") -> None:
        target = self.engine_dir(flavor)
        try:
            rel = self._release_for(flavor)
            asset, cudart = self._matching_assets(rel.get("assets", []))
            if not asset:
                raise RuntimeError(
                    "No usable build for this OS in the latest llama.cpp "
                    "release.")
            target.mkdir(parents=True, exist_ok=True)
            for a in [asset] + ([cudart] if cudart else []):
                self._progress(flavor, {
                    "stage": f"downloading {a['name']}", "error": None})
                dest = target / a["name"]
                urllib.request.urlretrieve(a["browser_download_url"], dest)
                self._progress(flavor, {"stage": f"extracting {a['name']}",
                                        "error": None})
                self._extract(dest, target)
                dest.unlink()
            # Releases nest the binaries a folder down (build/bin, or the
            # tag-named folder the fork's tarballs unpack to) — flatten.
            if not (target / _EXE).exists():
                nested = next(target.rglob(_EXE), None)
                if nested is not None:
                    for f in nested.parent.iterdir():
                        f.rename(target / f.name)
            if not (target / _EXE).exists():
                raise RuntimeError(f"{_EXE} not found after extraction.")
            if not hostos.IS_WSL:
                # zipfile drops the exec bit.
                for f in target.iterdir():
                    if f.is_file() and (f.name.startswith("llama")
                                        or f.suffix == ".so"):
                        f.chmod(f.stat().st_mode | 0o755)
            self._progress(flavor, None)
            log.info("llama.cpp engine installed (%s, %s)", flavor,
                     rel.get("tag_name"))
        except Exception as exc:  # noqa: BLE001
            log.exception("engine install failed (%s)", flavor)
            self._progress(flavor, {"stage": "failed", "error": str(exc)[:200]})

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
        # Projectors are companions, not models: they ride along with the
        # weights they belong to instead of appearing as something to load.
        return [{"file": p.name,
                 "size_gb": round(p.stat().st_size / 1e9, 1),
                 "engine": "prism" if needs_prism(p.name) else "stock",
                 "mmproj": mmproj_for(p.name) is not None}
                for p in sorted(GGUF_DIR.glob("*.gguf"))
                if "-mmproj-" not in p.name]

    def status(self) -> dict:
        alive = self.healthy(3, max_age=5)
        return {
            "engine_installed": self.engine_installed,
            "engine_install": self.engine_install,
            "prism_engine_installed": self.prism_engine_installed,
            "prism_engine_install": self.prism_engine_install,
            "engine_flavor": self.engine_flavor if alive else None,
            "running": alive,
            "model": self.model_file if alive else None,
            "port": PORT,
            "models": self.installed_models(),
            "picks": GGUF_PICKS,
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
            name = Path(model_file).name
            flavor = "prism" if needs_prism(name) else "stock"
            if not self.engine_ready(flavor):
                self.install_engine_async(flavor)
                raise RuntimeError(
                    "PrismML's llama.cpp fork is downloading — this ternary "
                    "model needs it; watch the Models page and try again "
                    "when it lands." if flavor == "prism" else
                    "The llama.cpp engine is downloading — watch the "
                    "Models page and try again when it lands.")
            engine_dir = self.engine_dir(flavor)
            if not (GGUF_DIR / name).exists():
                raise RuntimeError(f"No {name} in the model library.")
            if context is None:
                context = self.default_context(name)
            self.context = int(context)
            self._kill_instances()
            self._probe = None
            args = [str(engine_dir / _EXE),
                    "-m", _path_arg(GGUF_DIR / name),
                    "--host", "127.0.0.1", "--port", str(PORT),
                    "-ngl", "999", "-c", str(context), "--no-webui",
                    # Mirror the Mac's LlamaArguments: --jinja is what
                    # makes tool calls work without per-model cases.
                    "--jinja"]
            mmproj = mmproj_for(name)
            if mmproj is not None:
                # Image input for the 27B Bonsai family; loaded lazily by
                # llama-server, so text-only chat pays nothing for it.
                args += ["--mmproj", _path_arg(mmproj)]
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
                env["LD_LIBRARY_PATH"] = (str(engine_dir) + ":"
                                          + env.get("LD_LIBRARY_PATH", ""))
            self._logfile = engine_dir / "llama-server.log"
            logfh = open(self._logfile, "ab")  # noqa: SIM115
            self._proc = subprocess.Popen(
                args, stdout=logfh, stderr=subprocess.STDOUT,
                cwd=str(engine_dir), env=env)
            logfh.close()
            self.model_file = name
            self.engine_flavor = flavor
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
