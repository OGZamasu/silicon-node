"""ninfer-3090 LLM manager — Qwen3.8-27B as the node's `llm` capability.

The engine (github.com/Don-Chad/ninfer-3090) is a native Windows CUDA
binary serving OpenAI- and Anthropic-compatible APIs on 127.0.0.1:8080.
This service runs inside WSL, so we launch the exe through WSL interop
(it becomes a real Windows process) and health-check it via the Windows
host's NAT gateway IP. Remote (Mac) access goes through tailscale serve
tcp:8080 on the Windows side — the engine itself never binds beyond
localhost, and reaching it from off-box requires the tailnet.

GPU arbitration on the shared 24 GB card: the LLM (~20+ GB) and the 3D
pipelines (~10 GB) are mutually exclusive. 3D jobs preempt the LLM
(stop -> run job -> restore), and restoring first unloads the resident
TRELLIS pipeline. The trade: after any 3D job the next one repays the
~90 s TRELLIS load, but the LLM comes back automatically.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("silicon-node.llm")

NINFER_DIR = Path(os.environ.get(
    "NINFER_DIR",
    "/mnt/f/Windows Silicon Optimizer/ninfer-3090/dist/"
    "ninfer-rtx3090-windows-x64-0.6.0-rtx3090"))
NINFER_EXE = NINFER_DIR / "ninfer-serve.exe"
MODEL_FILE = NINFER_DIR / "models" / "qwen3_8_27b.ninfer"
MODEL_ID = "qwen3.8-27b"
# 8080 is taken by sftpgo on this machine; ninfer lives on 8081.
PORT = int(os.environ.get("NINFER_PORT", "8081"))
# How off-box clients reach this machine (tailnet/LAN address) — set in
# the node's private env, never hardcoded here.
PUBLIC_HOST = os.environ.get("SILICON_NODE_PUBLIC_HOST", "127.0.0.1")
AUTOSTART = os.environ.get("SILICON_NODE_LLM_AUTOSTART", "1") != "0"

# Windows paths as the exe needs to see them (interop passes argv through).
_WIN_MODEL = r"F:\Windows Silicon Optimizer\ninfer-3090\dist" \
    r"\ninfer-rtx3090-windows-x64-0.6.0-rtx3090\models\qwen3_8_27b.ninfer"

# Profile sizing is a measured trade (silicon-optimizer #11): the old c1
# asked for a 64K context envelope with 1024-token prefill chunks, which
# left free-after-startup at 1.2 GiB with 0 headroom — ANY real prefill
# then died mid-flight and took the process with it (the Mac's agent chat
# sends ~23K-token preambles). The workspace lever turned out to be the
# PREFILL CHUNK, not the envelope: at chunk 256 the same 64K envelope
# starts with 2.64 GiB free and swept clean — 25K in 29 s, 36.7K in 45 s,
# 60.6K in 87 s, over-context → clean HTTP 400, engine alive throughout
# at ~21.3 GiB. 128K is not physically available for this 27B on 24 GB
# (17 GiB weights + 5.5 GiB KV would leave less workspace than the config
# that crashed). Upstream 0.6.0 validates only 8K; 64K is OUR swept
# envelope — re-run the sweep before touching these numbers.
PROFILES = {
    # Agent-chat default: one 64K conversation, or two sharing the pool.
    "c1": {
        "context_length": 65536,
        "kv_pool_tokens": 65536,  # shared across the 2 lanes
        "max_concurrency": 2,
        "flags": ["--host", "127.0.0.1", "--port", str(PORT), "--cors",
                  "--max-context", "65536", "--kv-capacity", "65536",
                  "--max-concurrency", "2", "--max-pending-requests", "16",
                  "--prefill-chunk", "256", "--kv-dtype", "int8",
                  "--spec", "mtp", "--draft-tokens", "3",
                  "--lm-head-draft"],
    },
    # Throughput: the release's validated C8/8K MTP3 shape.
    "c8": {
        "context_length": 8192,
        "kv_pool_tokens": 65536,
        "max_concurrency": 8,
        "flags": ["--host", "127.0.0.1", "--port", str(PORT), "--cors",
                  "--max-context", "8192", "--kv-capacity", "65536",
                  "--max-concurrency", "8", "--max-pending-requests", "32",
                  "--prefill-chunk", "1024", "--kv-dtype", "int8",
                  "--spec", "mtp", "--draft-tokens", "3",
                  "--lm-head-draft"],
    },
}


# Context choice persistence (handoff 127): the Mac's Swarm page picks a
# serving context; later starts without the field reuse the last choice.
CONTEXT_FILE = Path(os.environ.get("SILICON_NODE_LLM_CONTEXT_FILE",
                                   "/opt/silicon/llm-context.json"))
# 64K is our swept ceiling for this engine (see PROFILES note); below
# ~4K the engine's own tooling was never exercised.
CONTEXT_MIN, CONTEXT_MAX = 4096, 65536


def _load_contexts() -> dict:
    try:
        import json  # noqa: PLC0415
        return json.loads(CONTEXT_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _ctx_lookup(model_id: str):
    """Stored context for a model, tolerant of id spelling: the filename
    stem says qwen3_8_27b while the engine serves qwen3.8-27b."""
    data = _load_contexts()
    if model_id in data:
        return data[model_id]
    def norm(s: str) -> str:
        return "".join(ch for ch in s if ch.isalnum())
    n = norm(model_id)
    for k, v in data.items():
        if norm(k) == n:
            return v
    return None


def _store_context(model_id: str, ctx: int) -> None:
    try:
        import json  # noqa: PLC0415
        data = _load_contexts()
        data[model_id] = ctx
        CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONTEXT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(CONTEXT_FILE)
    except Exception:  # noqa: BLE001
        log.exception("could not persist the context choice")


def _windows_host_ip() -> str:
    """The Windows host as seen from WSL2 (NAT default gateway)."""
    try:
        out = subprocess.run(["sh", "-c", "ip route | awk '/^default/ {print $3; exit}'"],
                             capture_output=True, text=True, timeout=5)
        ip = out.stdout.strip()
        if ip:
            return ip
    except Exception:  # noqa: BLE001
        pass
    return "127.0.0.1"


class LlmManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._profile = "c1"
        self._started_at: float | None = None
        self._was_running_before_job = False
        self._host_ip = _windows_host_ip()
        self._logfile = NINFER_DIR / "ninfer-serve.log"
        self._restore_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()
        self._expect_running = False   # armed by start(), cleared by stop()
        self._watchdog: threading.Thread | None = None
        self._last_watchdog_restart = 0.0

    # -- state ------------------------------------------------------------

    @property
    def installed(self) -> bool:
        # Size guard: the model downloads resumably straight to its final
        # name, so a partial file exists mid-download. Full size ~16.96 GiB.
        try:
            return (NINFER_EXE.exists()
                    and MODEL_FILE.stat().st_size > 18_000_000_000)
        except OSError:
            return False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def healthy(self, timeout: float = 4.0) -> bool:
        """Probe via Windows curl.exe through interop: the engine binds
        Windows loopback, which WSL cannot reach over the NAT gateway —
        an interop process's 127.0.0.1 IS the Windows loopback."""
        if not self.running:
            return False
        try:
            out = subprocess.run(
                ["/mnt/c/Windows/System32/curl.exe", "-s", "-m",
                 str(int(timeout)), "-o", "NUL", "-w", "%{http_code}",
                 f"http://127.0.0.1:{PORT}/v1/models"],
                capture_output=True, text=True, timeout=timeout + 3)
            return out.stdout.strip().startswith("2")
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> dict:
        models_dir = NINFER_DIR / "models"
        installed_files = sorted(
            p.name for p in models_dir.glob("*.ninfer")
            if p.stat().st_size > 10_000_000_000) if models_dir.is_dir() \
            else []
        return {
            "installed": self.installed,
            "installed_models": installed_files,
            "running": self.running,
            "healthy": self.healthy(1.5) if self.running else False,
            "model": getattr(self, "model_id", MODEL_ID)
            if self.installed else None,
            "profile": self._profile if self.running else None,
            # The Mac gateway advertises honest limits from this
            # (silicon-optimizer #11 ask #3).
            # The OPERATIVE serving context (chosen or clamped), not the
            # profile constant — handoffs 125/127.
            "context_length": getattr(
                self, "_ctx_effective",
                PROFILES[self._profile]["context_length"])
            if self.running else None,
            # Honesty note for budgeters: the KV pool is shared across
            # lanes — one conversation can use the full context_length
            # when alone; concurrent long sessions split this pool.
            "kv_pool_tokens": getattr(
                self, "_kv_effective",
                PROFILES[self._profile]["kv_pool_tokens"])
            if self.running else None,
            "max_concurrency": PROFILES[self._profile]["max_concurrency"]
            if self.running else None,
            "uptime_s": round(time.time() - self._started_at)
            if self.running and self._started_at else None,
            "api": {
                "openai": f"http://{PUBLIC_HOST}:{PORT}/v1",
                "anthropic": f"http://{PUBLIC_HOST}:{PORT} "
                             "(Messages API)",
                "note": "Engine binds 127.0.0.1 on the Windows host; "
                        "off-box access is tailnet-only via tailscale "
                        "serve.",
            },
        }

    # -- control ----------------------------------------------------------

    def start(self, profile: str = "c1", wait_healthy_s: float = 180.0,
              model_file: str | None = None,
              context_length: int | None = None) -> None:
        """model_file: bare filename of an installed .ninfer in the models
        dir (defaults to Qwen3.8-27B). The GUI's Models tab passes this.
        context_length (handoff 127): explicit → clamped, served, and
        persisted per model; absent → the model's last persisted choice,
        else the profile default."""
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; "
                             f"use one of {sorted(PROFILES)}.")
        with self._lock:
            if self.running:
                return
            if not self.installed:
                raise RuntimeError(
                    "ninfer is not installed: expected the exe and model at "
                    f"{NINFER_EXE} / {MODEL_FILE}.")
            win_model = _WIN_MODEL
            if model_file:
                name = Path(model_file).name  # no path traversal
                candidate = NINFER_DIR / "models" / name
                if not candidate.exists():
                    raise RuntimeError(
                        f"No model file named {name} in the models folder.")
                win_model = (r"F:\Windows Silicon Optimizer\ninfer-3090"
                             r"\dist\ninfer-rtx3090-windows-x64-0.6.0-"
                             r"rtx3090\models" + "\\" + name)
                # Provisional only — replaced by the engine's own served id
                # once healthy (deriving from the filename produced
                # "qwen3.8.27b" vs the served "qwen3.8-27b" and broke
                # remote chat: silicon-node issue #3).
                self.model_id = name.replace(".ninfer", "")
            else:
                self.model_id = MODEL_ID
            if context_length is not None:
                ctx = max(CONTEXT_MIN, min(CONTEXT_MAX, int(context_length)))
            else:
                stored = _ctx_lookup(self.model_id)
                ctx = (max(CONTEXT_MIN, min(CONTEXT_MAX, int(stored)))
                       if stored else PROFILES[profile]["context_length"])
            flags = list(PROFILES[profile]["flags"])
            i = flags.index("--max-context")
            flags[i + 1] = str(ctx)
            # The engine rejects a KV pool outside the usable range for
            # max_context × max_concurrency (learned from a live 16K
            # start) — scale the pool with the context, capped at the
            # measured-safe 65536.
            conc = PROFILES[profile]["max_concurrency"]
            k = flags.index("--kv-capacity")
            kv_pool = min(65536, ctx * conc)
            flags[k + 1] = str(kv_pool)
            self._ctx_effective = ctx
            self._kv_effective = kv_pool
            self._ctx_explicit = context_length is not None
            log.info("starting ninfer (%s, model %s, context %d)…",
                     profile, self.model_id, ctx)
            self._kill_all_instances()  # never allow a second instance
            logfh = open(self._logfile, "ab")  # noqa: SIM115
            self._proc = subprocess.Popen(
                [str(NINFER_EXE), win_model, *flags],
                stdout=logfh, stderr=subprocess.STDOUT,
                cwd=str(NINFER_DIR))
            logfh.close()
            self._profile = profile
            self._started_at = time.time()
        deadline = time.time() + wait_healthy_s
        while time.time() < deadline:
            if not self.running:
                tail = ""
                try:
                    tail = self._logfile.read_text(errors="replace")[-400:]
                except OSError:
                    pass
                raise RuntimeError(
                    "ninfer exited during startup (exit code "
                    f"{self._proc.returncode}). Log tail: {tail}")
            if self.healthy():
                served = self._served_model_id()
                if served:
                    self.model_id = served
                # Persist the context choice only for a start that
                # actually served, keyed by the engine's own model id —
                # a failed experiment must not become the new default.
                if getattr(self, "_ctx_explicit", False):
                    _store_context(self.model_id, self._ctx_effective)
                self._expect_running = True
                self._ensure_watchdog()
                log.info("ninfer healthy after %.0fs (model id %s)",
                         time.time() - self._started_at,
                         getattr(self, "model_id", MODEL_ID))
                return
            time.sleep(2)
        self.stop()
        raise RuntimeError(
            f"ninfer did not become healthy within {wait_healthy_s:.0f}s; "
            "stopped it again.")

    @staticmethod
    def _kill_all_instances() -> None:
        """Kill every ninfer-serve.exe on the Windows side.

        Terminating the WSL interop proxy does NOT kill the Windows
        process (learned the hard way: an orphan survived a preempt and a
        restore stacked a second 17 GB instance on the card). taskkill by
        image name is the only reliable teardown — and single-instance is
        exactly what we want anyway."""
        try:
            subprocess.run(
                ["/mnt/c/Windows/System32/taskkill.exe", "/IM",
                 "ninfer-serve.exe", "/F"],
                capture_output=True, timeout=20)
        except Exception:  # noqa: BLE001
            log.exception("taskkill failed")

    def _served_model_id(self) -> str | None:
        """The engine's own answer to /v1/models — the only id remote
        clients can trust."""
        try:
            out = subprocess.run(
                ["/mnt/c/Windows/System32/curl.exe", "-s", "-m", "5",
                 f"http://127.0.0.1:{PORT}/v1/models"],
                capture_output=True, text=True, timeout=10)
            import json as _json  # noqa: PLC0415
            data = _json.loads(out.stdout)
            return data["data"][0]["id"]
        except Exception:  # noqa: BLE001
            return None

    def _ensure_watchdog(self) -> None:
        """An engine death must not strand the port dead until someone
        notices (silicon-optimizer #11: a prefill crash left agent chat
        down). Process-exit is the trigger — health probes flap under
        prefill load, a vanished process does not."""
        if self._watchdog is not None:
            return
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="llm-watchdog", daemon=True)
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        while True:
            time.sleep(20)
            if self.running:
                continue
            # Two reasons the engine should be up: it crashed while
            # serving (_expect_running), or a GPU job preempted it and
            # the debounced restore lost a race and never fired
            # (_was_running_before_job — observed live: handoff 125/126
            # verification left the LLM down for good). Either way, once
            # the queue is quiet and no restore timer is pending, the
            # watchdog owns bringing it back.
            if not (self._expect_running or self._was_running_before_job):
                continue
            if time.time() - self._last_watchdog_restart < 60:
                continue
            from .jobs import STORE  # noqa: PLC0415
            if STORE.queue_depth() > 0:
                continue   # a job owns the GPU; restore comes later
            with self._timer_lock:
                if (self._restore_timer is not None
                        and self._restore_timer.is_alive()):
                    continue   # the debounced restore is still pending
            self._last_watchdog_restart = time.time()
            tail = ""
            try:
                tail = self._logfile.read_text(errors="replace")[-300:]
            except OSError:
                pass
            log.error("LLM should be up but is not; watchdog restarting. "
                      "Log tail: %s", tail)
            self._expect_running = False   # start() re-arms on success
            try:
                from . import pipeline as _pipeline  # noqa: PLC0415
                _pipeline.ENGINE.unload()  # 3D residency blocks the LLM
                self.start(self._profile)
                self._was_running_before_job = False
                log.info("watchdog restarted the LLM")
            except Exception:  # noqa: BLE001
                log.exception("watchdog restart failed; retrying on the "
                              "next quiet cycle")

    def stop(self) -> None:
        self._expect_running = False
        with self._lock:
            log.info("stopping ninfer…")
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                self._proc = None
            self._kill_all_instances()
            self._started_at = None
        # Give the driver a moment to actually release the VRAM.
        time.sleep(3)

    # -- GPU arbitration hooks (called by the job worker) -----------------

    def preempt_for_job(self) -> None:
        """Called before a GPU job runs: the job wins, the LLM yields.

        The restore intent is sticky across a batch of queued jobs — the
        worker only restores once the queue drains, so a later job in the
        batch must not erase the fact that the LLM was up before the
        first one."""
        with self._timer_lock:
            if self._restore_timer is not None:
                self._restore_timer.cancel()
                self._restore_timer = None
        self._was_running_before_job = (self._was_running_before_job
                                        or self.running)
        if self.running:
            log.info("preempting LLM for a GPU job")
            self.stop()
        # The GGUF engine shares the card too; jobs preempt it the same
        # way (no auto-restore for it yet — it restarts from the UI).
        try:
            from .llamacpp import LLAMACPP  # noqa: PLC0415
            if LLAMACPP.running:
                log.info("preempting llama-server for a GPU job")
                LLAMACPP.stop()
        except Exception:  # noqa: BLE001
            pass

    def schedule_restore(self, unload_pipelines, is_busy,
                         delay_s: float | None = None) -> None:
        """Debounced restore: bring the LLM back only after the job queue
        has been quiet for a while. Bursts of delegated 3D jobs would
        otherwise thrash a ~50 s LLM load between every pair of jobs."""
        if not self._was_running_before_job:
            return
        if delay_s is None:
            delay_s = float(os.environ.get(
                "SILICON_NODE_LLM_RESTORE_DELAY_S", "120"))

        def fire() -> None:
            if is_busy():
                return  # a new job owns the GPU; its finish reschedules us
            self.restore_after_job(unload_pipelines)

        with self._timer_lock:
            if self._restore_timer is not None:
                self._restore_timer.cancel()
            self._restore_timer = threading.Timer(delay_s, fire)
            self._restore_timer.daemon = True
            self._restore_timer.start()
        log.info("LLM restore scheduled in %.0fs (debounced)", delay_s)

    def restore_after_job(self, unload_pipelines) -> None:
        """Called after a GPU job: bring the LLM back if it was up.

        unload_pipelines: callable that frees the resident 3D pipelines
        first — the LLM and TRELLIS cannot share the card.
        """
        if not self._was_running_before_job:
            return
        try:
            unload_pipelines()
            self.start(self._profile)
            # Cleared only on success — a failed restore leaves the flag
            # armed so the watchdog retries instead of giving up forever.
            self._was_running_before_job = False
            log.info("LLM restored after GPU job")
        except Exception:  # noqa: BLE001
            log.exception("failed to restore the LLM after the job; "
                          "the watchdog retries once the GPU is quiet")


class ModelDownloads:
    """Resumable HF downloads into the ninfer models dir, driven from the
    web dashboard. One at a time; progress by file size."""

    CATALOG = {
        "qwen3.8-27b": "neroued/Qwen3.8-27B-NInfer",
        "qwen3.6-27b": "neroued/Qwen3.6-27B-NInfer",
        "qwen3.6-35b-a3b": "neroued/Qwen3.6-35B-A3B-NInfer",
    }

    def __init__(self) -> None:
        self.active: dict[str, dict] = {}   # filename -> {got,total,error}
        self._lock = threading.Lock()

    def start(self, model_id: str) -> str:
        import urllib.request  # noqa: PLC0415
        repo = self.CATALOG.get(model_id)
        if not repo:
            raise ValueError(f"Unknown model {model_id!r}.")
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repo}",
                timeout=20) as r:
            import json as _json  # noqa: PLC0415
            info = _json.loads(r.read().decode())
        names = [s["rfilename"] for s in info.get("siblings", [])
                 if s["rfilename"].endswith(".ninfer")]
        if not names:
            raise RuntimeError(f"No .ninfer file in {repo}.")
        name = names[0]
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        dest = NINFER_DIR / "models" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if name in self.active and not self.active[name].get("error"):
                return name
            self.active[name] = {"got": 0, "total": 0, "error": None}
        threading.Thread(target=self._worker, args=(url, dest, name),
                         daemon=True).start()
        return name

    def _worker(self, url: str, dest: Path, name: str) -> None:
        import urllib.request  # noqa: PLC0415
        try:
            have = dest.stat().st_size if dest.exists() else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get("Content-Length", 0))
                self.active[name]["total"] = total
                mode = "ab" if have else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        self.active[name]["got"] = dest.stat().st_size
            self.active[name]["got"] = dest.stat().st_size
        except Exception as exc:  # noqa: BLE001
            log.exception("model download failed")
            self.active[name]["error"] = str(exc)[:200]

    def progress(self) -> dict:
        out = {}
        for name, st in self.active.items():
            p = NINFER_DIR / "models" / name
            got = p.stat().st_size if p.exists() else 0
            out[name] = {"got": got, "total": st["total"],
                         "error": st["error"]}
        return out


DOWNLOADS = ModelDownloads()
LLM = LlmManager()
