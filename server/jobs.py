"""Job store and single GPU worker.

One job on the GPU at a time (queue + one worker thread — same idea as the
Mac's TRELLIS MCP threading.Semaphore(1)). Jobs run in the worker, never in
the HTTP handler, so a multi-minute render cannot die with the request.

Every job persists a status.json receipt in its own directory: request
params, per-stage timings, peak VRAM, artifact paths. The store reloads
those on startup, so a worker restart (e.g. after CUDA OOM) keeps history.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import config

log = logging.getLogger("silicon-node.jobs")


def new_id(prefix: str = "job") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


@dataclass
class Job:
    job_id: str
    capability: str
    params: dict[str, Any]
    state: str = "queued"  # queued | running | done | failed
    held: bool = False
    cancel_requested: bool = False
    progress: Optional[float] = None
    error: Optional[str] = None
    result_files: list[str] = field(default_factory=list)  # names under FILES_DIR
    stage: Optional[str] = None
    submitted_by: Optional[dict] = None  # {client, ip, proxied, user_agent}
    step: Optional[int] = None          # e.g. denoise step 6…
    steps_total: Optional[int] = None   # …of 8
    eta_seconds: Optional[float] = None  # smoothed estimate to completion
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    receipts: dict[str, Any] = field(default_factory=dict)  # timings, peak VRAM…

    @property
    def dir(self) -> Path:
        return config.JOBS_DIR / self.job_id

    def to_api(self) -> dict[str, Any]:
        """Shape returned by GET /v1/jobs/{id}, matching the Mac client.

        The client maps done/completed/… → success, failed/error → failure
        (message from "error"), anything else → still running. Queued jobs
        report "running" with no progress, per the contract.
        """
        status = {"queued": "running", "running": "running",
                  "done": "done", "failed": "failed"}[self.state]
        out: dict[str, Any] = {"job_id": self.job_id, "status": status}
        if self.state == "running" and self.progress is not None:
            out["progress"] = round(self.progress, 3)
            if self.started_at:
                out["elapsed_s"] = round(time.time() - self.started_at, 1)
            if self.step and self.steps_total:
                out["step"] = self.step
                out["steps_total"] = self.steps_total
            if self.eta_seconds is not None:
                out["eta_seconds"] = round(self.eta_seconds)
        if self.state == "done":
            out["progress"] = 1.0
            out["result_urls"] = [f"/v1/files/{n}" for n in self.result_files]
        if self.state in ("done", "failed") and self.started_at \
                and self.finished_at:
            out["elapsed_s"] = round(self.finished_at - self.started_at, 1)
        if self.state == "failed" and self.error:
            out["error"] = self.error
        if self.held:
            out["held"] = True
        if self.stage:
            out["stage"] = self.stage
        return out

    def to_disk(self) -> dict[str, Any]:
        # ETA bookkeeping attrs (_prog_*) are runtime-only.
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.to_disk(), indent=2, default=str))
        tmp.replace(self.dir / "status.json")


class JobCancelled(Exception):
    """Raised inside a handler's progress() when a cancel was requested;
    handlers get cancellation for free at every stage/step boundary."""


# A capability handler: (job, progress_cb) -> list of result file names.
Handler = Callable[[Job, Callable[[float, str], None]], list[str]]


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._cv = threading.Condition(self._lock)
        self._handlers: dict[str, Handler] = {}
        self._worker: Optional[threading.Thread] = None
        self._current: Optional[str] = None
        self._load_existing()

    # -- persistence ------------------------------------------------------

    def _load_existing(self) -> None:
        config.ensure_dirs()
        for status_file in config.JOBS_DIR.glob("*/status.json"):
            try:
                d = json.loads(status_file.read_text())
                job = Job(
                    job_id=d["job_id"], capability=d["capability"],
                    params=d.get("params", {}), state=d.get("state", "failed"),
                    progress=d.get("progress"), error=d.get("error"),
                    result_files=d.get("result_files", []),
                    stage=d.get("stage"),
                    submitted_by=d.get("submitted_by"),
                    created_at=d.get("created_at", 0.0),
                    started_at=d.get("started_at"),
                    finished_at=d.get("finished_at"),
                    receipts=d.get("receipts", {}),
                )
                # Anything that was mid-flight when the process died is failed.
                if job.state in ("queued", "running"):
                    job.state = "failed"
                    job.error = ("The service restarted while this job was "
                                 "in flight (likely a GPU out-of-memory "
                                 "restart). Please resubmit.")
                    job.save()
                self._jobs[job.job_id] = job
            except Exception:
                log.exception("could not reload job from %s", status_file)

    # -- registry ---------------------------------------------------------

    def register(self, capability: str, handler: Handler) -> None:
        self._handlers[capability] = handler

    def capabilities(self) -> list[str]:
        return sorted(self._handlers)

    # -- submission / lookup ---------------------------------------------

    def submit(self, capability: str, params: dict[str, Any],
               defer: bool = False) -> Job:
        """Create a job. With defer=True the caller must finish writing the
        job's input files and then call enqueue() — otherwise the worker
        could start before the upload is on disk."""
        if capability not in self._handlers:
            raise KeyError(capability)
        job = Job(job_id=new_id(), capability=capability, params=params)
        with self._lock:
            self._jobs[job.job_id] = job
        job.save()
        if not defer:
            self.enqueue(job)
        return job

    def enqueue(self, job: Job) -> None:
        job.save()
        with self._cv:
            self._pending.append(job.job_id)
            self._cv.notify()
        log.info("job %s queued (%s, params=%s)", job.job_id, job.capability,
                 {k: v for k, v in job.params.items()
                  if not k.endswith("_path")})

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending) + (1 if self._current else 0)

    # -- queue management (dashboard Activity controls) -------------------

    def cancel(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job_id in self._pending:
                self._pending.remove(job_id)
                job.state = "failed"
                job.error = "Cancelled before it started."
                job.finished_at = time.time()
                job.save()
                return True
            if job.state == "running":
                job.cancel_requested = True
                return True
            return False

    def hold(self, job_id: str, on: bool) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job_id not in self._pending:
                return False
            job.held = on
            job.save()
            if not on:
                self._cv.notify()
            return True

    def move(self, job_id: str, direction: str) -> bool:
        with self._cv:
            if job_id not in self._pending:
                return False
            i = self._pending.index(job_id)
            j = i - 1 if direction == "up" else i + 1
            if not 0 <= j < len(self._pending):
                return False
            self._pending[i], self._pending[j] = (self._pending[j],
                                                  self._pending[i])
            return True

    def cancel_queue(self, scope: str = "pending") -> int:
        """Drop every queued job at once (the Mac Swarm command center's
        Cancel Queue button, handoff 126). scope "all" also aborts the
        running job at its next progress checkpoint — cancelled jobs
        report failed/cancelled to their submitters, never vanish."""
        n = 0
        with self._cv:
            for jid in list(self._pending):
                job = self._jobs.get(jid)
                self._pending.remove(jid)
                if job is not None:
                    job.state = "failed"
                    job.error = "Cancelled by the swarm owner."
                    job.finished_at = time.time()
                    job.save()
                    n += 1
            if scope == "all" and self._current:
                running = self._jobs.get(self._current)
                if running is not None and running.state == "running":
                    running.cancel_requested = True
                    n += 1
        return n

    def retry(self, job_id: str) -> Optional[Job]:
        source = self.get(job_id)
        if source is None or source.state not in ("failed", "done"):
            return None
        return self.submit(source.capability, dict(source.params))

    # -- worker -----------------------------------------------------------

    def start_worker(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._worker_loop, name="gpu-worker", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                job_id = None
                while job_id is None:
                    for jid in self._pending:
                        j = self._jobs.get(jid)
                        if j is not None and not j.held:
                            job_id = jid
                            break
                    if job_id is None:
                        self._cv.wait()
                self._pending.remove(job_id)
                job = self._jobs.get(job_id)
            if job is not None:
                self._run_one(job)

    def _run_one(self, job: Job) -> None:
        handler = self._handlers[job.capability]
        self._current = job.job_id
        job.state = "running"
        job.started_at = time.time()
        job.progress = 0.0
        job.save()
        log.info("job %s started", job.job_id)

        # GPU arbitration: 3D jobs and the resident LLM cannot share the
        # 24 GB card — the job preempts, the LLM is restored afterwards.
        # Store installs are pure downloads; chat keeps running through
        # them (hub 137).
        from .llm import LLM  # noqa: PLC0415
        from . import pipeline as _pipeline  # noqa: PLC0415
        if job.capability != "store-install":
            LLM.preempt_for_job()

        def progress(frac: float, stage: str, step: Optional[int] = None,
                     steps_total: Optional[int] = None) -> None:
            if job.cancel_requested:
                raise JobCancelled()
            now = time.time()
            frac = max(0.0, min(1.0, frac))
            # ETA is per-stage (silicon-optimizer #8 feedback): seeding
            # the denoise rate with the model-load delta produced a first
            # estimate ~17x too high, and a finished stage's ETA went
            # stale into the next one. A stage boundary resets the EMA,
            # and until the new stage has its own rate signal there is NO
            # eta — honest absence beats a confident wrong number.
            if stage != job.stage:
                job._prog_prev = None
                job._prog_spu = None
                job.eta_seconds = None
            prev = getattr(job, "_prog_prev", None)
            if prev and frac > prev[1]:
                inst = (now - prev[0]) / (frac - prev[1])
                ema = getattr(job, "_prog_spu", None)
                job._prog_spu = inst if ema is None else 0.3 * inst + 0.7 * ema
            job._prog_prev = (now, frac)
            spu = getattr(job, "_prog_spu", None)
            if spu is not None and frac > 0:
                job.eta_seconds = max(0.0, spu * (1.0 - frac))
            job.progress = frac
            job.stage = stage
            job.step = step
            job.steps_total = steps_total
            job.save()

        try:
            job.result_files = handler(job, progress)
            job.state = "done"
            job.progress = 1.0
            job.finished_at = time.time()
            job.save()
            log.info("job %s done in %.1fs — receipts: %s", job.job_id,
                     job.finished_at - job.started_at, json.dumps(job.receipts))
        except JobCancelled:
            job.state = "failed"
            job.error = "Cancelled while running."
            job.finished_at = time.time()
            job.save()
            log.info("job %s cancelled", job.job_id)
        except Exception as exc:  # noqa: BLE001
            job.state = "failed"
            job.error = _human_error(exc)
            job.finished_at = time.time()
            job.save()
            log.error("job %s failed: %s\n%s", job.job_id, job.error,
                      traceback.format_exc())
            if _is_cuda_oom(exc):
                # Do not try to recover in-process: flush state and let the
                # supervisor restart us with a clean CUDA context.
                log.error("CUDA OOM — exiting for supervisor restart")
                logging.shutdown()
                os._exit(config.OOM_EXIT_CODE)
        finally:
            self._current = None
            LLM.schedule_restore(
                _pipeline.ENGINE.unload,
                is_busy=lambda: (len(self._pending) > 0
                                 or self._current is not None))


def _is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "CUDA out of memory" in text or "OutOfMemoryError" in text


def _human_error(exc: BaseException) -> str:
    """Non-2xx / error bodies are shown to the user — make them human."""
    if _is_cuda_oom(exc):
        return ("The GPU ran out of memory on this job. The service is "
                "restarting with a clean slate — please try again, or use a "
                "smaller input.")
    msg = str(exc).strip() or type(exc).__name__
    return msg[:500]


STORE = JobStore()
