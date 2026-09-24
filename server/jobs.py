"""Job store and single GPU worker.

One job on the GPU at a time (queue + one worker thread — same idea as the
Mac's TRELLIS MCP threading.Semaphore(1)). Jobs run in the worker, never in
the HTTP handler, so a multi-minute render cannot die with the request.

Every job persists a status.json receipt in its own directory: request
params, per-stage timings, peak VRAM, artifact paths. The store reloads
those on startup, so a worker restart (e.g. after CUDA OOM) keeps history.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
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
    state: str = "queued"  # queued | running | done | failed | cancelled
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

    def input_path(self, key: str) -> Path:
        """An input file this job was given (params[key]), as a path —
        and only ever one inside this job's own directory (hub 154).

        The server writes every *_path param itself, into job.dir, from
        an upload. A path from anywhere else means a caller wrote it, and
        opening it would hand them any file the node can read."""
        raw = self.params.get(key)
        if not raw:
            raise ValueError(f"This job was given no {key.removesuffix('_path')} "
                             "file.")
        path = Path(str(raw)).resolve()
        if not path.is_relative_to(self.dir.resolve()):
            raise ValueError(
                f"Refusing {key}: input files must be uploaded with the job, "
                "not named by path.")
        return path

    def to_api(self) -> dict[str, Any]:
        """Shape returned by GET /v1/jobs/{id}, matching the Mac client.

        The client maps done/completed/… → success, failed/error → failure
        (message from "error"), anything else → still running. Queued jobs
        report "running" with no progress, per the contract.
        """
        status = {"queued": "running", "running": "running",
                  "done": "done", "failed": "failed",
                  "cancelled": "cancelled"}[self.state]
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
        if self.state in ("done", "failed", "cancelled") and self.started_at \
                and self.finished_at:
            out["elapsed_s"] = round(self.finished_at - self.started_at, 1)
        if self.state in ("failed", "cancelled") and self.error:
            # A cancelled job keeps `error` too: Macs from before the
            # cancelled status read it as a terminal failure's message.
            out["error"] = self.error
        # The Mac reads `cancel` first (hub 158): stopped for good, or
        # asked to stop and heading there at the next checkpoint.
        if self.state == "cancelled":
            out["cancel"] = {"state": "cancelled",
                             "detail": self.error or "Cancelled."}
        elif self.cancel_requested and self.state == "running":
            out["cancel"] = {"state": "requested",
                             "detail": "Stopping at the next checkpoint."}
        if self.held:
            out["held"] = True
        if self.stage:
            out["stage"] = self.stage
        # What size the clip is, and whether it was rendered at it or
        # scaled up to it (hub 159) — video jobs only.
        if isinstance(self.params.get("delivery"), dict):
            out["delivery"] = self.params["delivery"]
        return out

    def to_disk(self) -> dict[str, Any]:
        # ETA bookkeeping attrs (_prog_*) are runtime-only.
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}

    def save(self) -> None:
        self._write(self.to_disk())

    def commit(self, **fields: Any) -> None:
        """Change fields on disk first, then in memory: whoever sees the
        new state in memory (a poller, the API) also finds it on disk.
        For terminal transitions."""
        self._write({**self.to_disk(), **fields})
        for key, value in fields.items():
            setattr(self, key, value)

    def _write(self, record: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Per-thread tmp name: the submitting API thread and the worker
        # can save concurrently, and a shared tmp made one of them
        # replace the other's already-moved file (FileNotFoundError).
        tmp = self.dir / f"status.json.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(record, indent=2, default=str))
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
                    cancel_requested=bool(d.get("cancel_requested")),
                    created_at=d.get("created_at", 0.0),
                    started_at=d.get("started_at"),
                    finished_at=d.get("finished_at"),
                    receipts=d.get("receipts", {}),
                )
                # Anything that was mid-flight when the process died is
                # failed — unless a cancel was accepted for it first: that
                # one is cancelled, never a failure a Mac might re-render.
                if job.state in ("queued", "running"):
                    if job.cancel_requested:
                        job.state = "cancelled"
                        job.error = "Cancelled."
                    else:
                        job.state = "failed"
                        job.error = ("The service restarted while this job "
                                     "was in flight (likely a GPU out-of-"
                                     "memory restart). Please resubmit.")
                    job.finished_at = job.finished_at or time.time()
                    job.save()
                self._jobs[job.job_id] = job
            except Exception:
                log.exception("could not reload job from %s", status_file)
        # Startup is the one moment a long-idle node is guaranteed to run
        # code, so it is where a disk that filled while nobody looked gets
        # its space back.
        self.prune()

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
            if job.state != "queued":
                return   # cancelled while its input was still arriving
            self._pending.append(job.job_id)
            self._cv.notify()
        log.info("job %s queued (%s, params=%s)", job.job_id, job.capability,
                 {k: v for k, v in job.params.items()
                  if not k.endswith("_path")})

    def abandon(self, job: Job, reason: str) -> None:
        """A deferred job whose input never reached the disk (upload
        refused, body undecodable). Nothing will ever enqueue it, so it
        must not sit in the table as queued forever — fail it with the
        reason the submitter was given, and retention will sweep it."""
        with self._cv:
            if job.state != "queued" or job.job_id in self._pending:
                return
            job.state = "failed"
            job.error = reason
            job.finished_at = time.time()
        job.save()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> list[Job]:
        """Every job the store knows, copied under the lock. Handlers that
        walk the table must use this: the worker adds to it and the
        retention sweep pops from it on other threads, and iterating the
        live dict while that happens raises mid-request."""
        with self._lock:
            return list(self._jobs.values())

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending) + (1 if self._current else 0)

    # -- queue management (dashboard Activity controls) -------------------

    def request_cancel(self, job_id: str) -> tuple[str, Optional[Job]]:
        """Stop one job — the one named, never "whatever is current"
        (hub 158). Returns the outcome the cancel route reports:

        cancelled  stopped for good; also the answer to every repeat
        requested  running: it stops at its next progress() checkpoint,
                   or at the commit if the handler has no more of them
        completed  already done (the commit is atomic, so a job that is
                   publishing its results is already done here)
        failed     already failed, and not by a cancel
        unknown    no job with that id
        """
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return "unknown", None
            if job.state in ("cancelled", "failed"):
                return job.state, job
            if job.state == "done":
                return "completed", job
            if job.state == "queued":
                # Waiting in the queue — or still receiving its upload,
                # in which case enqueue() will find it cancelled.
                if job_id in self._pending:
                    self._pending.remove(job_id)
                self._finish_cancelled(job, "Cancelled before it started.")
                return "cancelled", job
            job.cancel_requested = True
            job.save()   # a restart must not turn it into a failure
            return "requested", job

    def cancel(self, job_id: str) -> bool:
        """The dashboard's and the queue view's cancel: True when the job
        is stopped or stopping."""
        return self.request_cancel(job_id)[0] in ("cancelled", "requested")

    @staticmethod
    def _drop_artifacts(names: list[str]) -> None:
        """Delete files a job published to FILES_DIR before its cancel
        took effect — they are no one's results now."""
        for name in names:
            try:
                (config.FILES_DIR / Path(str(name)).name).unlink()
            except OSError:
                pass

    def _finish_cancelled(self, job: Job, detail: str) -> None:
        """Caller holds the lock. Results a cancelled job produced are not
        its results: they are dropped from the job (and from disk by the
        worker, which knows which ones it published)."""
        job.commit(state="cancelled", error=detail, result_files=[],
                   finished_at=time.time())

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
                    self._finish_cancelled(job, "Cancelled by the swarm owner.")
                    n += 1
            if scope == "all" and self._current:
                running = self._jobs.get(self._current)
                if running is not None and running.state == "running":
                    running.cancel_requested = True
                    running.save()
                    n += 1
        return n

    # -- retention --------------------------------------------------------

    def prune(self, keep: Optional[int] = None,
              max_age_days: Optional[float] = None) -> dict[str, Any]:
        """Delete the oldest finished jobs, with their inputs and artifacts.

        Only done/failed jobs are candidates: a queued or running job is
        someone's render in flight. A finished job goes only when it is
        BOTH outside the newest `keep` AND older than `max_age_days`, so
        a node idle for a month still has its last results to show. A
        zero (or less) in either limit means retention is off: nothing is
        deleted. Returns what went — a silent deleter of someone's renders
        is not something to ship.
        """
        keep = config.RETAIN_JOBS if keep is None else keep
        max_age_days = (config.RETAIN_DAYS if max_age_days is None
                        else max_age_days)

        with self._lock:
            finished = [j for j in self._jobs.values()
                        if j.state in ("done", "failed", "cancelled")]
        if keep <= 0 or max_age_days <= 0:
            return {"removed": [], "freed_bytes": 0, "kept": len(finished)}
        cutoff = time.time() - max_age_days * 86400
        # Oldest first, so "keep the newest N" is a tail slice.
        finished.sort(key=lambda j: j.finished_at or j.created_at or 0.0)
        doomed = [j for j in finished[:-keep]
                  if (j.finished_at or j.created_at or 0.0) < cutoff]

        removed, freed = [], 0
        for job in doomed:
            freed += self._delete(job)
            removed.append(job.job_id)
        if removed:
            log.info("pruned %d finished job(s), freed %.1f MB",
                     len(removed), freed / 1e6)
        return {"removed": removed, "freed_bytes": freed,
                "kept": len(finished) - len(removed)}

    def _delete(self, job: Job) -> int:
        """Remove one finished job's directory and published artifacts."""
        freed = 0
        for name in job.result_files:
            path = config.FILES_DIR / name
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError:
                pass
        for path in job.dir.rglob("*"):
            try:
                if path.is_file():
                    freed += path.stat().st_size
            except OSError:
                pass
        shutil.rmtree(job.dir, ignore_errors=True)
        with self._lock:
            self._jobs.pop(job.job_id, None)
        return freed

    def retry(self, job_id: str) -> Optional[Job]:
        source = self.get(job_id)
        if source is None or source.state not in ("failed", "done",
                                                  "cancelled"):
            return None
        params = dict(source.params)
        job = self.submit(source.capability, params, defer=True)
        job.submitted_by = source.submitted_by
        # Inputs live in the source job's directory and handlers only open
        # files inside their own (Job.input_path), so the retry gets its
        # own copies. An input retention already swept makes the retry
        # fail with that reason rather than point at another job's files.
        for key, raw in source.params.items():
            if not key.endswith("_path") or not raw:
                continue
            try:
                src = source.input_path(key)
                job.dir.mkdir(parents=True, exist_ok=True)
                dest = job.dir / src.name
                shutil.copyfile(src, dest)
                params[key] = str(dest)
            except (OSError, ValueError) as exc:
                self.abandon(job, f"Cannot retry: the {key} input is gone "
                                  f"({exc}).")
                return job
        self.enqueue(job)
        return job

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
                    # Claimed under the lock: from here on a cancel finds
                    # a running job, never one in neither place.
                    self._current = job_id
                    job.state = "running"
            if job is not None:
                self._run_one(job)

    def _run_one(self, job: Job) -> None:
        handler = self._handlers[job.capability]
        with self._cv:
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
            # The decision lane yields the card to real GPU work; it
            # rebuilds itself on the next decision.
            try:
                from .systemone import SYSTEMONE  # noqa: PLC0415
                SYSTEMONE.unload()
            except Exception:  # noqa: BLE001
                log.exception("could not release the decision lane")
            # HyperQwen wants the whole card, so a job stops it outright.
            # Unlike ninfer it is not auto-restored: the owner chose to
            # run it, and bringing a container back is their call.
            try:
                from .hyperqwen import HYPERQWEN  # noqa: PLC0415
                if HYPERQWEN.active:   # serving, or still loading
                    log.info("stopping HyperQwen for job %s", job.job_id)
                    HYPERQWEN.stop()
            except Exception:  # noqa: BLE001
                log.exception("could not release the HyperQwen engine")

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

        oom = False
        try:
            published = handler(job, progress) or []
            # The commit is one step under the store lock, so a cancel
            # lands either before it (the job is cancelled and what it
            # published is dropped) or after it (the job is done and the
            # cancel is answered "completed") — never lost between the
            # two, which is how a cancel accepted during the last export
            # used to vanish (hub 158).
            with self._cv:
                cancelled = job.cancel_requested
                if cancelled:
                    self._finish_cancelled(job, "Cancelled while running.")
                else:
                    job.commit(result_files=published, state="done",
                               progress=1.0, finished_at=time.time())
            if cancelled:
                self._drop_artifacts(published)
                log.info("job %s cancelled at the finish; results dropped",
                         job.job_id)
            else:
                log.info("job %s done in %.1fs — receipts: %s", job.job_id,
                         job.finished_at - job.started_at,
                         json.dumps(job.receipts))
        except JobCancelled:
            with self._cv:
                self._finish_cancelled(job, "Cancelled while running.")
            log.info("job %s cancelled", job.job_id)
        except Exception as exc:  # noqa: BLE001
            with self._cv:
                job.commit(state="failed", error=_human_error(exc),
                           finished_at=time.time())
            log.error("job %s failed: %s\n%s", job.job_id, job.error,
                      traceback.format_exc())
            oom = _is_cuda_oom(exc)
        finally:
            self._current = None
        # Out of memory fails THIS job, not the node (hub 159). Exiting for
        # a supervisor restart used to fail every queued job of every member
        # along with it — one oversized request, looped, kept the node
        # down. The allocator's OOM is recoverable; recover here, outside
        # the except block, so the traceback's frames (and the tensors
        # they hold) are already gone. Only a device that is still broken
        # afterwards takes the old exit.
        if oom:
            if _recover_from_oom():
                log.warning("CUDA OOM on job %s; freed the card, the queue "
                            "carries on", job.job_id)
            else:
                log.error("CUDA OOM left the device unusable — exiting for "
                          "supervisor restart")
                logging.shutdown()
                os._exit(config.OOM_EXIT_CODE)
        # Every finish is a chance to keep the disk inside its budget,
        # so retention does not depend on the node ever restarting.
        try:
            self.prune()
        except Exception:  # noqa: BLE001
            log.exception("pruning after job %s failed", job.job_id)
        LLM.schedule_restore(
            _pipeline.ENGINE.unload,
            is_busy=lambda: (len(self._pending) > 0
                             or self._current is not None))


def _is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "CUDA out of memory" in text or "OutOfMemoryError" in text


def _recover_from_oom() -> bool:
    """Free what an out-of-memory job left on the card, then check the
    device still works. False means it does not, and the caller falls
    back to a supervisor restart with a clean CUDA context."""
    import gc  # noqa: PLC0415
    # Resident pipelines may be half-built or hold the failed job's
    # buffers; the next job pays a reload, which beats a restart.
    for module in ("pipeline", "video", "image"):
        try:
            engine = getattr(importlib.import_module(
                f"{__package__}.{module}"), "ENGINE", None)
            if engine is not None:
                engine.unload()
        except Exception:  # noqa: BLE001
            log.exception("could not unload %s after an OOM", module)
    try:
        import torch  # noqa: PLC0415
        gc.collect()
        torch.cuda.empty_cache()
        probe = torch.ones(1024, device="cuda")
        ok = float(probe.sum().item()) == 1024.0
        del probe
        torch.cuda.synchronize()
        return ok
    except Exception:  # noqa: BLE001
        log.exception("the CUDA device failed its check after an OOM")
        return False


def _human_error(exc: BaseException) -> str:
    """Non-2xx / error bodies are shown to the user — make them human."""
    if _is_cuda_oom(exc):
        return ("The GPU ran out of memory on this job. Try a smaller "
                "size, fewer frames or a shorter input; other jobs in the "
                "queue are unaffected.")
    msg = str(exc).strip() or type(exc).__name__
    return msg[:500]


STORE = JobStore()
