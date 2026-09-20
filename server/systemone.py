"""System One decision lane — Laya on CUDA (silicon-optimizer Decisions).

The third lane behind the Mac's Decisions panel: TypeSafe's System One is
the cloud option, laya-mlx runs on Apple Silicon, and this serves the same
typed decisions from the node's GPU so the swarm can decide without the
cloud and without occupying the Mac.

Laya (github.com/NandhaKishorM/laya, Apache-2.0 for code and weights) is a
library, not a server: a non-autoregressive encoder that answers a whole
set of typed questions in ONE forward pass. Batching is therefore native —
sending eight questions together costs about as much as sending one — and
is the entire reason to prefer it over a generative model for routing.

Three question types, matching the Mac's wire shape:
    choice  -> a label from `criteria` + a probability per option
    score   -> the expected level on an ordinal rubric + per-level
               probabilities and the rubric legend
    noul    -> a calibrated P(true)
Every answer carries a confidence; the response carries the checkpoint
actually used, per-question latency, and Laya's own routing metadata.

VRAM: the checkpoints are small next to the video/3D lanes, but they are
not free (all three resident measured at ~4.5 GiB on this card). So this
lane loads lazily by default, keeps what it loads resident for speed,
unloads itself after an idle period, and yields the card outright when a
GPU job preempts. It never competes for VRAM while nobody is deciding.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger("silicon-node.systemone")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# The wire names the Mac sends -> Laya's own short checkpoint names. The
# wire names are the Hugging Face repo tails, which is what the Mac shows
# in its model picker; Laya's API speaks the short ones.
MODEL_ALIASES: dict[str, str] = {
    "laya": "english",
    "laya-multilingual": "multilingual",
    "laya-typed-decisions": "typed-decisions",
    # Accept Laya's own spelling too, so either side can send either.
    "english": "english",
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}
# Pinned checkpoint revisions — recorded here as well as in
# requirements-decisions.txt so a drift shows up in the advertisement.
CHECKPOINTS: dict[str, dict[str, str]] = {
    "english": {
        "repo": "convaiinnovations/laya",
        "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "encoder": "ModernBERT-large 421M", "context": "512"},
    "multilingual": {
        "repo": "convaiinnovations/laya-multilingual",
        "revision": "052592a15d198d9ad47da779604259b10b47b7aa",
        "encoder": "mmBERT-base 322M", "context": "1024"},
    "typed-decisions": {
        "repo": "convaiinnovations/laya-typed-decisions",
        "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "encoder": "ModernBERT-large 421M", "context": "1024"},
}

QUESTION_TYPES = ("choice", "score", "noul")

# Caps. A question set that would swap is refused, not served slowly.
MAX_QUESTIONS = _env_int("SILICON_NODE_LAYA_MAX_QUESTIONS", 32)
MAX_OPTIONS = _env_int("SILICON_NODE_LAYA_MAX_OPTIONS", 32)
MAX_STATE_CHARS = _env_int("SILICON_NODE_LAYA_MAX_STATE_CHARS", 20_000)
MAX_BODY_BYTES = _env_int("SILICON_NODE_LAYA_MAX_BODY_BYTES", 256 * 1024)

DEVICE = os.environ.get("SILICON_NODE_LAYA_DEVICE", "cuda")
# Lazy by default: the first decision pays the build, everything after it
# is warm. Set 1 to pay it at boot instead.
PRELOAD = os.environ.get("SILICON_NODE_LAYA_PRELOAD", "0") == "1"
# How many checkpoints stay resident. 3 = all of them (~4.5 GiB); 1 keeps
# ~1.8 GiB but rebuilds on every language switch (7-10 s).
MAX_LOADED = _env_int("SILICON_NODE_LAYA_MAX_LOADED", 3)
# Hand the VRAM back when nobody has decided anything for a while.
IDLE_UNLOAD_S = _env_int("SILICON_NODE_LAYA_IDLE_UNLOAD_S", 900)
ENABLED = os.environ.get("SILICON_NODE_LAYA", "1") != "0"


class DecisionError(ValueError):
    """A bad request — the caller can fix this one (400)."""


def validate(state: Any, questions: Any, model: str | None) -> str | None:
    """Check a request before any weights are touched. Returns the Laya
    checkpoint name, or None to let Laya route by script/language."""
    if model is not None:
        if not isinstance(model, str) or model not in MODEL_ALIASES:
            raise DecisionError(
                f"Unknown model {model!r}. This node serves: "
                f"laya, laya-multilingual, laya-typed-decisions.")
    if state is None or (isinstance(state, str) and not state.strip()):
        raise DecisionError("state is required.")
    if not isinstance(state, (str, dict)):
        raise DecisionError("state must be text or a JSON object.")
    size = len(state if isinstance(state, str) else str(state))
    if size > MAX_STATE_CHARS:
        raise DecisionError(
            f"state is {size} characters; this node caps it at "
            f"{MAX_STATE_CHARS}. Summarise or split it.")
    if not isinstance(questions, dict) or not questions:
        raise DecisionError("questions must be a non-empty object.")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionError(
            f"{len(questions)} questions; this node answers at most "
            f"{MAX_QUESTIONS} in one request. Split the set.")
    for qid, q in questions.items():
        where = f"questions.{qid}"
        if not isinstance(q, dict):
            raise DecisionError(f"{where} must be an object.")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise DecisionError(
                f"{where}.type is {qtype!r}; expected one of "
                f"{', '.join(QUESTION_TYPES)}.")
        if not str(q.get("instructions") or "").strip():
            raise DecisionError(f"{where}.instructions is required.")
        criteria = q.get("criteria")
        if qtype in ("choice", "score"):
            if not isinstance(criteria, (list, dict)) or not criteria:
                raise DecisionError(
                    f"{where}.criteria is required for {qtype} questions "
                    "(a list of options, or an object mapping each option "
                    "to what it means).")
            if len(criteria) > MAX_OPTIONS:
                raise DecisionError(
                    f"{where} has {len(criteria)} options; this node caps "
                    f"them at {MAX_OPTIONS}. Laya splits a fixed token "
                    "budget across options, so large sets also decide "
                    "badly — group them or use a two-step decision.")
    return MODEL_ALIASES[model] if model else None


class LayaEngine:
    """Owns the Router: one build, kept resident, released on idle or
    when a GPU job needs the card back."""

    def __init__(self) -> None:
        self._router: Any = None
        self._lock = threading.Lock()       # one forward pass at a time
        self._state_lock = threading.Lock()  # guards load/unload
        self._loaded_at: float | None = None
        self._last_used: float = 0.0
        self._vram_mib: int | None = None
        self._load_s: float | None = None
        self._latency_ms: dict[str, float] = {}   # per-kind median-ish
        self._counts: dict[str, int] = {}
        self._error: str | None = None
        self._idle_thread: threading.Thread | None = None

    # -- availability ------------------------------------------------------

    @staticmethod
    def installed() -> bool:
        try:
            import laya  # noqa: F401, PLC0415
            return True
        except Exception:  # noqa: BLE001
            return False

    @property
    def loaded(self) -> bool:
        return self._router is not None

    # -- load / unload -----------------------------------------------------

    def _build(self) -> Any:
        """Build the Router. Caller holds _state_lock."""
        if self._router is not None:
            return self._router
        import torch  # noqa: PLC0415
        from laya import Router  # noqa: PLC0415
        before = (torch.cuda.memory_allocated()
                  if torch.cuda.is_available() else 0)
        t0 = time.time()
        log.info("building Laya Router (device=%s, max_loaded=%d)…",
                 DEVICE, MAX_LOADED)
        router = Router(preload=True, device=DEVICE, max_loaded=MAX_LOADED)
        self._load_s = round(time.time() - t0, 1)
        if torch.cuda.is_available():
            self._vram_mib = int(
                (torch.cuda.memory_allocated() - before) / 2**20)
        self._router = router
        self._loaded_at = time.time()
        self._error = None
        log.info("Laya ready in %.1fs (%s MiB)", self._load_s,
                 self._vram_mib)
        self._start_idle_watch()
        return router

    def preload_async(self) -> None:
        """Eager build at boot when SILICON_NODE_LAYA_PRELOAD=1."""
        if not (ENABLED and PRELOAD and self.installed()):
            return

        def work() -> None:
            try:
                with self._state_lock:
                    self._build()
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"[:200]
                log.exception("Laya preload failed")
        threading.Thread(target=work, daemon=True,
                         name="laya-preload").start()

    def unload(self) -> None:
        """Release the checkpoints. Safe to call when not loaded — this is
        what a GPU job calls to reclaim the card."""
        with self._state_lock:
            if self._router is None:
                return
            log.info("unloading Laya (idle or preempted)")
            self._router = None
            self._loaded_at = None
            self._vram_mib = None
        import gc  # noqa: PLC0415
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _start_idle_watch(self) -> None:
        if IDLE_UNLOAD_S <= 0 or self._idle_thread is not None:
            return

        def watch() -> None:
            while True:
                time.sleep(30)
                if self._router is None:
                    continue
                if time.time() - self._last_used > IDLE_UNLOAD_S:
                    self.unload()
        self._idle_thread = threading.Thread(target=watch, daemon=True,
                                             name="laya-idle")
        self._idle_thread.start()

    # -- decide ------------------------------------------------------------

    def predict(self, state: Any, questions: dict,
                model: str | None = None) -> dict:
        """Answer a whole question set in one forward pass."""
        if not ENABLED:
            raise RuntimeError(
                "The decision lane is disabled on this node "
                "(SILICON_NODE_LAYA=0).")
        if not self.installed():
            raise RuntimeError(
                "Laya is not installed on this node — "
                "pip install -r requirements-decisions.txt.")
        with self._state_lock:
            router = self._build()
        n = len(questions)
        kind = "batch" if n > 1 else next(
            iter(questions.values())).get("type", "single")
        # One forward pass at a time: the point of the lane is predictable
        # latency, and two concurrent passes just make both slower.
        with self._lock:
            t0 = time.perf_counter()
            result = (router.predict(state, questions, model=model)
                      if model else router.predict(state, questions))
            elapsed_ms = (time.perf_counter() - t0) * 1000
        self._last_used = time.time()
        prev = self._latency_ms.get(kind)
        # Cheap running estimate; exact numbers live in the latency table.
        self._latency_ms[kind] = round(
            elapsed_ms if prev is None else 0.3 * elapsed_ms + 0.7 * prev, 1)
        self._counts[kind] = self._counts.get(kind, 0) + 1
        return self._shape(result, elapsed_ms, n)

    @staticmethod
    def _shape(result: dict, elapsed_ms: float, n: int) -> dict:
        """Laya's own dict, plus the fields the Mac's System One decoder
        expects. Additive: every key Laya returns is passed through."""
        answers = result.get("answers", {}) or {}
        per_q = round(elapsed_ms / max(n, 1), 2)
        for ans in answers.values():
            if isinstance(ans, dict):
                ans.setdefault("latency_ms", per_q)
        routing = result.get("routing", {}) or {}
        ckpt = CHECKPOINTS.get(routing.get("model") or "", {})
        return {
            "answers": answers,
            "model": routing.get("model"),
            "checkpoint": {
                "name": routing.get("model"),
                "repo": ckpt.get("repo") or routing.get("repo"),
                "revision": ckpt.get("revision"),
                "encoder": ckpt.get("encoder"),
            },
            "routing": routing,
            "usage": result.get("usage", {}),
            "latency_ms": round(elapsed_ms, 2),
            "latency_ms_per_question": per_q,
            "questions": n,
            "engine": "laya",
        }

    # -- advertisement -----------------------------------------------------

    def status(self) -> dict:
        """What /v1/node publishes so the Mac can choose between its own
        local lane and this one."""
        return {
            "engine": "laya",
            "available": bool(ENABLED and self.installed()),
            "loaded": self.loaded,
            "device": DEVICE,
            "preload": PRELOAD,
            "max_loaded": MAX_LOADED,
            "idle_unload_s": IDLE_UNLOAD_S,
            "models": list(MODEL_ALIASES)[:3],
            "checkpoints": CHECKPOINTS,
            "question_types": list(QUESTION_TYPES),
            "limits": {"max_questions": MAX_QUESTIONS,
                       "max_options": MAX_OPTIONS,
                       "max_state_chars": MAX_STATE_CHARS,
                       "max_body_bytes": MAX_BODY_BYTES},
            "vram_mib": self._vram_mib,
            "load_seconds": self._load_s,
            "latency_ms": self._latency_ms or None,
            "served": sum(self._counts.values()) or 0,
            "uptime_s": round(time.time() - self._loaded_at)
            if self._loaded_at else None,
            "error": self._error,
        }


SYSTEMONE = LayaEngine()
