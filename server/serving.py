"""Node-level serving switch (hub 138).

The owner can pause the whole node without stopping it: new job
submissions and member chats refuse with readable words, /v1/node stays
answerable so the Mac shows a truthful "paused" card, and jobs already
in the queue finish normally. Survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("silicon-node.serving")

SERVING_FILE = Path(os.environ.get(
    "SILICON_NODE_SERVING", "/opt/silicon/serving.json"))


class Serving:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict = {"paused": False, "reason": None, "since": None}
        try:
            self._data.update(json.loads(SERVING_FILE.read_text()))
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("unreadable %s — starting unpaused", SERVING_FILE)

    @property
    def paused(self) -> bool:
        with self._lock:
            return bool(self._data.get("paused"))

    def set(self, paused: bool, reason: str | None = None) -> None:
        with self._lock:
            self._data = {"paused": bool(paused),
                          "reason": (reason or "").strip() or None,
                          "since": time.time()}
            SERVING_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = SERVING_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2))
            tmp.replace(SERVING_FILE)
        log.info("serving %s%s", "PAUSED" if paused else "resumed",
                 f" ({reason})" if reason else "")

    def status(self) -> dict:
        with self._lock:
            return dict(self._data)

    def refusal(self) -> str:
        with self._lock:
            reason = self._data.get("reason")
        msg = "This node is paused by its owner"
        return f"{msg} — {reason}." if reason else f"{msg}."


SERVING = Serving()
