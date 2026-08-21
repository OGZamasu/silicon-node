"""Per-client swarm credentials (handoff request 125).

The shared swarm_token stays exactly what it is today and becomes the
admin credential; every paired member machine gets its own revocable
token minted here. Revoking one member then means deleting one row, not
rotating a secret shared by everyone.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path

log = logging.getLogger("silicon-node.clients")

CLIENTS_FILE = Path(os.environ.get("SILICON_NODE_CLIENTS_FILE",
                                   "/opt/silicon/clients.json"))


class ClientStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[dict] = []
        self._tokens: dict[str, dict] = {}
        self._last_persist = 0.0
        self._load()

    def _load(self) -> None:
        try:
            self._clients = json.loads(CLIENTS_FILE.read_text())
        except FileNotFoundError:
            self._clients = []
        except Exception:  # noqa: BLE001
            log.exception("could not read %s — starting empty",
                          CLIENTS_FILE)
            self._clients = []
        self._tokens = {c["token"]: c for c in self._clients}

    def _save(self) -> None:
        CLIENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CLIENTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._clients, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(CLIENTS_FILE)

    def mint(self, name: str) -> tuple[str, str]:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("A client needs a short, non-empty name.")
        with self._lock:
            if any(c["name"] == name for c in self._clients):
                raise KeyError(name)
            token = secrets.token_urlsafe(32)  # never logged
            c = {"name": name, "token": token,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "last_seen": None}
            self._clients.append(c)
            self._tokens[token] = c
            self._save()
            return name, token

    def revoke(self, name: str) -> bool:
        with self._lock:
            keep = [c for c in self._clients if c["name"] != name]
            if len(keep) == len(self._clients):
                return False
            self._clients = keep
            self._tokens = {c["token"]: c for c in self._clients}
            self._save()
            return True

    def listing(self) -> list[dict]:
        with self._lock:
            return [{"name": c["name"], "created": c["created"],
                     "last_seen": c["last_seen"]} for c in self._clients]

    def name_of(self, token: str) -> str | None:
        """The paired client's name for a live token, else None."""
        with self._lock:
            c = self._tokens.get(token)
            return c["name"] if c else None

    def accepts(self, token: str) -> bool:
        """True if the bearer is a live client token. Stamps last_seen,
        persisted at most once a minute — every request would otherwise
        rewrite the file."""
        if not token:
            return False
        with self._lock:
            c = self._tokens.get(token)
            if c is None:
                return False
            c["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            now = time.time()
            if now - self._last_persist > 60:
                self._last_persist = now
                self._save()
            return True


CLIENTS = ClientStore()
