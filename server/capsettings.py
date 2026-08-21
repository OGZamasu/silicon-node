"""Per-capability policy: enabled flag + remotely editable settings.

Handoff 129: the Mac's Swarm page shows an info popover per ability with
an enable toggle and simple settings. Only the knobs listed in DEFAULTS
are accepted from the API — everything else is reported back as ignored
rather than silently written.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from . import config

log = logging.getLogger("silicon-node.capsettings")

SETTINGS_FILE = Path(os.environ.get(
    "SILICON_NODE_CAP_SETTINGS", "/opt/silicon/capability-settings.json"))

# The remotely changeable knobs per ability, with their defaults. These
# are the safe subset: things a job already accepts per-request.
DEFAULTS: dict[str, dict] = {
    "image-to-mesh": {"vert_num": config.VERT_NUM_DEFAULT},
    "retopologize": {"vert_num": config.VERT_NUM_DEFAULT},
    "text-to-video": {"resolution": "720p", "wan_steps": 30,
                      "ltx_steps": 8},
    "portrait-animate": {},
    "talking-head": {},
}


class CapSettings:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        try:
            self._data = json.loads(SETTINGS_FILE.read_text())
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("unreadable %s — starting from defaults",
                          SETTINGS_FILE)

    def _save(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(SETTINGS_FILE)

    def enabled(self, cap: str) -> bool:
        with self._lock:
            return bool(self._data.get(cap, {}).get("enabled", True))

    def settings(self, cap: str) -> dict:
        with self._lock:
            merged = dict(DEFAULTS.get(cap, {}))
            merged.update(self._data.get(cap, {}).get("settings", {}))
            return merged

    def update(self, cap: str, enabled=None, settings=None) -> list[str]:
        """Apply a partial update; returns the setting keys that were
        ignored (unknown or non-scalar)."""
        ignored: list[str] = []
        with self._lock:
            entry = self._data.setdefault(cap, {})
            if enabled is not None:
                entry["enabled"] = bool(enabled)
            if settings:
                allowed = DEFAULTS.get(cap, {})
                cur = entry.setdefault("settings", {})
                for k, v in settings.items():
                    if k in allowed and isinstance(v, (str, int, float,
                                                       bool)):
                        cur[k] = v
                    else:
                        ignored.append(str(k))
            self._save()
        return ignored


CAPS = CapSettings()
