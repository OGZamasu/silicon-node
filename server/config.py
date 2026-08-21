"""Silicon Node configuration — all overridable via environment."""

import os
from pathlib import Path

SERVER_NAME = "silicon-node"
SERVER_VERSION = "0.1.0"
PLATFORM = "windows-wsl2-cuda"

# Paths (inside WSL)
DATA_DIR = Path(os.environ.get("SILICON_NODE_DATA", "/opt/silicon/data"))
JOBS_DIR = DATA_DIR / "jobs"
FILES_DIR = DATA_DIR / "files"
LOG_DIR = DATA_DIR / "logs"

LATO2_ROOT = Path(os.environ.get("LATO2_ROOT", "/opt/silicon/LATO.2"))
TRELLIS2_ROOT = Path(os.environ.get("TRELLIS2_ROOT", "/opt/silicon/TRELLIS.2"))

# Network
HOST = os.environ.get("SILICON_NODE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SILICON_NODE_PORT", "8790"))

# Auth: empty/unset = no auth required (LAN-only milestone).
# When set, every /v1/* request must carry "Authorization: Bearer <token>".
# /health stays open — it is a reachability probe.
TOKEN = os.environ.get("SILICON_NODE_TOKEN", "").strip()

# Swarm registry (mirrors the Mac's swarm.json): shared token + peer list.
# The swarm token is always *accepted* as a valid bearer once present;
# hard *enforcement* (reject missing tokens) flips on via
# SILICON_NODE_REQUIRE_AUTH=1 only after both ends confirm their clients
# send it — the Mac's Phase-1 LATO client historically sent no header.
SWARM_FILE = Path(os.environ.get("SILICON_NODE_SWARM_FILE",
                                 "/opt/silicon/swarm.json"))
SWARM_TOKEN = ""
PEERS: list[dict] = []
if SWARM_FILE.is_file():
    try:
        import json as _json
        _swarm = _json.loads(SWARM_FILE.read_text())
        SWARM_TOKEN = str(_swarm.get("swarm_token", "")).strip()
        PEERS = [p for p in _swarm.get("peers", [])
                 if isinstance(p, dict) and p.get("base_url")]
    except Exception:  # noqa: BLE001
        pass

REQUIRE_AUTH = os.environ.get("SILICON_NODE_REQUIRE_AUTH", "0") == "1"
VALID_TOKENS = {t for t in (TOKEN, SWARM_TOKEN) if t}

# Contract limits (mirrors the Mac client and LATO.2 defaults)
VERT_NUM_MIN = 200
VERT_NUM_MAX = 5000
VERT_NUM_DEFAULT = 2000

# Exit code used when the worker hits CUDA OOM: the supervisor loop in
# run-server.sh restarts the whole process rather than recovering in-process.
OOM_EXIT_CODE = 3


def ensure_dirs() -> None:
    for d in (DATA_DIR, JOBS_DIR, FILES_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
