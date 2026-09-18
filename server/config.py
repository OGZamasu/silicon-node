"""Silicon Node configuration — all overridable via environment."""

import os
import secrets
from pathlib import Path

SERVER_NAME = "silicon-node"


def _version() -> str:
    """The VERSION file is what the release scripts stamp, so it is the
    only place a version may live — /health and /v1/node advertise this
    and the Mac reads it for compatibility messages."""
    try:
        return (Path(__file__).resolve().parent.parent
                / "VERSION").read_text().strip() or "0"
    except OSError:
        return "0"


SERVER_VERSION = _version()
from .hostos import IS_WSL as _IS_WSL  # noqa: E402
PLATFORM = "windows-wsl2-cuda" if _IS_WSL else "linux-cuda"

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
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

# Headers that mean "someone forwarded this": a proxy on this machine
# hands us its own loopback address as the source, so the address alone
# cannot say the sender is the owner at the console.
FORWARDED_HEADERS = ("x-forwarded-for", "x-forwarded-host",
                     "x-real-ip", "forwarded")

# Auth: a token is required for every off-box /v1/* request. Requests
# arriving on the loopback interface are the owner's own dashboard and
# tooling, and stay open. /health stays open everywhere — it is a
# reachability probe.
TOKEN = os.environ.get("SILICON_NODE_TOKEN", "").strip()

# Swarm registry (mirrors the Mac's swarm.json): shared token + peer list.
# The swarm token is accepted as a valid bearer once present, and is also
# the admin credential for token management.
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

# Strict mode: also demand a token from loopback callers. Off by default
# because the node's own dashboard and tray GUI are loopback clients that
# send no header; a remote caller is required to carry one regardless.
REQUIRE_AUTH = os.environ.get("SILICON_NODE_REQUIRE_AUTH", "0") == "1"
VALID_TOKENS = {t for t in (TOKEN, SWARM_TOKEN) if t}


def is_loopback(ip: str | None) -> bool:
    return bool(ip) and ip in LOOPBACK_HOSTS


def is_forwarded(headers) -> bool:
    return any(str(headers.get(h, "")).strip() for h in FORWARDED_HEADERS)


def same_token(supplied: str, secret: str) -> bool:
    """Constant-time equality: a token check must not leak the secret one
    comparison at a time. compare_digest rejects non-ASCII strings, and a
    token that isn't ASCII is not one of ours anyway."""
    if not supplied or not secret:
        return False
    try:
        return secrets.compare_digest(supplied, secret)
    except TypeError:
        return False


def token_valid(supplied: str) -> bool:
    return any(same_token(supplied, t) for t in VALID_TOKENS)


def is_swarm_token(supplied: str) -> bool:
    return same_token(supplied, SWARM_TOKEN)


def is_node_token(supplied: str) -> bool:
    return same_token(supplied, TOKEN)


def effective_host() -> str:
    """The address the service may actually bind.

    The one rule, in code rather than in the README: a node with no token
    is not allowed to listen beyond this machine, because an
    unauthenticated jobs API is an unauthenticated remote-execution
    service. The Mac enforces the same rule in ControlServer.start.
    """
    if HOST in LOOPBACK_HOSTS or VALID_TOKENS:
        return HOST
    return "127.0.0.1"


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
