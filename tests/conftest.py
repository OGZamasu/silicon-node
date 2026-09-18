"""Test-wide setup: point every on-disk location at a scratch dir before
the server package is imported, so no test can touch /opt/silicon.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_SCRATCH = Path(tempfile.mkdtemp(prefix="silicon-node-tests-"))
os.environ.setdefault("SILICON_NODE_DATA", str(_SCRATCH / "data"))
os.environ.setdefault("SILICON_NODE_SWARM_FILE", str(_SCRATCH / "swarm.json"))
os.environ.setdefault("SILICON_NODE_CLIENTS_FILE",
                      str(_SCRATCH / "clients.json"))
os.environ.setdefault("SILICON_NODE_CAP_SETTINGS",
                      str(_SCRATCH / "capability-settings.json"))
os.environ.setdefault("SILICON_NODE_SERVING",
                      str(_SCRATCH / "serving.json"))
os.environ.setdefault("SILICON_NODE_STORE_STATE",
                      str(_SCRATCH / "store-state.json"))

import pytest  # noqa: E402

from server import config  # noqa: E402


@pytest.fixture
def scratch() -> Path:
    return _SCRATCH


@pytest.fixture
def tokens(monkeypatch):
    """A node with both tokens set, plus one paired member."""
    from server.clients import CLIENTS

    monkeypatch.setattr(config, "TOKEN", "node-token-value")
    monkeypatch.setattr(config, "SWARM_TOKEN", "swarm-token-value")
    monkeypatch.setattr(config, "VALID_TOKENS",
                        {"node-token-value", "swarm-token-value"})
    name, member = CLIENTS.mint("test-member")
    yield {"node": "node-token-value", "swarm": "swarm-token-value",
           "member": member}
    CLIENTS.revoke(name)
