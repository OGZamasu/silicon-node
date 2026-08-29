"""The VERSION file is the single source of the advertised version."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server import config
from server.main import app

VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def test_config_reports_the_version_file():
    assert config.SERVER_VERSION == VERSION_FILE.read_text().strip()


def test_health_advertises_it():
    body = TestClient(app, client=("127.0.0.1", 1)).get("/health").json()
    assert body["server"]["version"] == config.SERVER_VERSION
