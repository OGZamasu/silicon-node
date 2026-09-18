"""The small persistent stores: member credentials, capability policy,
pause state, and the host-OS abstraction that lets one codebase serve
Windows-WSL2 and native Linux.
"""

from __future__ import annotations

import json

import pytest

from server.capsettings import CAPS, DEFAULTS, SETTINGS_FILE
from server.clients import CLIENTS
from server.serving import SERVING


# --- member credentials ---------------------------------------------------

def test_minting_gives_a_long_token_and_never_repeats_a_name():
    name, token = CLIENTS.mint("mac-studio")
    try:
        assert name == "mac-studio"
        assert len(token) >= 32
        assert CLIENTS.name_of(token) == "mac-studio"
        with pytest.raises(KeyError):
            CLIENTS.mint("mac-studio")
    finally:
        CLIENTS.revoke(name)


def test_a_name_must_be_short_and_non_empty():
    for bad in ("", "   ", "x" * 81):
        with pytest.raises(ValueError):
            CLIENTS.mint(bad)


def test_revoking_kills_only_that_token():
    _, keep = CLIENTS.mint("keeper")
    name, gone = CLIENTS.mint("leaver")
    try:
        assert CLIENTS.revoke(name)
        assert not CLIENTS.accepts(gone)
        assert CLIENTS.accepts(keep)
        assert not CLIENTS.revoke(name)
    finally:
        CLIENTS.revoke("keeper")


def test_the_listing_never_shows_a_token():
    name, token = CLIENTS.mint("nosy")
    try:
        CLIENTS.count_job(name, "text-to-image")
        CLIENTS.count_llm(name)
        row = next(r for r in CLIENTS.listing() if r["name"] == name)
        assert token not in json.dumps(CLIENTS.listing())
        assert row["jobs_total"] == 1
        assert row["jobs_by_kind"] == {"text-to-image": 1}
        assert row["llm_requests"] == 1
    finally:
        CLIENTS.revoke(name)


# --- capability policy ----------------------------------------------------

def test_abilities_are_enabled_until_someone_says_otherwise():
    assert CAPS.enabled("text-to-image")
    CAPS.update("text-to-image", enabled=False)
    try:
        assert not CAPS.enabled("text-to-image")
    finally:
        CAPS.update("text-to-image", enabled=True)


def test_only_the_exposed_knobs_are_writable():
    ignored = CAPS.update("text-to-image", settings={
        "qwen_steps": 12, "sudo": True, "sana_steps": {"nested": 1}})
    assert set(ignored) == {"sudo", "sana_steps"}
    assert CAPS.settings("text-to-image")["qwen_steps"] == 12
    assert CAPS.settings("text-to-image")["sana_steps"] == \
        DEFAULTS["text-to-image"]["sana_steps"]


def test_settings_survive_a_restart():
    CAPS.update("image-to-mesh", settings={"vert_num": 1234})
    on_disk = json.loads(SETTINGS_FILE.read_text())
    assert on_disk["image-to-mesh"]["settings"]["vert_num"] == 1234


# --- pause state ----------------------------------------------------------

def test_pausing_records_the_reason_and_explains_itself():
    SERVING.set(True, "  rendering for the owner  ")
    try:
        assert SERVING.paused
        assert SERVING.status()["reason"] == "rendering for the owner"
        assert SERVING.refusal().endswith("rendering for the owner.")
    finally:
        SERVING.set(False)
    assert not SERVING.paused
    assert SERVING.refusal() == "This node is paused by its owner."


# --- host OS --------------------------------------------------------------

def test_the_host_abstraction_agrees_with_itself():
    from server import config, hostos
    assert isinstance(hostos.IS_WSL, bool)
    assert config.PLATFORM == ("windows-wsl2-cuda" if hostos.IS_WSL
                               else "linux-cuda")


def test_orca_adapter_is_pinned_and_verified(tmp_path, monkeypatch):
    """The adapter goes into the model's graph, so its URL is a commit,
    not a branch, and a downloaded file that does not match the digest
    is discarded rather than kept."""
    from server import llamacpp
    meta = llamacpp.ADAPTERS[llamacpp.ORCA_LORA_FILE]
    assert "/947a80cd1d3b4f9a97417025e6c2c62223571287/" in meta["url"]
    assert meta["sha256"] and len(meta["sha256"]) == 64

    monkeypatch.setattr(llamacpp, "GGUF_DIR", tmp_path)
    dl = llamacpp.GGUFDownloads()
    good = tmp_path / "a.gguf"
    good.write_bytes(b"adapter")
    digest = llamacpp.sha256_of(good)
    # Present and matching: nothing is queued.
    dl.start_url("http://127.0.0.1:9/never", "a.gguf", sha256=digest)
    assert "a.gguf" not in dl.active

    # A worker that "downloaded" the wrong bytes must throw the file away.
    class _Resp:
        headers = {"Content-Length": "5"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n):
            data, self.data = getattr(self, "data", b"wrong"), b""
            return data
    monkeypatch.setattr(llamacpp.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp())
    bad = tmp_path / "b.gguf"
    dl.active["b.gguf"] = {"got": 0, "total": 0, "error": None, "repo": "x"}
    dl._worker("http://127.0.0.1:9/x", "b.gguf", sha256=digest)
    assert not bad.exists()
    assert "digest" in dl.active["b.gguf"]["error"]
