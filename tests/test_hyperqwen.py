"""The HyperQwen engine as a setting: off unless asked for, exclusive
with the other chat engines, and operator-only to switch.

Docker-free: docker.exe is never invoked. What is exercised is the policy
around the container — the default-off setting, the knobs the dashboard
writes, who may start it, and the arbitration that keeps one language
engine on the card.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import hyperqwen
from server.capsettings import CAPS, DEFAULTS
from server.main import app

LOCAL = ("127.0.0.1", 51234)
REMOTE = ("192.168.1.50", 51234)


@pytest.fixture(autouse=True)
def _reset_setting():
    """Each test starts from the shipped default (off)."""
    CAPS.update("hyperqwen", settings=dict(DEFAULTS["hyperqwen"]))
    yield
    CAPS.update("hyperqwen", settings=dict(DEFAULTS["hyperqwen"]))


@pytest.fixture
def local() -> TestClient:
    return TestClient(app, client=LOCAL)


def set_enabled(c: TestClient, on: bool):
    return c.post("/v1/capabilities/hyperqwen",
                  json={"settings": {"enabled": on}})


# -- it is a setting, and it is off ---------------------------------------

def test_the_engine_ships_switched_off():
    assert DEFAULTS["hyperqwen"]["enabled"] is False
    assert hyperqwen.enabled() is False


def test_the_owner_can_turn_it_on_and_off_through_the_settings_route(local):
    assert set_enabled(local, True).status_code == 200
    assert hyperqwen.enabled() is True
    assert set_enabled(local, False).status_code == 200
    assert hyperqwen.enabled() is False


def test_settings_that_are_not_exposed_are_reported_not_written(local):
    r = local.post("/v1/capabilities/hyperqwen",
                   json={"settings": {"mode": "batch", "nonsense": 1}})
    assert r.status_code == 200
    assert "nonsense" in r.json().get("warning", "")
    assert CAPS.settings("hyperqwen")["mode"] == "batch"


def test_the_knobs_the_dashboard_offers_are_the_ones_that_exist(local):
    st = local.get("/v1/hyperqwen").json()
    assert set(st["modes"]) == set(hyperqwen.MODES)
    assert set(st["contexts"]) == set(hyperqwen.CONTEXTS)
    assert set(st["specs"]) == set(hyperqwen.SPECS)
    for knob in ("mode", "context", "spec", "prefix_cache", "gpu_util"):
        assert knob in st["settings"]


# -- refusals --------------------------------------------------------------

def test_starting_it_while_switched_off_says_so(local, monkeypatch):
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "docker_ready",
                        staticmethod(lambda: (True, "29.0")))
    r = local.post("/v1/hyperqwen/start", json={})
    assert r.status_code == 500
    assert "switched off" in r.json()["error"]


def test_it_reports_docker_being_down_rather_than_failing_obscurely(
        local, monkeypatch):
    set_enabled(local, True)
    monkeypatch.setattr(
        hyperqwen.HyperQwenManager, "docker_ready",
        staticmethod(lambda: (False, "Docker Desktop is installed but not "
                                     "running — start it, then load this "
                                     "engine again.")))
    r = local.post("/v1/hyperqwen/start", json={})
    assert r.status_code == 500
    assert "not running" in r.json()["error"]
    # …and the install route says the same thing with a 503.
    assert local.post("/v1/hyperqwen/install").status_code == 503


def test_an_unknown_mode_is_refused(local, monkeypatch):
    set_enabled(local, True)
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "docker_ready",
                        staticmethod(lambda: (True, "29.0")))
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "checked_out", True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "image_present",
                        lambda: True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "prepared", lambda: True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "write_env", lambda: {})
    r = local.post("/v1/hyperqwen/start", json={"mode": "turbo"})
    assert r.status_code == 500
    assert "single" in r.json()["error"] and "batch" in r.json()["error"]


def test_a_queued_gpu_job_blocks_the_start(local, monkeypatch):
    set_enabled(local, True)
    from server.jobs import STORE
    monkeypatch.setattr(STORE, "queue_depth", lambda: 1)
    r = local.post("/v1/hyperqwen/start", json={})
    assert r.status_code == 409
    assert "whole card" in r.json()["error"]


def test_starting_it_uninstalled_kicks_off_the_install_and_says_so(
        local, monkeypatch):
    set_enabled(local, True)
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "docker_ready",
                        staticmethod(lambda: (True, "29.0")))
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "checked_out", False)
    called: list[bool] = []
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "install_async",
                        lambda: called.append(True))
    r = local.post("/v1/hyperqwen/start", json={})
    assert r.status_code == 500
    assert "installing" in r.json()["error"]
    assert called


# -- one language engine per card -----------------------------------------

def test_starting_ninfer_stops_hyperqwen(monkeypatch):
    import server.main as main
    stopped: list[bool] = []
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "checked_out", True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "healthy",
                        lambda *a, **k: True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "stop",
                        lambda: stopped.append(True))
    main._stop_hyperqwen_if_running()
    assert stopped


def test_freeing_the_card_is_cheap_when_it_was_never_installed(monkeypatch):
    import server.main as main
    monkeypatch.setattr(hyperqwen.HyperQwenManager, "checked_out", False)

    def explode(*a, **k):
        raise AssertionError("must not probe docker when not installed")
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "healthy", explode)
    main._stop_hyperqwen_if_running()   # no raise


def test_the_env_it_writes_matches_the_settings(local, tmp_path,
                                                monkeypatch):
    monkeypatch.setattr(hyperqwen, "CHECKOUT", tmp_path)
    CAPS.update("hyperqwen", settings={"spec": "mtp", "context": "huge",
                                       "dflash_tokens": 15,
                                       "prefix_cache": True})
    hyperqwen.HYPERQWEN.write_env()
    env = (tmp_path / ".env").read_text()
    assert "SPEC=mtp" in env
    assert "CTX=huge" in env
    assert "DFLASH_TOKENS=15" in env
    assert f"PORT={hyperqwen.PORT}" in env
    # Docker Desktop on WSL2 aborts without this one.
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY=1" in env
    # Defaults are left to the project rather than restated.
    assert "GPU_UTIL" not in env


def test_the_default_context_is_not_written_as_an_override(local, tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(hyperqwen, "CHECKOUT", tmp_path)
    hyperqwen.HYPERQWEN.write_env()
    assert "CTX=" not in (tmp_path / ".env").read_text()


# -- advertisement ---------------------------------------------------------

def test_the_node_advertises_the_engine_compactly(local):
    hq = local.get("/v1/node").json()["hyperqwen"]
    assert hq["engine"] == "hyperqwen"
    assert hq["enabled"] is False
    assert hq["running"] is False
    # The advertisement is polled every 2.5 s — it must not carry the
    # whole status page (settings, docker probe, container state).
    assert "settings" not in hq and "docker" not in hq


def test_a_broken_docker_probe_does_not_break_the_advertisement(
        local, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("docker exploded")
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "status", boom)
    body = local.get("/v1/node").json()
    assert body["hyperqwen"]["running"] is False
    assert body["capabilities"]


# -- auth ------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/v1/hyperqwen/start",
                                  "/v1/hyperqwen/stop",
                                  "/v1/hyperqwen/install"])
def test_a_member_may_not_operate_the_engine(tokens, path):
    c = TestClient(app, client=REMOTE)
    r = c.post(path, json={},
               headers={"Authorization": f"Bearer {tokens['member']}"})
    assert r.status_code == 403
    assert "owner" in r.json()["error"] or "admin" in r.json()["error"]


def test_status_needs_a_token_off_box(tokens):
    c = TestClient(app, client=REMOTE)
    assert c.get("/v1/hyperqwen").status_code == 401
    ok = c.get("/v1/hyperqwen",
               headers={"Authorization": f"Bearer {tokens['member']}"})
    assert ok.status_code == 200


def test_install_covers_model_preparation_not_just_the_image(monkeypatch,
                                                             tmp_path):
    """The ~19.5 GB preparation belongs to install, not to start: a start
    that blocked on it would sit there for tens of minutes with nothing
    to show, which is exactly what compose does if you let it."""
    monkeypatch.setattr(hyperqwen, "CHECKOUT", tmp_path)
    (tmp_path / "docker-compose.yml").write_text("{}")
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "image_present", lambda: True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "prepared", lambda: False)
    # docker.exe only sees mounted drives, so _compose_argv refuses a
    # scratch path — that rule has its own test below.
    monkeypatch.setattr(hyperqwen, "_compose_argv",
                        lambda *a: ["docker", "compose", *a])
    ran: list[list[str]] = []

    def fake_run(argv, timeout=120.0):
        ran.append(argv)
        import subprocess as sp
        return sp.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(hyperqwen, "_run", fake_run)
    hyperqwen.HYPERQWEN.install_state = None
    hyperqwen.HYPERQWEN.install_async()
    for _ in range(50):
        if any("prepare" in a for argv in ran for a in argv):
            break
        time.sleep(0.1)
    assert any("prepare" in a for argv in ran for a in argv), ran


def test_an_unprepared_model_counts_as_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(hyperqwen, "CHECKOUT", tmp_path)
    (tmp_path / "docker-compose.yml").write_text("{}")
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "image_present", lambda: True)
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "prepared", lambda: False)
    assert hyperqwen.HYPERQWEN.installed() is False
    monkeypatch.setattr(hyperqwen.HYPERQWEN, "prepared", lambda: True)
    assert hyperqwen.HYPERQWEN.installed() is True


def test_a_checkout_windows_cannot_see_is_refused_in_words(monkeypatch):
    """docker.exe runs on Windows; a path inside the distro is invisible
    to it, so say that rather than raising a path-conversion error."""
    if not hyperqwen.hostos.IS_WSL:
        pytest.skip("only meaningful on the WSL node")
    monkeypatch.setattr(hyperqwen, "CHECKOUT", Path("/opt/silicon/hq"))
    with pytest.raises(RuntimeError, match="Windows cannot see"):
        hyperqwen._compose_argv("ps")
