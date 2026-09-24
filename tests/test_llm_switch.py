"""Switching the chat model: validate first, act second (hub 156/162).

The engine is never launched here. What is exercised is the order of
things around it: the name is resolved before the serving model is
touched, every spelling the Mac has ever sent resolves, and a start that
fails after the old model was stopped puts the old model back.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import llm
from server.llm import LLM
from server.main import app

LOCAL = ("127.0.0.1", 51234)   # the owner's console: an operator
DEFAULT = "qwen3_8_27b.ninfer"
OTHER = "other_model.ninfer"


class FakeProc:
    returncode = None

    def poll(self):
        return None


@pytest.fixture
def engine(monkeypatch):
    """Two installed models, the default one serving, and an engine that
    records what it was asked to do instead of launching anything."""
    import server.main as main

    monkeypatch.setattr(llm, "installed_model_files",
                        lambda: sorted([DEFAULT, OTHER]))
    monkeypatch.setattr(main, "_stop_hyperqwen_if_running", lambda: None)
    monkeypatch.setattr(main.pipeline.ENGINE, "unload", lambda: None)
    monkeypatch.setattr(LLM, "_ensure_watchdog", lambda: None)
    monkeypatch.setattr(LLM, "healthy", lambda *a, **k: LLM.running)
    monkeypatch.setattr(LLM, "_proc", FakeProc())
    monkeypatch.setattr(LLM, "_model_path", llm.NINFER_DIR / "models" / DEFAULT)
    monkeypatch.setattr(LLM, "_last_good", ("c1", DEFAULT, None))
    monkeypatch.setattr(LLM, "last_error", None)
    monkeypatch.setattr(LLM, "_expect_running", True)
    monkeypatch.setattr(LLM, "model_id", llm.MODEL_ID, raising=False)
    calls = {"start": [], "stop": 0, "broken": set()}

    def stop():
        calls["stop"] += 1
        LLM._proc = None
        LLM._expect_running = False

    def start(profile="c1", wait_healthy_s=180.0, model_file=None,
              context_length=None):
        calls["start"].append(model_file)
        if model_file in calls["broken"]:
            LLM.last_error = f"{model_file} would not load"
            raise RuntimeError(LLM.last_error)
        LLM._proc = FakeProc()
        LLM._model_path = llm.NINFER_DIR / "models" / model_file
        LLM._last_good = (profile, model_file, context_length)
        LLM.last_error = None
    monkeypatch.setattr(LLM, "stop", stop)
    monkeypatch.setattr(LLM, "start", start)
    return calls


def post(body):
    return TestClient(app, client=LOCAL).post("/v1/llm/start", json=body)


# -- validate first --------------------------------------------------------

def test_an_unknown_model_is_a_404_and_the_serving_model_keeps_answering(
        engine):
    r = post({"model": "no-such-model"})
    assert r.status_code == 404
    assert "no-such-model" in r.json()["error"]
    # The answer lists what IS installed, by the id the node advertises.
    assert f"{llm.MODEL_ID} ({DEFAULT})" in r.json()["models"]
    assert engine["stop"] == 0 and engine["start"] == []
    assert LLM.running


def test_an_unknown_profile_is_refused_before_anything_stops(engine):
    r = post({"model": llm.MODEL_ID, "profile": "c99"})
    assert r.status_code == 400
    assert engine["stop"] == 0 and LLM.running


def test_a_bad_context_is_refused_before_anything_stops(engine):
    r = post({"model": llm.MODEL_ID, "context_length": "lots"})
    assert r.status_code == 400
    assert engine["stop"] == 0 and LLM.running


@pytest.mark.parametrize("spelling", [
    "qwen3.8-27b",            # the advertised id (the Mac, from now on)
    "Qwen3.8-27B",
    "qwen3_8_27b.ninfer",     # the listed filename (the Mac, until now)
    "qwen3_8_27b",
])
@pytest.mark.parametrize("field", ["model", "model_file"])
def test_every_spelling_of_an_installed_model_resolves(engine, spelling,
                                                       field):
    assert LLM.resolve_model(**{field: spelling}).name == DEFAULT


def test_a_path_is_only_ever_a_name(engine):
    with pytest.raises(llm.UnknownModel):
        LLM.resolve_model(model_file="../../etc/passwd")
    assert LLM.resolve_model(
        model_file="/elsewhere/qwen3_8_27b.ninfer").name == DEFAULT


def test_model_file_is_tried_before_model(engine):
    path = LLM.resolve_model(model=llm.MODEL_ID, model_file=OTHER)
    assert path.name == OTHER


def test_naming_nothing_keeps_the_model_that_is_serving(engine):
    """A context-only change from the Mac's picker must not swap the
    owner's chosen model for the default."""
    LLM._model_path = llm.NINFER_DIR / "models" / OTHER
    assert LLM.resolve_model().name == OTHER


def test_a_good_switch_starts_the_named_file(engine):
    r = post({"model": "other_model", "model_file": OTHER})
    assert r.status_code == 200, r.text
    assert engine["stop"] == 1
    assert engine["start"] == [OTHER]


# -- act second, and recover ----------------------------------------------

def test_a_failed_switch_puts_the_previous_model_back(engine):
    engine["broken"].add(OTHER)
    r = post({"model_file": OTHER})
    assert r.status_code == 500
    assert "would not load" in r.json()["error"]
    assert f"{DEFAULT} was restarted" in r.json()["error"]
    assert engine["start"] == [OTHER, DEFAULT]
    assert LLM.running
    body = TestClient(app, client=LOCAL).get("/v1/node").json()["llm"]
    assert body["running"] is True and body["error"] is None


def test_a_failed_switch_and_a_failed_restore_say_chat_is_down(engine):
    engine["broken"].update({OTHER, DEFAULT})
    r = post({"model_file": OTHER})
    assert r.status_code == 500
    assert "watchdog keeps trying" in r.json()["error"]
    assert not LLM.running
    # Armed, so the watchdog brings the old model back on a quiet cycle.
    assert LLM._expect_running is True
    body = TestClient(app, client=LOCAL).get("/v1/node").json()["llm"]
    assert body["running"] is False
    assert body["error"]


def test_the_watchdog_brings_back_the_model_that_was_serving(engine):
    LLM._last_good = ("c1", OTHER, 32768)
    LLM._proc = None
    LLM.start_last_good()
    assert engine["start"] == [OTHER]
    assert LLM._last_good == ("c1", OTHER, 32768)
