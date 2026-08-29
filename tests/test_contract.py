"""The Mac↔node wire contract, checked against the live app.

contract/*.json pins the shape the Mac parses. These tests assert the
real responses still carry those keys with those types — extra fields are
fine (the Mac parses leniently), a rename or a removal is not. The same
files are parsed from the Swift side in silicon-optimizer, so the two
repositories cannot drift without one of the two suites failing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import config
from server.jobs import Job
from server.main import app

CONTRACT = Path(__file__).resolve().parent.parent / "contract"

NUMBER = (int, float)


def load(name: str) -> dict:
    return json.loads((CONTRACT / f"{name}.json").read_text())


def conforms(expected, actual, where: str = "") -> list[str]:
    """Every key in the fixture, present in the response with a
    compatible type.

    Three deliberate relaxations, matching how the Mac actually parses:
    a null in the fixture means "key optional, type unchecked"; a null in
    the response satisfies any type, because most of these fields are
    hardware probes that legitimately have no answer; and 3 and 3.0 are
    the same type, because no client here distinguishes them.
    """
    problems: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{where or 'root'}: expected an object, got "
                    f"{type(actual).__name__}"]
        for key, want in expected.items():
            at = f"{where}.{key}" if where else key
            if want is None:
                continue
            if key not in actual:
                problems.append(f"{at}: missing from the response")
                continue
            if actual[key] is None:
                continue
            problems += conforms(want, actual[key], at)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            problems.append(f"{where}: expected an array")
        elif expected and actual:
            problems += conforms(expected[0], actual[0], f"{where}[0]")
    elif isinstance(expected, bool):
        if not isinstance(actual, bool):
            problems.append(f"{where}: expected a boolean")
    elif isinstance(expected, NUMBER):
        if isinstance(actual, bool) or not isinstance(actual, NUMBER):
            problems.append(f"{where}: expected a number")
    elif isinstance(expected, str):
        if not isinstance(actual, str):
            problems.append(f"{where}: expected a string")
    return problems


@pytest.fixture
def local() -> TestClient:
    """A caller on this machine, so the untokened dashboard path applies."""
    return TestClient(app, client=("127.0.0.1", 51234))


def test_the_fixtures_are_valid_json_and_stable_on_disk():
    names = sorted(p.name for p in CONTRACT.glob("*.json"))
    assert names == ["health.json", "job-done.json", "job-failed.json",
                     "job-queued.json", "job-running.json", "node.json"]
    for p in CONTRACT.glob("*.json"):
        assert json.loads(p.read_text())


def test_the_checksum_matches_the_fixtures():
    """silicon-optimizer keeps a byte-identical copy and checks the same
    digest, so a one-sided edit fails on both sides rather than passing
    quietly on one."""
    digest = hashlib.sha256()
    for p in sorted(CONTRACT.glob("*.json")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    recorded = (CONTRACT / "VERSION.sha256").read_text().split()[0]
    assert digest.hexdigest() == recorded, (
        "contract/ changed — update contract/VERSION.sha256 here and copy "
        "the whole directory into silicon-optimizer/contract/")


def test_health_answers_the_contract_shape(local):
    body = local.get("/health").json()
    assert conforms(load("health"), body) == []
    assert body["server"]["version"] == config.SERVER_VERSION


def test_the_node_advertisement_answers_the_contract_shape(local):
    body = local.get("/v1/node").json()
    fixture = load("node")
    # metrics/profile keys are GPU-probe output — absent without a card,
    # so only the cross-platform envelope is contractual here.
    for optional in ("metrics", "profile"):
        fixture.pop(optional)
    assert conforms(fixture, body) == []
    assert body["platform"] in ("windows-wsl2-cuda", "linux-cuda")
    assert conforms(load("node")["capabilities"][0],
                    body["capabilities"][0], "capabilities[0]") == []


def test_queue_depth_is_reported_in_both_places(local):
    assert local.get("/health").json()["queue_depth"] == \
        local.get("/v1/node").json()["metrics"]["queue_depth"]


def job(**kw) -> Job:
    return Job(job_id="job-4f2c1a9b7d3e5f608a12", capability="text-to-video",
               params={}, **kw)


def test_a_queued_job_reports_running_with_no_progress():
    api = job().to_api()
    assert conforms(load("job-queued"), api) == []
    assert api["status"] == "running"
    assert "progress" not in api, (
        "a queued job with a 0% bar reads as a stall on the Mac")


def test_a_running_job_reports_progress_step_and_eta():
    api = job(state="running", progress=0.42, started_at=1.0, step=13,
              steps_total=30, eta_seconds=121.4,
              stage="video-denoise").to_api()
    assert conforms(load("job-running"), api) == []
    assert 0.0 <= api["progress"] <= 1.0, "progress is 0–1, never 0–100"


def test_a_finished_job_reports_node_relative_result_urls():
    api = job(state="done", started_at=1.0, finished_at=215.7,
              stage="video-denoise", result_files=["clip.mp4"]).to_api()
    assert conforms(load("job-done"), api) == []
    assert api["progress"] == 1.0
    assert api["result_urls"] == ["/v1/files/clip.mp4"]


def test_a_failed_job_reports_a_sentence_a_person_can_act_on():
    api = job(state="failed", started_at=1.0, finished_at=13.1,
              error="The GPU ran out of memory at 720p.").to_api()
    assert conforms(load("job-failed"), api) == []
    assert api["error"].endswith(".")


def test_only_three_statuses_ever_reach_the_client():
    seen = {job(state=s).to_api()["status"]
            for s in ("queued", "running", "done", "failed")}
    assert seen == {"running", "done", "failed"}
