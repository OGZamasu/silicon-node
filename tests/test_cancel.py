"""Per-job cancellation, as the Mac's Cancel render speaks it (hub 158).

POST /v1/jobs/{id}/cancel answers in the body's `cancel` field:
cancelled 200 | requested 202 | completed 409 | failed 409 | unknown 404.
A cancelled job is its own terminal state — not a failure the Mac might
re-render — and a cancel is never lost in the gap between a handler
finishing and its results being published.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from server import config
from server.jobs import Job, JobStore, new_id
from server.main import app

REMOTE = ("192.168.1.50", 51234)


def drain(store: JobStore, jid: str, timeout: float = 10.0) -> Job:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(jid)
        if job.state in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"{jid} never finished: {store.get(jid)}")


def wait_running(store: JobStore, jid: str) -> None:
    deadline = time.time() + 10
    while store.get(jid).state != "running":
        assert time.time() < deadline, "never started"
        time.sleep(0.02)


@pytest.fixture
def api(tokens, monkeypatch):
    """A fresh store with a worker behind the real routes, and a caller
    for each token."""
    import server.main as main
    store = JobStore()
    ran: list[str] = []
    release = threading.Event()

    def loop(job, progress):
        for i in range(500):
            progress(i / 500, "working")
            time.sleep(0.02)
        return []

    def gate(job, progress):          # holds the worker until released
        release.wait(10)
        progress(0.5, "working")
        return []

    def record(job, progress):
        ran.append(job.job_id)
        return []

    def boom(job, progress):
        raise RuntimeError("the model refused")
    for name, fn in (("loop", loop), ("gate", gate), ("record", record),
                     ("boom", boom)):
        store.register(name, fn)
    store.start_worker()
    monkeypatch.setattr(main, "STORE", store)

    def cancel(jid, who="swarm"):
        return TestClient(app, client=REMOTE).post(
            f"/v1/jobs/{jid}/cancel",
            headers={"Authorization": f"Bearer {tokens[who]}"})

    def submit(cap, owner="swarm"):
        job = store.submit(cap, {}, defer=True)
        job.submitted_by = {"owner": owner}
        store.enqueue(job)
        return job
    yield {"store": store, "cancel": cancel, "submit": submit, "ran": ran,
           "release": release, "tokens": tokens}
    release.set()


# -- a queued job ------------------------------------------------------------

def test_a_queued_job_is_cancelled_and_never_runs(api):
    blocker = api["submit"]("gate")                 # occupies the worker
    wait_running(api["store"], blocker.job_id)
    queued = api["submit"]("record")
    r = api["cancel"](queued.job_id)
    assert r.status_code == 200
    assert r.json() == {"job_id": queued.job_id, "cancel": "cancelled",
                        "status": "cancelled",
                        "detail": r.json()["detail"]}
    assert queued.job_id not in api["store"]._pending
    api["release"].set()
    drain(api["store"], blocker.job_id)
    time.sleep(0.2)
    assert queued.job_id not in api["ran"]
    # Idempotent: the repeat gets the same answer.
    again = api["cancel"](queued.job_id)
    assert again.status_code == 200 and again.json()["cancel"] == "cancelled"


# -- a running job -----------------------------------------------------------

def test_a_running_job_stops_and_the_job_behind_it_still_runs(api):
    running = api["submit"]("loop")
    wait_running(api["store"], running.job_id)
    behind = api["submit"]("record")
    r = api["cancel"](running.job_id)
    assert r.status_code in (200, 202)
    assert drain(api["store"], running.job_id).state == "cancelled"
    assert drain(api["store"], behind.job_id).state == "done"
    assert behind.job_id in api["ran"]
    assert api["cancel"](running.job_id).json()["cancel"] == "cancelled"


def test_a_job_between_checkpoints_answers_requested(api, monkeypatch):
    import server.main as main
    monkeypatch.setattr(main, "CANCEL_WAIT_S", 0.2)
    blocker = api["submit"]("gate")       # no checkpoint until released
    wait_running(api["store"], blocker.job_id)
    r = api["cancel"](blocker.job_id)
    assert r.status_code == 202
    assert r.json()["cancel"] == "requested"
    status = TestClient(app, client=REMOTE).get(
        f"/v1/jobs/{blocker.job_id}",
        headers={"Authorization": f"Bearer {api['tokens']['swarm']}"}).json()
    assert status["cancel"]["state"] == "requested"
    api["release"].set()
    assert drain(api["store"], blocker.job_id).state == "cancelled"


# -- the race with completion -------------------------------------------------

def test_a_cancel_after_the_last_checkpoint_still_cancels(tmp_path):
    """The cancel lands after the handler's last progress() but before the
    commit: the job ends cancelled and what it published is dropped."""
    store = JobStore()
    config.FILES_DIR.mkdir(parents=True, exist_ok=True)

    def publish_then_get_cancelled(job, progress):
        progress(0.9, "export")
        name = f"{job.job_id}-clip.mp4"
        (config.FILES_DIR / name).write_bytes(b"mp4")
        assert store.request_cancel(job.job_id)[0] == "requested"
        return [name]
    store.register("late", publish_then_get_cancelled)
    store.start_worker()
    job = store.submit("late", {})
    done = drain(store, job.job_id)
    assert done.state == "cancelled"
    assert done.result_files == []
    assert "result_urls" not in done.to_api()
    assert not (config.FILES_DIR / f"{job.job_id}-clip.mp4").exists()
    assert json.loads((job.dir / "status.json").read_text())["state"] \
        == "cancelled"


def test_a_cancel_after_the_commit_is_completed_and_keeps_results(api):
    job = api["submit"]("record")
    assert drain(api["store"], job.job_id).state == "done"
    for _ in range(2):
        r = api["cancel"](job.job_id)
        assert r.status_code == 409
        assert r.json()["cancel"] == "completed"
    assert api["store"].get(job.job_id).state == "done"


# -- terminal states and unknown ids -----------------------------------------

def test_a_failed_job_says_failed_with_its_error(api):
    job = api["submit"]("boom")
    assert drain(api["store"], job.job_id).state == "failed"
    for _ in range(2):
        r = api["cancel"](job.job_id)
        assert r.status_code == 409
        assert r.json()["cancel"] == "failed"
        assert "refused" in r.json()["detail"]


def test_an_unknown_job_is_404_unknown(api):
    r = api["cancel"](new_id(), who="member")
    assert r.status_code == 404
    assert r.json()["cancel"] == "unknown"


# -- restart ---------------------------------------------------------------

def test_a_cancel_accepted_before_a_restart_reloads_as_cancelled():
    job = Job(job_id=new_id(), capability="text-to-video", params={},
              state="running", cancel_requested=True, started_at=1.0)
    job.save()
    try:
        reloaded = JobStore().get(job.job_id)
        assert reloaded.state == "cancelled"
        assert reloaded.to_api()["status"] == "cancelled"
    finally:
        import shutil
        shutil.rmtree(job.dir, ignore_errors=True)


def test_a_restart_without_a_cancel_is_still_a_failure():
    job = Job(job_id=new_id(), capability="text-to-video", params={},
              state="running", started_at=1.0)
    job.save()
    try:
        assert JobStore().get(job.job_id).state == "failed"
    finally:
        import shutil
        shutil.rmtree(job.dir, ignore_errors=True)


# -- what the Mac reads -----------------------------------------------------

def test_status_reports_cancelled_with_a_cancel_object(api):
    job = api["submit"]("loop")
    wait_running(api["store"], job.job_id)
    api["cancel"](job.job_id)
    drain(api["store"], job.job_id)
    body = TestClient(app, client=REMOTE).get(
        f"/v1/jobs/{job.job_id}",
        headers={"Authorization": f"Bearer {api['tokens']['swarm']}"}).json()
    assert body["status"] == "cancelled"
    assert body["cancel"]["state"] == "cancelled"
    assert body["error"]            # what older Macs show


def test_video_advertises_the_cancel_action(tokens):
    caps = TestClient(app, client=REMOTE).get(
        "/v1/node",
        headers={"Authorization": f"Bearer {tokens['member']}"}).json()
    tv = next(c for c in caps["capabilities"] if c["id"] == "text-to-video")
    assert tv["supported_job_actions"] == ["cancel"]


# -- only the named job, only by its owner --------------------------------------

def test_a_member_cannot_cancel_someone_elses_job(api):
    from server.clients import CLIENTS
    other, _tok = CLIENTS.mint("test-cancel-other")
    try:
        blocker = api["submit"]("gate")
        wait_running(api["store"], blocker.job_id)
        theirs = api["submit"]("record", owner="client:test-cancel-other")
        r = api["cancel"](theirs.job_id, who="member")
        assert r.status_code == 403
        assert api["store"].get(theirs.job_id).state == "queued"
        assert theirs.job_id in api["store"]._pending
    finally:
        CLIENTS.revoke(other)


def test_a_cancel_touches_nothing_but_its_job(monkeypatch):
    """Not the chat engines, not HyperQwen, not the job running now."""
    from server.hyperqwen import HYPERQWEN
    from server.llm import LLM

    def explode(*a, **k):
        raise AssertionError("a cancel reached another tenant")
    monkeypatch.setattr(LLM, "stop", explode)
    monkeypatch.setattr(HYPERQWEN, "stop", explode)
    store = JobStore()
    store.register("noop", lambda job, progress: [])
    a = store.submit("noop", {})
    b = store.submit("noop", {})
    with store._lock:
        store._current = b.job_id           # pretend b is on the card
        b.state = "running"
    assert store.request_cancel(a.job_id)[0] == "cancelled"
    assert b.state == "running" and not b.cancel_requested
