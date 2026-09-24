"""Video sizes and out-of-memory failures (hub 159).

The Mac sends "480p" / "720p" / "1080p"; the route read only WxH, so
every Mac clip came out at 1280x704 whatever was picked. And any WxH was
rendered as asked — an oversized one ran the card out of memory, the
service exited, and every member's queued jobs failed with it.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from server import jobs, video
from server.jobs import STORE, Job, JobStore
from server.main import app

REMOTE = ("192.168.1.50", 51234)


# -- the size names map to real canvases ----------------------------------

@pytest.mark.parametrize(("name", "render", "deliver", "scaling"), [
    ("480p", (832, 480), (832, 480), "native"),
    ("720p", (1280, 704), (1280, 704), "native"),
    ("1080p", (1280, 704), (1920, 1080), "upscaled"),
    (" 720P ", (1280, 704), (1280, 704), "native"),
])
def test_each_size_name_has_its_canvas(name, render, deliver, scaling):
    spec = video.canvas(name)
    assert (spec["internal_width"], spec["internal_height"]) == render
    assert (spec["width"], spec["height"]) == deliver
    assert spec["scaling"] == scaling


def test_an_unknown_size_names_the_ones_that_exist():
    with pytest.raises(ValueError) as err:
        video.canvas("4k")
    for size in ("480p", "720p", "1080p"):
        assert size in str(err.value)


@pytest.mark.parametrize(("w", "h"), [(1280, 704), (704, 1280), (832, 480),
                                      (512, 512), (256, 1280)])
def test_an_explicit_size_the_card_takes_is_rendered_as_asked(w, h):
    spec = video.canvas(width=w, height=h)
    assert (spec["internal_width"], spec["internal_height"]) == (w, h)
    assert spec["scaling"] == "native"


@pytest.mark.parametrize(("w", "h"), [(8192, 8192), (1920, 1080),
                                      (1280, 1280), (1000, 704), (128, 704),
                                      ("big", 704)])
def test_an_explicit_size_the_card_cant_take_is_refused(w, h):
    with pytest.raises(ValueError):
        video.canvas(width=w, height=h)


def test_the_handler_checks_the_size_again_before_any_gpu_work():
    job = Job(job_id="job-x", capability="text-to-video",
              params={"prompt": "p", "width": 8192, "height": 8192})
    with pytest.raises(ValueError, match="outside what this node renders"):
        video.canvas_for_job(job)


# -- through the Mac's route ------------------------------------------------

@pytest.fixture
def video_route(tokens, monkeypatch):
    """The route with the engine reported ready and nothing ever run: jobs
    are queued without a worker, then removed again."""
    import server.main as main
    monkeypatch.setattr(main, "_video_ready", lambda: True)
    monkeypatch.setattr(STORE, "_handlers",
                        {**STORE._handlers, "text-to-video": lambda j, p: []})
    made: list[str] = []

    def post(body):
        r = TestClient(app, client=REMOTE).post(
            "/v1/text-to-video", json=body,
            headers={"Authorization": f"Bearer {tokens['member']}"})
        if r.status_code == 200:
            made.append(r.json()["job_id"])
        return r
    yield post, tokens
    with STORE._lock:
        for jid in made:
            STORE._jobs.pop(jid, None)
            if jid in STORE._pending:
                STORE._pending.remove(jid)


def test_the_macs_480p_is_rendered_at_480p(video_route):
    post, _ = video_route
    r = post({"model": "wan22-ti2v-5b", "prompt": "a boat", "seconds": 2,
              "resolution": "480p"})
    assert r.status_code == 200, r.text
    job = STORE.get(r.json()["job_id"])
    assert (job.params["width"], job.params["height"]) == (832, 480)


def test_1080p_is_rendered_at_720p_and_says_it_will_be_upscaled(video_route):
    post, tokens = video_route
    r = post({"prompt": "a boat", "resolution": "1080p"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    job = STORE.get(jid)
    assert (job.params["width"], job.params["height"]) == (1280, 704)
    status = TestClient(app, client=REMOTE).get(
        f"/v1/jobs/{jid}",
        headers={"Authorization": f"Bearer {tokens['member']}"}).json()
    assert status["delivery"]["requested"] == "1080p"
    assert (status["delivery"]["width"],
            status["delivery"]["height"]) == (1920, 1080)
    assert status["delivery"]["scaling"] == "upscaled"


@pytest.mark.parametrize("resolution", ["8192x8192", "4k", "1920x1080"])
def test_a_size_the_node_cant_do_is_a_400_and_nothing_queues(video_route,
                                                            resolution):
    post, _ = video_route
    before = STORE.queue_depth()
    r = post({"prompt": "a boat", "resolution": resolution})
    assert r.status_code == 400
    assert "480p" in r.json()["error"]
    assert STORE.queue_depth() == before


def test_an_explicit_size_still_works_through_the_route(video_route):
    post, _ = video_route
    r = post({"prompt": "a boat", "resolution": "704x1280"})
    assert r.status_code == 200, r.text
    job = STORE.get(r.json()["job_id"])
    assert (job.params["width"], job.params["height"]) == (704, 1280)


def test_the_generic_route_is_held_to_the_same_limit(tokens, monkeypatch):
    before = STORE.queue_depth()
    r = TestClient(app, client=REMOTE).post(
        "/v1/jobs", json={"capability": "text-to-video", "prompt": "x",
                          "width": 8192, "height": 8192},
        headers={"Authorization": f"Bearer {tokens['member']}"})
    assert r.status_code == 400
    assert STORE.queue_depth() == before


def test_the_capability_advertises_its_sizes(tokens):
    caps = TestClient(app, client=REMOTE).get(
        "/v1/capabilities",
        headers={"Authorization": f"Bearer {tokens['member']}"}).json()
    tv = next(c for c in caps if c["id"] == "text-to-video")
    assert tv["supported_resolutions"] == ["480p", "720p", "1080p"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_an_upscaled_delivery_is_really_1920x1080(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=1280x704:rate=24:duration=0.25",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    job = Job(job_id="job-deliver", capability="text-to-video", params={})
    video._deliver(clip, job, video.canvas("1080p"))
    dims = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert dims == "1920,1080"
    assert job.to_api()["delivery"]["scaling"] == "upscaled"


# -- out of memory fails one job, not the queue ---------------------------

class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError."""


def _drain(store: JobStore, jid: str, timeout: float = 10.0) -> Job:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(jid)
        if job.state in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"{jid} never finished")


def _oom_store(monkeypatch, recovers: bool):
    exits: list[int] = []
    monkeypatch.setattr(jobs, "_recover_from_oom", lambda: recovers)
    monkeypatch.setattr(jobs.os, "_exit", lambda code: exits.append(code))
    store = JobStore()

    def oom(job, progress):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
    store.register("oom", oom)
    store.register("fine", lambda job, progress: [])
    return store, exits


def test_an_oom_fails_only_its_job_and_the_next_one_runs(monkeypatch):
    store, exits = _oom_store(monkeypatch, recovers=True)
    big = store.submit("oom", {})
    after = store.submit("fine", {})
    store.start_worker()
    failed = _drain(store, big.job_id)
    assert failed.state == "failed"
    assert "ran out of memory" in failed.error
    assert _drain(store, after.job_id).state == "done"
    assert exits == []


def test_a_device_still_broken_after_an_oom_takes_the_restart(monkeypatch):
    store, exits = _oom_store(monkeypatch, recovers=False)
    big = store.submit("oom", {})
    store.start_worker()
    _drain(store, big.job_id)
    deadline = time.time() + 5
    while not exits and time.time() < deadline:
        time.sleep(0.02)
    assert exits == [jobs.config.OOM_EXIT_CODE]
