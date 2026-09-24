"""Files a job reads are ones the server wrote (hub 154).

Every *_path param is written by the server, into the job's own folder,
from an upload. A caller that could name one would make the node read
any file it can see — another member's inputs, the owner's pictures —
and hand back the render; a URL there made the video lane fetch it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import pipeline, talkinghead, video
from server.jobs import Job, JobStore
from server.main import _fetchable_media, app

REMOTE = ("192.168.1.50", 51234)
LOCAL = ("127.0.0.1", 51234)


def member(tokens) -> dict:
    return {"Authorization": f"Bearer {tokens['member']}"}


def job_with(tmp_path, **params) -> Job:
    return Job(job_id="t" + tmp_path.name[-8:], capability="x", params=params)


# -- the generic route never takes a path ---------------------------------

@pytest.mark.parametrize(("cap", "key"), [
    ("image-to-mesh", "image_path"),
    ("retopologize", "mesh_path"),
    ("talking-head", "audio_path"),
    ("portrait-animate", "driving_path"),
    ("text-to-video", "image_path"),
])
def test_a_generic_submit_naming_a_file_is_refused(tokens, cap, key):
    from server.jobs import STORE
    before = len(STORE.snapshot())
    r = TestClient(app, client=REMOTE).post(
        "/v1/jobs", headers=member(tokens),
        json={"capability": cap, "prompt": "x",
              key: "/mnt/c/Users/someone/Pictures/private.jpg"})
    assert r.status_code == 400
    assert key in r.json()["error"]
    assert len(STORE.snapshot()) == before     # nothing was queued


def test_a_file_capability_through_the_generic_route_names_its_own(tokens):
    r = TestClient(app, client=REMOTE).post(
        "/v1/jobs", headers=member(tokens),
        json={"capability": "image-to-mesh"})
    assert r.status_code == 400
    assert "/v1/image-to-mesh" in r.json()["error"]


# -- and the handlers only open files inside their own job ----------------

def test_input_path_accepts_the_jobs_own_upload(tmp_path):
    job = job_with(tmp_path)
    upload = job.dir / "input.png"
    job.params["image_path"] = str(upload)
    assert job.input_path("image_path") == upload.resolve()


@pytest.mark.parametrize("raw", [
    "/etc/passwd",
    "/mnt/c/Users/someone/Pictures/private.jpg",
    "http://169.254.169.254/latest/meta-data",
    "relative.png",
])
def test_input_path_refuses_anything_else(tmp_path, raw):
    job = job_with(tmp_path, image_path=raw)
    with pytest.raises(ValueError, match="uploaded with the job"):
        job.input_path("image_path")


def test_input_path_refuses_climbing_out_of_the_job(tmp_path):
    job = job_with(tmp_path)
    job.params["image_path"] = str(job.dir / ".." / "other-job" / "in.png")
    with pytest.raises(ValueError):
        job.input_path("image_path")


def test_the_mesh_handler_refuses_before_loading_anything(tmp_path,
                                                          monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the engine was reached")
    monkeypatch.setattr(pipeline.ENGINE, "retopo", explode)
    job = job_with(tmp_path, mesh_path="/etc/passwd", vert_num="1000")
    with pytest.raises(ValueError):
        pipeline.retopologize(job, lambda *a, **k: None)


def test_the_talking_head_handler_refuses_before_running(tmp_path,
                                                         monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the subprocess ran")
    monkeypatch.setattr(talkinghead.subprocess, "run", explode)
    job = job_with(tmp_path, image_path="/etc/hostname",
                   audio_path="/etc/hostname")
    with pytest.raises(ValueError):
        talkinghead.talking_head(job, lambda *a, **k: None)


def test_the_video_start_image_is_never_fetched(tmp_path):
    job = job_with(tmp_path, image_path="http://example.invalid/x.png")
    with pytest.raises(ValueError):
        video._start_image(job)


def test_the_video_start_image_is_opened_from_the_job(tmp_path):
    from PIL import Image
    job = job_with(tmp_path)
    job.dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (8, 6), (255, 0, 0, 128)).save(job.dir / "start.png")
    job.params["image_path"] = str(job.dir / "start.png")
    img = video._start_image(job)
    assert img.mode == "RGB" and img.size == (8, 6)


# -- a retry gets its own copy of the inputs ------------------------------

def test_a_retry_copies_its_inputs_into_its_own_job():
    store = JobStore()
    seen: list[str] = []
    store.register("reads", lambda job, progress: seen.append(
        job.input_path("image_path").read_text()) or {"files": []})
    first = store.submit("reads", {}, defer=True)
    first.dir.mkdir(parents=True, exist_ok=True)
    (first.dir / "input.png").write_text("pixels")
    first.params["image_path"] = str(first.dir / "input.png")
    first.state = "failed"
    again = store.retry(first.job_id)
    assert again is not None and again.job_id != first.job_id
    assert again.input_path("image_path").read_text() == "pixels"
    assert again.input_path("image_path").parent == again.dir.resolve()


def test_a_retry_whose_input_was_swept_fails_with_the_reason():
    store = JobStore()
    store.register("reads", lambda job, progress: {"files": []})
    first = store.submit("reads", {}, defer=True)
    first.params["image_path"] = str(first.dir / "gone.png")
    first.state = "failed"
    again = store.retry(first.job_id)
    assert again.state == "failed"
    assert "input is gone" in again.error


# -- chat never makes the node fetch a URL ---------------------------------

def _chat(url: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": url}}]}]}


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "https://example.com/cat.png",
    "file:///etc/passwd",
    "  HTTP://10.0.0.1/",
])
def test_remote_media_in_a_chat_is_found(url):
    assert _fetchable_media(_chat(url)) == ["image_url"]


def test_inline_media_and_plain_text_pass():
    assert _fetchable_media(_chat("data:image/png;base64,iVBORw0KGgo=")) == []
    assert _fetchable_media({"messages": [
        {"role": "user", "content": "http://example.com is a site"}]}) == []
    assert _fetchable_media({"messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "rtsp://cam/1"}},
        {"type": "input_audio", "audio_url": "https://x/a.wav"}]}]}) == \
        ["video_url", "audio_url"]
    assert _fetchable_media(None) == []
    assert _fetchable_media({"messages": "nonsense"}) == []


def test_the_chat_route_refuses_a_remote_image(tokens, monkeypatch):
    from server.llm import LLM

    class Running:
        returncode = None

        def poll(self):
            return None
    monkeypatch.setattr(LLM, "_proc", Running())

    async def never(*a, **k):
        raise AssertionError("the request reached the engine")
    import asyncio
    monkeypatch.setattr(asyncio, "create_subprocess_exec", never)
    r = TestClient(app, client=REMOTE).post(
        "/v1/chat/completions", headers=member(tokens),
        json=_chat("http://169.254.169.254/"))
    assert r.status_code == 400
    assert "data: URL" in r.json()["error"]
