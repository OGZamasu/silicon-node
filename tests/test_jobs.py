"""The queue: one job at a time, every ending recorded on disk.

These run the real worker thread with trivial handlers — no GPU, no
models — because the interesting behaviour is the bookkeeping, and a
job that vanishes silently is the failure mode that matters.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from server.jobs import Job, JobCancelled, JobStore


@pytest.fixture
def store():
    s = JobStore()
    s.register("noop", lambda job, progress: {"files": []})

    def _slow(job, progress):
        for i in range(50):
            progress(i / 50, "working")
            time.sleep(0.02)
        return {"files": []}

    def _boom(job, progress):
        raise RuntimeError("the model refused")

    s.register("slow", _slow)
    s.register("boom", _boom)
    return s


def drain(store: JobStore, job_id: str, timeout: float = 10.0) -> Job:
    """Wait for a job to reach a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job is not None and job.state in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never finished: {store.get(job_id)}")


def test_an_unknown_capability_is_refused(store):
    with pytest.raises(KeyError):
        store.submit("no-such-capability", {})


def test_a_job_runs_and_is_recorded_on_disk(store):
    store.start_worker()
    job = store.submit("noop", {"a": 1})
    done = drain(store, job.job_id)
    assert done.state == "done"
    assert done.finished_at and done.started_at
    on_disk = json.loads((done.dir / "status.json").read_text())
    assert on_disk["state"] == "done"
    assert on_disk["params"] == {"a": 1}


def test_a_failing_handler_leaves_a_readable_error(store):
    store.start_worker()
    job = drain(store, store.submit("boom", {}).job_id)
    assert job.state == "failed"
    assert "refused" in job.error


def test_a_deferred_job_does_not_run_until_enqueued(store):
    store.start_worker()
    job = store.submit("noop", {}, defer=True)
    time.sleep(0.2)
    assert store.get(job.job_id).state == "queued"
    store.enqueue(job)
    assert drain(store, job.job_id).state == "done"


def test_a_held_job_is_skipped_until_released(store):
    blocked = threading.Event()
    store.register("block", lambda job, progress: blocked.wait(10) or
                   {"files": []})
    store.start_worker()
    first = store.submit("block", {})          # occupies the worker
    held = store.submit("noop", {})
    assert store.hold(held.job_id, True)
    after = store.submit("noop", {})
    blocked.set()
    assert drain(store, first.job_id).state == "done"
    assert drain(store, after.job_id).state == "done"
    assert store.get(held.job_id).state == "queued"
    assert store.hold(held.job_id, False)
    assert drain(store, held.job_id).state == "done"


def test_cancelling_a_queued_job_fails_it_rather_than_dropping_it(store):
    job = store.submit("noop", {})             # no worker started
    assert store.cancel(job.job_id)
    assert store.get(job.job_id).state == "failed"
    assert "Cancelled" in store.get(job.job_id).error
    assert not store.cancel(job.job_id)        # already gone from the queue


def test_cancel_queue_reports_every_job_it_dropped(store):
    ids = [store.submit("noop", {}).job_id for _ in range(3)]
    assert store.cancel_queue() == 3
    assert all(store.get(i).state == "failed" for i in ids)


def test_a_running_job_sees_the_cancel_request(store):
    store.start_worker()
    job = store.submit("slow", {})
    while store.get(job.job_id).state != "running":
        time.sleep(0.02)
    assert store.cancel(job.job_id)
    assert drain(store, job.job_id).state == "failed"


def test_retry_resubmits_the_same_work_as_a_new_job(store):
    store.start_worker()
    first = drain(store, store.submit("boom", {"seed": 7}).job_id)
    again = store.retry(first.job_id)
    assert again.job_id != first.job_id
    assert again.params == {"seed": 7}
    assert store.retry("no-such-job") is None


def test_queued_jobs_can_be_reordered(store):
    a, b = store.submit("noop", {}), store.submit("noop", {})
    assert store.move(b.job_id, "up")
    assert store._pending == [b.job_id, a.job_id]
    assert not store.move(b.job_id, "up")      # already first
    assert not store.move("no-such-job", "up")


def test_an_interrupted_job_comes_back_as_failed(store):
    """A job still marked running on disk means the process died — an OOM
    restart, usually. It must not look queued forever."""
    job = Job(job_id="job-interrupted", capability="noop", params={},
              state="running")
    job.save()
    reloaded = JobStore().get("job-interrupted")
    assert reloaded.state == "failed"
    assert "restarted" in reloaded.error


def test_cancellation_is_its_own_exception_type():
    assert issubclass(JobCancelled, Exception)
