"""Retention and upload limits.

A node that renders 720p clips fills its disk in weeks, and the failure
lands on a machine nobody is sitting at: every capability fails at once,
for no reason the Mac can explain. These tests are the disk budget.
"""

import time

import pytest
from fastapi.testclient import TestClient

from server import config
from server.jobs import STORE, Job, JobStore
from server.main import app


@pytest.fixture
def store():
    """The process-wide store the endpoints use, emptied of test jobs
    afterwards so one test's fixtures cannot survive into another's counts.
    """
    before = set(STORE._jobs)
    yield STORE
    for job_id in set(STORE._jobs) - before:
        job = STORE._jobs.pop(job_id)
        STORE._delete(job)


@pytest.fixture
def client():
    # Loopback, so the auth middleware is not what these tests are about.
    return TestClient(app, client=("127.0.0.1", 51234))


def finished_job(store: JobStore, *, age_days: float = 0.0,
                 state: str = "done", artifacts: int = 1) -> Job:
    """A job as it looks after the worker is done with it: a directory with
    inputs, a status.json receipt, and published files under FILES_DIR."""
    job = Job(job_id=f"job-test-{time.time_ns()}", capability="image-to-mesh",
              params={}, state=state)
    job.created_at = time.time() - age_days * 86400
    job.finished_at = job.created_at + 10
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "input.png").write_bytes(b"x" * 1024)
    names = []
    for i in range(artifacts):
        name = f"{job.job_id}-out{i}.glb"
        (config.FILES_DIR / name).write_bytes(b"y" * 2048)
        names.append(name)
    job.result_files = names
    job.save()
    store._jobs[job.job_id] = job
    return job


class TestRetention:

    def test_an_old_finished_job_and_its_artifacts_are_deleted(self, store):
        old = finished_job(store, age_days=30)
        result = store.prune(keep=0, max_age_days=14)

        assert old.job_id in result["removed"]
        assert not old.dir.exists()
        assert not (config.FILES_DIR / old.result_files[0]).exists()
        assert store.get(old.job_id) is None
        # Both the 1 KB input and the 2 KB artifact are accounted for, so
        # the dashboard can say what it actually reclaimed.
        assert result["freed_bytes"] >= 1024 + 2048

    def test_a_recent_job_survives_the_age_sweep(self, store):
        fresh = finished_job(store, age_days=1)
        store.prune(keep=0, max_age_days=14)

        assert fresh.dir.exists()
        assert store.get(fresh.job_id) is not None

    def test_the_newest_jobs_survive_however_old_they_are(self, store):
        jobs = [finished_job(store, age_days=days) for days in (90, 60, 30)]
        # An idle node still has its last results to show.
        store.prune(keep=2, max_age_days=14)

        assert store.get(jobs[0].job_id) is None
        assert store.get(jobs[1].job_id) is not None
        assert store.get(jobs[2].job_id) is not None

    def test_a_count_limit_prunes_without_any_age_limit(self, store):
        jobs = [finished_job(store, age_days=d) for d in (0.3, 0.2, 0.1)]
        store.prune(keep=1, max_age_days=0)

        assert [j.job_id for j in jobs if store.get(j.job_id)] \
            == [jobs[2].job_id]

    def test_failed_jobs_are_pruned_too(self, store):
        failed = finished_job(store, age_days=30, state="failed")
        store.prune(keep=0, max_age_days=14)

        assert store.get(failed.job_id) is None

    def test_a_queued_or_running_job_is_never_pruned(self, store):
        queued = finished_job(store, age_days=99, state="queued")
        running = finished_job(store, age_days=99, state="running")
        store.prune(keep=0, max_age_days=0.0001)

        # Someone is waiting on both of these, whatever their timestamps say.
        assert store.get(queued.job_id) is not None
        assert store.get(running.job_id) is not None

    def test_pruning_nothing_is_not_an_error(self, store):
        assert store.prune(keep=0, max_age_days=14) == {
            "removed": [], "freed_bytes": 0, "kept": 0}

    def test_defaults_come_from_config(self, store, monkeypatch):
        monkeypatch.setattr(config, "RETAIN_JOBS", 2)
        monkeypatch.setattr(config, "RETAIN_DAYS", 7)
        old = finished_job(store, age_days=30)
        for _ in range(2):
            finished_job(store, age_days=0)
        # Called the way the worker calls it: no arguments.
        store.prune()

        assert store.get(old.job_id) is None

    def test_a_missing_artifact_does_not_stop_the_sweep(self, store):
        job = finished_job(store, age_days=30)
        (config.FILES_DIR / job.result_files[0]).unlink()
        # Half-deleted state is normal after a crash mid-prune; it must not
        # leave the job directory behind forever.
        store.prune(keep=0, max_age_days=14)

        assert store.get(job.job_id) is None
        assert not job.dir.exists()


class TestPruneEndpoint:

    def test_a_member_cannot_delete_other_peoples_results(
            self, client, tokens):
        response = client.post(
            "/v1/jobs/prune", json={},
            headers={"Authorization": f"Bearer {tokens['member']}"})
        assert response.status_code == 403

    def test_the_owner_can_reclaim_disk_on_demand(self, client, tokens, store):
        job = finished_job(store, age_days=30)
        response = client.post(
            "/v1/jobs/prune", json={"keep": 0, "max_age_days": 14},
            headers={"Authorization": f"Bearer {tokens['node']}"})

        assert response.status_code == 200
        assert job.job_id in response.json()["removed"]

    def test_nonsense_limits_are_refused_with_a_reason(self, client, tokens):
        response = client.post(
            "/v1/jobs/prune", json={"keep": "lots"},
            headers={"Authorization": f"Bearer {tokens['node']}"})
        assert response.status_code == 400
        assert "whole number" in response.json()["error"]


class TestUploadLimits:

    def test_an_oversized_body_is_refused_before_it_is_read(
            self, client, tokens):
        # No body sent at all: the Content-Length claim alone is enough,
        # which is the point — the disk is never touched.
        response = client.post(
            "/v1/image-to-mesh",
            headers={"Authorization": f"Bearer {tokens['node']}",
                     "Content-Length": str(config.MAX_UPLOAD_BYTES + 1),
                     "Content-Type": "application/octet-stream"})
        assert response.status_code == 413
        assert "MB" in response.text

    def test_a_normal_body_is_not_refused_by_the_size_check(
            self, client, tokens):
        from server import uploads
        uploads.check_declared_size(str(config.MAX_UPLOAD_BYTES))
        uploads.check_declared_size(None)
        uploads.check_declared_size("not-a-number")

    def test_an_upload_is_streamed_off_the_event_loop(self, tmp_path):
        import asyncio
        import io

        from fastapi import UploadFile

        from server import uploads

        payload = b"z" * (uploads.CHUNK_BYTES + 7)
        upload = UploadFile(filename="big.png", file=io.BytesIO(payload))
        destination = tmp_path / "input.png"
        written = asyncio.run(uploads.save_upload(upload, destination))

        assert written == len(payload)
        assert destination.read_bytes() == payload

    def test_an_upload_over_the_limit_leaves_no_partial_file(self, tmp_path):
        import asyncio
        import io

        from fastapi import HTTPException, UploadFile

        from server import uploads

        upload = UploadFile(filename="big.png", file=io.BytesIO(b"z" * 4096))
        destination = tmp_path / "input.png"
        try:
            asyncio.run(uploads.save_upload(upload, destination, limit=1024))
            raise AssertionError("expected the upload to be refused")
        except HTTPException as exc:
            assert exc.status_code == 413
        # A truncated PNG on disk would be enqueued as a real job input.
        assert not destination.exists()

    def test_decoded_bodies_are_held_to_the_same_ceiling(self, tmp_path):
        import asyncio

        from fastapi import HTTPException

        from server import uploads

        assert asyncio.run(
            uploads.write_bytes(tmp_path / "a.bin", b"ok", limit=8)) == 2
        try:
            asyncio.run(
                uploads.write_bytes(tmp_path / "b.bin", b"z" * 9, limit=8))
            raise AssertionError("expected the write to be refused")
        except HTTPException as exc:
            assert exc.status_code == 413
