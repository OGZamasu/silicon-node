"""Members see their own work (hub 155).

Writes were already owner-scoped; reads were not — any member could list
every other member's prompts and inputs and download their renders. And
ownership was compared by display name, which a joining machine chooses.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import config, hostos
from server.jobs import STORE, Job, new_id
from server.main import _guard_loopback_forwarders, app

REMOTE = ("192.168.1.50", 51234)
LOCAL = ("127.0.0.1", 51234)


@pytest.fixture
def two_members(tokens):
    """The fixture's member plus a second one, each with a finished job
    that has an input path and an artifact on disk."""
    from server.clients import CLIENTS
    name_b, tok_b = CLIENTS.mint("test-member-b")
    config.FILES_DIR.mkdir(parents=True, exist_ok=True)
    made = {}
    for who, owner in (("a", "client:test-member"),
                       ("b", "client:test-member-b")):
        job = Job(job_id=new_id(), capability="text-to-image",
                  params={"prompt": f"{who}'s secret prompt",
                          "image_path": f"/opt/silicon/data/jobs/x/{who}.png"},
                  submitted_by={"client": owner.split(":", 1)[1],
                                "owner": owner})
        job.state = "done"
        artifact = f"{job.job_id}-image.png"
        (config.FILES_DIR / artifact).write_bytes(b"png")
        job.result_files = [artifact]
        with STORE._lock:
            STORE._jobs[job.job_id] = job
        made[who] = job
    yield {"a": tokens["member"], "b": tok_b, "swarm": tokens["swarm"],
           "node": tokens["node"], "jobs": made}
    CLIENTS.revoke(name_b)
    with STORE._lock:
        for job in made.values():
            STORE._jobs.pop(job.job_id, None)
    for job in made.values():
        for f in job.result_files:
            (config.FILES_DIR / f).unlink(missing_ok=True)


def get(path, token):
    return TestClient(app, client=REMOTE).get(
        path, headers={"Authorization": f"Bearer {token}"})


def test_a_member_lists_only_its_own_jobs(two_members):
    ids = {j["job_id"] for j in get("/v1/jobs", two_members["a"]).json()}
    assert two_members["jobs"]["a"].job_id in ids
    assert two_members["jobs"]["b"].job_id not in ids


@pytest.mark.parametrize("role", ["swarm", "node"])
def test_operators_list_everyones(two_members, role):
    ids = {j["job_id"] for j in get("/v1/jobs", two_members[role]).json()}
    assert {two_members["jobs"]["a"].job_id,
            two_members["jobs"]["b"].job_id} <= ids


@pytest.mark.parametrize("suffix", ["", "/detail"])
def test_someone_elses_job_answers_like_a_missing_one(two_members, suffix):
    other = two_members["jobs"]["b"].job_id
    r = get(f"/v1/jobs/{other}{suffix}", two_members["a"])
    assert r.status_code == 404
    assert "secret" not in r.text
    assert get(f"/v1/jobs/{two_members['jobs']['a'].job_id}{suffix}",
               two_members["a"]).status_code == 200


def test_a_members_detail_names_files_without_their_paths(two_members):
    own = two_members["jobs"]["a"].job_id
    member_view = get(f"/v1/jobs/{own}/detail", two_members["a"]).json()
    assert member_view["params"]["image_path"] == "a.png"
    admin_view = get(f"/v1/jobs/{own}/detail", two_members["swarm"]).json()
    assert admin_view["params"]["image_path"].startswith("/opt/silicon/")


def test_artifacts_go_to_their_owner_and_the_operators(two_members):
    theirs = two_members["jobs"]["b"].result_files[0]
    mine = two_members["jobs"]["a"].result_files[0]
    assert get(f"/v1/files/{theirs}", two_members["a"]).status_code == 404
    assert get(f"/v1/files/{mine}", two_members["a"]).status_code == 200
    assert get(f"/v1/files/{theirs}", two_members["swarm"]).status_code == 200


def test_an_artifact_with_no_job_left_is_the_operators(two_members):
    orphan = f"{new_id()}-image.png"
    (config.FILES_DIR / orphan).write_bytes(b"png")
    try:
        assert get(f"/v1/files/{orphan}", two_members["a"]).status_code == 404
        assert get(f"/v1/files/{orphan}",
                   two_members["node"]).status_code == 200
    finally:
        (config.FILES_DIR / orphan).unlink()


# -- ownership is the credential ------------------------------------------

@pytest.mark.parametrize("label", ["swarm (shared token)",
                                   "This node's token", "mac (admin)"])
def test_the_nodes_own_labels_cannot_be_minted(tokens, label):
    from server.clients import CLIENTS
    with pytest.raises(ValueError, match="reserved"):
        CLIENTS.mint(label)


def test_a_client_wearing_the_shared_label_owns_nothing_it_did_not_send(
        tokens, monkeypatch):
    """A name from before the reservation (or from a hand-edited
    clients.json) still can't manage jobs sent with the shared token."""
    from server.clients import CLIENTS
    impostor = "impostor-token"
    real_accepts, real_name_of = CLIENTS.accepts, CLIENTS.name_of
    monkeypatch.setattr(CLIENTS, "accepts",
                        lambda t: t == impostor or real_accepts(t))
    monkeypatch.setattr(CLIENTS, "name_of", lambda t: (
        "swarm (shared token)" if t == impostor else real_name_of(t)))
    monkeypatch.setattr(CLIENTS, "role_of", lambda t: "member")
    shared = Job(job_id=new_id(), capability="text-to-image", params={},
                 submitted_by={"client": "swarm (shared token)"})  # legacy
    with STORE._lock:
        STORE._jobs[shared.job_id] = shared
    try:
        r = TestClient(app, client=REMOTE).delete(
            f"/v1/queue/{shared.job_id}",
            headers={"Authorization": f"Bearer {impostor}"})
        assert r.status_code == 403
        r = TestClient(app, client=REMOTE).get(
            f"/v1/jobs/{shared.job_id}",
            headers={"Authorization": f"Bearer {impostor}"})
        assert r.status_code == 404
    finally:
        with STORE._lock:
            STORE._jobs.pop(shared.job_id, None)


# -- a loopback forwarder turns strict mode on ------------------------------

def test_a_raw_tcp_forwarder_makes_loopback_carry_a_token(tokens,
                                                           monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_AUTH", False)
    assert TestClient(app, client=LOCAL).get(
        "/v1/capabilities").status_code == 200
    monkeypatch.setattr(hostos, "loopback_tcp_forwards",
                        lambda port: [f"127.0.0.1:{port}"])
    assert _guard_loopback_forwarders() is True
    assert TestClient(app, client=LOCAL).get(
        "/v1/capabilities").status_code == 401


def test_no_forwarder_changes_nothing(tokens, monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_AUTH", False)
    monkeypatch.setattr(hostos, "loopback_tcp_forwards", lambda port: [])
    assert _guard_loopback_forwarders() is False
    assert config.REQUIRE_AUTH is False


@pytest.mark.parametrize(("serve", "found"), [
    ({"TCP": {"8790": {"TCPForward": "127.0.0.1:8790"}}}, True),
    ({"Foreground": {"s1": {"TCP": {"8790": {
        "TCPForward": "localhost:8790"}}}}}, True),
    ({"TCP": {"9": {"TCPForward": "[::1]:8790"}}}, True),
    # HTTP mode adds X-Forwarded-For, which the node already reads.
    ({"TCP": {"443": {"HTTPS": True}}, "Web": {"n:443": {"Handlers": {
        "/": {"Proxy": "http://127.0.0.1:8790"}}}}}, False),
    ({"TCP": {"8081": {"TCPForward": "127.0.0.1:8081"}}}, False),
    ({"TCP": {"8790": {"TCPForward": "192.168.1.9:8790"}}}, False),
    ({}, False),
])
def test_tailscale_serve_tcp_forwards_are_recognised(serve, found):
    assert bool(hostos.tcp_forwards_to(serve, 8790)) is found
