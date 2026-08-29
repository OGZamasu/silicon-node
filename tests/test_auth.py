"""The node's access rules, as executable statements of policy.

Three roles: the swarm admin, the node owner, and a paired member. The
first two operate the machine; a member submits jobs and chats. On top
of that, one bind rule: an untokened node stays on loopback.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import config
from server.main import app

LOCAL = ("127.0.0.1", 51234)
REMOTE = ("192.168.1.50", 51234)

# Every route a member must not reach. Bodies are deliberately invalid or
# name nothing real, so an operator's request is refused by the route
# itself instead of downloading weights or opening a file manager.
OPERATOR_ROUTES = [
    ("post", "/v1/capabilities/no-such-ability", {"enabled": False}),
    ("post", "/v1/models/no-such-model/reveal", None),
    ("delete", "/v1/models/no-such-model", None),
    ("post", "/v1/gguf/download", {}),
    ("post", "/v1/llm/models/download", {"model_id": ""}),
    ("post", "/v1/serving", {"paused": True, "reason": "test"}),
    ("post", "/v1/llm/stop", None),
    ("post", "/v1/gguf/stop", None),
]


def client(addr=REMOTE) -> TestClient:
    return TestClient(app, client=addr)


def call(c: TestClient, method: str, path: str, body, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return c.request(method.upper(), path, json=body, headers=headers)


# --- who may talk to the node at all -------------------------------------

def test_loopback_needs_no_token(tokens):
    """The node's own dashboard sends no header and must keep working."""
    r = client(LOCAL).get("/v1/capabilities")
    assert r.status_code == 200


def test_remote_without_token_is_rejected(tokens):
    r = client(REMOTE).get("/v1/capabilities")
    assert r.status_code == 401


def test_wrong_token_is_rejected_from_anywhere(tokens):
    for addr in (LOCAL, REMOTE):
        r = call(client(addr), "get", "/v1/capabilities", None, "not-a-token")
        assert r.status_code == 401


def test_strict_mode_also_demands_a_token_from_loopback(tokens, monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_AUTH", True)
    assert client(LOCAL).get("/v1/capabilities").status_code == 401
    r = call(client(LOCAL), "get", "/v1/capabilities", None, tokens["node"])
    assert r.status_code == 200


def test_health_is_open(tokens):
    assert client(REMOTE).get("/health").status_code == 200


def test_every_token_reaches_the_read_only_api(tokens):
    for role in ("node", "swarm", "member"):
        r = call(client(REMOTE), "get", "/v1/capabilities", None,
                 tokens[role])
        assert r.status_code == 200, role


# --- who may operate the machine -----------------------------------------

@pytest.mark.parametrize(("method", "path", "body"), OPERATOR_ROUTES)
def test_members_cannot_operate_the_node(tokens, method, path, body):
    r = call(client(REMOTE), method, path, body, tokens["member"])
    assert r.status_code == 403, f"{method} {path} let a member through"


@pytest.mark.parametrize(("method", "path", "body"), OPERATOR_ROUTES)
def test_operators_are_not_blocked(tokens, method, path, body):
    """The owner and the admin get past the guard — whatever the route
    then decides about the state of the machine."""
    for role in ("node", "swarm"):
        r = call(client(REMOTE), method, path, body, tokens[role])
        assert r.status_code != 403, f"{method} {path} blocked the {role}"
    if path == "/v1/serving":  # leave the node accepting work again
        call(client(REMOTE), "post", path, {"paused": False}, tokens["node"])


def test_unauthenticated_remote_never_reaches_an_operator_route(tokens):
    for method, path, body in OPERATOR_ROUTES:
        r = call(client(REMOTE), method, path, body, None)
        assert r.status_code == 401


def test_client_management_needs_the_admin_token(tokens):
    c = client(REMOTE)
    assert call(c, "get", "/swarm/clients", None, None).status_code == 401
    assert call(c, "get", "/swarm/clients", None,
                tokens["member"]).status_code == 403
    # Even the node token is not the credential store's admin.
    assert call(c, "get", "/swarm/clients", None,
                tokens["node"]).status_code == 401
    assert call(c, "get", "/swarm/clients", None,
                tokens["swarm"]).status_code == 200


# --- token comparison and bind policy ------------------------------------

def test_token_helpers_only_accept_the_real_thing(tokens):
    assert config.token_valid("node-token-value")
    assert config.token_valid("swarm-token-value")
    assert not config.token_valid("")
    assert not config.token_valid("node-token-valu")
    assert not config.token_valid("node-token-value ")
    assert config.is_swarm_token("swarm-token-value")
    assert not config.is_swarm_token("node-token-value")
    assert config.is_node_token("node-token-value")
    assert not config.is_node_token("swarm-token-value")


def test_a_non_ascii_token_is_refused_not_a_500(tokens):
    """compare_digest rejects non-ASCII input; that must read as "wrong
    token", not as a server error."""
    assert not config.token_valid("nöde-token-value")
    # Raw bytes: a real client can put latin-1 on the wire, which
    # Starlette hands us back as a non-ASCII str.
    r = client(REMOTE).get("/v1/capabilities", headers={
        "Authorization": "Bearer tökén".encode("latin-1")})
    assert r.status_code == 401


def test_no_token_means_loopback_only(monkeypatch):
    monkeypatch.setattr(config, "VALID_TOKENS", set())
    monkeypatch.setattr(config, "HOST", "0.0.0.0")  # noqa: S104
    assert config.effective_host() == "127.0.0.1"


def test_a_tokened_node_may_serve_the_network(monkeypatch):
    monkeypatch.setattr(config, "VALID_TOKENS", {"t"})
    monkeypatch.setattr(config, "HOST", "0.0.0.0")  # noqa: S104
    assert config.effective_host() == "0.0.0.0"  # noqa: S104


def test_an_explicit_loopback_host_is_left_alone(monkeypatch):
    monkeypatch.setattr(config, "VALID_TOKENS", set())
    monkeypatch.setattr(config, "HOST", "::1")
    assert config.effective_host() == "::1"


def test_client_tokens_are_matched_exactly(tokens):
    from server.clients import CLIENTS
    assert CLIENTS.name_of(tokens["member"]) == "test-member"
    assert CLIENTS.name_of(tokens["member"][:-1]) is None
    assert CLIENTS.name_of("") is None
    assert not CLIENTS.accepts("")
