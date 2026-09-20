"""The System One decision lane (POST /v1/systemone), as policy.

GPU-free: the checkpoints are ~4.5 GiB and a build takes seconds, so the
engine's predict() is stubbed. What is exercised for real is everything
around it — the wire shape, the caps that refuse a request instead of
swapping, checkpoint naming, and who is allowed to decide. The measured
latency table lives in the deploy notes, not here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import systemone
from server.main import app

LOCAL = ("127.0.0.1", 51234)
REMOTE = ("192.168.1.50", 51234)

STATE = {"subject": "Duplicate charge on invoice #4411",
         "body": "We were billed twice for March. Please refund it."}
CHOICE = {"department": {"type": "choice",
                         "instructions": "Which team should handle this?",
                         "criteria": ["billing", "technical", "sales"]}}
SCORE = {"urgency": {"type": "score",
                     "instructions": "How urgent is this?",
                     "criteria": ["not urgent", "soon", "critical"]}}
NOUL = {"churn_risk": {"type": "noul",
                       "instructions": "Does the user threaten to leave?"}}


def _fake_result(_state, questions, model=None):
    """Laya's own return shape, as captured from laya 0.3.4 on CUDA."""
    answers = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            opts = (list(q["criteria"]) if isinstance(q["criteria"], list)
                    else list(q["criteria"]))
            answers[qid] = {"type": "choice", "choice": opts[0],
                            "probabilities": {o: 1.0 / len(opts)
                                              for o in opts},
                            "confidence": 0.91,
                            "action": {"act_probability": 1.0}}
        elif q["type"] == "score":
            levels = list(q["criteria"])
            answers[qid] = {"type": "score", "score": 1.45,
                            "legend": {str(i): v
                                       for i, v in enumerate(levels)},
                            "probabilities": {str(i): 1.0 / len(levels)
                                              for i in range(len(levels))},
                            "confidence": 0.15,
                            "action": {"act_probability": 1.0}}
        else:
            answers[qid] = {"type": "noul", "noul": 0.88,
                            "confidence": 0.88,
                            "action": {"act_probability": 1.0}}
    return {"model": "laya-rl-agent", "answers": answers,
            "usage": {"input_tokens": 249, "output_tokens": 0},
            "routing": {"model": model or "english",
                        "repo": "convaiinnovations/laya",
                        "reason": "English Latin text"}}


@pytest.fixture
def lane(monkeypatch):
    """A node whose decision lane answers without touching the GPU."""
    monkeypatch.setattr(systemone, "ENABLED", True)
    monkeypatch.setattr(systemone.SYSTEMONE, "installed",
                        staticmethod(lambda: True))

    class _Router:
        def predict(self, state, questions, model=None):
            return _fake_result(state, questions, model)

    monkeypatch.setattr(systemone.SYSTEMONE, "_build",
                        lambda: _Router())
    return TestClient(app, client=LOCAL)


def decide(c: TestClient, body: dict, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return c.post("/v1/systemone", json=body, headers=headers)


# -- one request of each kind ---------------------------------------------

def test_a_choice_question_returns_a_label_and_every_option(lane):
    r = decide(lane, {"state": STATE, "questions": CHOICE})
    assert r.status_code == 200, r.text
    body = r.json()
    ans = body["answers"]["department"]
    assert ans["choice"] == "billing"
    assert set(ans["probabilities"]) == {"billing", "technical", "sales"}
    assert 0.0 <= ans["confidence"] <= 1.0
    # The Mac needs to know which checkpoint answered, and how fast.
    assert body["checkpoint"]["repo"] == "convaiinnovations/laya"
    assert body["checkpoint"]["revision"]
    assert ans["latency_ms"] >= 0
    assert body["questions"] == 1


def test_a_score_question_returns_an_expected_level_and_the_rubric(lane):
    r = decide(lane, {"state": STATE, "questions": SCORE})
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]["urgency"]
    assert isinstance(ans["score"], float)
    assert ans["legend"] == {"0": "not urgent", "1": "soon",
                             "2": "critical"}
    assert set(ans["probabilities"]) == {"0", "1", "2"}


def test_a_noul_question_returns_a_probability(lane):
    r = decide(lane, {"state": STATE, "questions": NOUL})
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]["churn_risk"]
    assert 0.0 <= ans["noul"] <= 1.0
    assert ans["confidence"] == pytest.approx(ans["noul"])


# -- batching --------------------------------------------------------------

def test_a_batch_answers_every_question_in_one_pass(lane):
    questions = {**CHOICE, **SCORE, **NOUL}
    r = decide(lane, {"state": STATE, "questions": questions})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["answers"]) == set(questions)
    assert body["questions"] == 3
    # The batch is one forward pass, so per-question time is the whole
    # request divided across it — that is the lane's selling point.
    assert (body["latency_ms_per_question"]
            == pytest.approx(body["latency_ms"] / 3, rel=0.01))


def test_a_plain_string_state_is_accepted(lane):
    r = decide(lane, {"state": "We were billed twice.", "questions": NOUL})
    assert r.status_code == 200, r.text


# -- refusals: too large, rather than swapping -----------------------------

def test_too_many_questions_is_refused(lane):
    many = {f"q{i}": dict(NOUL["churn_risk"])
            for i in range(systemone.MAX_QUESTIONS + 1)}
    r = decide(lane, {"state": STATE, "questions": many})
    assert r.status_code == 400
    assert str(systemone.MAX_QUESTIONS) in r.json()["error"]


def test_too_many_options_is_refused(lane):
    q = {"dept": {"type": "choice", "instructions": "Which?",
                  "criteria": [f"opt{i}"
                               for i in range(systemone.MAX_OPTIONS + 1)]}}
    r = decide(lane, {"state": STATE, "questions": q})
    assert r.status_code == 400
    assert "options" in r.json()["error"]


def test_an_oversized_state_is_refused(lane):
    r = decide(lane, {"state": "x" * (systemone.MAX_STATE_CHARS + 1),
                      "questions": NOUL})
    assert r.status_code == 400
    assert "characters" in r.json()["error"]


def test_an_oversized_body_is_refused_before_it_is_parsed(lane):
    r = lane.post("/v1/systemone",
                  content=b'{"state":"' + b"x" * systemone.MAX_BODY_BYTES
                  + b'"}', headers={"Content-Type": "application/json"})
    assert r.status_code == 413


@pytest.mark.parametrize("questions, missing", [
    ({"q": {"type": "choice", "instructions": "Which?"}}, "criteria"),
    ({"q": {"type": "choice", "criteria": ["a", "b"]}}, "instructions"),
    ({"q": {"type": "nonsense", "instructions": "?",
            "criteria": ["a"]}}, "type"),
])
def test_a_malformed_question_names_what_is_wrong(lane, questions, missing):
    r = decide(lane, {"state": STATE, "questions": questions})
    assert r.status_code == 400
    assert missing in r.json()["error"]


def test_an_empty_question_set_is_refused(lane):
    assert decide(lane, {"state": STATE,
                         "questions": {}}).status_code == 400


def test_a_missing_state_is_refused(lane):
    assert decide(lane, {"questions": NOUL}).status_code == 400


# -- checkpoints -----------------------------------------------------------

def test_an_unknown_checkpoint_is_refused_by_name(lane):
    r = decide(lane, {"state": STATE, "questions": NOUL,
                      "model": "laya-enormous"})
    assert r.status_code == 400
    body = r.json()["error"]
    assert "laya-enormous" in body and "laya-multilingual" in body


def test_the_wire_names_map_onto_layas_own(lane):
    r = decide(lane, {"state": STATE, "questions": NOUL,
                      "model": "laya-typed-decisions"})
    assert r.status_code == 200, r.text
    assert r.json()["routing"]["model"] == "typed-decisions"


def test_every_advertised_checkpoint_is_pinned_to_a_revision():
    for name, ckpt in systemone.CHECKPOINTS.items():
        assert ckpt["repo"].startswith("convaiinnovations/"), name
        assert len(ckpt["revision"]) == 40, name


def test_an_uninstalled_engine_answers_503_not_500(lane, monkeypatch):
    monkeypatch.setattr(systemone.SYSTEMONE, "installed",
                        staticmethod(lambda: False))
    r = decide(lane, {"state": STATE, "questions": NOUL})
    assert r.status_code == 503
    assert "requirements-decisions" in r.json()["error"]


# -- the advertisement -----------------------------------------------------

def test_the_node_advertises_the_decision_lane(lane):
    d = lane.get("/v1/node").json()["decisions"]
    assert d["engine"] == "laya"
    assert d["question_types"] == ["choice", "score", "noul"]
    assert set(d["models"]) == {"laya", "laya-multilingual",
                                "laya-typed-decisions"}
    assert d["limits"]["max_questions"] == systemone.MAX_QUESTIONS
    assert "1c5edc17a7acd8701df6fc341c0d179f1c62c982" in str(
        d["checkpoints"])


def test_a_broken_lane_does_not_take_the_advertisement_down(monkeypatch):
    import server.main as main

    def boom():
        raise RuntimeError("no torch here")
    monkeypatch.setattr(main, "_decisions_status",
                        lambda: {"engine": "laya", "available": False,
                                 "error": "RuntimeError"})
    body = TestClient(app, client=LOCAL).get("/v1/node").json()
    assert body["decisions"]["available"] is False
    assert body["capabilities"]          # the rest still advertises


# -- auth ------------------------------------------------------------------

def test_a_member_may_decide(lane, tokens):
    r = decide(TestClient(app, client=REMOTE),
               {"state": STATE, "questions": NOUL}, tokens["member"])
    assert r.status_code == 200, r.text


def test_an_off_box_request_without_a_token_is_refused(lane, tokens):
    r = decide(TestClient(app, client=REMOTE),
               {"state": STATE, "questions": NOUL}, None)
    assert r.status_code == 401


def test_a_bad_token_is_refused(lane, tokens):
    r = decide(TestClient(app, client=REMOTE),
               {"state": STATE, "questions": NOUL}, "not-a-real-token")
    assert r.status_code == 401


def test_a_paused_node_refuses_members_but_still_serves_the_owner(
        lane, tokens, monkeypatch):
    from server.serving import SERVING
    SERVING.set(True, "owner is rendering")
    try:
        member = decide(TestClient(app, client=REMOTE),
                        {"state": STATE, "questions": NOUL},
                        tokens["member"])
        assert member.status_code == 503
        assert "paused" in member.json()["error"].lower()
        owner = decide(TestClient(app, client=REMOTE),
                       {"state": STATE, "questions": NOUL}, tokens["swarm"])
        assert owner.status_code == 200, owner.text
    finally:
        SERVING.set(False)
