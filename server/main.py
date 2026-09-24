"""Silicon Node — CUDA job service (Windows/WSL2 counterpart of Silicon Optimizer).

Phase 1 contract (the Mac's Lato2Runtime client builds against this):
    GET  /health                    2xx probe; includes a "server" field
    POST /v1/image-to-mesh          multipart: image, vert_num, seed -> {job_id}
    GET  /v1/jobs/{job_id}          {status, progress, result_urls, error}
    GET  /v1/files/{name}           artifact bytes

Phase 2:
    GET  /v1/capabilities           what this node can run, with measured numbers
    POST /v1/jobs                   {"capability": id, ...params}
    GET  /v1/node                   swarm advertisement (name/platform/profile/metrics)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import config, pipeline, uploads
from .jobs import STORE
from .llm import DOWNLOADS, LLM, MODEL_ID as LLM_MODEL_ID, PORT as LLM_PORT
from .video import SUPPORTED_RESOLUTIONS as _VIDEO_RESOLUTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("silicon-node")

app = FastAPI(title=config.SERVER_NAME, version=config.SERVER_VERSION)

STARTED_AT = time.time()


def _is_owner_local(request: Request) -> bool:
    """The owner at this machine's console: a loopback source that no
    proxy has relayed."""
    return (config.is_loopback(request.client.host if request.client else None)
            and not config.is_forwarded(request.headers))


# ---------------------------------------------------------------------------
# Auth: every off-box /v1/ request carries a bearer token — the node token,
# the swarm token, or a paired client's own. Loopback callers (the node's
# dashboard and tray GUI, which send no header) stay open unless strict
# mode is on. /health stays open everywhere as a probe.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    # Gate on scope["path"], the path the router dispatches on. request.url
    # is rebuilt from the Host header, which the caller writes: on
    # starlette <= 1.0.0 a Host such as "x?" moved the path out from under
    # this check while the router still served the /v1/ route.
    if request.scope["path"].startswith("/v1/"):
        header = request.headers.get("authorization", "")
        supplied = header.removeprefix("Bearer ").strip() if header else ""
        from .clients import CLIENTS  # noqa: PLC0415
        if supplied and not config.token_valid(supplied) \
                and not CLIENTS.accepts(supplied):
            # A wrong token is always rejected, whatever the interface —
            # it catches misconfiguration (and revoked members) early.
            return PlainTextResponse(
                "That bearer token does not match this node's node token, "
                "the swarm token, or any paired client.", status_code=401)
        if not supplied:
            local = _is_owner_local(request)
            if config.REQUIRE_AUTH or not local:
                return PlainTextResponse(
                    "This Silicon node requires a bearer token. Send "
                    "'Authorization: Bearer <token>'.", status_code=401)
        # Body size is checked here rather than in each handler so no
        # submit route can be added without one.
        try:
            uploads.check_declared_size(
                request.headers.get("content-length"))
        except HTTPException as exc:
            return PlainTextResponse(str(exc.detail), status_code=413)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Swarm client credentials (handoff 125): the shared swarm token is the
# admin credential; members get individually revocable tokens. These
# endpoints demand the admin token ALWAYS — minting or revoking
# credentials must never be open, whatever REQUIRE_AUTH says.
# ---------------------------------------------------------------------------

def _require_swarm_admin(request: Request) -> None:
    header = request.headers.get("authorization", "")
    supplied = header.removeprefix("Bearer ").strip() if header else ""
    if not config.SWARM_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="This node has no swarm token configured yet.")
    if config.is_swarm_token(supplied):
        return
    from .clients import CLIENTS  # noqa: PLC0415
    if supplied and CLIENTS.accepts(supplied):
        raise HTTPException(
            status_code=403,
            detail="Client tokens cannot manage clients — this needs "
                   "the swarm admin token.")
    raise HTTPException(
        status_code=401,
        detail="This endpoint needs the swarm admin token.")


@app.post("/swarm/clients")
async def swarm_client_mint(request: Request):
    _require_swarm_admin(request)
    from .clients import CLIENTS  # noqa: PLC0415
    body = await request.json()
    name = str(body.get("name", ""))
    role = str(body.get("role") or "member")
    try:
        name, token = CLIENTS.mint(name, role=role)
    except KeyError:
        raise HTTPException(
            status_code=409,
            detail=f"A client named {name!r} already exists.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"name": name, "token": token, "role": role}


@app.get("/swarm/clients")
def swarm_client_list(request: Request):
    _require_swarm_admin(request)
    from .clients import CLIENTS  # noqa: PLC0415
    return CLIENTS.listing()


@app.delete("/swarm/clients/{name}")
def swarm_client_revoke(name: str, request: Request):
    _require_swarm_admin(request)
    from .clients import CLIENTS  # noqa: PLC0415
    if not CLIENTS.revoke(name):
        raise HTTPException(status_code=404,
                            detail=f"No client named {name!r}.")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Phase 1 endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "server": {"name": config.SERVER_NAME,
                   "version": config.SERVER_VERSION},
        "uptime_s": round(time.time() - STARTED_AT),
        "queue_depth": STORE.queue_depth(),
    }


def _bearer_of(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header.removeprefix("Bearer ").strip() if header else ""


def _actor(request: Request):
    """Who is acting: "admin" (swarm token), "node" (node token, or an
    untokened loopback caller — the owner's own dashboard), or a paired
    client's name.

    An unrecognised token is never the owner: the middleware rejects it
    outright, and if it somehow arrives here it gets a member's rights,
    not an operator's.
    """
    tok = _bearer_of(request)
    if not tok:
        if _is_owner_local(request):
            return "node"
        return "member"
    if config.is_swarm_token(tok):
        return "admin"
    if config.is_node_token(tok):
        return "node"
    from .clients import CLIENTS  # noqa: PLC0415
    return CLIENTS.name_of(tok) or "member"


def _role(request: Request) -> str:
    """admin | node | member — from the credential alone. A paired
    client's *name* never grants rights, however it was spelled."""
    tok = _bearer_of(request)
    if not tok:
        return "node" if _is_owner_local(request) else "member"
    if config.is_swarm_token(tok):
        return "admin"
    if config.is_node_token(tok):
        return "node"
    from .clients import CLIENTS  # noqa: PLC0415
    if CLIENTS.role_of(tok) == "admin":
        return "admin"           # minted as such by the swarm admin
    return "member"


def _owner_key(request: Request) -> str:
    """Who owns what this request submits, by credential kind and id
    (hub 155): "swarm", "node", "client:<name>", or "anonymous". Never a
    display string — a joining machine picks its own name, and one named
    "swarm (shared token)" must not own the jobs sent with that token."""
    tok = _bearer_of(request)
    if not tok:
        return "node" if _is_owner_local(request) else "anonymous"
    if config.is_swarm_token(tok):
        return "swarm"
    if config.is_node_token(tok):
        return "node"
    from .clients import CLIENTS  # noqa: PLC0415
    name = CLIENTS.name_of(tok)
    return f"client:{name}" if name else "anonymous"


def _job_owner(job) -> str:
    """The owner key a job was submitted under. Jobs from before owner
    keys existed carry only the display label; map the two fixed labels
    back to their credential kinds and read anything else as a client."""
    sb = job.submitted_by or {}
    if sb.get("owner"):
        return sb["owner"]
    label = sb.get("client")
    if label in (None, "this node's token"):
        return "node"
    if label == "swarm (shared token)":
        return "swarm"
    return f"client:{label}"


def _can_see(request: Request, job) -> bool:
    """Operators see every job; a member sees the jobs it submitted."""
    return (_role(request) in ("admin", "node")
            or _job_owner(job) == _owner_key(request))


def _submitter(request: Request, cap: Optional[str] = None) -> dict:
    """Who sent this job: the paired client's name when they used their
    own token, the shared/node token labels otherwise, plus source IP.
    Tailnet and LAN traffic is proxied through the Windows host, so that
    source IP is flagged rather than shown as if it were the sender.
    When cap is given and the sender is a paired client, their lifetime
    job counters tick (handoff 132)."""
    tok = _bearer_of(request)
    who = None
    if tok:
        if config.is_swarm_token(tok):
            who = "swarm (shared token)"
        elif config.is_node_token(tok):
            who = "this node's token"
        else:
            from .clients import CLIENTS  # noqa: PLC0415
            who = CLIENTS.name_of(tok)
            if who and cap:
                CLIENTS.count_job(who, cap)
    ip = request.client.host if request.client else None
    from .llm import _windows_host_ip  # noqa: PLC0415
    return {"client": who, "owner": _owner_key(request), "ip": ip,
            "proxied": ip == _windows_host_ip(),
            "user_agent": request.headers.get("user-agent", "")[:120]}


@contextlib.contextmanager
def _staging(job):
    """The window between STORE.submit(defer=True) and STORE.enqueue():
    the job exists but its input is still being written. If that write
    is refused (413 over the upload ceiling, undecodable base64, an empty
    file) the submitter gets the error — and the job must not stay queued
    forever, because nothing will ever enqueue it."""
    try:
        yield
    except BaseException as exc:
        detail = getattr(exc, "detail", None) or str(exc) or "input rejected"
        STORE.abandon(job, f"Input rejected: {detail}")
        raise


@app.post("/v1/image-to-mesh")
async def image_to_mesh(
    request: Request,
    image: UploadFile = File(...),
    vert_num: str = Form(str(config.VERT_NUM_DEFAULT)),
    seed: Optional[str] = Form(None),
):
    params = {"vert_num": vert_num, "seed": seed}
    try:
        pipeline._parse_vert_num(params)
        pipeline._parse_seed(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    trellis_ok, trellis_detail = pipeline.ENGINE.trellis_available()
    if not trellis_ok:
        raise HTTPException(status_code=503, detail=trellis_detail)
    _require_enabled("image-to-mesh")

    job = STORE.submit("image-to-mesh", params, defer=True)
    job.submitted_by = _submitter(request, "image-to-mesh")
    job.dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "input.png").suffix or ".png"
    image_path = job.dir / f"input{suffix}"
    with _staging(job):
        await uploads.save_upload(image, image_path)
        if image_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty. Please send a PNG or "
                       "JPEG.")
    job.params["image_path"] = str(image_path)
    STORE.enqueue(job)
    return {"job_id": job.job_id}


@app.get("/v1/jobs")
def jobs_list(request: Request):
    """Recent jobs, newest first (for the dashboard). A member sees its
    own; operators see everyone's (hub 155)."""
    jobs = sorted((j for j in STORE.snapshot() if _can_see(request, j)),
                  key=lambda j: j.created_at, reverse=True)[:20]
    return [{**j.to_api(), "capability": j.capability, "state": j.state,
             "created_at": j.created_at, "started_at": j.started_at,
             "finished_at": j.finished_at,
             "submitted_by": j.submitted_by,
             # The prompt (or input file name) so the Activity feed can
             # say WHAT each job is, not just its kind.
             "prompt": (str(j.params.get("prompt"))[:300]
                        if j.params.get("prompt") else None),
             "input_name": next(
                 (Path(str(j.params[k])).name for k in
                  ("image_path", "mesh_path", "driving_path", "audio_path")
                  if j.params.get(k)), None)} for j in jobs]


def _visible_job(request: Request, job_id: str):
    """The job, if this caller may see it. Someone else's job answers
    exactly like a missing one, so ids can't be probed for."""
    job = STORE.get(job_id)
    if job is None or not _can_see(request, job):
        raise HTTPException(
            status_code=404,
            detail=f"No job named {job_id} on this node. It may predate a "
                   "service restart.")
    return job


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    return _visible_job(request, job_id).to_api()


@app.get("/v1/jobs/{job_id}/detail")
def job_detail(job_id: str, request: Request):
    job = _visible_job(request, job_id)
    operator = _role(request) in ("admin", "node")
    params = {k: (v if not isinstance(v, str) or len(v) < 200 else "…")
              for k, v in job.params.items() if not k.endswith("_b64")}
    if not operator:
        # File names, not where the node keeps them.
        params = {k: (Path(str(v)).name if k.endswith("_path") and v else v)
                  for k, v in params.items()}
    return {**job.to_api(), "capability": job.capability, "params": params,
            "created_at": job.created_at, "started_at": job.started_at,
            "finished_at": job.finished_at, "receipts": job.receipts,
            "state": job.state, "held": job.held,
            "submitted_by": job.submitted_by}


@app.post("/v1/queue/cancel")
async def queue_cancel(request: Request):
    """The Mac Swarm command center's Cancel Queue button (handoff 126).
    scope "pending" drops queued jobs; "all" also aborts the running one
    at its next progress checkpoint."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if _role(request) not in ("admin", "node"):
        raise HTTPException(
            status_code=403,
            detail="Clearing the whole queue is for the swarm admin or "
                   "the node owner; members can cancel their own jobs.")
    scope = body.get("scope", "pending")
    if scope not in ("pending", "all"):
        raise HTTPException(status_code=400,
                            detail='scope must be "pending" or "all".')
    return {"cancelled": STORE.cancel_queue(scope)}


@app.post("/v1/jobs/prune")
async def jobs_prune(request: Request):
    """Reclaim disk now, rather than waiting for the next finished job.

    Retention runs on its own (at startup and after every job), so this is
    for the case where the disk is full *today*: the dashboard's Free space
    button, and `keep`/`max_age_days` for a one-off deeper sweep. Both
    limits must agree before a job goes, and a zero in either means no
    sweep at all — the same rule as the SILICON_NODE_RETAIN_* settings.
    """
    _require_operator(request, "Deleting finished jobs and their artifacts")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    keep = body.get("keep")
    days = body.get("max_age_days")
    try:
        keep_n = None if keep is None else max(0, int(keep))
        max_age = None if days is None else max(0.0, float(days))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="keep must be a whole number of jobs and max_age_days "
                   "a number of days.") from None
    return await asyncio.to_thread(STORE.prune, keep=keep_n,
                                   max_age_days=max_age)


_CANCEL_HTTP = {"cancelled": 200, "requested": 202, "completed": 409,
                "failed": 409, "unsupported": 409, "unknown": 404}
# How long the cancel route waits for a running job to reach a
# checkpoint before answering "requested" (the Mac waits 60 s).
CANCEL_WAIT_S = 15.0


def _cancel_answer(request: Request, job_id: str) -> JSONResponse:
    """POST /v1/jobs/{id}/cancel — the Mac's per-job Cancel render (hub
    158). The answer is in the body's `cancel` field; the HTTP code
    agrees with it. Idempotent: a repeat gets the same answer."""
    if STORE.get(job_id) is None:
        outcome, job = "unknown", None
    else:
        _require_job_owner(request, job_id)   # 403 for someone else's
        outcome, job = STORE.request_cancel(job_id)
        if outcome == "requested":
            # Give the worker until its next checkpoint to stop, so the
            # common case answers "cancelled" in one round trip.
            deadline = time.time() + CANCEL_WAIT_S
            while time.time() < deadline and job.state == "running":
                time.sleep(0.1)
            outcome, job = STORE.request_cancel(job_id)
    details = {
        "cancelled": "The job is stopped and will not publish results.",
        "requested": "The job is stopping at its next checkpoint.",
        "completed": "The job had already finished; its results are kept.",
        "unknown": f"No job named {job_id} on this node.",
    }
    detail = (job.error if outcome == "failed" and job is not None
              else details.get(outcome, ""))
    return JSONResponse(status_code=_CANCEL_HTTP[outcome], content={
        "job_id": job_id, "cancel": outcome,
        "status": job.to_api()["status"] if job is not None else None,
        "detail": detail})


@app.post("/v1/jobs/{job_id}/{action}")
def job_action(job_id: str, action: str, request: Request,
               direction: str = "up"):
    if action == "cancel":
        return _cancel_answer(request, job_id)
    _require_job_owner(request, job_id)
    ok = False
    if action == "hold":
        ok = STORE.hold(job_id, True)
    elif action == "resume":
        ok = STORE.hold(job_id, False)
    elif action == "move":
        ok = STORE.move(job_id, direction)
    elif action == "retry":
        ok = STORE.retry(job_id) is not None
    else:
        raise HTTPException(status_code=404, detail="Unknown action.")
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"{action} doesn't apply to this job's current state.")
    return {"ok": True}


@app.get("/v1/files/{name}")
def get_file(name: str, request: Request):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    path = config.FILES_DIR / name
    # Every artifact is named <job_id>-<label>.<ext>; it belongs to that
    # job's owner (and the operators). One with no job left is operators'.
    job = STORE.get("-".join(name.split("-")[:2]))
    visible = (_can_see(request, job) if job is not None
               else _role(request) in ("admin", "node"))
    if not visible or not path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"No artifact named {name}.")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Phase 2: capabilities + generic submit
# ---------------------------------------------------------------------------

# Human descriptions for the Mac's ability popovers (handoff 129) —
# what it does and what backs it, in sentences rather than wire shorthand.
_CAP_DESCRIPTIONS = {
    "image-to-mesh":
        "Turns a single photo or rendered image into a game-ready 3D "
        "model. TRELLIS.2 builds a dense textured mesh, then LATO.2 "
        "retopologizes it into clean low-poly geometry — both the dense "
        "GLB and the clean OBJ come back.",
    "retopologize":
        "Rebuilds an existing mesh (GLB/OBJ/PLY) as clean low-poly "
        "geometry with LATO.2 — topology fit for rigging and editing, "
        "200–5,000 vertices.",
    "text-to-video":
        "Generates short video clips from a text prompt, optionally "
        "starting from an image. Wan 2.2 TI2V-5B is the quality engine; "
        "LTX-2 distilled renders the same clip about 2.6× faster for "
        "iteration; LTX-2.3 Uncensored v1.4 (installed from the store) is "
        "the adult-content merge and its clips carry audio — choose with "
        "the model field.",
    "text-to-image":
        "Renders still images from a text prompt. Qwen-Image 20B is the "
        "quality engine (it can put readable text in images); Sana is "
        "the seconds-per-image iteration lane — choose with the model "
        "field. More image models install from the model store.",
    "portrait-animate":
        "Animates a still portrait with a recorded video performance "
        "via LivePortrait: the driving clip's expression and head "
        "motion transfer onto the photo.",
    "talking-head":
        "Makes a photo speak: a portrait plus a speech recording "
        "becomes a lip-synced clip via SadTalker — the audio-driven "
        "sibling of portrait animation.",
}


def capability_list() -> list[dict]:
    """Measured numbers get filled in from receipts as jobs run."""
    trellis_ok, trellis_detail = pipeline.ENGINE.trellis_available()
    i2m_detail = ("Route A: TRELLIS.2 densify (dense textured GLB) then "
                  "LATO.2 retopology (clean OBJ, vert_num 200–5000).")
    if not trellis_ok:
        i2m_detail = trellis_detail
    caps = [
        {
            "id": "image-to-mesh",
            "name": "Image → clean low-poly mesh (TRELLIS.2 + LATO.2)",
            "kind": "mesh",
            "peak_vram_gb": _measured("image-to-mesh", "peak_vram_gb"),
            "typical_seconds": _measured("image-to-mesh", "typical_seconds"),
            "ready": pipeline.ENGINE.lato_ready and trellis_ok,
            "detail": i2m_detail,
        },
        {
            "id": "retopologize",
            "name": "Mesh → clean low-poly mesh (LATO.2)",
            "kind": "retopo",
            "peak_vram_gb": _measured("retopologize", "peak_vram_gb"),
            "typical_seconds": _measured("retopologize", "typical_seconds"),
            "ready": pipeline.ENGINE.lato_ready,
            "detail": "LATO.2 without the densify stage: mesh in (GLB/OBJ), "
                      "clean OBJ out. vert_num 200–5000.",
        },
        {
            "id": "portrait-animate",
            "name": "Portrait + performance → animated clip (LivePortrait)",
            # The Mac's peer filter matches on kind, not id.
            "kind": "portrait-animate",
            "peak_vram_gb": _measured("portrait-animate", "peak_vram_gb"),
            "typical_seconds": _measured("portrait-animate",
                                         "typical_seconds"),
            "ready": _portrait_ready(),
            "detail": "LivePortrait on CUDA: POST /v1/portrait-animate "
                      "{image_b64, image_name, driving_b64, driving_name} "
                      "→ job; result is the animated .mp4."
                      if _portrait_ready() else
                      "LivePortrait still installing — flips ready "
                      "automatically.",
        },
        {
            "id": "text-to-video",
            "name": "Text → video clip (Wan 2.2 TI2V-5B)",
            "kind": "video",
            # Every model behind this capability takes all three (hub
            # 159); 1080p is rendered at 720p and scaled up, which each
            # job's `delivery` states.
            "supported_resolutions": list(_VIDEO_RESOLUTIONS),
            # POST /v1/jobs/{id}/cancel answers the Mac's protocol for
            # every job; the Mac offers Cancel render where this is listed
            # (hub 158). Video is the one it reads today.
            "supported_job_actions": ["cancel"],
            "peak_vram_gb": _measured("text-to-video", "peak_vram_gb"),
            "typical_seconds": _measured("text-to-video",
                                         "typical_seconds"),
            "ready": _video_ready(),
            "detail": "Wan 2.2 TI2V-5B via diffusers, CPU-offloaded for "
                      "the shared 24 GB card. Submit POST /v1/jobs "
                      '{"capability":"text-to-video","prompt":...} with '
                      "optional negative_prompt/frames/width/height/steps/"
                      "seed; result_urls carries the .mp4. Models: "
                      + ", ".join(_installed_video_models()) + "."
                      if _video_ready() else
                      "Wan 2.2 weights or diffusers still downloading — "
                      "flips ready automatically.",
        },
        {
            "id": "text-to-image",
            "name": "Text → image (Qwen-Image 20B + Sana)",
            "kind": "image",
            "peak_vram_gb": _measured("text-to-image", "peak_vram_gb"),
            "typical_seconds": _measured("text-to-image",
                                         "typical_seconds"),
            "ready": _image_ready(),
            "detail": ("POST /v1/text-to-image {prompt, width, height, "
                       "steps?, seed?, negative_prompt?, model?} → job; "
                       "result_urls carries the .png. Models: "
                       + ", ".join(_installed_image_models() or ["none"])
                       + "." if _image_ready() else
                       "Image model weights still installing — flips "
                       "ready automatically (see the model store)."),
        },
        {
            "id": "talking-head",
            "name": "Portrait + speech audio → lip-synced clip (SadTalker)",
            "kind": "talking-head",
            "peak_vram_gb": _measured("talking-head", "peak_vram_gb"),
            "typical_seconds": _measured("talking-head", "typical_seconds"),
            "ready": _talkinghead_ready(),
            "detail": "SadTalker on CUDA: POST /v1/talking-head "
                      "{image_b64, image_name, audio_b64, audio_name} "
                      "→ job; result is the lip-synced .mp4."
                      if _talkinghead_ready() else
                      "SadTalker still installing — flips ready "
                      "automatically.",
        },
        {
            "id": f"llm-{LLM_MODEL_ID}",
            "name": "Qwen3.8-27B chat/completions (ninfer-3090, INT8)",
            "kind": "llm",
            "peak_vram_gb": None,  # measured after first sustained run
            "typical_seconds": None,
            "ready": LLM.installed,
            "detail": ("OpenAI Chat Completions + Anthropic Messages API on "
                       f"port {LLM_PORT} (tailnet-only off-box). Not a jobs-"
                       "API capability — talk to the LLM endpoint directly; "
                       "manage via GET/POST /v1/llm. Mutually exclusive "
                       "with 3D jobs on the 24 GB card: jobs preempt the "
                       "LLM and it auto-restores when the queue drains."
                       if LLM.installed else
                       "ninfer-3090 engine or Qwen3.8-27B model file not "
                       "present yet."),
        },
    ]
    from .capsettings import CAPS  # noqa: PLC0415
    for c in caps:
        cid = c["id"]
        if cid in _CAP_DESCRIPTIONS:
            c["description"] = _CAP_DESCRIPTIONS[cid]
            c["enabled"] = CAPS.enabled(cid)
            settings = CAPS.settings(cid)
            if cid == "text-to-image":
                # The Mac parses the installed lanes out of settings
                # (hub 136); informational, not writable.
                settings = dict(settings)
                settings["models"] = ",".join(_installed_image_models())
            if settings:
                c["settings"] = settings
            # A disabled ability is still listed (so it can be
            # re-enabled) but must not attract delegated jobs.
            if not c["enabled"]:
                c["ready"] = False
        else:  # the LLM entry — managed via /v1/llm, not this API
            c["description"] = (
                "Chat and agent completions on Qwen3.8-27B via the "
                "ninfer engine — OpenAI- and Anthropic-compatible APIs. "
                "Managed through /v1/llm rather than the jobs API.")
            c["enabled"] = True
    return caps


def _installed_video_models() -> list[str]:
    """The model ids POST /v1/text-to-video accepts right now."""
    from . import video  # noqa: PLC0415
    out = ["wan22-ti2v-5b"] if video.ENGINE.weights_present() else []
    if video.ENGINE.ltx_ready():
        out.append("ltx2-distilled")
    if video.ENGINE.ltx_uncensored_ready():
        out.append("ltx23-uncensored")
    return out


def _decisions_status() -> dict:
    """Never let a broken decision lane take the advertisement down —
    the Mac reads /v1/node to find every other capability too."""
    try:
        from .systemone import SYSTEMONE  # noqa: PLC0415
        return SYSTEMONE.status()
    except Exception as exc:  # noqa: BLE001
        return {"engine": "laya", "endpoint": "/v1/systemone",
                "available": False, "loaded": False,
                "error": f"{type(exc).__name__}"}


def _stop_hyperqwen_if_running() -> None:
    """One language engine per card. Cheap when the engine was never
    installed, so the other two lanes can always call it."""
    try:
        from .hyperqwen import HYPERQWEN  # noqa: PLC0415
        if HYPERQWEN.checked_out and HYPERQWEN.active:
            log.info("stopping HyperQwen to free the card")
            HYPERQWEN.stop()
    except Exception:  # noqa: BLE001
        log.exception("could not stop the HyperQwen engine")


def _hyperqwen_status() -> dict:
    """Never let a docker probe take the advertisement down."""
    try:
        from .hyperqwen import HYPERQWEN  # noqa: PLC0415
        # Polled every 2.5 s by the dashboard: the compact block, built
        # without the docker calls the full status page makes.
        return HYPERQWEN.advertisement()
    except Exception as exc:  # noqa: BLE001
        return {"engine": "hyperqwen", "enabled": False, "running": False,
                "error": f"{type(exc).__name__}"}


def _video_ready() -> bool:
    from . import video
    return video.ENGINE.ready()


def _image_ready() -> bool:
    from . import image
    return image.ENGINE.ready()


def _installed_image_models() -> list[str]:
    from . import image
    return image.ENGINE.installed_models()


def _portrait_ready() -> bool:
    from . import portrait
    return portrait.ready()


def _talkinghead_ready() -> bool:
    from . import talkinghead
    return talkinghead.ready()


def _measured(cap: str, field: str):
    """Aggregate receipts of finished jobs; None until first measurement."""
    samples = []
    for job in STORE.snapshot():
        if job.capability != cap or job.state != "done":
            continue
        if field == "typical_seconds" and job.started_at and job.finished_at:
            samples.append(job.finished_at - job.started_at)
        elif field == "peak_vram_gb":
            peaks = [job.receipts.get("trellis_peak_vram_gb"),
                     job.receipts.get("lato_peak_vram_gb"),
                     job.receipts.get("video_peak_vram_gb"),
                     job.receipts.get("image_peak_vram_gb")]
            peaks = [p for p in peaks if p]
            if peaks:
                samples.append(max(peaks))
    if not samples:
        return None
    return round(sorted(samples)[len(samples) // 2], 1)  # median


@app.get("/v1/capabilities")
def capabilities():
    return capability_list()


def _require_enabled(cap: str) -> None:
    from .serving import SERVING  # noqa: PLC0415
    if SERVING.paused:
        raise HTTPException(status_code=503, detail=SERVING.refusal())
    from .capsettings import CAPS  # noqa: PLC0415
    if not CAPS.enabled(cap):
        raise HTTPException(
            status_code=503,
            detail=f"The {cap} ability is currently disabled on this "
                   f"node. Re-enable it from the Swarm page or via "
                   f"POST /v1/capabilities/{cap}.")


# The exact enable-flag sets behind each named profile (hub 138). The
# chat capability is deliberately untouched — it has its own lifecycle
# tier via /v1/llm.
_JOB_CAPS = ("image-to-mesh", "retopologize", "text-to-video",
             "text-to-image", "portrait-animate", "talking-head")
_PROFILES = {
    "images-only": {"text-to-image"},
    "video-only": {"text-to-video", "portrait-animate", "talking-head"},
    "everything": set(_JOB_CAPS),
    "nothing": set(),
}


def _current_profile() -> Optional[str]:
    from .capsettings import CAPS  # noqa: PLC0415
    on = {c for c in _JOB_CAPS if CAPS.enabled(c)}
    for name, want in _PROFILES.items():
        if on == want:
            return name
    return None


@app.get("/v1/serving")
def serving_status():
    from .serving import SERVING  # noqa: PLC0415
    st = SERVING.status()
    st["profile"] = _current_profile()
    return st


@app.post("/v1/serving")
async def serving_update(request: Request):
    """The owner's pause switch (hub 138): while paused, submissions and
    member chats refuse with words, queued jobs finish, and /v1/node
    stays readable with a serving.paused card."""
    _require_operator(request, "Pausing or resuming the node")
    from .serving import SERVING  # noqa: PLC0415
    body = await request.json()
    if not isinstance(body.get("paused"), bool):
        raise HTTPException(status_code=400,
                            detail='"paused" must be true or false.')
    SERVING.set(body["paused"], body.get("reason"))
    return serving_status()


@app.post("/v1/capabilities/profile")
async def capability_profile(request: Request):
    """Named presets over the per-ability toggles (hub 138): one call
    sets every job ability's enable flag atomically."""
    _require_operator(request, "Switching the capability profile")
    from .capsettings import CAPS  # noqa: PLC0415
    body = await request.json()
    name = str(body.get("name") or "")
    if name not in _PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile {name!r}; this node offers "
                   f"{', '.join(sorted(_PROFILES))}. The chat model is "
                   "not part of profiles — manage it via /v1/llm.")
    applied = {}
    for cap in _JOB_CAPS:
        want = cap in _PROFILES[name]
        CAPS.update(cap, enabled=want)
        applied[cap] = want
    return {"ok": True, "profile": name, "applied": applied}


@app.post("/v1/capabilities/{cap_id}")
async def capability_update(cap_id: str, request: Request):
    """Enable/disable an ability or change its exposed settings
    (handoff 129). Partial updates; unknown setting keys are reported
    back as ignored rather than written."""
    _require_operator(request, "Changing an ability")
    from .capsettings import CAPS, DEFAULTS  # noqa: PLC0415
    if cap_id not in DEFAULTS:
        raise HTTPException(
            status_code=404,
            detail=f"No configurable ability named {cap_id!r} on this "
                   "node. The chat model is managed via /v1/llm.")
    body = await request.json()
    ignored = CAPS.update(cap_id, body.get("enabled"),
                          body.get("settings"))
    # Some configurable ids are engines rather than job capabilities
    # (hyperqwen), so they carry settings without appearing in the
    # advertised capability list — report their settings instead.
    entry = next((c for c in capability_list() if c["id"] == cap_id), None)
    if entry is None:
        entry = {"id": cap_id, "enabled": CAPS.enabled(cap_id),
                 "settings": CAPS.settings(cap_id)}
    out = {"ok": True, "capability": entry}
    if ignored:
        out["warning"] = ("These setting keys are not exposed on this "
                          f"node and were ignored: {', '.join(ignored)}")
    return out


# ---------------------------------------------------------------------------
# Model inventory — every model on this node, with the service it powers.
# One list across all panes/tools so the dashboard can show what the node
# is actually offering (Wan/LTX video, LivePortrait, TRELLIS/LATO 3D,
# ninfer + GGUF LLMs) instead of just the LLM catalogs.
# ---------------------------------------------------------------------------

_SIZE_CACHE: dict[str, tuple[float, Optional[float]]] = {}


def _sized(key: str, compute) -> Optional[float]:
    """Directory walks are slow and sizes barely change; 5-minute cache."""
    hit = _SIZE_CACHE.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    try:
        val = compute()
    except Exception:  # noqa: BLE001
        val = None
    _SIZE_CACHE[key] = (time.time(), val)
    return val


def _hf_installed(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _hf_size_gb(repo: str) -> Optional[float]:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    root = Path(snapshot_download(repo, local_files_only=True))
    seen: set[Path] = set()
    total = 0
    for f in root.rglob("*"):
        if f.is_file():
            real = f.resolve()  # snapshots symlink into the blob store
            if real not in seen:
                seen.add(real)
                total += real.stat().st_size
    return round(total / 1e9, 1) if total else None


def _dir_size_gb(path: Path) -> Optional[float]:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 1) if total else None


def _hf_repo_root(repo: str) -> Optional[Path]:
    """The whole models--org--name cache dir (snapshots + blobs)."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        snap = Path(snapshot_download(repo, local_files_only=True))
        return snap.parents[1]
    except Exception:  # noqa: BLE001
        return None


def _win_visible(path: Path) -> str:
    """A path Explorer can open: drive letters for /mnt/X, the WSL share
    for distro-internal paths (the HF cache lives there)."""
    s = str(path)
    if s.startswith("/mnt/") and len(s) > 6:
        return s[5].upper() + ":" + s[6:].replace("/", "\\")
    return r"\\wsl.localhost\SiliconNode" + s.replace("/", "\\")


def _model_admin() -> dict:
    """Lifecycle facts per inventory model: where it lives, whether it
    can be uninstalled, and whether it is busy right now."""
    import os  # noqa: PLC0415
    from . import portrait, talkinghead, video  # noqa: PLC0415
    from .llamacpp import GGUF_DIR, LLAMACPP  # noqa: PLC0415
    from .llm import MODEL_FILE as NINFER_MODEL_FILE  # noqa: PLC0415
    trellis_repo = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
    trellis_loaded = pipeline.ENGINE.trellis_ready
    admin = {
        "trellis2-4b": {
            "kind": "hf", "loc": _hf_repo_root(trellis_repo),
            "cache_key": "trellis2", "deletable": True,
            "busy": "the TRELLIS.2 pipeline is loaded on the GPU right "
                    "now" if trellis_loaded else None},
        "dinov3-vitl16": {
            "kind": "hf", "loc": _hf_repo_root(pipeline.DINOV3_REPO),
            "cache_key": "dinov3", "deletable": True,
            "busy": "the TRELLIS.2 pipeline (which uses this "
                    "conditioner) is loaded right now"
                    if trellis_loaded else None},
        "lato2": {
            "kind": "dir", "loc": config.LATO2_ROOT / "ckpt",
            "cache_key": "lato2", "deletable": False,
            "refuse": "LATO.2's checkpoint came with the repo install "
                      "and has no automatic reinstall — remove "
                      "/opt/silicon/LATO.2/ckpt by hand if you truly "
                      "mean it.",
            "busy": None},
        "wan22-ti2v-5b": {
            "kind": "hf", "loc": _hf_repo_root(video.MODEL_REPO),
            "cache_key": "wan", "deletable": True,
            "busy": "the Wan pipeline is loaded right now"
                    if (video.ENGINE._pipe is not None
                        or getattr(video.ENGINE, "_pipe_i2v", None)
                        is not None) else None},
        "ltx2-distilled": {
            "kind": "hf", "loc": _hf_repo_root(video.LTX_REPO),
            "cache_key": "ltx", "deletable": True,
            "busy": "the LTX-2 pipeline is loaded right now"
                    if getattr(video.ENGINE, "_pipe_ltx", None)
                    is not None else None},
        "ltx23-uncensored": {
            "kind": "hf", "loc": _hf_repo_root(video.LTX_UNCENSORED_REPO),
            "cache_key": "ltx-unc", "deletable": True,
            "busy": "the LTX-2.3 uncensored pipeline is loaded right now"
                    if getattr(video.ENGINE, "_pipe_ltx_unc", None)
                    is not None else None},
        "liveportrait": {
            "kind": "dir",
            "loc": portrait.LP_ROOT / "pretrained_weights",
            "cache_key": "liveportrait", "deletable": True,
            "busy": None},
        "sadtalker": {
            "kind": "dir", "loc": talkinghead.ST_ROOT / "checkpoints",
            "cache_key": "sadtalker", "deletable": True,
            "busy": None},
        LLM_MODEL_ID: {
            "kind": "file", "loc": NINFER_MODEL_FILE,
            "cache_key": f"ninfer:{NINFER_MODEL_FILE.name}",
            "deletable": True,
            "busy": "the chat engine is serving this model right now"
                    if LLM.running else None},
    }
    llama = LLAMACPP.status()
    for entry in llama.get("models", []):
        admin[entry["file"]] = {
            "kind": "file", "loc": GGUF_DIR / entry["file"],
            "cache_key": None, "deletable": True,
            "busy": "llama-server is serving this file right now"
                    if (llama.get("running")
                        and llama.get("model") == entry["file"])
                    else None}
    return admin


@app.post("/v1/models/{model_id}/reveal")
def model_reveal(model_id: str, request: Request):
    """Open the model's folder in Windows Explorer."""
    _require_operator(request, "Opening a model folder on the host desktop")
    adm = _model_admin().get(model_id)
    if adm is None or adm["loc"] is None:
        raise HTTPException(status_code=404,
                            detail=f"No model named {model_id!r}.")
    target = adm["loc"] if adm["kind"] != "file" else adm["loc"].parent
    from .hostos import IS_WSL  # noqa: PLC0415
    if IS_WSL:
        win = _win_visible(Path(target))
        subprocess.Popen(["/mnt/c/Windows/explorer.exe", win])
        return {"ok": True, "path": win}
    subprocess.Popen(["xdg-open", str(target)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True, "path": str(target)}


@app.delete("/v1/models/{model_id}")
def model_delete(model_id: str, request: Request):
    """Uninstall a model's weights from disk. Refused while the model
    is loaded, and for the one checkpoint with no reinstall path."""
    _require_operator(request, "Uninstalling a model")
    adm = _model_admin().get(model_id)
    if adm is None:
        raise HTTPException(status_code=404,
                            detail=f"No model named {model_id!r}.")
    if not adm["deletable"]:
        raise HTTPException(status_code=403, detail=adm.get(
            "refuse", "This model cannot be uninstalled from here."))
    if adm["busy"]:
        raise HTTPException(
            status_code=409,
            detail=f"Not while {adm['busy']} — unload it first.")
    loc = adm["loc"]
    if loc is None or not Path(loc).exists():
        raise HTTPException(status_code=404,
                            detail="Its files are already gone.")
    if adm["kind"] == "file":
        Path(loc).unlink()
    else:
        shutil.rmtree(loc)
    if adm.get("cache_key"):
        _SIZE_CACHE.pop(adm["cache_key"], None)
    log.info("uninstalled model %s (%s)", model_id, loc)
    return {"ok": True}


@app.get("/v1/models")
def models_inventory():
    import os  # noqa: PLC0415
    from . import portrait, talkinghead, video  # noqa: PLC0415
    from .lato_engine import LATO_ENGINE  # noqa: PLC0415
    from .llamacpp import LLAMACPP  # noqa: PLC0415
    from .llm import NINFER_DIR  # noqa: PLC0415

    trellis_repo = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
    trellis_ok, _detail = pipeline.ENGINE.trellis_available()
    wan_inst = video.ENGINE.weights_present()
    ltx_inst = video.ENGINE.ltx_ready()
    ltx_unc_inst = video.ENGINE.ltx_uncensored_ready()
    llama = LLAMACPP.status()

    models = [
        {
            "id": "trellis2-4b", "name": "TRELLIS.2 4B",
            "capability": "image-to-mesh", "engine": "trellis2 (resident)",
            "repo": trellis_repo,
            "installed": _hf_installed(trellis_repo), "ready": trellis_ok,
            "loaded": pipeline.ENGINE.trellis_ready,
            "size_gb": _sized("trellis2", lambda: _hf_size_gb(trellis_repo)),
        },
        {
            "id": "dinov3-vitl16", "name": "DINOv3 ViT-L/16 (conditioner)",
            "capability": "image-to-mesh", "engine": "trellis2 (resident)",
            "repo": pipeline.DINOV3_REPO,
            "installed": _hf_installed(pipeline.DINOV3_REPO),
            "ready": trellis_ok, "loaded": pipeline.ENGINE.trellis_ready,
            "size_gb": _sized("dinov3",
                              lambda: _hf_size_gb(pipeline.DINOV3_REPO)),
        },
        {
            "id": "lato2", "name": "LATO.2 retopology",
            "capability": "retopologize", "engine": "lato2 (resident)",
            "repo": "checkpoint at " + str(config.LATO2_ROOT / "ckpt"),
            "installed": pipeline.ENGINE.lato_ready,
            "ready": pipeline.ENGINE.lato_ready,
            "loaded": getattr(LATO_ENGINE, "_m", None) is not None,
            "size_gb": _sized("lato2", lambda: _dir_size_gb(
                config.LATO2_ROOT / "ckpt")),
        },
        {
            "id": "wan22-ti2v-5b", "name": "Wan 2.2 TI2V-5B",
            "capability": "text-to-video", "engine": "diffusers",
            "repo": video.MODEL_REPO,
            "installed": wan_inst, "ready": video.ENGINE.ready(),
            "loaded": (video.ENGINE._pipe is not None
                       or getattr(video.ENGINE, "_pipe_i2v", None)
                       is not None),
            "size_gb": _sized("wan", lambda: _hf_size_gb(video.MODEL_REPO)),
        },
        {
            "id": "ltx2-distilled", "name": "LTX-2 distilled",
            "capability": "text-to-video", "engine": "diffusers",
            "repo": video.LTX_REPO,
            "installed": ltx_inst, "ready": ltx_inst,
            "loaded": getattr(video.ENGINE, "_pipe_ltx", None) is not None,
            "size_gb": _sized("ltx", lambda: _hf_size_gb(video.LTX_REPO)),
        },
        {
            "id": "ltx23-uncensored", "name": "LTX-2.3 Uncensored v1.4",
            "capability": "text-to-video", "engine": "diffusers",
            "repo": (video.LTX_UNCENSORED_REPO + " ("
                     + Path(video.LTX_UNCENSORED_FILE).name + ")"),
            "installed": ltx_unc_inst, "ready": ltx_unc_inst,
            "loaded": getattr(video.ENGINE, "_pipe_ltx_unc", None)
            is not None,
            "size_gb": _sized("ltx-unc", lambda: _hf_size_gb(
                video.LTX_UNCENSORED_REPO)),
        },
        {
            "id": "liveportrait", "name": "LivePortrait",
            "capability": "portrait-animate", "engine": "subprocess",
            "repo": str(portrait.LP_ROOT / "pretrained_weights"),
            "installed": portrait.ready(), "ready": portrait.ready(),
            "loaded": None,  # per-take subprocess, never resident
            "size_gb": _sized("liveportrait", lambda: _dir_size_gb(
                portrait.LP_ROOT / "pretrained_weights")),
        },
        {
            "id": "sadtalker", "name": "SadTalker",
            "capability": "talking-head", "engine": "subprocess",
            "repo": str(talkinghead.ST_ROOT / "checkpoints"),
            "installed": talkinghead.ready(),
            "ready": talkinghead.ready(),
            "loaded": None,  # per-take subprocess, never resident
            "size_gb": _sized("sadtalker", lambda: _dir_size_gb(
                talkinghead.ST_ROOT / "checkpoints")),
        },
    ]

    llm_status = LLM.status()
    for fname in llm_status.get("installed_models", []):
        # The id /v1/llm/start accepts back (as does the filename).
        mid = LLM.model_id_for(fname)
        models.append({
            "id": mid, "name": f"{mid} (ninfer INT8)",
            "capability": "llm", "engine": "ninfer-3090",
            "repo": str(NINFER_DIR / "models" / fname),
            "installed": True,
            "ready": llm_status.get("installed", False),
            "loaded": bool(llm_status.get("running")
                           and llm_status.get("healthy")),
            "size_gb": _sized(f"ninfer:{fname}", lambda f=fname: round(
                (NINFER_DIR / "models" / f).stat().st_size / 1e9, 1)),
        })

    from .llamacpp import GGUF_DIR_WIN  # noqa: PLC0415
    for entry in llama.get("models", []):
        prism = entry.get("engine") == "prism"
        models.append({
            "id": entry["file"], "name": entry["file"],
            "capability": "llm-gguf",
            "engine": "llama.cpp (PrismML fork)" if prism else "llama.cpp",
            "repo": GGUF_DIR_WIN,
            "installed": True,
            "ready": llama.get("prism_engine_installed" if prism
                               else "engine_installed", False),
            "loaded": bool(llama.get("running")
                           and llama.get("model") == entry["file"]),
            "size_gb": entry.get("size_gb"),
        })

    admin = _model_admin()
    for m in models:
        adm = admin.get(m["id"])
        if adm:
            m["deletable"] = bool(adm["deletable"] and not adm["busy"])
            m["busy_reason"] = adm["busy"]
            m["refuse_reason"] = adm.get("refuse")
            m["path"] = (_win_visible(Path(adm["loc"]))
                         if adm["loc"] else None)
    return {"models": models}


# Where each capability that takes a file accepts it.
_FILE_ROUTES = {
    "image-to-mesh": "POST /v1/image-to-mesh (multipart image)",
    "retopologize": "POST /v1/retopologize (multipart mesh)",
    "portrait-animate": "POST /v1/portrait-animate (image_b64 + driving_b64)",
    "talking-head": "POST /v1/talking-head (image_b64 + audio_b64)",
    "text-to-video": "POST /v1/text-to-video (image_b64 for a start image)",
}


@app.post("/v1/jobs")
async def submit_generic(request: Request):
    body = await request.json()
    cap = body.get("capability")
    ids = {c["id"] for c in capability_list()}
    if cap not in ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown capability {cap!r}. This node offers: "
                   f"{', '.join(sorted(ids))}.")
    params = {k: v for k, v in body.items() if k != "capability"}
    # A job param that names a file is only ever written by the server,
    # into the job's own folder, from an upload (hub 154). Taken from a
    # caller, it would make the node read any file it can see — and a
    # URL there made the video lane fetch it.
    named = sorted(k for k in params if k.endswith("_path"))
    if named:
        raise HTTPException(
            status_code=400,
            detail=f"{', '.join(named)}: files can't be named by path. "
                   "Send them with the job instead"
                   + (f" — {_FILE_ROUTES[cap]}." if cap in _FILE_ROUTES
                      else "."))
    if cap in _FILE_ROUTES and cap != "text-to-video":
        raise HTTPException(
            status_code=400,
            detail=f"{cap} needs an input file; use {_FILE_ROUTES[cap]}, "
                   "which accepts it directly.")
    if cap == "text-to-video" and ("width" in params or "height" in params):
        from . import video as _video  # noqa: PLC0415
        try:
            _video.canvas(width=params.get("width"),
                          height=params.get("height"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    params.pop("delivery", None)   # the node's to write, not a caller's
    _require_enabled(cap)
    job = STORE.submit(cap, params)
    job.submitted_by = _submitter(request, cap)
    job.save()
    return {"job_id": job.job_id}


@app.post("/v1/retopologize")
async def retopologize_upload(
    request: Request,
    mesh: UploadFile = File(...),
    vert_num: str = Form(str(config.VERT_NUM_DEFAULT)),
    seed: Optional[str] = Form(None),
):
    """Multipart sugar for the retopologize capability (mesh file upload)."""
    params = {"vert_num": vert_num, "seed": seed}
    try:
        pipeline._parse_vert_num(params)
        pipeline._parse_seed(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    suffix = Path(mesh.filename or "input.glb").suffix or ".glb"
    if suffix.lower() not in (".glb", ".obj", ".ply", ".gltf"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mesh format {suffix}; send GLB, OBJ, PLY "
                   "or GLTF.")
    _require_enabled("retopologize")
    job = STORE.submit("retopologize", params, defer=True)
    job.submitted_by = _submitter(request, "retopologize")
    job.dir.mkdir(parents=True, exist_ok=True)
    mesh_path = job.dir / f"input{suffix}"
    with _staging(job):
        await uploads.save_upload(mesh, mesh_path)
    job.params["mesh_path"] = str(mesh_path)
    STORE.enqueue(job)
    return {"job_id": job.job_id}


# ---------------------------------------------------------------------------
# LLM management (ninfer-3090)
# ---------------------------------------------------------------------------

def _require_operator(request: Request, what: str) -> None:
    """Engine control is for the swarm admin or the node owner —
    members chat with the model, they don't restart it (handoff 132)."""
    if _role(request) not in ("admin", "node"):
        raise HTTPException(
            status_code=403,
            detail=f"{what} is for the swarm admin or the node owner.")


@app.get("/v1/llm")
def llm_status():
    return LLM.status()


@app.post("/v1/systemone")
async def systemone_decide(request: Request):
    """Typed decisions on the GPU — the node's System One lane (Laya).

    Same wire shape the Mac and TypeSafe already speak, so the Decisions
    panel can pick this lane over its own without a second decoder:
    {state, model?, questions:{id:{type, instructions, criteria?}}}.
    Members may decide (it is a service like chat, not an operator
    action); the bearer is required off-box like every /v1/ route, and
    the state's text is never logged."""
    from .systemone import (  # noqa: PLC0415
        DecisionError, MAX_BODY_BYTES, SYSTEMONE, validate)
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request is {len(raw)} bytes; this node caps the "
                   f"decision body at {MAX_BODY_BYTES}.")
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(status_code=400,
                            detail="Body must be JSON.") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400,
                            detail="Body must be a JSON object.")
    role = _role(request)
    if role not in ("admin", "node"):
        from .serving import SERVING  # noqa: PLC0415
        if SERVING.paused:
            raise HTTPException(status_code=503, detail=SERVING.refusal())
        from .clients import CLIENTS  # noqa: PLC0415
        CLIENTS.count_llm(_actor(request))
    try:
        model = validate(body.get("state"), body.get("questions"),
                         body.get("model"))
    except DecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        # Off the event loop: a forward pass is ~30 ms of GPU work, and a
        # cold build is seconds.
        return await asyncio.to_thread(
            SYSTEMONE.predict, body["state"], body["questions"], model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        # Never echo the state back in an error.
        log.exception("decision failed")
        raise HTTPException(
            status_code=500,
            detail=f"The decision engine failed: "
                   f"{type(exc).__name__}") from None


def _fetchable_media(body) -> list[str]:
    """Message parts carrying a URL the engine would fetch itself.

    vLLM (HyperQwen) and llama-server fetch http(s) image/video/audio URLs
    from the node's network; a member's chat must not make the node do
    that (hub 154, the same SSRF the Mac closed in its own chat route).
    Inline data: URLs are the only form accepted."""
    found = []
    messages = body.get("messages") if isinstance(body, dict) else None
    for msg in messages if isinstance(messages, list) else ():
        content = msg.get("content") if isinstance(msg, dict) else None
        for part in content if isinstance(content, list) else ():
            if not isinstance(part, dict):
                continue
            for key, val in part.items():
                url = (val.get("url") if isinstance(val, dict)
                       else val if key.endswith("url") else None)
                if isinstance(url, str) and \
                        not url.strip().lower().startswith("data:"):
                    found.append(key)
    return found


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """The OpenAI chat surface THROUGH the node, so member usage is
    attributable (handoff 132). The engines bind the Windows loopback,
    which WSL sockets cannot reach — interop curl bridges it, streaming
    included. Clients that should be counted point at :8790/v1 instead
    of the engine port; the engine ports keep working unchanged."""
    import uuid  # noqa: PLC0415
    from .llamacpp import LLAMACPP, PORT as GGUF_PORT  # noqa: PLC0415
    from .hyperqwen import HYPERQWEN, PORT as HQ_PORT  # noqa: PLC0415
    if LLM.running:
        port = LLM_PORT
    elif LLAMACPP.running:
        port = GGUF_PORT
    elif HYPERQWEN.running:
        # Same OpenAI surface, one more port; members reach it through
        # this route exactly like the other two.
        port = HQ_PORT
    else:
        raise HTTPException(
            status_code=503,
            detail="No chat engine is running — start one via "
                   "/v1/llm/start, /v1/gguf/start or "
                   "/v1/hyperqwen/start.")
    role = _role(request)
    if role not in ("admin", "node"):
        from .serving import SERVING  # noqa: PLC0415
        if SERVING.paused:
            raise HTTPException(status_code=503,
                                detail=SERVING.refusal())
        from .clients import CLIENTS  # noqa: PLC0415
        CLIENTS.count_llm(_actor(request))
    body = await request.body()
    try:
        parsed = json.loads(body or b"{}")
    except ValueError:
        parsed = None   # the engine answers a malformed body itself
    remote = _fetchable_media(parsed)
    if remote:
        raise HTTPException(
            status_code=400,
            detail=f"{', '.join(sorted(set(remote)))}: send media inline "
                   "as a data: URL — this node does not fetch URLs on a "
                   "caller's behalf.")
    stream = b'"stream": true' in body or b'"stream":true' in body
    from .hostos import bridge_curl_argv, chat_spool_dir  # noqa: PLC0415
    wsl_tmp = chat_spool_dir()
    name = f"{uuid.uuid4().hex}.json"

    def _spool():  # drvfs I/O can stall — never run it on the event loop
        wsl_tmp.mkdir(parents=True, exist_ok=True)
        (wsl_tmp / name).write_bytes(body)
    await asyncio.to_thread(_spool)
    # Guards (hub 135): a stuck engine must cost one chat, never the node.
    # --connect-timeout bounds the dial; -m caps the whole exchange; the
    # pump's idle timeout below catches an engine that answers then hangs
    # (non-stream bodies arrive all at once, so idle stays generous).
    proc = await asyncio.create_subprocess_exec(
        *bridge_curl_argv(f"http://127.0.0.1:{port}/v1/chat/completions",
                          wsl_tmp / name),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)

    async def pump():
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(4096), timeout=600)
                except asyncio.TimeoutError:
                    log.warning("chat bridge idle >600s; dropping this "
                                "chat (engine stuck or unreachable)")
                    break
                if not chunk:
                    break
                yield chunk
        finally:
            if proc.returncode is None:
                proc.kill()
            await asyncio.to_thread(
                (wsl_tmp / name).unlink, missing_ok=True)

    from fastapi.responses import StreamingResponse  # noqa: PLC0415
    return StreamingResponse(
        pump(),
        media_type="text/event-stream" if stream else "application/json")


@app.get("/v1/llm/models")
def llm_models():
    """The Mac gateway polls this route for the node's LLM inventory —
    it showed up as steady 404s in the access log before it existed."""
    st = LLM.status()
    return {
        "models": st.get("installed_models", []),
        "active": st.get("model") if st.get("running") else None,
        "context_length": st.get("context_length"),
        "max_concurrency": st.get("max_concurrency"),
    }


@app.post("/v1/llm/start")
async def llm_start(request: Request):
    """Start or switch the chat model: {model?, model_file?, profile?,
    context_length?}. Everything the request names is checked BEFORE the
    serving model is touched — an unknown model is a 404 and the old one
    keeps answering. A start that fails after the old model was stopped
    puts the old model back."""
    _require_operator(request, "Starting or switching the chat model")
    from .llm import PROFILES as _PROFILES, UnknownModel  # noqa: PLC0415
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    profile = body.get("profile") or "c1"
    if profile not in _PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile {profile!r}; use one of "
                   f"{', '.join(sorted(_PROFILES))}.")
    context_length = None
    if body.get("context_length") is not None:
        try:
            context_length = int(body["context_length"])
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="context_length must be a number.") from None
    try:
        # The Mac sends the advertised id as `model` and the listed file
        # as `model_file` (hub 156/162); older Macs sent the filename as
        # `model`. All of them resolve here.
        model_path = LLM.resolve_model(body.get("model"),
                                       body.get("model_file"))
    except UnknownModel as exc:
        return JSONResponse(status_code=404, content={
            "error": str(exc), "models": exc.choices})
    if STORE.queue_depth() > 0:
        raise HTTPException(
            status_code=409,
            detail="A GPU job is queued or running; the LLM will not start "
                   "until the job queue drains. Try again shortly.")

    def _switch():
        was_serving = LLM.running
        if was_serving:
            LLM.stop()  # switching model/profile/context
        _stop_hyperqwen_if_running()
        pipeline.ENGINE.unload()
        try:
            LLM.start(profile, model_file=model_path.name,
                      context_length=context_length)
        except (ValueError, RuntimeError) as exc:
            if not was_serving:
                raise
            raise RuntimeError(f"{exc} — {LLM.fall_back()}.") from None
    try:
        # start() blocks up to ~3 min waiting healthy; on the event loop
        # that froze every route for the duration (hub 135).
        await asyncio.to_thread(_switch)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return LLM.status()


@app.post("/v1/llm/stop")
def llm_stop(request: Request):
    _require_operator(request, "Stopping the chat model")
    LLM.stop()
    return LLM.status()


@app.get("/v1/harness")
def harness_status():
    from .harness import HARNESS
    return HARNESS.status()


@app.post("/v1/harness/start")
def harness_start(request: Request):
    _require_operator(request, "Starting the agent harness")
    from .harness import HARNESS
    try:
        HARNESS.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return HARNESS.status()


@app.post("/v1/harness/stop")
def harness_stop(request: Request):
    _require_operator(request, "Stopping the agent harness")
    from .harness import HARNESS
    HARNESS.stop()
    return HARNESS.status()


@app.post("/v1/text-to-video")
async def text_to_video_submit(request: Request):
    """The Mac's Video pane submit (NodeVideoRuntime.generate): JSON
    {model, prompt, seconds, resolution, image_b64?, image_name?}."""
    import base64
    body = await request.json()
    model = (body.get("model") or "wan22-ti2v-5b").strip()
    from . import video as _video
    engines = {"ltx2-distilled": "ltx", "ltx23-uncensored": "ltx-uncensored"}
    if model == "ltx2-distilled":
        if not _video.ENGINE.ltx_ready():
            raise HTTPException(
                status_code=503,
                detail="LTX-2 distilled is downloading to this node right "
                       "now — try again in a few minutes, or use Wan 2.2 "
                       "5B meanwhile.")
    elif model == "ltx23-uncensored":
        if not _video.ENGINE.ltx_uncensored_ready():
            raise HTTPException(
                status_code=503,
                detail="LTX-2.3 Uncensored isn't installed on this node yet "
                       "— install it from the node's Store page (it needs "
                       "LTX-2 distilled installed too), or use another "
                       "model meanwhile.")
    elif model not in ("wan22-ti2v-5b", "text-to-video", "wan2.2-ti2v-5b"):
        raise HTTPException(
            status_code=400,
            detail=f"This node serves wan22-ti2v-5b, ltx2-distilled and "
                   f"ltx23-uncensored; {model} isn't one of them.")
    engine = engines.get(model, "wan")
    prompt = (body.get("prompt") or "").strip()
    if not prompt and not body.get("image_b64"):
        raise HTTPException(status_code=400,
                            detail="A prompt (or an image) is required.")
    if not _video_ready():
        raise HTTPException(
            status_code=503,
            detail="Video weights are still installing on this node.")
    try:
        seconds = float(body.get("seconds", 2))
    except (TypeError, ValueError):
        seconds = 2.0
    # Wan and distilled LTX top out at five seconds; the uncensored merge
    # takes the Mac's full ten (its card says 40 s, the 24 GB card doesn't).
    max_frames = 241 if engine == "ltx-uncensored" else 121
    frames = max(17, min(max_frames, int(round(seconds * 24)) + 1))
    # The Mac sends "480p" / "720p" / "1080p"; agents may send "WxH". Both
    # were read as WxH-or-1280x704 before, so every Mac clip came out at
    # 1280x704 whatever was picked, and any WxH at all was rendered — up
    # to an out-of-memory crash that took the queue with it (hub 159).
    from .capsettings import CAPS  # noqa: PLC0415
    res = str(body.get("resolution")
              or CAPS.settings("text-to-video").get("resolution", "720p"))
    try:
        if "x" in res.lower():
            w, _, h = res.lower().partition("x")
            spec = _video.canvas(width=w.strip(), height=h.strip())
        else:
            spec = _video.canvas(res)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    params = {"prompt": prompt, "frames": frames,
              "width": spec["internal_width"],
              "height": spec["internal_height"],
              "delivery": spec,
              "seed": body.get("seed"),
              "engine": engine}
    # Optional knobs the Mac doesn't send but an agent might.
    for key in ("steps", "guidance", "negative_prompt"):
        if body.get(key) not in (None, ""):
            params[key] = body[key]
    _require_enabled("text-to-video")
    job = STORE.submit("text-to-video", params, defer=True)
    job.submitted_by = _submitter(request, "text-to-video")
    job.dir.mkdir(parents=True, exist_ok=True)
    if body.get("image_b64"):
        with _staging(job):
            try:
                img = base64.b64decode(body["image_b64"])
            except ValueError:
                raise HTTPException(status_code=400,
                                    detail="image_b64 is not valid base64."
                                    ) from None
            suffix = (Path(body.get("image_name", "start.png")).suffix
                      or ".png")
            await uploads.write_bytes(job.dir / f"start{suffix}", img)
        job.params["image_path"] = str(job.dir / f"start{suffix}")
    STORE.enqueue(job)
    return {"job_id": job.job_id}


@app.post("/v1/text-to-image")
async def text_to_image_submit(request: Request):
    """The Mac's NodeImageRuntime contract (hub 136): JSON {prompt,
    width, height, steps?, seed?, negative_prompt?, model?} → {job_id};
    poll GET /v1/jobs/{id}; result_urls carries the .png."""
    body = await request.json()
    from . import image as _image
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400,
                            detail="A prompt is required.")
    model = (body.get("model") or "").strip() or None
    if model and model not in _image.MODEL_REPOS:
        raise HTTPException(
            status_code=400,
            detail=f"This node serves "
                   f"{', '.join(sorted(_image.MODEL_REPOS))}; "
                   f"{model} isn't one of them.")
    installed = _image.ENGINE.installed_models()
    if not installed:
        raise HTTPException(
            status_code=503,
            detail="Image model weights are still installing on this "
                   "node — try again in a few minutes.")
    if model and model not in installed:
        raise HTTPException(
            status_code=503,
            detail=f"The {model} weights are not installed on this node "
                   "— install them from the model store, or omit the "
                   f"model field to use {', '.join(installed)}.")
    _require_enabled("text-to-image")
    params = {"prompt": prompt,
              "width": body.get("width", 1024),
              "height": body.get("height", 1024),
              "steps": body.get("steps"),
              "seed": body.get("seed"),
              "negative_prompt": body.get("negative_prompt"),
              "model": model}
    params = {k: v for k, v in params.items() if v is not None}
    job = STORE.submit("text-to-image", params)
    job.submitted_by = _submitter(request, "text-to-image")
    job.save()
    return {"job_id": job.job_id}


@app.get("/v1/store")
def store_list():
    """Every model this node knows how to host (hub 137). Any bearer can
    read it — members see what could be asked for."""
    from . import modelstore  # noqa: PLC0415
    return modelstore.listing()


@app.post("/v1/store/install")
async def store_install(request: Request):
    _require_operator(request, "Installing models")
    from . import modelstore  # noqa: PLC0415
    body = await request.json()
    mid = str(body.get("model_id") or "")
    entry = modelstore._catalog().get(mid)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No model named {mid!r} in this node's store — "
                   "GET /v1/store lists what exists.")
    if not entry["installable"]:
        raise HTTPException(status_code=400, detail=entry["note"])
    if modelstore._installed(entry):
        return {"ok": True, "already_installed": True,
                "detail": f"{entry['name']} is already installed."}
    refusal = modelstore.disk_refusal(entry)
    if refusal:
        raise HTTPException(status_code=507, detail=refusal)
    for j in STORE.snapshot():
        if (j.capability == "store-install" and j.state in
                ("queued", "running") and j.params.get("model_id") == mid):
            return {"ok": True, "job_id": j.job_id,
                    "detail": "That install is already in the queue."}
    job = STORE.submit("store-install", {"model_id": mid})
    job.submitted_by = _submitter(request, "store-install")
    job.save()
    modelstore._record("installs", mid, _actor(request))
    return {"job_id": job.job_id}


@app.delete("/v1/store/{model_id}")
def store_delete(model_id: str, request: Request):
    _require_operator(request, "Deleting models")
    from . import modelstore  # noqa: PLC0415
    entry = modelstore._catalog().get(model_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No model named {model_id!r} in this node's store.")
    if not modelstore._installed(entry):
        return {"ok": True, "already_absent": True,
                "detail": f"{entry['name']} is not installed."}
    for j in STORE.snapshot():
        if j.state not in ("queued", "running"):
            continue
        if j.capability == entry["capability"]:
            raise HTTPException(
                status_code=409,
                detail=f"A {entry['capability']} job is queued or "
                       "running and may need these weights — wait for "
                       "the queue to drain, then delete.")
        if (j.capability == "store-install"
                and j.params.get("model_id") == model_id):
            raise HTTPException(
                status_code=409,
                detail=f"{entry['name']} is being installed right now — "
                       "cancel that job first if you want it gone.")
    freed = modelstore.delete(model_id)
    modelstore._record("deletes", model_id, _actor(request))
    return {"ok": True, "freed_bytes": freed}


@app.post("/v1/portrait-animate")
async def portrait_animate_submit(request: Request):
    """The Mac's persona take → animated clip. JSON body per their
    VideoRuntime.animatePortrait: image_b64/image_name + driving_b64/
    driving_name."""
    import base64
    body = await request.json()
    try:
        image = base64.b64decode(body["image_b64"])
        driving = base64.b64decode(body["driving_b64"])
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="image_b64 and driving_b64 (base64) are required."
        ) from None
    from . import portrait
    if not portrait.ready():
        raise HTTPException(
            status_code=503,
            detail="LivePortrait is still installing on this node.")
    _require_enabled("portrait-animate")
    job = STORE.submit("portrait-animate", {}, defer=True)
    job.submitted_by = _submitter(request, "portrait-animate")
    job.dir.mkdir(parents=True, exist_ok=True)
    img_suffix = Path(body.get("image_name", "p.jpg")).suffix or ".jpg"
    drv_suffix = Path(body.get("driving_name", "d.mp4")).suffix or ".mp4"
    with _staging(job):
        await uploads.write_bytes(job.dir / f"portrait{img_suffix}", image)
        await uploads.write_bytes(job.dir / f"driving{drv_suffix}", driving)
    job.params["image_path"] = str(job.dir / f"portrait{img_suffix}")
    job.params["driving_path"] = str(job.dir / f"driving{drv_suffix}")
    STORE.enqueue(job)
    return {"job_id": job.job_id}


@app.post("/v1/talking-head")
async def talking_head_submit(request: Request):
    """Photo + audio → lip-synced clip (silicon-node #5). Body mirrors
    /v1/portrait-animate's idiom: image_b64/image_name +
    audio_b64/audio_name."""
    import base64
    body = await request.json()
    try:
        image = base64.b64decode(body["image_b64"])
        audio = base64.b64decode(body["audio_b64"])
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="image_b64 and audio_b64 (base64) are required."
        ) from None
    from . import talkinghead
    if not talkinghead.ready():
        raise HTTPException(
            status_code=503,
            detail="SadTalker is still installing on this node.")
    _require_enabled("talking-head")
    job = STORE.submit("talking-head", {}, defer=True)
    job.submitted_by = _submitter(request, "talking-head")
    job.dir.mkdir(parents=True, exist_ok=True)
    img_suffix = Path(body.get("image_name", "p.jpg")).suffix or ".jpg"
    aud_suffix = Path(body.get("audio_name", "a.wav")).suffix or ".wav"
    with _staging(job):
        await uploads.write_bytes(job.dir / f"portrait{img_suffix}", image)
        await uploads.write_bytes(job.dir / f"speech{aud_suffix}", audio)
    job.params["image_path"] = str(job.dir / f"portrait{img_suffix}")
    job.params["audio_path"] = str(job.dir / f"speech{aud_suffix}")
    STORE.enqueue(job)
    return {"job_id": job.job_id}


@app.get("/v1/hf/search")
def hf_search(q: str = ""):
    """Search Hugging Face for GGUF models (the Mac's Models-tab search)."""
    import httpx  # noqa: PLC0415
    q = q.strip()
    if not q:
        return []
    r = httpx.get("https://huggingface.co/api/models",
                  params={"search": q, "filter": "gguf", "limit": 12,
                          "sort": "downloads", "direction": "-1"},
                  timeout=15)
    out = []
    for m in r.json():
        out.append({"repo": m.get("id"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0)})
    return out


@app.get("/v1/hf/files")
def hf_files(repo: str):
    """GGUF files (with sizes) inside one HF repo."""
    import httpx  # noqa: PLC0415
    r = httpx.get(f"https://huggingface.co/api/models/{repo}/tree/main",
                  timeout=15)
    files = []
    for f in r.json():
        name = f.get("path", "")
        if name.endswith(".gguf"):
            files.append({"file": name,
                          "size_gb": round((f.get("size") or 0) / 1e9, 1)})
    return sorted(files, key=lambda f: f["size_gb"])


@app.get("/v1/gguf")
def gguf_status():
    from .llamacpp import GGUF_DL, LLAMACPP
    return {**LLAMACPP.status(), "downloads": GGUF_DL.progress()}


@app.post("/v1/gguf/download")
async def gguf_download(request: Request):
    _require_operator(request, "Downloading a GGUF model")
    from .llamacpp import GGUF_DL, LLAMACPP, needs_prism
    body = await request.json()
    if not body.get("repo") or not body.get("file"):
        raise HTTPException(status_code=400,
                            detail="repo and file are required.")
    # Fetch the engine alongside the model — PrismML's fork for its
    # ternary packings, the stock build for everything else.
    LLAMACPP.install_engine_async(
        "prism" if needs_prism(body["file"]) else "stock")
    GGUF_DL.start(body["repo"], body["file"])
    if body.get("mmproj"):
        # The vision projector that belongs with the weights (27B Bonsai).
        GGUF_DL.start(body["repo"], body["mmproj"])
    if body.get("lora"):
        # A runtime adapter named in the server-side registry — the URL
        # comes from there, never from the request.
        from .llamacpp import ADAPTERS  # noqa: PLC0415
        meta = ADAPTERS.get(str(body["lora"]))
        if meta is None:
            raise HTTPException(
                status_code=404,
                detail=f"No adapter named {body['lora']!r} on this node.")
        GGUF_DL.start_url(meta["url"], str(body["lora"]),
                          sha256=meta.get("sha256"))
    return {"ok": True}


@app.post("/v1/gguf/start")
async def gguf_start(request: Request):
    _require_operator(request, "Starting or switching the GGUF engine")
    from .llamacpp import LLAMACPP
    body = await request.json()
    if STORE.queue_depth() > 0:
        raise HTTPException(
            status_code=409,
            detail="A GPU job is queued or running; try again shortly.")
    def _switch():
        if LLM.running:
            LLM.stop()  # one language engine at a time on this card
        _stop_hyperqwen_if_running()
        pipeline.ENGINE.unload()
        ctx = body.get("context")
        LLAMACPP.start(body.get("file", ""),
                       int(ctx) if ctx is not None else None,
                       lora=body.get("lora"))
    try:
        # Same event-loop guard as /v1/llm/start (hub 135).
        await asyncio.to_thread(_switch)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return LLAMACPP.status()


@app.post("/v1/gguf/stop")
def gguf_stop(request: Request):
    _require_operator(request, "Stopping the GGUF engine")
    from .llamacpp import LLAMACPP
    LLAMACPP.stop()
    return LLAMACPP.status()


@app.get("/v1/hyperqwen")
def hyperqwen_status():
    from .hyperqwen import HYPERQWEN
    return HYPERQWEN.status()


@app.post("/v1/hyperqwen/install")
def hyperqwen_install(request: Request):
    """Clone the checkout and pull the 9.5 GB image. The one-time ~20 GB
    requantization happens inside the container on first start."""
    _require_operator(request, "Installing the HyperQwen engine")
    from .hyperqwen import HYPERQWEN
    ok, detail = HYPERQWEN.docker_ready()
    if not ok:
        raise HTTPException(status_code=503, detail=detail)
    HYPERQWEN.install_async()
    return HYPERQWEN.status()


@app.post("/v1/hyperqwen/start")
async def hyperqwen_start(request: Request):
    _require_operator(request, "Starting the HyperQwen engine")
    from .hyperqwen import HYPERQWEN
    body = {}
    with contextlib.suppress(Exception):
        body = await request.json()
    if STORE.queue_depth() > 0:
        raise HTTPException(
            status_code=409,
            detail="A GPU job is queued or running; this engine wants the "
                   "whole card. Try again when the queue drains.")

    def _switch():
        pipeline.ENGINE.unload()
        from .systemone import SYSTEMONE  # noqa: PLC0415
        SYSTEMONE.unload()
        HYPERQWEN.start(body.get("mode"))
    try:
        # Returns once the container is up; the engine then loads for
        # minutes and /v1/hyperqwen reports starting until it serves.
        await asyncio.to_thread(_switch)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return HYPERQWEN.status()


@app.post("/v1/hyperqwen/stop")
async def hyperqwen_stop(request: Request):
    _require_operator(request, "Stopping the HyperQwen engine")
    from .hyperqwen import HYPERQWEN
    await asyncio.to_thread(HYPERQWEN.stop)
    return HYPERQWEN.status()


@app.post("/v1/llm/models/download")
async def llm_model_download(request: Request):
    _require_operator(request, "Downloading a chat model")
    body = await request.json()
    try:
        name = DOWNLOADS.start(body.get("model_id", ""))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"filename": name}


@app.get("/v1/llm/models/downloads")
def llm_model_downloads():
    return DOWNLOADS.progress()


# ---------------------------------------------------------------------------
# Phase 4 groundwork: node advertisement (read-only, no delegation yet)
# ---------------------------------------------------------------------------

@app.get("/v1/node")
def node():
    met = _gpu_metrics()
    consumer = _gpu_consumer(met)
    if consumer:
        met["gpu_consumer"] = consumer
    from .llm import PROFILES as _LLM_PROFILES  # noqa: PLC0415
    from .serving import SERVING  # noqa: PLC0415
    serving = SERVING.status()
    # NOTE for clients: the top-level "profile" is the GPU hardware
    # profile (predates hub 138); the capability preset lives at
    # serving.profile.
    serving["profile"] = _current_profile()
    return {
        "name": config.SERVER_NAME,
        "platform": config.PLATFORM,
        "profile": _gpu_profile(),
        "serving": serving,
        "capabilities": capability_list(),
        "metrics": met,
        "queue": _queue_view(),
        # The decision lane, so the Mac's Decisions panel can choose
        # between its own laya-mlx lane and this GPU one.
        "decisions": _decisions_status(),
        # The optional third chat engine, so the swarm can see it exists
        # and whether it is the one currently serving.
        "hyperqwen": _hyperqwen_status(),
        "llm": {
            "running": LLM.running,
            "model": getattr(LLM, "model_id", LLM_MODEL_ID)
            if LLM.installed else None,
            "context_length": getattr(LLM, "_ctx_effective", None)
            if LLM.running else None,
            "max_concurrency":
                _LLM_PROFILES[LLM._profile]["max_concurrency"]
            if LLM.running else None,
            # Set when a start failed and nothing is serving: chat is
            # down, not merely unloaded.
            "error": None if LLM.running else LLM.last_error,
        },
        "peers": [{"name": p.get("name", "?"), "base_url": p["base_url"]}
                  for p in config.PEERS],
    }


def _gpu_consumer(met: dict) -> Optional[str]:
    """What owns the GPU right now, for the Mac's meter caption
    (handoff 128): a job, the chat model, something external, or
    nothing worth naming."""
    cur = STORE._current
    if cur:
        j = STORE.get(cur)
        # Store installs are downloads, not GPU tenants — fall through
        # so the caption names whatever actually owns the card.
        if j is not None and j.capability != "store-install":
            return f"job:{j.capability}"
    if LLM.running:
        return "llm"
    try:
        from .llamacpp import LLAMACPP  # noqa: PLC0415
        if LLAMACPP.running:
            return "llm"
    except Exception:  # noqa: BLE001
        pass
    if (met.get("gpu_util_pct") or 0) > 15:
        return "external"
    return None


def _queue_view() -> dict:
    """Per-job queue rows for the Mac's Swarm page (handoff 128)."""
    running = None
    cur = STORE._current
    if cur:
        j = STORE.get(cur)
        if j is not None and j.state == "running":
            running = {"id": j.job_id, "kind": j.capability,
                       "progress": round(j.progress or 0.0, 3),
                       "started_at": j.started_at,
                       "submitted_by": (j.submitted_by or {}).get("client")}
    with STORE._lock:
        pending_ids = list(STORE._pending)
    pending = []
    for jid in pending_ids:
        j = STORE.get(jid)
        if j is not None:
            pending.append({"id": j.job_id, "kind": j.capability,
                            "submitted_by":
                                (j.submitted_by or {}).get("client")})
    return {"running": running, "pending": pending}


def _require_job_owner(request: Request, job_id: str) -> None:
    """Members touch only their own jobs; the swarm admin and the node
    owner touch anything (handoff 132)."""
    if _role(request) in ("admin", "node"):
        return
    job = STORE.get(job_id)
    # Compared by credential (kind + id), never by display name (hub 155).
    if job is None or _job_owner(job) != _owner_key(request):
        raise HTTPException(
            status_code=403,
            detail="Members can manage only their own jobs — this one "
                   "was submitted by someone else.")


@app.delete("/v1/queue/{job_id}")
def queue_delete(job_id: str, request: Request):
    """Cancel one job by id — the Mac's per-row ✕ button (handoff 128).
    Pending jobs are dropped; the running job aborts at its next
    progress checkpoint."""
    _require_job_owner(request, job_id)
    if STORE.cancel(job_id):
        return {"ok": True}
    raise HTTPException(
        status_code=404,
        detail=f"No cancellable job {job_id!r} in the queue — it may "
               "have already finished.")


def _nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return [s.strip() for s in out.stdout.strip().split(",")]
    except Exception:  # noqa: BLE001
        return []

_GPU_SPECS = {
    # name substring -> (cuda cores, memory bandwidth GB/s)
    "3090 Ti": (10752, 1008),
    "3090": (10496, 936),
}


def _gpu_profile() -> dict:
    vals = _nvidia_smi("name,memory.total,driver_version")
    if len(vals) < 3:
        return {}
    prof = {"gpu": vals[0], "vram_mb": int(float(vals[1])),
            "driver": vals[2]}
    for key, (cores, bw) in _GPU_SPECS.items():
        if key in vals[0]:
            prof["cuda_cores"] = cores
            prof["bandwidth_gbps"] = bw
            break
    try:
        import shutil as _sh
        from .hostos import IS_WSL as _wsl, WIN_HOME as _win_home  # noqa: PLC0415
        prof["disk_free_gb"] = round(
            _sh.disk_usage(_win_home if _wsl else "/").free / 1e9)
    except OSError:
        pass
    return prof

def _gpu_metrics() -> dict:
    vals = _nvidia_smi("memory.used,memory.free,utilization.gpu")
    if len(vals) >= 3:
        return {"vram_used_mb": int(float(vals[0])),
                "vram_free_mb": int(float(vals[1])),
                # The swarm router's one normalized cross-platform field
                # (agreed on silicon-optimizer #7).
                "headroom_gb": round(float(vals[1]) / 1024, 1),
                "gpu_util_pct": int(float(vals[2])),
                "queue_depth": STORE.queue_depth()}
    return {"queue_depth": STORE.queue_depth()}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def human_errors(request: Request, exc: HTTPException):
    """Non-2xx bodies are shown to the user by the Mac client (first 300
    bytes) — return the message as plain readable JSON, not a stack dump."""
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail})


_UI_DIR = Path(__file__).parent / "ui"
_UI_FILES = {"manifest.json": "application/manifest+json",
             "icon.png": "image/png"}


@app.get("/")
@app.get("/ui")
def ui():
    page = _UI_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="UI not deployed.")
    return FileResponse(page, media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/ui/{name}")
def ui_asset(name: str):
    if name not in _UI_FILES or not (_UI_DIR / name).is_file():
        raise HTTPException(status_code=404, detail="No such asset.")
    return FileResponse(_UI_DIR / name, media_type=_UI_FILES[name])


def _guard_loopback_forwarders() -> bool:
    """Turn strict auth on if a raw TCP forwarder delivers outside
    traffic from loopback (hub 155). Fail-safe in one direction only:
    once on, it stays on until the service restarts without one."""
    if config.REQUIRE_AUTH:
        return False
    from .hostos import loopback_tcp_forwards  # noqa: PLC0415
    forwards = loopback_tcp_forwards(config.PORT)
    if not forwards:
        return False
    config.REQUIRE_AUTH = True
    log.warning("tailscale serve forwards raw TCP to this service (%s): "
                "those requests arrive from loopback with nothing to mark "
                "them, so loopback callers now need a token too "
                "(SILICON_NODE_REQUIRE_AUTH behaviour). Serve it over "
                "HTTP instead to keep the console open.", ", ".join(forwards))
    return True


def create_app() -> FastAPI:
    config.ensure_dirs()
    from .hostos import IS_WSL  # noqa: PLC0415
    if not _guard_loopback_forwarders() and not IS_WSL \
            and not config.REQUIRE_AUTH:
        # A forwarder can be added while the node runs; look again now
        # and then (one `tailscale serve status` every five minutes).
        import threading  # noqa: PLC0415

        def _watch_forwarders():
            while not config.REQUIRE_AUTH:
                time.sleep(300)
                _guard_loopback_forwarders()
        threading.Thread(target=_watch_forwarders, daemon=True,
                         name="forwarder-guard").start()
    # Off by default: the decision lane builds on its first request so it
    # costs no VRAM on a node nobody is deciding on.
    from .systemone import SYSTEMONE  # noqa: PLC0415
    SYSTEMONE.preload_async()
    STORE.register("image-to-mesh", pipeline.image_to_mesh)
    STORE.register("retopologize", pipeline.retopologize)
    from . import video
    STORE.register("text-to-video", video.text_to_video)
    from . import image
    STORE.register("text-to-image", image.text_to_image)
    from . import modelstore
    STORE.register("store-install", modelstore.install_job)
    from . import portrait
    STORE.register("portrait-animate", portrait.portrait_animate)
    from . import talkinghead
    STORE.register("talking-head", talkinghead.talking_head)
    STORE.start_worker()
    from .llamacpp import ensure_sharp_template_async
    ensure_sharp_template_async()
    from .llm import AUTOSTART
    if AUTOSTART and LLM.installed:
        def _boot_llm():
            try:
                LLM.start("c1")
            except Exception:  # noqa: BLE001
                log.exception("LLM autostart failed; start manually via "
                              "POST /v1/llm/start")
        import threading
        threading.Thread(target=_boot_llm, daemon=True,
                         name="llm-autostart").start()
    host = config.effective_host()
    if host != config.HOST:
        log.warning("No node or swarm token is set, so %s would be an "
                    "unauthenticated remote job API — binding %s instead. "
                    "Set SILICON_NODE_TOKEN (or a swarm token) to serve "
                    "the network.", config.HOST, host)
    log.info("%s v%s ready on %s:%d (auth: %s)", config.SERVER_NAME,
             config.SERVER_VERSION, host, config.PORT,
             "bearer token" if config.VALID_TOKENS else "loopback only")
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host=config.effective_host(),
                port=config.PORT)
