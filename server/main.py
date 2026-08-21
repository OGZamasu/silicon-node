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

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import config, pipeline
from .jobs import STORE
from .llm import DOWNLOADS, LLM, MODEL_ID as LLM_MODEL_ID, PORT as LLM_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("silicon-node")

app = FastAPI(title=config.SERVER_NAME, version=config.SERVER_VERSION)

STARTED_AT = time.time()


# ---------------------------------------------------------------------------
# Auth: optional bearer token. Unset -> open (LAN-only milestone). Set ->
# required on everything under /v1/. /health stays open as a probe.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    if request.url.path.startswith("/v1/") and config.VALID_TOKENS:
        header = request.headers.get("authorization", "")
        supplied = header.removeprefix("Bearer ").strip() if header else ""
        from .clients import CLIENTS  # noqa: PLC0415
        if supplied and supplied not in config.VALID_TOKENS \
                and not CLIENTS.accepts(supplied):
            # A wrong token is always rejected — catches misconfiguration
            # (and revoked members) early even while enforcement is off.
            return PlainTextResponse(
                "That bearer token does not match this node's node token, "
                "the swarm token, or any paired client.", status_code=401)
        if not supplied and config.REQUIRE_AUTH:
            return PlainTextResponse(
                "This Silicon node requires a bearer token. Send "
                "'Authorization: Bearer <token>'.", status_code=401)
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
    if supplied == config.SWARM_TOKEN:
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
    try:
        name, token = CLIENTS.mint(name)
    except KeyError:
        raise HTTPException(
            status_code=409,
            detail=f"A client named {name!r} already exists.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"name": name, "token": token}


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
    """Who is acting: "admin" (swarm token), "node" (node token or
    unauthenticated local use while enforcement is off), or a paired
    client's name."""
    tok = _bearer_of(request)
    if not tok:
        return "node"   # the local dashboard, pre-enforcement
    if config.SWARM_TOKEN and tok == config.SWARM_TOKEN:
        return "admin"
    if config.TOKEN and tok == config.TOKEN:
        return "node"
    from .clients import CLIENTS  # noqa: PLC0415
    return CLIENTS.name_of(tok) or "node"


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
        if config.SWARM_TOKEN and tok == config.SWARM_TOKEN:
            who = "swarm (shared token)"
        elif config.TOKEN and tok == config.TOKEN:
            who = "this node's token"
        else:
            from .clients import CLIENTS  # noqa: PLC0415
            who = CLIENTS.name_of(tok)
            if who and cap:
                CLIENTS.count_job(who, cap)
    ip = request.client.host if request.client else None
    from .llm import _windows_host_ip  # noqa: PLC0415
    return {"client": who, "ip": ip,
            "proxied": ip == _windows_host_ip(),
            "user_agent": request.headers.get("user-agent", "")[:120]}


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
    with image_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    if image_path.stat().st_size == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty. Please send a PNG or JPEG.")
    job.params["image_path"] = str(image_path)
    STORE.enqueue(job)
    return {"job_id": job.job_id}


@app.get("/v1/jobs")
def jobs_list():
    """Recent jobs, newest first (for the dashboard)."""
    jobs = sorted(STORE._jobs.values(), key=lambda j: j.created_at,
                  reverse=True)[:20]
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


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job named {job_id} on this node. It may predate a "
                   "service restart.")
    return job.to_api()


@app.get("/v1/jobs/{job_id}/detail")
def job_detail(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    params = {k: (v if not isinstance(v, str) or len(v) < 200 else "…")
              for k, v in job.params.items() if not k.endswith("_b64")}
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
    if _actor(request) not in ("admin", "node"):
        raise HTTPException(
            status_code=403,
            detail="Clearing the whole queue is for the swarm admin or "
                   "the node owner; members can cancel their own jobs.")
    scope = body.get("scope", "pending")
    if scope not in ("pending", "all"):
        raise HTTPException(status_code=400,
                            detail='scope must be "pending" or "all".')
    return {"cancelled": STORE.cancel_queue(scope)}


@app.post("/v1/jobs/{job_id}/{action}")
def job_action(job_id: str, action: str, request: Request,
               direction: str = "up"):
    _require_job_owner(request, job_id)
    ok = False
    if action == "cancel":
        ok = STORE.cancel(job_id)
    elif action == "hold":
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
def get_file(name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    path = config.FILES_DIR / name
    if not path.is_file():
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
        "iteration — choose with the model field.",
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
            "peak_vram_gb": _measured("text-to-video", "peak_vram_gb"),
            "typical_seconds": _measured("text-to-video",
                                         "typical_seconds"),
            "ready": _video_ready(),
            "detail": "Wan 2.2 TI2V-5B via diffusers, CPU-offloaded for "
                      "the shared 24 GB card. Submit POST /v1/jobs "
                      '{"capability":"text-to-video","prompt":...} with '
                      "optional negative_prompt/frames/width/height/steps/"
                      "seed; result_urls carries the .mp4."
                      if _video_ready() else
                      "Wan 2.2 weights or diffusers still downloading — "
                      "flips ready automatically.",
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
    from .capsettings import CAPS, DEFAULTS  # noqa: PLC0415
    for c in caps:
        cid = c["id"]
        if cid in _CAP_DESCRIPTIONS:
            c["description"] = _CAP_DESCRIPTIONS[cid]
            c["enabled"] = CAPS.enabled(cid)
            settings = CAPS.settings(cid)
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


def _video_ready() -> bool:
    from . import video
    return video.ENGINE.ready()


def _portrait_ready() -> bool:
    from . import portrait
    return portrait.ready()


def _talkinghead_ready() -> bool:
    from . import talkinghead
    return talkinghead.ready()


def _measured(cap: str, field: str):
    """Aggregate receipts of finished jobs; None until first measurement."""
    samples = []
    for job in list(STORE._jobs.values()):
        if job.capability != cap or job.state != "done":
            continue
        if field == "typical_seconds" and job.started_at and job.finished_at:
            samples.append(job.finished_at - job.started_at)
        elif field == "peak_vram_gb":
            peaks = [job.receipts.get("trellis_peak_vram_gb"),
                     job.receipts.get("lato_peak_vram_gb"),
                     job.receipts.get("video_peak_vram_gb")]
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
    from .capsettings import CAPS  # noqa: PLC0415
    if not CAPS.enabled(cap):
        raise HTTPException(
            status_code=503,
            detail=f"The {cap} ability is currently disabled on this "
                   f"node. Re-enable it from the Swarm page or via "
                   f"POST /v1/capabilities/{cap}.")


@app.post("/v1/capabilities/{cap_id}")
async def capability_update(cap_id: str, request: Request):
    """Enable/disable an ability or change its exposed settings
    (handoff 129). Partial updates; unknown setting keys are reported
    back as ignored rather than written."""
    from .capsettings import CAPS, DEFAULTS  # noqa: PLC0415
    if cap_id not in DEFAULTS:
        raise HTTPException(
            status_code=404,
            detail=f"No configurable ability named {cap_id!r} on this "
                   "node. The chat model is managed via /v1/llm.")
    body = await request.json()
    ignored = CAPS.update(cap_id, body.get("enabled"),
                          body.get("settings"))
    entry = next(c for c in capability_list() if c["id"] == cap_id)
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
def model_reveal(model_id: str):
    """Open the model's folder in Windows Explorer."""
    adm = _model_admin().get(model_id)
    if adm is None or adm["loc"] is None:
        raise HTTPException(status_code=404,
                            detail=f"No model named {model_id!r}.")
    target = adm["loc"] if adm["kind"] != "file" else adm["loc"].parent
    win = _win_visible(Path(target))
    subprocess.Popen(["/mnt/c/Windows/explorer.exe", win])
    return {"ok": True, "path": win}


@app.delete("/v1/models/{model_id}")
def model_delete(model_id: str):
    """Uninstall a model's weights from disk. Refused while the model
    is loaded, and for the one checkpoint with no reinstall path."""
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
        mid = LLM_MODEL_ID  # single-model engine today
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
        models.append({
            "id": entry["file"], "name": entry["file"],
            "capability": "llm-gguf", "engine": "llama.cpp",
            "repo": GGUF_DIR_WIN,
            "installed": True,
            "ready": llama.get("engine_installed", False),
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
    if cap == "image-to-mesh" and "image_path" not in params:
        raise HTTPException(
            status_code=400,
            detail="image-to-mesh needs an image; use POST /v1/image-to-mesh "
                   "(multipart) which accepts the file directly.")
    if cap == "retopologize" and "mesh_path" not in params:
        raise HTTPException(
            status_code=400,
            detail="retopologize needs a mesh; use POST /v1/retopologize "
                   "(multipart) which accepts the file directly.")
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
    with mesh_path.open("wb") as f:
        shutil.copyfileobj(mesh.file, f)
    job.params["mesh_path"] = str(mesh_path)
    STORE.enqueue(job)
    return {"job_id": job.job_id}


# ---------------------------------------------------------------------------
# LLM management (ninfer-3090)
# ---------------------------------------------------------------------------

def _require_operator(request: Request, what: str) -> None:
    """Engine control is for the swarm admin or the node owner —
    members chat with the model, they don't restart it (handoff 132)."""
    if _actor(request) not in ("admin", "node"):
        raise HTTPException(
            status_code=403,
            detail=f"{what} is for the swarm admin or the node owner.")


@app.get("/v1/llm")
def llm_status():
    return LLM.status()


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """The OpenAI chat surface THROUGH the node, so member usage is
    attributable (handoff 132). The engines bind the Windows loopback,
    which WSL sockets cannot reach — interop curl bridges it, streaming
    included. Clients that should be counted point at :8790/v1 instead
    of the engine port; the engine ports keep working unchanged."""
    import asyncio  # noqa: PLC0415
    import os  # noqa: PLC0415
    import uuid  # noqa: PLC0415
    from .llamacpp import LLAMACPP, PORT as GGUF_PORT  # noqa: PLC0415
    if LLM.running:
        port = LLM_PORT
    elif LLAMACPP.running:
        port = GGUF_PORT
    else:
        raise HTTPException(
            status_code=503,
            detail="No chat engine is running — start one via "
                   "/v1/llm/start or /v1/gguf/start.")
    actor = _actor(request)
    if actor not in ("admin", "node"):
        from .clients import CLIENTS  # noqa: PLC0415
        CLIENTS.count_llm(actor)
    body = await request.body()
    stream = b'"stream": true' in body or b'"stream":true' in body
    # Windows curl reads a Windows path (interop passes argv verbatim).
    win_tmp = os.environ.get("SILICON_NODE_WIN_TMP", r"C:\Windows\Temp")
    wsl_tmp = Path("/mnt/" + win_tmp[0].lower()
                   + win_tmp[2:].replace("\\", "/")) / "silicon-chat"
    wsl_tmp.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.json"
    (wsl_tmp / name).write_bytes(body)
    proc = await asyncio.create_subprocess_exec(
        "/mnt/c/Windows/System32/curl.exe", "-s", "-N", "-X", "POST",
        f"http://127.0.0.1:{port}/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "--data-binary", "@" + win_tmp + "\\silicon-chat\\" + name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)

    async def pump():
        try:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            (wsl_tmp / name).unlink(missing_ok=True)
            if proc.returncode is None:
                proc.kill()

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
    _require_operator(request, "Starting or switching the chat model")
    profile, model_file, context_length = "c1", None, None
    try:
        body = await request.json()
        profile = body.get("profile", "c1")
        model_file = body.get("model_file")
        # The Mac's Swarm page sends the model ID, not a filename
        # (handoff 127): "qwen3.8-27b" → qwen3_8_27b.ninfer.
        if not model_file and body.get("model"):
            stem = str(body["model"]).replace(".", "_").replace("-", "_")
            model_file = f"{stem}.ninfer"
        if body.get("context_length") is not None:
            context_length = int(body["context_length"])
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="context_length must be a number.") from None
    except Exception:  # noqa: BLE001
        pass
    if STORE.queue_depth() > 0:
        raise HTTPException(
            status_code=409,
            detail="A GPU job is queued or running; the LLM will not start "
                   "until the job queue drains. Try again shortly.")
    try:
        if LLM.running:
            LLM.stop()  # switching model/profile/context
        pipeline.ENGINE.unload()
        LLM.start(profile, model_file=model_file,
                  context_length=context_length)
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
def harness_start():
    from .harness import HARNESS
    try:
        HARNESS.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return HARNESS.status()


@app.post("/v1/harness/stop")
def harness_stop():
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
    if model == "ltx2-distilled":
        if not _video.ENGINE.ltx_ready():
            raise HTTPException(
                status_code=503,
                detail="LTX-2 distilled is downloading to this node right "
                       "now — try again in a few minutes, or use Wan 2.2 "
                       "5B meanwhile.")
    elif model not in ("wan22-ti2v-5b", "text-to-video", "wan2.2-ti2v-5b"):
        raise HTTPException(
            status_code=400,
            detail=f"This node serves wan22-ti2v-5b and ltx2-distilled; "
                   f"{model} isn't one of them.")
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
    frames = max(17, min(121, int(round(seconds * 24)) + 1))
    res = str(body.get("resolution", "720p")).lower()
    width, height = (1280, 704)
    if "x" in res:
        try:
            width, height = (int(v) for v in res.split("x", 1))
        except ValueError:
            pass
    params = {"prompt": prompt, "frames": frames,
              "width": width, "height": height,
              "seed": body.get("seed"),
              "engine": "ltx" if model == "ltx2-distilled" else "wan"}
    _require_enabled("text-to-video")
    job = STORE.submit("text-to-video", params, defer=True)
    job.submitted_by = _submitter(request, "text-to-video")
    job.dir.mkdir(parents=True, exist_ok=True)
    if body.get("image_b64"):
        try:
            img = base64.b64decode(body["image_b64"])
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="image_b64 is not valid base64."
                                ) from None
        suffix = Path(body.get("image_name", "start.png")).suffix or ".png"
        (job.dir / f"start{suffix}").write_bytes(img)
        job.params["image_path"] = str(job.dir / f"start{suffix}")
    STORE.enqueue(job)
    return {"job_id": job.job_id}


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
    (job.dir / f"portrait{img_suffix}").write_bytes(image)
    (job.dir / f"driving{drv_suffix}").write_bytes(driving)
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
    (job.dir / f"portrait{img_suffix}").write_bytes(image)
    (job.dir / f"speech{aud_suffix}").write_bytes(audio)
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
    from .llamacpp import GGUF_DL, LLAMACPP
    body = await request.json()
    if not body.get("repo") or not body.get("file"):
        raise HTTPException(status_code=400,
                            detail="repo and file are required.")
    LLAMACPP.install_engine_async()  # fetch the engine alongside the model
    GGUF_DL.start(body["repo"], body["file"])
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
    try:
        if LLM.running:
            LLM.stop()  # one language engine at a time on this card
        pipeline.ENGINE.unload()
        ctx = body.get("context")
        LLAMACPP.start(body.get("file", ""),
                       int(ctx) if ctx is not None else None)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return LLAMACPP.status()


@app.post("/v1/gguf/stop")
def gguf_stop(request: Request):
    _require_operator(request, "Stopping the GGUF engine")
    from .llamacpp import LLAMACPP
    LLAMACPP.stop()
    return LLAMACPP.status()


@app.post("/v1/llm/models/download")
async def llm_model_download(request: Request):
    body = await request.json()
    try:
        name = DOWNLOADS.start(body.get("model_id", ""))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"filename": name}


@app.get("/v1/llm/models/downloads")
def llm_model_downloads():
    return DOWNLOADS.progress()


@app.post("/v1/swarm/image")
async def swarm_image(request: Request):
    """Delegate image generation to a peer (the Mac's image-flux)."""
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="A prompt is required.")
    if not config.PEERS or not config.SWARM_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="No swarm registry on this node (swarm.json).")
    peer = config.PEERS[0]
    import httpx  # noqa: PLC0415
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{peer['base_url']}/v1/jobs",
                json={"capability": "image-flux", "prompt": prompt},
                headers={"Authorization":
                         f"Bearer {config.SWARM_TOKEN}"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach {peer.get('name')}: {exc}") from None
    if r.status_code // 100 != 2:
        raise HTTPException(
            status_code=502,
            detail=f"{peer.get('name')} answered {r.status_code}: the Mac "
                   "hasn't exposed image jobs to the swarm yet (asked for "
                   "on the hub).")
    return {"peer": peer.get("name"), **r.json()}


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
    return {
        "name": config.SERVER_NAME,
        "platform": config.PLATFORM,
        "profile": _gpu_profile(),
        "capabilities": capability_list(),
        "metrics": met,
        "queue": _queue_view(),
        "llm": {
            "running": LLM.running,
            "model": getattr(LLM, "model_id", LLM_MODEL_ID)
            if LLM.installed else None,
            "context_length": getattr(LLM, "_ctx_effective", None)
            if LLM.running else None,
            "max_concurrency":
                _LLM_PROFILES[LLM._profile]["max_concurrency"]
            if LLM.running else None,
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
        if j is not None:
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
    actor = _actor(request)
    if actor in ("admin", "node"):
        return
    job = STORE.get(job_id)
    owner = (job.submitted_by or {}).get("client") if job else None
    if owner != actor:
        raise HTTPException(
            status_code=403,
            detail="Members can manage only their own jobs — this one "
                   f"was submitted by {owner or 'someone else'}.")


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
        prof["disk_free_gb"] = round(
            _sh.disk_usage("/mnt/f").free / 1e9)
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


def create_app() -> FastAPI:
    config.ensure_dirs()
    STORE.register("image-to-mesh", pipeline.image_to_mesh)
    STORE.register("retopologize", pipeline.retopologize)
    from . import video
    STORE.register("text-to-video", video.text_to_video)
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
    log.info("%s v%s ready on %s:%d (auth: %s)", config.SERVER_NAME,
             config.SERVER_VERSION, config.HOST, config.PORT,
             "bearer token" if config.TOKEN else "open (LAN-only)")
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host=config.HOST, port=config.PORT)
