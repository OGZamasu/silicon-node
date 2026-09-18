"""silicon-node-mcp — MCP stdio server for the Silicon Node.

Thin adapter over the node's HTTP API (one implementation of the work, two
doors into it — same split as the Mac's silicon-mcp). Point any MCP client
at this script and it can drive the node from any machine that reaches the
service URL.

Config via env:
    SILICON_NODE_URL    default http://127.0.0.1:8790
    SILICON_NODE_TOKEN  optional bearer token (must match the service's)

Run:  python mcp_server.py   (requires: pip install mcp httpx)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

try:  # MCP SDK 2.x
    from mcp.server import MCPServer as _Server
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

BASE = os.environ.get("SILICON_NODE_URL", "http://127.0.0.1:8790").rstrip("/")
TOKEN = os.environ.get("SILICON_NODE_TOKEN", "").strip()

mcp = _Server("silicon-node")


def _client() -> httpx.Client:
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    return httpx.Client(base_url=BASE, headers=headers, timeout=30.0)


@mcp.tool()
def node_status() -> dict[str, Any]:
    """Health, GPU profile, live metrics and queue depth of the Silicon node."""
    with _client() as c:
        health = c.get("/health").json()
        node = c.get("/v1/node").json()
    return {"health": health, "node": node}


@mcp.tool()
def list_capabilities() -> list[dict[str, Any]]:
    """List what this node can run, with measured VRAM/timing numbers."""
    with _client() as c:
        return c.get("/v1/capabilities").json()


@mcp.tool()
def generate_3d(image_path: str, vert_num: int = 2000,
                seed: Optional[int] = None) -> dict[str, Any]:
    """Submit an image for 3D generation (TRELLIS.2 densify + LATO.2
    retopology). Returns the job_id to poll with get_status / wait the job
    out with wait_for_job. vert_num controls the low-poly output (200-5000).
    """
    p = Path(image_path).expanduser()
    if not p.is_file():
        return {"error": f"No image at {image_path}"}
    data = {"vert_num": str(vert_num)}
    if seed is not None:
        data["seed"] = str(seed)
    with _client() as c:
        r = c.post("/v1/image-to-mesh", data=data,
                   files={"image": (p.name, p.read_bytes(),
                                    "application/octet-stream")})
    if r.status_code // 100 != 2:
        return {"error": r.text[:300]}
    return r.json()


@mcp.tool()
def retopologize(mesh_path: str, vert_num: int = 2000,
                 seed: Optional[int] = None) -> dict[str, Any]:
    """Submit an existing mesh (GLB/OBJ/PLY) for LATO.2 retopology into a
    clean low-poly mesh, without the image densify stage."""
    p = Path(mesh_path).expanduser()
    if not p.is_file():
        return {"error": f"No mesh at {mesh_path}"}
    data = {"vert_num": str(vert_num)}
    if seed is not None:
        data["seed"] = str(seed)
    with _client() as c:
        r = c.post("/v1/retopologize", data=data,
                   files={"mesh": (p.name, p.read_bytes(),
                                   "application/octet-stream")})
    if r.status_code // 100 != 2:
        return {"error": r.text[:300]}
    return r.json()


@mcp.tool()
def llm_status() -> dict[str, Any]:
    """State of the node's Qwen3.8-27B LLM (ninfer-3090): installed,
    running, and the OpenAI/Anthropic-compatible endpoints to call."""
    with _client() as c:
        return c.get("/v1/llm").json()


@mcp.tool()
def llm_start(profile: str = "c1") -> dict[str, Any]:
    """Start the LLM server (profile c1 = single user low latency, c8 =
    concurrent throughput). 3D jobs preempt it; it restores afterwards."""
    with _client() as c:
        r = c.post("/v1/llm/start", json={"profile": profile},
                   timeout=240.0)
    if r.status_code // 100 != 2:
        return {"error": r.text[:300]}
    return r.json()


@mcp.tool()
def llm_stop() -> dict[str, Any]:
    """Stop the LLM server, freeing ~20 GB VRAM for 3D work."""
    with _client() as c:
        return c.post("/v1/llm/stop").json()


@mcp.tool()
def get_status(job_id: str) -> dict[str, Any]:
    """Current status/progress/results of a job."""
    with _client() as c:
        return c.get(f"/v1/jobs/{job_id}").json()


@mcp.tool()
def wait_for_job(job_id: str, timeout_s: int = 1800) -> dict[str, Any]:
    """Poll a job to completion (3 s cadence, like the Mac client)."""
    deadline = time.time() + timeout_s
    with _client() as c:
        while time.time() < deadline:
            status = c.get(f"/v1/jobs/{job_id}").json()
            if status.get("status") in ("done", "failed"):
                return status
            time.sleep(3)
    return {"status": "timeout", "job_id": job_id,
            "error": f"Job still running after {timeout_s}s."}


@mcp.tool()
def download_result(job_id: str, dest_dir: str) -> dict[str, Any]:
    """Download a finished job's artifacts (.glb/.obj) into dest_dir."""
    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    with _client() as c:
        status = c.get(f"/v1/jobs/{job_id}").json()
        if status.get("status") != "done":
            return {"error": f"Job is {status.get('status')}, not done."}
        saved = []
        for url in status.get("result_urls", []):
            name = url.rsplit("/", 1)[-1]
            r = c.get(url)
            if r.status_code // 100 == 2:
                (dest / name).write_bytes(r.content)
                saved.append(str(dest / name))
    return {"saved": saved}


if __name__ == "__main__":
    mcp.run()
