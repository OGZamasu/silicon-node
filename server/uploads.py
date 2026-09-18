"""Getting bytes onto disk without stalling the service.

The submit handlers are `async def`, so anything they do synchronously runs
on the event loop — and a 300 MB driving clip written with a blocking
`copyfileobj` freezes every other request for as long as the disk takes,
including `/health` and the progress polls of jobs already on the GPU.
The dashboard reads that as a hung node.

So: streamed in chunks, each write handed to a thread, and a size ceiling
enforced while writing rather than after, because "reject it once it is
already on the disk" is not a limit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException, UploadFile

from . import config

CHUNK_BYTES = 4 * 1024 * 1024


def _too_large(limit: int) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"That upload is larger than this node accepts "
               f"({limit // (1024 * 1024)} MB). Raise "
               f"SILICON_NODE_MAX_UPLOAD_MB on the node, or send less.")


async def save_upload(upload: UploadFile, destination: Path,
                      limit: int | None = None) -> int:
    """Stream a multipart upload to `destination`; return bytes written."""
    limit = config.MAX_UPLOAD_BYTES if limit is None else limit
    written = 0
    handle = await asyncio.to_thread(destination.open, "wb")
    try:
        while True:
            chunk = await upload.read(CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if limit and written > limit:
                raise _too_large(limit)
            await asyncio.to_thread(handle.write, chunk)
    except BaseException:
        await asyncio.to_thread(handle.close)
        # A partial file is worse than none: the handler would enqueue a
        # job whose input is a truncated PNG.
        destination.unlink(missing_ok=True)
        raise
    await asyncio.to_thread(handle.close)
    return written


async def write_bytes(destination: Path, data: bytes,
                      limit: int | None = None) -> int:
    """Write an already-decoded body (base64 JSON payloads) off the loop."""
    limit = config.MAX_UPLOAD_BYTES if limit is None else limit
    if limit and len(data) > limit:
        raise _too_large(limit)
    await asyncio.to_thread(destination.write_bytes, data)
    return len(data)


def check_declared_size(content_length: str | None,
                        limit: int | None = None) -> None:
    """Refuse an oversized body before reading a byte of it.

    Content-Length is a claim, not a fact — `save_upload` still counts what
    arrives. This just means an honest 40 GB post costs one round trip
    instead of 40 GB of disk.
    """
    limit = config.MAX_UPLOAD_BYTES if limit is None else limit
    if not content_length or not limit:
        return
    try:
        declared = int(content_length)
    except ValueError:
        return
    if declared > limit:
        raise _too_large(limit)
