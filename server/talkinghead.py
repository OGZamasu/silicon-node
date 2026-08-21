"""talking-head capability — SadTalker on CUDA (silicon-node #5).

Photo + audio in, lip-synced clip out — the audio-driven sibling of
portrait-animate (LivePortrait handles a recorded video performance;
this handles speech). The Mac's peer filter already matches capability
kind "talking-head", so advertising it lights the path up as soon as
their audio client lands.

Same subprocess model as portrait.py: SadTalker's own inference.py in a
dedicated conda env — correct by construction and OOM-isolated.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import config
from .jobs import Job

log = logging.getLogger("silicon-node.talkinghead")

ST_ROOT = Path("/opt/silicon/SadTalker")
ST_PY = Path("/opt/miniforge3/envs/sadtalker/bin/python")


def ready() -> bool:
    return (ST_PY.exists()
            and (ST_ROOT / "checkpoints"
                 / "SadTalker_V0.0.2_256.safetensors").exists())


def talking_head(job: Job, progress) -> list[str]:
    image = Path(job.params["image_path"])
    audio = Path(job.params["audio_path"])
    out_dir = job.dir / "st_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress(0.10, "animating the portrait from audio")
    t0 = time.time()
    # --still + full preprocess keeps the whole photo and only moves the
    # face — the persona-card look. Enhancer off: ~2x faster and the
    # quality question belongs to a later pass.
    # The service env sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments
    # for its own torch 2.6; SadTalker's torch 2.0.1 rejects that option
    # at CUDA init, so the subprocess gets a scrubbed environment.
    env = {k: v for k, v in os.environ.items()
           if k != "PYTORCH_CUDA_ALLOC_CONF"}
    proc = subprocess.run(
        [str(ST_PY), "inference.py",
         "--driven_audio", str(audio),
         "--source_image", str(image),
         "--result_dir", str(out_dir),
         "--still", "--preprocess", "full"],
        cwd=str(ST_ROOT), env=env,
        capture_output=True, text=True, timeout=30 * 60)
    job.receipts["sadtalker_s"] = round(time.time() - t0, 1)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        log.error("SadTalker failed:\n%s", tail)
        if "CUDA out of memory" in tail:
            raise RuntimeError("CUDA out of memory while animating.")
        raise RuntimeError(
            "The talking-head render failed. Last output: " + tail[-300:])

    # inference.py writes <timestamp>.mp4 (sometimes inside a
    # <timestamp>/ folder); take the newest clip it produced.
    clips = sorted(out_dir.rglob("*.mp4"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not clips:
        raise RuntimeError(
            "SadTalker produced no clip — the portrait may not contain "
            "a detectable face.")
    progress(0.95, "export")
    name = f"{job.job_id}-talking.mp4"
    shutil.copy2(clips[0], config.FILES_DIR / name)
    job.receipts["artifacts"] = [name]
    return [name]
