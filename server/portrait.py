"""portrait-animate capability — LivePortrait on CUDA.

The Mac's persona/VTuber stack delegates photoreal takes to any swarm
node advertising capability kind "talking-head"/"portrait-animate"
(their VideoRuntime.animatePortrait): a portrait image plus a recorded
driving performance in, an animated clip back. They run LivePortrait on
MPS at ~740 ms/frame; this node's job is to be the fast lane.

Subprocess model (their own inference.py CLI in a dedicated conda env) —
correct by construction and OOM-isolated; residency is a later
optimization if take latency matters.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from . import config
from .jobs import Job

log = logging.getLogger("silicon-node.portrait")

LP_ROOT = Path("/opt/silicon/LivePortrait")
LP_PY = Path("/opt/miniforge3/envs/liveportrait/bin/python")


def ready() -> bool:
    return (LP_PY.exists()
            and (LP_ROOT / "pretrained_weights" / "liveportrait").exists())


def portrait_animate(job: Job, progress) -> list[str]:
    image = Path(job.params["image_path"])
    driving = Path(job.params["driving_path"])
    out_dir = job.dir / "lp_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress(0.10, "animating the portrait")
    t0 = time.time()
    proc = subprocess.run(
        [str(LP_PY), "inference.py",
         "-s", str(image), "-d", str(driving), "-o", str(out_dir)],
        cwd=str(LP_ROOT), capture_output=True, text=True, timeout=30 * 60)
    job.receipts["liveportrait_s"] = round(time.time() - t0, 1)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        log.error("LivePortrait failed:\n%s", tail)
        if "CUDA out of memory" in tail:
            raise RuntimeError("CUDA out of memory while animating.")
        raise RuntimeError(
            "The portrait animation failed. Last output: " + tail[-300:])

    # inference.py writes <source>--<driving>.mp4 (and a _concat variant);
    # the plain one is the product.
    clips = sorted(out_dir.glob("*.mp4"),
                   key=lambda p: ("concat" in p.name, p.name))
    if not clips:
        raise RuntimeError(
            "LivePortrait produced no clip — the portrait may not contain "
            "a detectable face.")
    progress(0.95, "export")
    name = f"{job.job_id}-animated.mp4"
    shutil.copy2(clips[0], config.FILES_DIR / name)
    job.receipts["artifacts"] = [name]
    return [name]
