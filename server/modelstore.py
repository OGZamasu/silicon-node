"""Managed model store (hub 137): install/delete the node's heavyweight
model checkpoints per model, recommended-only by default.

Installs run through the jobs queue so progress/%/eta ride the existing
machinery and show up in every queue view. Installed-ness is read from
the Hugging Face cache itself (the disk is the truth); this file's JSON
only records who installed/deleted what and when.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from .jobs import Job

log = logging.getLogger("silicon-node.store")

HF_HUB = Path(os.environ.get(
    "HF_HUB_CACHE",
    str(Path(os.environ.get("HF_HOME",
                            Path.home() / ".cache/huggingface")) / "hub")))
STATE_FILE = Path(os.environ.get(
    "SILICON_NODE_STORE_STATE", "/opt/silicon/store-state.json"))
# Refuse installs that would leave less than this free (the Mac keeps a
# 10 GB floor; the guest matches it, and the Windows drive hosting the
# growable VHDX keeps a bigger one).
GUEST_RESERVE = 10 * 1024**3
HOST_RESERVE = 25 * 1024**3


def _catalog() -> dict[str, dict]:
    from . import image, video  # noqa: PLC0415
    return {
        "qwen-image": {
            "name": "Qwen-Image 20B",
            "capability": "text-to-image",
            "repos": [image.QWEN_REPO],
            "est_bytes": 58_000_000_000,
            "license": "Apache-2.0",
            "recommended": True,
            "installable": True,
            "note": "Readable text in images — the default image lane "
                    "(NF4-quantized for the 24 GB card).",
        },
        "sana": {
            "name": "Sana 1600M 1024px",
            "capability": "text-to-image",
            "repos": [image.SANA_REPO],
            "est_bytes": 8_000_000_000,
            "license": "Apache-2.0 (Gemma-2 encoder terms apply)",
            "recommended": True,
            "installable": True,
            "note": "The fast-iteration lane: seconds per 1024px image.",
        },
        "sdxl": {
            "name": "Stable Diffusion XL base 1.0",
            "capability": "text-to-image",
            "repos": ["stabilityai/stable-diffusion-xl-base-1.0"],
            "est_bytes": 7_000_000_000,
            "license": "CreativeML OpenRAIL++-M",
            "recommended": False,
            "installable": True,
            "note": "Earns its disk when someone wants its LoRA/"
                    "ControlNet world; not installed by default.",
        },
        "flux2-dev": {
            "name": "FLUX.2 [dev]",
            "capability": "text-to-image",
            "repos": ["black-forest-labs/FLUX.2-dev"],
            "est_bytes": 110_000_000_000,
            "license": "BFL non-commercial",
            "recommended": False,
            "installable": False,
            "note": "Deliberate download only: non-commercial license, "
                    "and at ~110 GB bf16 it has no runtime lane on this "
                    "24 GB card yet — ask for one before installing.",
        },
        "wan22-ti2v-5b": {
            "name": "Wan 2.2 TI2V-5B",
            "capability": "text-to-video",
            "repos": [video.MODEL_REPO],
            "est_bytes": 34_000_000_000,
            "license": "Apache-2.0",
            "recommended": True,
            "installable": True,
            "note": "The quality video engine.",
        },
        "ltx2-distilled": {
            "name": "LTX-2.3 distilled",
            "capability": "text-to-video",
            "repos": [video.LTX_REPO],
            "est_bytes": 95_000_000_000,
            "license": "LTX-2 Community License",
            "recommended": True,
            "installable": True,
            "note": "Same clip ~2.6x faster than Wan for iteration "
                    "(NF4 at load).",
        },
        "trellis2": {
            "name": "TRELLIS.2-4B + DINOv3",
            "capability": "image-to-mesh",
            "repos": ["microsoft/TRELLIS.2-4B",
                      "facebook/dinov3-vitl16-pretrain-lvd1689m"],
            "est_bytes": 18_000_000_000,
            "license": "MIT (DINOv3 under the Meta DINOv3 license)",
            "recommended": True,
            "installable": True,
            "note": "The densify stage of image-to-mesh.",
        },
    }


def _repo_dir(repo: str) -> Path:
    return HF_HUB / ("models--" + repo.replace("/", "--"))


def _installed(entry: dict) -> bool:
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        for repo in entry["repos"]:
            snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


_DU_CACHE: dict[str, tuple[float, int]] = {}


def _du(entry: dict) -> int:
    """Bytes on disk across the entry's repos (5-minute cache)."""
    key = "+".join(entry["repos"])
    hit = _DU_CACHE.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    total = 0
    for repo in entry["repos"]:
        d = _repo_dir(repo)
        if d.is_dir():
            for p in d.rglob("*"):
                try:
                    if p.is_file() and not p.is_symlink():
                        total += p.stat().st_size
                except OSError:
                    continue
    _DU_CACHE[key] = (time.time(), total)
    return total


def _du_bust(entry: dict) -> None:
    _DU_CACHE.pop("+".join(entry["repos"]), None)


def _cache_volume() -> str:
    """The filesystem the model cache lives on (HF_HUB may not exist
    yet on a fresh box — walk up to something that does)."""
    p = HF_HUB
    while not p.exists() and p != p.parent:
        p = p.parent
    return str(p)


def listing() -> dict:
    guest = shutil.disk_usage(_cache_volume())
    out = {"disk_free_bytes": guest.free,
           "reserve_bytes": GUEST_RESERVE,
           "models": []}
    from .hostos import IS_WSL  # noqa: PLC0415
    if IS_WSL:
        try:
            # The VHDX grows on the Windows drive; show that budget too.
            out["disk_free_windows_bytes"] = (
                shutil.disk_usage("/mnt/f").free)
        except OSError:
            pass
    for mid, entry in _catalog().items():
        inst = _installed(entry)
        out["models"].append({
            "id": mid,
            "name": entry["name"],
            "capability": entry["capability"],
            "installed": inst,
            "recommended": entry["recommended"],
            "installable": entry["installable"],
            "size_bytes": _du(entry) if inst else entry["est_bytes"],
            "size_is_estimate": not inst,
            "license": entry["license"],
            "note": entry["note"],
        })
    return out


def disk_refusal(entry: dict) -> str | None:
    """Readable refusal when the install would blow the disk budget."""
    need = entry["est_bytes"] - _du(entry)  # resumed installs need less
    guest_free = shutil.disk_usage(_cache_volume()).free
    if guest_free - need < GUEST_RESERVE:
        return (f"Not enough space in the node's disk: {entry['name']} "
                f"needs ~{need / 1e9:.0f} GB and only "
                f"{max(0, guest_free - GUEST_RESERVE) / 1e9:.0f} GB is "
                "available above the 10 GB reserve.")
    from .hostos import IS_WSL  # noqa: PLC0415
    if not IS_WSL:
        return None
    try:
        host_free = shutil.disk_usage("/mnt/f").free
    except OSError:
        return None
    if host_free - need < HOST_RESERVE:
        return (f"Not enough space on the Windows drive hosting this "
                f"node's disk image: {entry['name']} needs ~"
                f"{need / 1e9:.0f} GB and only "
                f"{max(0, host_free - HOST_RESERVE) / 1e9:.0f} GB is "
                "available above the 25 GB reserve.")
    return None


def _record(kind: str, model_id: str, actor) -> None:
    data = {}
    try:
        data = json.loads(STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        pass
    data.setdefault(kind, {})[model_id] = {
        "at": time.time(), "by": actor}
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(STATE_FILE)


def install_job(job: Job, progress) -> list[str]:
    """Queue handler for capability "store-install". Downloads run in a
    side thread while this loop reports bytes-on-disk progress, so %/eta
    ride the normal job machinery. Cancelling the job stops the progress
    (and the queue slot); an already-running fetch finishes filling the
    cache harmlessly and resumes instantly on the next install."""
    entry = _catalog()[job.params["model_id"]]
    progress(0.01, "download")
    err: list[Exception] = []
    done = threading.Event()

    def work() -> None:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            for repo in entry["repos"]:
                snapshot_download(repo)
        except Exception as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=work, daemon=True,
                     name=f"store-{job.params['model_id']}").start()
    est = max(entry["est_bytes"], 1)
    while not done.wait(5):
        _du_bust(entry)
        frac = min(0.98, 0.02 + 0.96 * (_du(entry) / est))
        progress(frac, "download")
    if err:
        raise RuntimeError(f"Download failed: {err[0]}")
    _du_bust(entry)
    if not _installed(entry):
        raise RuntimeError("Download finished but the snapshot does not "
                           "verify — try the install again.")
    job.receipts["installed_bytes"] = _du(entry)
    progress(1.0, "installed")
    return []


def delete(model_id: str) -> int:
    """Remove the model's weights; returns bytes freed. Callers guard
    against in-use models before calling."""
    entry = _catalog()[model_id]
    freed = _du(entry)
    # Drop any resident pipeline that could hold these weights open.
    from . import image, pipeline, video  # noqa: PLC0415
    if entry["capability"] == "text-to-image":
        image.ENGINE.unload()
    elif entry["capability"] == "text-to-video":
        video.ENGINE.unload()
    elif entry["capability"] == "image-to-mesh":
        pipeline.ENGINE.unload()
    for repo in entry["repos"]:
        d = _repo_dir(repo)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    _du_bust(entry)
    return freed
