"""The CUDA pipelines behind the jobs API.

image-to-mesh (Route A):
    TRELLIS.2 densify  (image -> dense textured GLB, resident in-process)
    LATO.2 retopology  (dense mesh -> clean low-poly OBJ)

retopologize:
    LATO.2 only (mesh in, clean mesh out).

The TRELLIS.2 pipeline stays loaded between jobs (reload is the expensive
part). The LATO.2 stage currently shells out to the repo's own
scripts/e2e_inference.py — correct by construction and OOM-isolated; moving
it in-process/resident is a planned optimization once VRAM numbers are in.

Every stage records receipts: wall time and (in-process stages) peak VRAM
via torch.cuda.max_memory_allocated.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from . import config
from .jobs import Job

log = logging.getLogger("silicon-node.pipeline")

Progress = Callable[[float, str], None]

_engine_lock = threading.Lock()


DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class Engine:
    """Owns the resident TRELLIS.2 pipeline; LATO.2 runs as a subprocess."""

    def __init__(self) -> None:
        self._trellis = None
        self.trellis_ready = False
        self.lato_ready = (config.LATO2_ROOT / "ckpt" / "vflow.pt").exists()
        self._dinov3_ok: bool | None = None

    def trellis_available(self) -> tuple[bool, str]:
        """TRELLIS.2 needs the gated DINOv3 conditioner from HuggingFace.

        Returns (available, detail). Cached after the first check; a token
        arriving via HF_TOKEN needs a service restart to be picked up.
        """
        if self.trellis_ready:
            return True, "loaded"
        if self._dinov3_ok is None:
            try:
                from huggingface_hub import auth_check  # noqa: PLC0415
                auth_check(DINOV3_REPO)
                self._dinov3_ok = True
            except Exception as exc:  # noqa: BLE001
                log.warning("DINOv3 gate check failed: %s", exc)
                self._dinov3_ok = False
        if self._dinov3_ok:
            return True, "ready"
        return False, (
            f"Blocked on the gated HuggingFace repo {DINOV3_REPO} "
            "(TRELLIS.2's image conditioner). Accept the license on that "
            "repo page with a HuggingFace account, then give this service "
            "a read token via the HF_TOKEN environment variable and "
            "restart it.")

    # ---- TRELLIS.2 (resident) ------------------------------------------

    def _ensure_trellis(self) -> None:
        if self._trellis is not None:
            return
        with _engine_lock:
            if self._trellis is not None:
                return
            log.info("loading TRELLIS.2 pipeline (first job pays this once)…")
            t0 = time.time()
            sys.path.insert(0, str(config.TRELLIS2_ROOT))
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: PLC0415

            pipe = Trellis2ImageTo3DPipeline.from_pretrained(
                os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B"))
            pipe.cuda()
            self._trellis = pipe
            self.trellis_ready = True
            log.info("TRELLIS.2 loaded in %.1fs", time.time() - t0)

    def unload(self) -> None:
        """Free the resident 3D pipelines (GPU handover to the LLM).

        The next 3D job repays the load (~90 s TRELLIS, ~30 s LATO) — the
        deliberate trade for giving the LLM the card back."""
        from .lato_engine import LATO_ENGINE  # noqa: PLC0415
        LATO_ENGINE.unload()
        try:
            from .video import ENGINE as VIDEO_ENGINE  # noqa: PLC0415
            VIDEO_ENGINE.unload()
        except Exception:  # noqa: BLE001
            pass
        with _engine_lock:
            if self._trellis is None:
                return
            log.info("unloading TRELLIS.2 pipeline to free VRAM")
            self._trellis = None
            self.trellis_ready = False
        import gc  # noqa: PLC0415
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def densify(self, image_path: Path, seed: int | None, out_glb: Path,
                progress: Progress, receipts: dict) -> None:
        """image -> dense textured GLB via TRELLIS.2."""
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._ensure_trellis()
        import o_voxel  # noqa: PLC0415

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        progress(0.10, "trellis-sample")
        image = Image.open(image_path).convert("RGBA")
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        mesh = self._trellis.run(image, **kwargs)[0]
        mesh.simplify(16_777_216)  # nvdiffrast face-count limit
        receipts["trellis_sample_s"] = round(time.time() - t0, 1)

        progress(0.50, "trellis-export")
        t1 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=500_000,
            texture_size=2048,
            remesh=True,
        )
        glb.export(str(out_glb))
        receipts["trellis_export_s"] = round(time.time() - t1, 1)
        receipts["trellis_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)

    # ---- LATO.2 (subprocess for now) -----------------------------------

    def retopo(self, mesh_path: Path, vert_num: int, seed: int | None,
               work_dir: Path, progress: Progress, receipts: dict) -> Path:
        """dense mesh -> clean low-poly OBJ via LATO.2.

        Resident in-process engine by default (models load once); the
        original subprocess path stays available via
        SILICON_NODE_LATO_SUBPROCESS=1 as a fallback."""
        mesh_dir = work_dir / "lato_in"
        out_dir = work_dir / "lato_out"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged = mesh_dir / mesh_path.name
        if not staged.exists():
            shutil.copy2(mesh_path, staged)

        if os.environ.get("SILICON_NODE_LATO_SUBPROCESS", "0") != "1":
            from .lato_engine import LATO_ENGINE  # noqa: PLC0415
            progress(0.65, "lato-retopology")
            return LATO_ENGINE.retopo(mesh_dir, out_dir, vert_num, seed,
                                      receipts)

        cmd = [
            sys.executable, str(config.LATO2_ROOT / "scripts" / "e2e_inference.py"),
            "--mesh_dir", str(mesh_dir),
            "--out_dir", str(out_dir),
            "--vert_num", str(vert_num),
            "--batch_size", "1",
            "--num_workers", "1",
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        progress(0.65, "lato-retopology")
        t0 = time.time()
        sampler = _VramSampler()
        sampler.start()
        try:
            proc = subprocess.run(
                cmd, cwd=str(config.LATO2_ROOT),
                capture_output=True, text=True, timeout=25 * 60)
        finally:
            sampler.stop()
        receipts["lato_e2e_s"] = round(time.time() - t0, 1)
        if sampler.peak_mb:
            receipts["lato_peak_vram_gb"] = round(sampler.peak_mb / 1024, 2)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            log.error("LATO.2 subprocess failed:\n%s", tail)
            if "CUDA out of memory" in tail:
                raise RuntimeError("CUDA out of memory in the retopology stage")
            raise RuntimeError(
                "The retopology stage failed. Last output: " + tail[-300:])

        stem = staged.stem
        pred = out_dir / f"{stem}_pred.obj"
        if not pred.exists():
            raise RuntimeError(
                "Retopology produced no mesh (the model generated no usable "
                "topology for this input). Try a different seed or vertex count.")
        return pred


class _VramSampler:
    """Approximate peak VRAM of a subprocess stage by polling nvidia-smi.

    Device-wide (includes the desktop's ~1.6 GB), which is the number that
    actually matters for "does this fit next to everything else".
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        self.peak_mb = 0
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                self.peak_mb = max(self.peak_mb,
                                   int(float(out.stdout.strip())))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


ENGINE = Engine()


def _stage_files(job: Job, named: dict[str, Path]) -> list[str]:
    """Copy artifacts into FILES_DIR under stable, unguessable names."""
    out: list[str] = []
    for label, src in named.items():
        name = f"{job.job_id}-{label}{src.suffix}"
        shutil.copy2(src, config.FILES_DIR / name)
        out.append(name)
    job.receipts["artifacts"] = out
    return out


def _parse_vert_num(params: dict) -> int:
    raw = params.get("vert_num", config.VERT_NUM_DEFAULT)
    try:
        v = int(str(raw))
    except (TypeError, ValueError):
        raise ValueError(
            f"vert_num must be an integer between {config.VERT_NUM_MIN} and "
            f"{config.VERT_NUM_MAX}; got {raw!r}") from None
    return max(config.VERT_NUM_MIN, min(config.VERT_NUM_MAX, v))


def _parse_seed(params: dict) -> int | None:
    raw = params.get("seed")
    if raw in (None, ""):
        return None
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        raise ValueError(f"seed must be an integer; got {raw!r}") from None


def image_to_mesh(job: Job, progress: Progress) -> list[str]:
    """Phase-1 capability: image -> dense GLB + clean low-poly OBJ."""
    vert_num = _parse_vert_num(job.params)
    seed = _parse_seed(job.params)
    image_path = Path(job.params["image_path"])
    work = job.dir

    progress(0.05, "model-load")
    dense_glb = work / "dense.glb"
    ENGINE.densify(image_path, seed, dense_glb, progress, job.receipts)

    pred_obj = ENGINE.retopo(dense_glb, vert_num, seed, work, progress,
                             job.receipts)

    progress(0.97, "export")
    return _stage_files(job, {"dense": dense_glb, "retopo": pred_obj})


def retopologize(job: Job, progress: Progress) -> list[str]:
    """Phase-2 capability: mesh in, clean low-poly mesh out (no densify)."""
    vert_num = _parse_vert_num(job.params)
    seed = _parse_seed(job.params)
    mesh_path = Path(job.params["mesh_path"])

    progress(0.05, "model-load")
    pred_obj = ENGINE.retopo(mesh_path, vert_num, seed, job.dir, progress,
                             job.receipts)
    progress(0.97, "export")
    return _stage_files(job, {"retopo": pred_obj})
