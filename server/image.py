"""text-to-image capability — Qwen-Image + Sana via diffusers (hub 136).

Same jobs pattern as text-to-video: submit via POST /v1/text-to-image or
POST /v1/jobs {"capability": "text-to-image", "prompt": ...}, poll
GET /v1/jobs/{id}, and result_urls carries the .png. GPU-arbitrated like
every other tenant (the job preempts the chat engine; it auto-restores).

Two lanes, chosen with the model field:
  qwen-image  — Qwen-Image 20B, the readable-text-in-images pick. bf16 is
                ~41 GB transformer + ~16 GB text encoder, both beyond the
                24 GB card, so NF4 like the LTX video path.  DEFAULT.
  sana        — Sana 1600M, the fast-iteration lane; small enough to sit
                fully resident while its job runs.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from . import config
from .jobs import Job

log = logging.getLogger("silicon-node.image")

QWEN_REPO = os.environ.get("SILICON_NODE_QWEN_IMAGE_MODEL",
                           "Qwen/Qwen-Image")
SANA_REPO = os.environ.get(
    "SILICON_NODE_SANA_MODEL",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers")

# API-facing model ids (also the store ids, hub 137). sdxl has no
# weights by default — it appears in installed_models() only after a
# store install.
SDXL_REPO = os.environ.get("SILICON_NODE_SDXL_MODEL",
                           "stabilityai/stable-diffusion-xl-base-1.0")
MODEL_REPOS = {"qwen-image": QWEN_REPO, "sana": SANA_REPO,
               "sdxl": SDXL_REPO}

_lock = threading.RLock()


def _weights_present(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


class ImageEngine:
    def __init__(self) -> None:
        self._pipe_qwen = None
        self._pipe_sana = None

    def installed_models(self) -> list[str]:
        return [mid for mid, repo in MODEL_REPOS.items()
                if _weights_present(repo)]

    def ready(self) -> bool:
        try:
            import diffusers  # noqa: F401, PLC0415
        except ImportError:
            return False
        return bool(self.installed_models())

    def unload(self) -> None:
        with _lock:
            if (self._pipe_qwen is None and self._pipe_sana is None
                    and getattr(self, "_pipe_sdxl", None) is None):
                return
            log.info("unloading image pipeline(s)")
            self._pipe_qwen = None
            self._pipe_sana = None
            self._pipe_sdxl = None
        import gc  # noqa: PLC0415
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_qwen(self):
        with _lock:
            if self._pipe_qwen is not None:
                return self._pipe_qwen
            import torch  # noqa: PLC0415
            from diffusers import (  # noqa: PLC0415
                QwenImagePipeline, PipelineQuantizationConfig)
            self.unload()  # one image pipeline resident at a time
            log.info("loading %s (NF4-quantized)…", QWEN_REPO)
            q = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16},
                components_to_quantize=["transformer", "text_encoder"])
            p = QwenImagePipeline.from_pretrained(
                QWEN_REPO, torch_dtype=torch.bfloat16,
                quantization_config=q)
            p.enable_model_cpu_offload()
            try:
                p.vae.enable_tiling()
            except Exception:  # noqa: BLE001
                pass
            self._pipe_qwen = p
            return p

    def _ensure_sana(self):
        with _lock:
            if self._pipe_sana is not None:
                return self._pipe_sana
            import torch  # noqa: PLC0415
            from diffusers import SanaPipeline  # noqa: PLC0415
            self.unload()
            log.info("loading %s…", SANA_REPO)
            try:
                p = SanaPipeline.from_pretrained(
                    SANA_REPO, variant="bf16", torch_dtype=torch.bfloat16)
            except Exception:  # noqa: BLE001
                p = SanaPipeline.from_pretrained(
                    SANA_REPO, torch_dtype=torch.bfloat16)
            # Small enough for full residency while its job owns the card.
            p.to("cuda")
            self._pipe_sana = p
            return p

    def _ensure_sdxl(self):
        with _lock:
            if getattr(self, "_pipe_sdxl", None) is not None:
                return self._pipe_sdxl
            import torch  # noqa: PLC0415
            from diffusers import StableDiffusionXLPipeline  # noqa: PLC0415
            self.unload()
            log.info("loading %s…", SDXL_REPO)
            try:
                p = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_REPO, variant="fp16", torch_dtype=torch.bfloat16)
            except Exception:  # noqa: BLE001
                p = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_REPO, torch_dtype=torch.bfloat16)
            p.to("cuda")
            self._pipe_sdxl = p
            return p

    def generate(self, job: Job, progress) -> list[str]:
        import torch  # noqa: PLC0415

        prompt = (job.params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("A prompt is required for text-to-image.")
        from .capsettings import CAPS  # noqa: PLC0415
        cs = CAPS.settings("text-to-image")
        model = (job.params.get("model")
                 or cs.get("default_model", "qwen-image"))
        model = {"text-to-image": cs.get("default_model", "qwen-image"),
                 "qwen": "qwen-image"}.get(model, model)
        if model not in MODEL_REPOS:
            raise ValueError(
                f"This node serves {', '.join(sorted(MODEL_REPOS))}; "
                f"{model!r} isn't one of them.")
        if not _weights_present(MODEL_REPOS[model]):
            raise ValueError(
                f"The {model} weights are not installed on this node — "
                "install them from the model store.")

        # Snap to the pipelines' size granularity rather than erroring.
        snap = 32
        width = max(512, min(2048, int(job.params.get("width", 1024))))
        height = max(512, min(2048, int(job.params.get("height", 1024))))
        width, height = width // snap * snap, height // snap * snap
        negative = (job.params.get("negative_prompt") or "").strip() or None
        seed = job.params.get("seed")
        gen = None
        if seed not in (None, ""):
            gen = torch.Generator("cuda").manual_seed(int(seed))

        progress(0.05, "model-load")
        if model == "sana":
            pipe = self._ensure_sana()
            steps = min(50, max(5, int(job.params.get(
                "steps", cs.get("sana_steps", 20)))))
        elif model == "sdxl":
            pipe = self._ensure_sdxl()
            steps = min(60, max(10, int(job.params.get(
                "steps", cs.get("sdxl_steps", 30)))))
        else:
            pipe = self._ensure_qwen()
            steps = min(60, max(10, int(job.params.get(
                "steps", cs.get("qwen_steps", 30)))))
            job.receipts["image_quant"] = "nf4"

        def _cb(pipeline, i, t, kw):
            progress(0.10 + 0.85 * (i + 1) / steps, "image-denoise",
                     step=i + 1, steps_total=steps)
            return kw

        kwargs = dict(prompt=prompt, width=width, height=height,
                      num_inference_steps=steps, generator=gen,
                      callback_on_step_end=_cb)
        if negative:
            kwargs["negative_prompt"] = negative
            if model == "qwen-image":
                # Qwen-Image only applies the negative with true CFG on.
                kwargs["true_cfg_scale"] = 4.0

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        progress(0.10, "image-denoise")
        result = pipe(**kwargs)
        job.receipts["image_denoise_s"] = round(time.time() - t0, 1)
        job.receipts["image_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)
        job.receipts["image_model"] = model

        progress(0.97, "image-export")
        out = job.dir / "image.png"
        result.images[0].save(out)
        name = f"{job.job_id}-image.png"
        import shutil  # noqa: PLC0415
        shutil.copy2(out, config.FILES_DIR / name)
        job.receipts["artifacts"] = [name]
        return [name]


ENGINE = ImageEngine()


def text_to_image(job: Job, progress) -> list[str]:
    return ENGINE.generate(job, progress)
