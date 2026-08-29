"""text-to-video capability — Wan 2.2 TI2V-5B via diffusers.

The Mac app's "make a video clip" delegates over the swarm to whichever
peer advertises {"id": "text-to-video", "kind": "video", "ready": true}
(silicon-node issue #4). Same jobs pattern as image-to-mesh: submit via
POST /v1/jobs {"capability": "text-to-video", "prompt": ...}, poll, and
result_urls carries the .mp4. GPU-arbitrated like every other tenant.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from . import config
from .jobs import Job

log = logging.getLogger("silicon-node.video")

MODEL_REPO = os.environ.get("SILICON_NODE_VIDEO_MODEL",
                            "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
LTX_REPO = os.environ.get("SILICON_NODE_LTX_MODEL",
                          "dg845/LTX-2.3-Distilled-Diffusers")
# LTX-2.3 Uncensored v1.4 (ChrisColeTech): LTX-2.3 with the Eros10 NSFW,
# DMD-distilled and in-context detailer LoRAs merged into the DiT, shipped
# ComfyUI-style as single-file GGUF/FP8 transformers. The node loads just
# that one file into the distilled diffusers pipeline above — text encoder,
# connectors, VAEs and vocoder are the stock LTX-2.3 ones, which is what
# the repo's own split ships too (its projections/VAE files are re-exports;
# their headers were read to check). Q6_K is the author's recommended
# quant; point the FILE env at the _Q4_K_M sibling if the card runs out.
LTX_UNCENSORED_REPO = os.environ.get(
    "SILICON_NODE_LTX_UNCENSORED_REPO",
    "ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8")
LTX_UNCENSORED_FILE = os.environ.get(
    "SILICON_NODE_LTX_UNCENSORED_FILE",
    "split/diffusion_models/ltxv23_uncensored_v1.4_Q6_K.gguf")

_lock = threading.RLock()  # the i2v branch unloads inside the lock


class VideoEngine:
    def __init__(self) -> None:
        self._pipe = None

    def weights_present(self) -> bool:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            snapshot_download(MODEL_REPO, local_files_only=True)
            return True
        except Exception:  # noqa: BLE001
            return False

    def ready(self) -> bool:
        try:
            import diffusers  # noqa: F401, PLC0415
        except ImportError:
            return False
        return self.weights_present()

    def unload(self) -> None:
        with _lock:
            if (self._pipe is None
                    and getattr(self, "_pipe_i2v", None) is None
                    and getattr(self, "_pipe_ltx", None) is None
                    and getattr(self, "_pipe_ltx_unc", None) is None):
                return
            log.info("unloading video pipeline(s)")
            self._pipe = None
            self._pipe_i2v = None
            self._pipe_ltx = None
            self._pipe_ltx_unc = None
            self._pipe_ltx_unc_i2v = None
            self._ltx_unc_offload_owner = None
        import gc  # noqa: PLC0415
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure(self) -> None:
        with _lock:
            if self._pipe is not None:
                return
            import torch  # noqa: PLC0415
            from diffusers import WanPipeline  # noqa: PLC0415
            log.info("loading %s (first video job pays this once)…",
                     MODEL_REPO)
            t0 = time.time()
            pipe = WanPipeline.from_pretrained(
                MODEL_REPO, torch_dtype=torch.bfloat16)
            # Measured, not assumed: full residency was tried and the
            # pipeline (umt5-xxl text encoder + transformer + VAE) does
            # NOT fit 24 GB WDDM — it spills to shared memory, GPU util
            # collapses to ~15%, and the driver errors out. Model-level
            # offload keeps each sub-model on-GPU for its whole phase
            # (text encode once, then the transformer stays resident for
            # the entire denoise), so denoise-time utilization is real.
            if os.environ.get("SILICON_NODE_VIDEO_RESIDENT", "0") == "1":
                pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
            self._pipe = pipe
            log.info("Wan pipeline ready in %.0fs", time.time() - t0)

    def generate(self, job: Job, progress) -> list[str]:
        import shutil  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers.utils import export_to_video  # noqa: PLC0415

        prompt = (job.params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("A prompt is required for text-to-video.")
        negative = job.params.get("negative_prompt") or None
        seed = job.params.get("seed")
        # Node-level defaults are remotely tunable (handoff 129) —
        # explicit per-request values always win.
        from .capsettings import CAPS  # noqa: PLC0415
        vs = CAPS.settings("text-to-video")
        _res = {"480p": (832, 480), "720p": (1280, 704)}
        dw, dh = _res.get(str(vs.get("resolution", "720p")), (1280, 704))
        frames = min(121, max(17, int(job.params.get("frames", 49))))
        width = int(job.params.get("width", dw))
        height = int(job.params.get("height", dh))
        steps = min(50, max(10, int(job.params.get(
            "steps", vs.get("wan_steps", 30)))))

        image_path = job.params.get("image_path")
        engine = job.params.get("engine", "wan")
        if engine == "ltx":
            return self._generate_ltx(job, progress)
        if engine == "ltx-uncensored":
            return self._generate_ltx_uncensored(job, progress)

        progress(0.05, "model-load")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gen = None
        if seed not in (None, ""):
            gen = torch.Generator("cuda").manual_seed(int(seed))

        def _cb(pipe, i, t, kw):
            progress(0.10 + 0.80 * (i + 1) / steps, "video-denoise",
                     step=i + 1, steps_total=steps)
            return kw

        kwargs = dict(prompt=prompt, negative_prompt=negative,
                      num_frames=frames, width=width, height=height,
                      num_inference_steps=steps, generator=gen,
                      callback_on_step_end=_cb)
        progress(0.10, "video-denoise")
        if image_path:
            # TI2V's image-conditioned path (the Mac sends a start image
            # for image-to-video requests).
            from diffusers import WanImageToVideoPipeline  # noqa: PLC0415
            from diffusers.utils import load_image  # noqa: PLC0415
            with _lock:
                if getattr(self, "_pipe_i2v", None) is None:
                    self.unload()  # one Wan pipeline resident at a time
                    log.info("loading Wan image-to-video pipeline…")
                    p = WanImageToVideoPipeline.from_pretrained(
                        MODEL_REPO, torch_dtype=torch.bfloat16)
                    if os.environ.get("SILICON_NODE_VIDEO_RESIDENT",
                                      "0") == "1":
                        p.to("cuda")
                    else:
                        p.enable_model_cpu_offload()
                    self._pipe_i2v = p
            kwargs["image"] = load_image(image_path)
            result = self._pipe_i2v(**kwargs)
        else:
            self._ensure()
            result = self._pipe(**kwargs)
        job.receipts["video_denoise_s"] = round(time.time() - t0, 1)

        progress(0.93, "video-export")
        t1 = time.time()
        out = job.dir / "clip.mp4"
        export_to_video(result.frames[0], str(out), fps=24)
        job.receipts["video_export_s"] = round(time.time() - t1, 1)
        job.receipts["video_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)
        job.receipts["video_frames"] = frames

        name = f"{job.job_id}-clip.mp4"
        shutil.copy2(out, config.FILES_DIR / name)
        job.receipts["artifacts"] = [name]
        return [name]


    def ltx_ready(self) -> bool:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            snapshot_download(LTX_REPO, local_files_only=True)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _generate_ltx(self, job: Job, progress) -> list[str]:
        """LTX-2 distilled — the iteration pick: few steps, fast clips."""
        import shutil  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers import (  # noqa: PLC0415
            LTX2Pipeline, PipelineQuantizationConfig)
        from diffusers.utils import export_to_video  # noqa: PLC0415

        progress(0.05, "model-load")
        with _lock:
            if getattr(self, "_pipe_ltx", None) is None:
                self.unload()
                # Measured, not assumed: in bf16 this checkpoint's text
                # encoder is 46 GB and its transformer 36 GB — each one
                # alone exceeds the 24 GB card, so no offload scheme can
                # move them whole (WDDM spills, the driver errors out).
                # NF4 brings them to ~12 GB + ~10 GB, which fit one at a
                # time under model offload.
                log.info("loading %s (NF4-quantized)…", LTX_REPO)
                q = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16},
                    components_to_quantize=["transformer", "text_encoder"])
                p = LTX2Pipeline.from_pretrained(
                    LTX_REPO, torch_dtype=torch.bfloat16,
                    quantization_config=q)
                p.enable_model_cpu_offload()
                try:
                    p.vae.enable_tiling()
                except Exception:  # noqa: BLE001
                    pass
                self._pipe_ltx = p
                job.receipts["ltx_quant"] = "nf4"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        from .capsettings import CAPS  # noqa: PLC0415
        vs = CAPS.settings("text-to-video")
        steps = min(12, max(4, int(job.params.get(
            "steps", vs.get("ltx_steps", 8)))))
        frames = min(161, max(17, int(job.params.get("frames", 49))))
        gen = None
        seed = job.params.get("seed")
        if seed not in (None, ""):
            gen = torch.Generator("cuda").manual_seed(int(seed))

        def _cb(pipe, i, t, kw):
            progress(0.10 + 0.80 * (i + 1) / steps, "video-denoise",
                     step=i + 1, steps_total=steps)
            return kw

        progress(0.10, "video-denoise")
        result = self._pipe_ltx(
            prompt=job.params.get("prompt", ""),
            num_frames=frames,
            width=int(job.params.get("width", 1280)),
            height=int(job.params.get("height", 704)),
            num_inference_steps=steps, generator=gen,
            callback_on_step_end=_cb)
        job.receipts["ltx_denoise_s"] = round(time.time() - t0, 1)
        progress(0.93, "video-export")
        out = job.dir / "clip.mp4"
        export_to_video(result.frames[0], str(out), fps=24)
        job.receipts["video_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)
        name = f"{job.job_id}-clip.mp4"
        shutil.copy2(out, config.FILES_DIR / name)
        job.receipts["artifacts"] = [name]
        return [name]

    # -- LTX-2.3 Uncensored v1.4 --------------------------------------------

    def ltx_uncensored_ready(self) -> bool:
        """The transformer file is cached AND the distilled pipeline it
        plugs into is installed (that install supplies everything else)."""
        if not self.ltx_ready():
            return False
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            hf_hub_download(LTX_UNCENSORED_REPO, LTX_UNCENSORED_FILE,
                            local_files_only=True)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _ensure_ltx_uncensored(self, job: Job) -> None:
        import torch  # noqa: PLC0415
        from diffusers import (  # noqa: PLC0415
            GGUFQuantizationConfig, LTX2Pipeline,
            LTX2VideoTransformer3DModel, PipelineQuantizationConfig)
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        with _lock:
            if getattr(self, "_pipe_ltx_unc", None) is not None:
                return
            self.unload()
            path = hf_hub_download(LTX_UNCENSORED_REPO, LTX_UNCENSORED_FILE,
                                   local_files_only=True)
            log.info("loading %s transformer from %s…",
                     LTX_UNCENSORED_REPO, path)
            t0 = time.time()
            # The GGUF carries Lightricks' original tensor names; the
            # single-file loader renames them and drops the connector
            # blocks the file also carries (those live in the pipeline's
            # own `connectors` component). `config=` must point at the
            # 2.3 layout — the loader's default guess is LTX-2.0's.
            transformer = LTX2VideoTransformer3DModel.from_single_file(
                path, config=LTX_REPO, subfolder="transformer",
                local_files_only=True,
                quantization_config=GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16)
            # Everything else is the distilled install, text encoder NF4
            # exactly as in _generate_ltx (46 GB bf16 does not fit).
            q = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16},
                components_to_quantize=["text_encoder"])
            p = LTX2Pipeline.from_pretrained(
                LTX_REPO, transformer=transformer,
                torch_dtype=torch.bfloat16, quantization_config=q)
            p.enable_model_cpu_offload()
            try:
                p.vae.enable_tiling()
            except Exception:  # noqa: BLE001
                pass
            self._pipe_ltx_unc = p
            self._pipe_ltx_unc_i2v = None
            self._ltx_unc_offload_owner = p
            job.receipts["ltx_uncensored_file"] = Path(LTX_UNCENSORED_FILE).name
            job.receipts["ltx_quant"] = "gguf+nf4"
            log.info("LTX-2.3 uncensored pipeline ready in %.0fs",
                     time.time() - t0)

    def _ltx_uncensored_pipe(self, image_to_video: bool):
        """The t2v pipeline, or an i2v one sharing its weights. Model
        offload hooks belong to one pipeline at a time, so whichever is
        about to run takes them over."""
        base = self._pipe_ltx_unc
        if image_to_video:
            if getattr(self, "_pipe_ltx_unc_i2v", None) is None:
                from diffusers import LTX2ImageToVideoPipeline  # noqa: PLC0415
                self._pipe_ltx_unc_i2v = LTX2ImageToVideoPipeline(
                    **base.components)
            pipe = self._pipe_ltx_unc_i2v
        else:
            pipe = base
        if getattr(self, "_ltx_unc_offload_owner", None) is not pipe:
            pipe.enable_model_cpu_offload()
            self._ltx_unc_offload_owner = pipe
        return pipe

    def _generate_ltx_uncensored(self, job: Job, progress) -> list[str]:
        """LTX-2.3 Uncensored v1.4 — the distilled pipeline with the merged
        transformer swapped in. Clips carry the audio track the model
        generates alongside the frames, and a start image is honoured."""
        import shutil  # noqa: PLC0415
        import torch  # noqa: PLC0415

        progress(0.05, "model-load")
        self._ensure_ltx_uncensored(job)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        from .capsettings import CAPS  # noqa: PLC0415
        vs = CAPS.settings("text-to-video")
        # Model card: 8 steps minimum, 16 max; CFG 1.0 (slow, quiet) to
        # 4.0 (action, dialogue), 3.5 in its samples.
        steps = min(16, max(4, int(job.params.get(
            "steps", vs.get("ltx_uncensored_steps", 8)))))
        guidance = min(8.0, max(1.0, float(job.params.get(
            "guidance", vs.get("ltx_uncensored_cfg", 3.5)))))
        # 241 = the Mac's ten-second ceiling at 24 fps (8k+1, as LTX wants).
        frames = min(241, max(17, int(job.params.get("frames", 49))))
        width = int(job.params.get("width", 1280))
        height = int(job.params.get("height", 704))
        gen = None
        seed = job.params.get("seed")
        if seed not in (None, ""):
            gen = torch.Generator("cuda").manual_seed(int(seed))

        def _cb(pipe, i, t, kw):
            progress(0.10 + 0.80 * (i + 1) / steps, "video-denoise",
                     step=i + 1, steps_total=steps)
            return kw

        kwargs = dict(prompt=job.params.get("prompt", ""),
                      negative_prompt=job.params.get("negative_prompt") or None,
                      num_frames=frames, width=width, height=height,
                      frame_rate=24.0, num_inference_steps=steps,
                      guidance_scale=guidance, generator=gen,
                      callback_on_step_end=_cb)
        image_path = job.params.get("image_path")
        with _lock:
            pipe = self._ltx_uncensored_pipe(image_to_video=bool(image_path))
        if image_path:
            from diffusers.utils import load_image  # noqa: PLC0415
            kwargs["image"] = load_image(image_path)
        progress(0.10, "video-denoise")
        result = pipe(**kwargs)
        job.receipts["ltx_denoise_s"] = round(time.time() - t0, 1)
        job.receipts["ltx_steps"] = steps
        job.receipts["ltx_cfg"] = guidance

        progress(0.93, "video-export")
        t1 = time.time()
        out = job.dir / "clip.mp4"
        self._export_with_audio(pipe, result, out, job)
        job.receipts["video_export_s"] = round(time.time() - t1, 1)
        job.receipts["video_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)
        job.receipts["video_frames"] = frames
        name = f"{job.job_id}-clip.mp4"
        shutil.copy2(out, config.FILES_DIR / name)
        job.receipts["artifacts"] = [name]
        return [name]

    @staticmethod
    def _export_with_audio(pipe, result, out: Path, job: Job) -> None:
        """Mux the generated audio under the frames; fall back to a silent
        clip — and say so in the receipts — if PyAV is missing."""
        audio = getattr(result, "audio", None)
        try:
            if audio is None:
                raise ValueError("the pipeline returned no audio")
            from diffusers.utils import encode_video  # noqa: PLC0415
            encode_video(
                result.frames[0], fps=24, output_path=str(out),
                audio=audio[0].float().cpu(),
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate)
            job.receipts["video_audio"] = "muxed"
        except Exception as exc:  # noqa: BLE001
            log.warning("audio mux skipped (%s); writing a silent clip", exc)
            from diffusers.utils import export_to_video  # noqa: PLC0415
            export_to_video(result.frames[0], str(out), fps=24)
            job.receipts["video_audio"] = f"dropped ({exc})"


ENGINE = VideoEngine()


def text_to_video(job: Job, progress) -> list[str]:
    return ENGINE.generate(job, progress)


def download_weights_async() -> None:
    def work() -> None:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            log.info("downloading %s…", MODEL_REPO)
            snapshot_download(MODEL_REPO)
            log.info("video weights complete")
        except Exception:  # noqa: BLE001
            log.exception("video weight download failed")
    threading.Thread(target=work, daemon=True).start()
