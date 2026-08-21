"""Resident LATO.2 engine — the retopology models loaded once, reused.

Faithful port of LATO.2's scripts/e2e_inference.py main() into a
load-once/run-many engine. The subprocess path paid a ~30 s model +
DINOv2 reload on every job; resident, a retopology is just the ~10 s of
actual sampling. The conditioning-view render keeps the script's own
proven pattern: a spawn-context DataLoader worker (EGL rendering hangs in
*forked* children of a CUDA-initialized parent; spawn is safe).

VRAM: ~7 GB resident. The LLM arbitration hooks unload this engine too —
LATO + Qwen3.8-27B cannot share the 24 GB card.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from . import config

log = logging.getLogger("silicon-node.lato")

# Must be set before LATO's modules (open3d/EGL) are imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

_ROOT = str(config.LATO2_ROOT)


class LatoEngine:
    """load() once; retopo() many. Thread-safety: the single GPU worker is
    the only caller of retopo(); load/unload also happen on that thread or
    behind the same coarse lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._m: SimpleNamespace | None = None  # models + config bundle

    @property
    def loaded(self) -> bool:
        return self._m is not None

    # ------------------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self._m is not None:
                return
            t0 = time.time()
            os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
            if _ROOT not in sys.path:
                sys.path.insert(0, _ROOT)
            import torch  # noqa: PLC0415
            from models import (  # noqa: PLC0415
                DinoV2Encoder, OffsetHead, TopoFlowEulerSampler,
                TopologySiTFlow, TopologyVAE, VertexSLatFlowModel,
                VertFlowEulerCfgSampler, VertexVAE, VoxelFieldConditioner,
            )
            from utils.load import load_latov2_model  # noqa: PLC0415

            device = torch.device("cuda")
            ckpt = config.LATO2_ROOT / "ckpt"
            vflow, vflow_cfg = load_latov2_model(
                VertexSLatFlowModel, str(ckpt / "vflow.pt"), device)
            vvae, vvae_cfg = load_latov2_model(
                VertexVAE, str(ckpt / "vvae.pt"), device)
            offset_head, _ = load_latov2_model(
                OffsetHead, str(ckpt / "offset_head.pt"), device)
            tflow, tflow_cfg = load_latov2_model(
                TopologySiTFlow, str(ckpt / "tflow.pt"), device)
            tvae, _ = load_latov2_model(
                TopologyVAE, str(ckpt / "tvae.pt"), device)
            voxel_encoder, venc_cfg = load_latov2_model(
                VoxelFieldConditioner, str(ckpt / "voxel_encoder.pt"), device)

            res = vvae_cfg["resolution"]
            min_res = vvae_cfg["min_resolution"]
            num_discrete = int(tflow_cfg["args"]["num_discrete"])
            voxel_res = int(venc_cfg["resolution"])
            if num_discrete != res:
                raise ValueError(
                    f"T-Flow num_discrete={num_discrete} != V-VAE "
                    f"resolution={res}")
            if voxel_res != min_res:
                raise ValueError(
                    f"voxel encoder resolution={voxel_res} != V-VAE "
                    f"min_resolution={min_res}")

            dino = (DinoV2Encoder(
                model_name=vflow_cfg["dino_version"],
                hub_dir=str(ckpt / "dinov2"),
                img_res=vflow_cfg["image_resolution"],
            ).to(device).eval())

            self._m = SimpleNamespace(
                device=device, vflow=vflow, vvae=vvae,
                offset_head=offset_head, tflow=tflow, tvae=tvae,
                voxel_encoder=voxel_encoder, dino=dino,
                vertex_sampler=VertFlowEulerCfgSampler(),
                topo_sampler=TopoFlowEulerSampler(),
                res=res, min_res=min_res,
                latent_dim=vflow_cfg["latent_dim"],
                density_max=vflow_cfg["max_vertex_num"],
                z_dim=int(tflow_cfg["args"]["z_dim"]),
                max_vertices=int(tflow_cfg["args"]["max_vertices"]),
                latent_scale=float(tflow_cfg["latent_scale"]),
                voxel_res=voxel_res,
            )
            log.info("LATO.2 engine loaded in %.1fs", time.time() - t0)

    def unload(self) -> None:
        with self._lock:
            if self._m is None:
                return
            log.info("unloading LATO.2 engine to free VRAM")
            self._m = None
        import gc  # noqa: PLC0415
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def retopo(self, mesh_dir: Path, out_dir: Path, vert_num: int,
               seed: int | None, receipts: dict) -> Path:
        """One mesh dir in (single file), <stem>_pred.obj out.

        Body mirrors e2e_inference.py's per-batch flow with the script's
        defaults (vflow_steps 24, cfg 3.0, tflow_steps 50, threshold 0.5,
        fill_quad_rings on)."""
        self.load()
        m = self._m
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from torch.utils.data import DataLoader  # noqa: PLC0415
        from dataset.voxel_dataset import (  # noqa: PLC0415
            VoxelVertexDataset, collate_fn)
        from modules.sparse import SparseTensor  # noqa: PLC0415
        from utils.export import export_vertex  # noqa: PLC0415
        from utils.inference import (  # noqa: PLC0415
            build_voxel_fields, compute_density, decode_vertices,
            edges_to_faces, pad_verts, worker_init)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        torch.manual_seed(seed if seed is not None else 42)
        np.random.seed(seed if seed is not None else 42)
        out_dir.mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace(
            vert_num=vert_num, use_gt_vert_count=False, scaler=1.0,
            min_verts=200.0, max_verts=5000.0)

        dataset = VoxelVertexDataset(
            root_dir=str(mesh_dir), resolution=m.res,
            min_resolution=m.min_res, need_encoder_inputs=False,
            num_samples=None, render=True, img_res=518,
            render_azimuth=45.0, render_elevation=30.0)
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=partial(collate_fn, resolution=m.res,
                               min_resolution=m.min_res),
            num_workers=1, pin_memory=True,
            multiprocessing_context="spawn", worker_init_fn=worker_init)

        pred_path: Path | None = None
        for batch in loader:
            for err in batch["errors"]:
                raise RuntimeError(
                    "Mesh preprocessing failed: "
                    + err["error"].splitlines()[0][:200])
            if "name" not in batch:
                continue
            name = batch["name"][0]

            density = compute_density(batch, args, m.density_max, m.device)
            with torch.no_grad():
                cond = m.dino(np.stack(batch["image"])).float()
                neg_cond = torch.zeros_like(cond)

            min_active = batch[f"active_voxels_{m.min_res}"]
            with torch.no_grad(), torch.autocast("cuda",
                                                 dtype=torch.bfloat16):
                coords = min_active.to(m.device)
                noise = SparseTensor(
                    coords=coords.int(),
                    feats=torch.randn(coords.shape[0], m.latent_dim,
                                      device=m.device))
                z_pred = m.vertex_sampler.sample(
                    model=m.vflow, noise=noise, cond=cond,
                    neg_cond=neg_cond, steps=24, cfg_strength=3.0,
                    rescale_t=1.0, density=density)
                pred_coords, pred_offsets = decode_vertices(
                    m.vvae, m.offset_head, z_pred, 0.5)

            sel = pred_coords[:, 0] == 0
            vert_int = pred_coords[sel, 1:].long()
            vert_off = pred_offsets[sel]
            export_vertex(str(out_dir), name, type_name="pred",
                          vert_int=vert_int.numpy(),
                          vert_offsets=vert_off.numpy(), resolution=m.res)
            Image.fromarray(batch["image"][0]).save(
                str(out_dir / f"{name}_render.png"))
            n = int(vert_int.shape[0])
            if n < 3 or n > m.max_vertices:
                raise RuntimeError(
                    f"V-Flow produced {n} vertices, outside the usable "
                    "range — try a different seed or vertex count.")

            with torch.no_grad():
                verts, mask, lengths = pad_verts([vert_int], m.device)
                voxel_list = [min_active[min_active[:, 0] == 0, 1:].long()]
                field = build_voxel_fields(voxel_list, m.voxel_res,
                                           m.device)
                cond_vox = m.voxel_encoder(field)
                z0 = torch.randn(verts.shape[0], verts.shape[1], m.z_dim,
                                 device=m.device)
                z_flow = m.topo_sampler.sample(
                    model=m.tflow, noise=z0, verts=verts, mask=mask,
                    cond=cond_vox, steps=50)
                z = z_flow.float() / m.latent_scale
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    edges_list = m.tvae.decode(
                        z, verts=verts, verts_mask=mask,
                        chunk_size=20000, threshold=0.0)

            faces = edges_to_faces(edges_list[0], lengths[0], True)
            if faces.shape[0] == 0:
                raise RuntimeError(
                    "No topology decoded for this input — try a different "
                    "seed or vertex count.")

            res_f = float(m.res)
            with_offset = (vert_int.numpy().astype(np.float64) / res_f - 0.5
                           + vert_off.numpy().astype(np.float64)
                           / (res_f * 2.0))
            pred_path = out_dir / f"{name}_pred.obj"
            trimesh.Trimesh(vertices=with_offset, faces=faces).export(
                str(pred_path))

        if pred_path is None:
            raise RuntimeError("No mesh found in the input directory.")
        receipts["lato_resident_s"] = round(time.time() - t0, 1)
        receipts["lato_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)
        return pred_path


LATO_ENGINE = LatoEngine()
