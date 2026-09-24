# Setting up the models

This is the long part — but it's honest about it, and it's built so you
can't really get lost:

- **Every section is optional.** Only sections 1, 2, and 7 are required.
  Skip any ability you don't want; add it later any time.
- **You can stop whenever.** Finish any section and you have a working
  node — abilities light up on the Mac automatically the moment their
  files exist. No "finish everything or nothing works."
- **One section at a sitting is fine.** Most are copy-paste-wait.

Your path:

- [ ] 1. WSL2 + CUDA (required, ~30 min)
- [ ] 2. Python environment (required, ~10 min)
- [ ] 3. 3D models — image → mesh
- [ ] 4. Video models — text → clip
- [ ] 5. Portrait models — talking + animated photos
- [ ] 6. Chat models
- [ ] 7. The service itself (required, ~5 min)

Throughout: the WSL distro is named `SiliconNode`, the service tree lives
at `/opt/silicon/silicon-node`, data at `/opt/silicon/data`.

## 1. WSL2 distro + CUDA

```powershell
wsl --install -d Ubuntu-22.04
wsl --shutdown; wsl --manage Ubuntu-22.04 --set-default-user root  # or your user
# rename/import as SiliconNode if you keep multiple distros
```

Inside the distro: enable systemd (`/etc/wsl.conf` → `[boot] systemd=true`),
install the CUDA 12.4 **toolkit** (the driver comes from Windows), and
[miniforge](https://github.com/conda-forge/miniforge) at `/opt/miniforge3`.

Give WSL enough memory in `C:\Users\<you>\.wslconfig` (LTX-2's offloaded
weights want it):

```ini
[wsl2]
memory=48GB
```

Two gotchas that will bite otherwise:

- After **any NVIDIA driver update**, run `wsl --shutdown` — WSL CUDA
  breaks until the VM restarts.
- WSL idles out without a Windows-side keepalive process;
  `setup-lan-exposure.ps1` installs one as a logon task.

## 2. Python env

```bash
conda create -n lato2 python=3.10
conda run -n lato2 pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n lato2 pip install "transformers==4.57.1" diffusers accelerate bitsandbytes \
    fastapi uvicorn httpx pillow huggingface_hub
```

`transformers` stays pinned at 4.x: 5.x renames DINOv3 internals and
breaks TRELLIS.2's extractor.

## 3. 3D pipeline (image → mesh, retopology)

- Clone [LATO.2](https://github.com/LoHhhha/LATO.2) to `/opt/silicon/LATO.2`,
  fetch its checkpoints into `ckpt/`, build its extensions in the env.
- Clone TRELLIS.2 to `/opt/silicon/TRELLIS.2`; weights
  (`microsoft/TRELLIS.2-4B`) download on first use.
- TRELLIS.2's conditioner (`facebook/dinov3-vitl16`) is **gated**: accept
  the license on HuggingFace, then run `set-tokens.ps1` on the Windows
  side to store your `hf_` token.

## 4. Video (text → clip)

Weights land in the distro's HF cache on first job, or prefetch:

```bash
conda run -n lato2 python -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers')"
conda run -n lato2 python -c "from huggingface_hub import snapshot_download; snapshot_download('dg845/LTX-2.3-Distilled-Diffusers')"
```

LTX-2 is 95 GB across 51 shards — the systemd unit below raises the file
descriptor limit for it, and its transformer + text encoder load
NF4-quantized automatically (bf16 does not fit a 24 GB card).

Optional: the adult-content LTX-2.3 merge is one 18 GB GGUF file, not the
whole 240 GB repo — it reuses the distilled install's text encoder and
decoders, so install that first. The dashboard's Store page does the same
fetch.

```bash
conda run -n lato2 python -c "from huggingface_hub import hf_hub_download; hf_hub_download('ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8', 'split/diffusion_models/ltxv23_uncensored_v1.4_Q6_K.gguf')"
```

Its loader needs `gguf` and `av` from requirements-media.txt. If the card
runs out of memory at load, set `SILICON_NODE_LTX_UNCENSORED_FILE` to the
`_Q4_K_M.gguf` sibling in the same folder (14 GB) and fetch that instead.

## 5. Portrait abilities

- **LivePortrait** (video-driven): clone to `/opt/silicon/LivePortrait`,
  own conda env `liveportrait` (their requirements omit torch — install
  torch 2.6 cu124 yourself), `apt install ffmpeg`, fetch pretrained
  weights per their README.
- **SadTalker** (audio-driven): clone to `/opt/silicon/SadTalker`, env
  `sadtalker` with `torch==2.0.1 torchvision==0.15.2` (cu118 wheels are
  fine), `pip install "setuptools<81"` (81 removed `pkg_resources`, which
  librosa needs), then `bash scripts/download_models.sh`.

## 6. Chat engines

- **ninfer** (Qwen3.8-27B INT8): download the
  [ninfer-3090 release](https://github.com/Don-Chad/ninfer-3090) and the
  model artifact into `ninfer-3090/dist/.../models/` on the Windows side.
  The service manages it (start/stop/context) via `/v1/llm`. Serving
  profile notes live in `server/llm.py` — the KV pool and prefill chunk
  values there are measured, not guessed; re-sweep before changing them.
- **llama.cpp** (any GGUF): nothing to do — the dashboard's Models page
  downloads the engine and GGUFs on demand. Small Qwens serve at 128K
  context by default.

## 7. The service

```bash
mkdir -p /opt/silicon/silicon-node   # copy this repo's tree there
cat > /etc/systemd/system/silicon-node.service <<'EOF'
[Unit]
Description=Silicon Node CUDA job service
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash /opt/silicon/silicon-node/run-server.sh
Restart=on-failure
RestartSec=5
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now silicon-node
```

`LimitNOFILE` is load-bearing: many-shard model loads exhaust the default
1024 soft limit and the failure masquerades as a CUDA driver error.

Then tell the service where the Windows side keeps things. The chat
engines are Windows programs, so they live on a Windows drive, which the
distro sees under `/mnt/<drive>`. Write your own paths into
`/opt/silicon/secrets.env` (read at every start, never committed):

```bash
cat >> /opt/silicon/secrets.env <<'EOF'
# The folder holding this repository's Windows checkout (its runtime/
# gets the llama.cpp builds, the agent harness and HyperQwen) and ninfer.
SILICON_NODE_WIN_HOME="/mnt/d/SiliconNode"
# The model library: GGUFs and chat templates.
SILICON_NODE_WIN_MODELS="/mnt/d/SiliconNode/models"
# Only if ninfer isn't at $SILICON_NODE_WIN_HOME/ninfer:
# NINFER_DIR="/mnt/d/SiliconNode/ninfer-3090/dist/ninfer-rtx3090-windows-x64-0.6.0-rtx3090"
EOF
```

Unset, they default to `/mnt/c/SiliconNode` and its `models/` folder.
Put the distro's disk image on the same drive: the model store checks
that drive's free space before an install.

Windows side: `install.ps1` adds the tray app to the Start menu and
startup; `setup-lan-exposure.ps1` (run as admin) adds the LAN port proxy,
firewall rule, and WSL keepalive if you want LAN access — with the swarm
token set first, per [SECURITY.md](../SECURITY.md).

## Bonsai 2 27B on the GGUF lane

PrismML's Ternary Bonsai 2 27B (`prism-ml/Ternary-Bonsai-2-27B-gguf`, the
`PTQ1_0` file plus its `mmproj-Q8_0` projector) is a one-click pick on the
dashboard's Models page. Its packing is refused by stock llama.cpp, so the
node fetches PrismML's fork (`PrismML-Eng/llama.cpp`, the Windows CUDA 12.4
build plus its cudart) into `runtime/llamacpp-prism` beside the stock engine
the first time such a file is downloaded, and serves those files from it.
`SILICON_NODE_LLAMACPP_PRISM_DIR` moves that folder. Everything else keeps
using the stock engine.
