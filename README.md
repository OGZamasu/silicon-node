# Silicon Node

The **Windows/CUDA node** for [Silicon Optimizer](https://github.com/OGZamasu/silicon-optimizer)
swarms: a GPU job service that runs what a Mac can't, and lends its
RTX-class GPU to every machine in your swarm.

A Mac running Silicon Optimizer delegates work here over your own network —
nothing leaves it — and renders live progress for every job. One node can
serve several Macs; one Mac can use several nodes.

## What it serves

| Ability | Backed by | Peak VRAM (measured) |
|---|---|---|
| Image → clean low-poly 3D model | TRELLIS.2 + LATO.2 retopology | ~10–13 GB |
| Mesh retopology | LATO.2 (resident, ~16 s warm) | ~7 GB |
| Text → video clip | Wan 2.2 TI2V-5B **or** LTX-2 distilled (~2.6× faster, NF4) | 12–16 GB |
| Portrait + video performance → animated clip | LivePortrait | ~4 GB |
| Portrait + speech audio → lip-synced clip | SadTalker | ~5 GB |
| Chat / agent completions (OpenAI + Anthropic APIs) | ninfer (Qwen3.8-27B INT8, up to 64K context) or llama.cpp GGUFs (128K on small models) | 17–21 GB |

One GPU job runs at a time; jobs preempt the chat model and it restores
itself when the queue drains. Every job reports progress, stage, step,
ETA, and who submitted it. A web dashboard (with tray app) shows and
controls all of it, including per-model configuration, enable/disable per
ability, and remote settings the Mac's Swarm page can edit.

## Hardware you need

- **NVIDIA GPU** — 24 GB VRAM (RTX 3090/4090 class) runs everything.
  Smaller cards can serve subsets: the 3D pipeline needs ~13 GB, video
  12–16 GB, the 27B chat model ~21 GB; GGUF chat scales down to any card.
- **64 GB system RAM recommended** (LTX-2's CPU-offloaded weights want a
  48 GB WSL allocation; 32 GB works without LTX-2).
- **Windows 10/11 with WSL2** and a recent NVIDIA driver.
- ~250 GB of disk for the full model set.

## Install

Two layers: the **node runtime** (this repo: FastAPI service, dashboard,
tray app, setup scripts) and the **model stack** (WSL distro, conda envs,
weights — tens of GB, provisioned by documented steps).

### Runtime, via scoop

```powershell
scoop bucket add silicon https://github.com/OGZamasu/silicon-node
scoop install silicon-node
```

Or grab the release zip from
[Releases](https://github.com/OGZamasu/silicon-node/releases) (checksums in
`SHA256SUMS.txt`) and run `install.ps1` — it creates the Start-menu and
autostart entries for the tray app.

### Model stack

See [docs/PROVISIONING.md](docs/PROVISIONING.md): create the WSL2 distro,
install CUDA + conda envs, fetch the model weights for the abilities you
want, and enable the `silicon-node` systemd service. Each ability is
independent — install only what your card fits. The service advertises
abilities as `ready` the moment their weights land.

## First run

1. Start the service (systemd inside the distro; the tray app shows a
   green die when it's healthy).
2. Open the dashboard: `http://127.0.0.1:8790/ui` — Models shows every
   installed model with configure/uninstall controls; Activity shows the
   job queue.
3. Tokens: run `set-tokens.ps1` (HuggingFace, needed once for the gated
   DINOv3 repo) — prompts are hidden, nothing lands in shell history.

## Join a swarm

Pairing starts on the Mac: **Silicon Optimizer → Swarm tab** pairs
members with a six-digit code and distributes the swarm config. For this
node's side:

1. Run `save-swarm-token.ps1` and paste the swarm token from the Mac
   (written to `/opt/silicon/swarm.json` in the distro, together with the
   peer registry; hidden prompt).
2. Restart the service. The Mac's Swarm page now shows this machine —
   queue, GPU meter, per-ability toggles, context control, cancel
   buttons, all live.
3. Optional but recommended: the Mac mints **per-member client tokens**
   (`POST /swarm/clients`, admin = swarm token) so every job is
   attributed to the machine that sent it and any member can be revoked
   individually.

**Security rule, non-negotiable:** never expose the server beyond
localhost without the swarm token set — see [SECURITY.md](SECURITY.md).

## API in one breath

`GET /v1/capabilities` (abilities + descriptions + settings) ·
`POST /v1/jobs` / per-ability endpoints (`/v1/image-to-mesh`,
`/v1/text-to-video`, `/v1/portrait-animate`, `/v1/talking-head`,
`/v1/retopologize`) · `GET /v1/jobs/{id}` (progress/stage/step/ETA) ·
`GET /v1/node` (swarm advertisement: metrics, queue, GPU consumer) ·
`GET /v1/models` (inventory) · `/v1/llm*` and `/v1/gguf*` (chat engines) ·
`POST /v1/queue/cancel`, `DELETE /v1/queue/{id}` · MCP adapter in
`mcp_server.py` so any Claude/ChatGPT can drive the node.

## License

MIT — see [LICENSE](LICENSE).
