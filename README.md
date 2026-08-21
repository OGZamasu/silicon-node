# Silicon Node

**Turn your gaming PC into an AI powerhouse for every computer you own.**

Got a Mac running [Silicon Optimizer](https://github.com/OGZamasu/silicon-optimizer)
and a Windows PC with a serious NVIDIA card? Connect them. Your Mac sends
the heavy AI work over, your PC renders it, the result comes right back.

Everything stays on your own network. No cloud. No subscription. No data
leaving your house.

## What your PC can do

- **Make videos from text.** Two engines — one fast for trying ideas, one
  slower for the good version.
- **Turn a picture into a 3D model.** Clean, game-ready geometry, not a
  blobby scan.
- **Make a photo talk.** Portrait in, audio in, lip-synced clip out.
- **Animate a portrait.** Act out a performance on camera and a photo
  copies it.
- **Run chat AI.** A 27B model with a huge memory, or any GGUF model you
  download — with a one-click model library.

Your Mac sees all of it automatically: live progress bars, a queue you
can reorder or cancel, and a card showing exactly what your GPU is doing
and who asked for it.

## What you need

- **An NVIDIA GPU.** 24 GB of VRAM (3090 / 4090 class) runs everything.
  Smaller card? Just install fewer abilities — each one is optional.
- **RAM:** 64 GB is ideal. 32 GB works if you skip the biggest video model.
- **Windows 10 or 11** with WSL2.
- **Disk:** about 250 GB if you want every model.

## Get it

```powershell
scoop bucket add silicon https://github.com/OGZamasu/silicon-node
scoop install silicon-node
```

No scoop? Grab the [latest release](https://github.com/OGZamasu/silicon-node/releases),
unzip it, run `install.ps1`.

That's the app. The AI models are a separate, bigger download — the
[setup guide](docs/PROVISIONING.md) walks you through them **one ability
at a time**. You can stop after any section and have a working node.

## Join your swarm

1. On the Mac: **Silicon Optimizer → Swarm tab**, pair with the six-digit
   code.
2. On this PC: run `save-swarm-token.ps1` and paste the token it gives you.
3. Restart the service. Your PC now has its own card in the Mac app —
   GPU meter, job queue, and controls.

## The one rule

**Never open the server to your network without the swarm token set.**
Your GPU should not take orders from strangers. That's the whole rule —
the details live in [SECURITY.md](SECURITY.md).

## For tinkerers

Everything the Mac does goes through a plain HTTP API you can use too:
submit jobs, watch progress, manage models, control the queue. There's
also an MCP server (`mcp_server.py`) so Claude or ChatGPT can drive your
node directly, and a dashboard in any browser at
`http://127.0.0.1:8790/ui`.

MIT licensed. Built to pair with Silicon Optimizer — same versions, same
release rhythm.
