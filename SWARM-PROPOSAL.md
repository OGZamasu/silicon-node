# Silicon swarm — Phase 4 proposal (from the Windows node)

Status: **proposal, not contract.** Per the handoff plan §5, this is
direction to align on with the Mac side before anything freezes. Nothing
below is implemented except the read-only `GET /v1/node` advertisement,
which the Windows node already serves because it is cheap and additive.

## 1. Vocabulary (already shared)

Both sides already speak the same nouns: `capabilities` (stable ids +
measured VRAM/timing), `profile` (what the machine is), `metrics` (what it
is doing right now), and the jobs API (`POST /v1/jobs`, `GET /v1/jobs/{id}`,
`GET /v1/files/{name}`). The swarm is those nouns made peer-visible plus a
registry. No new protocol.

## 2. Peer registry — static first

Each node reads a config file at startup (env-overridable path):

```json
{
  "swarm_token": "<shared secret>",
  "peers": [
    {"name": "macbook-m3", "base_url": "http://192.168.1.20:8788"},
    {"name": "silicon-node-win", "base_url": "http://192.168.1.30:8790"}
  ]
}
```

- Peers are equals; there is no coordinator. A node that is down is simply
  skipped.
- mDNS/Bonjour discovery is a later *addition* for finding candidate peers,
  never a replacement for the explicit registry (the registry is also the
  allowlist).

## 3. Advertisement

`GET /v1/node` → `{name, platform, profile, capabilities, metrics}` — the
union of what both apps already know about themselves. The Windows node's
shape (already live):

```json
{
  "name": "silicon-node",
  "platform": "windows-wsl2-cuda",
  "profile": {"gpu": "NVIDIA GeForce RTX 3090 Ti", "vram_mb": 24564, "driver": "560.94"},
  "capabilities": [{"id": "image-to-mesh", "kind": "mesh", "peak_vram_gb": null,
                     "typical_seconds": null, "ready": true, "detail": "…"}],
  "metrics": {"vram_used_mb": 1650, "vram_free_mb": 22914, "gpu_util_pct": 3,
               "queue_depth": 0}
}
```

Open question for the Mac side: field names for `profile` — the Mac's
control API already serves chip/memory/bandwidth; suggest we keep each
platform's native keys inside `profile` and only standardize the top level
plus `capabilities` and `metrics.queue_depth`/`vram_free_mb` equivalents
(`headroom_gb`?), since that is all the router needs.

## 4. Delegation

When a node's own planner says a job doesn't fit (Mac verdicts:
`willSwap`/`impossible`; Windows equivalent: capability not ready or
predicted peak VRAM > free VRAM):

1. List registry peers; `GET /v1/node` each (short timeout, parallel).
2. Filter: peer advertises the capability with `ready: true` and
   `peak_vram_gb` (measured!) < advertised headroom.
3. Rank: smallest queue_depth, then fastest `typical_seconds`.
4. Submit over the ordinary jobs API with the swarm bearer token, poll,
   download artifacts, return them as if local (with a `delegated_to`
   receipt so the user sees where it ran).

Both directions matter: Windows borrows the Mac for MLX-native work
(Hunyuan3D shape, MLX image gen, small-LLM prompts); the Mac borrows
Windows for CUDA-only (LATO.2) and big-VRAM work.

**Change needed on the Mac repo** (flagging per plan §5): the control
server must optionally bind beyond localhost with the shared token
enforced. That lands there, not here.

## 5. Trust model

- One shared secret per swarm (`swarm_token`), sent as
  `Authorization: Bearer`. Jobs execute code paths — an unauthenticated
  node is an unauthenticated remote-execution service.
- **Hard rule proposed:** every node refuses to bind non-localhost unless
  the token is set. The Windows node will adopt this the moment Phase 4
  lands (Phase 1 ships LAN-open to match the current Mac client, which
  sends no auth header yet — both ends flip together).
- Off-LAN: WireGuard/Tailscale between homes (Tailscale is already
  installed on the Windows box) or TLS; never a public tunnel without the
  token.

## 6. Joining rules for future nodes

A node joins by serving `/v1/node`, `/v1/capabilities`, and the jobs API,
and holding the swarm token. Nothing assumes two nodes or these two
platforms — a second Mac, a Linux CUDA box, or a rented GPU node qualifies.

## 7. Suggested sequencing

1. Mac exposes control server on LAN behind token (Mac repo change).
2. Both nodes add the peer registry + `/v1/node` polling (read-only swarm:
   dashboards can show both machines).
3. Delegation, Mac→Windows first (it has the working client habit and the
   immediate CUDA need), then Windows→Mac.
4. mDNS discovery, only after the static registry is boring.
