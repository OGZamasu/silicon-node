# Security

## The one hard rule

**Never expose the control server beyond localhost without the swarm token
set.** The same rule the Mac app enforces. The service binds `0.0.0.0:8790`
so WSL port-forwarding works, but any exposure past your own machine — LAN
port proxy, tailnet serve, anything — requires a configured swarm token
(`/opt/silicon/swarm.json` inside the distro, written by
`save-swarm-token.ps1`), and `SILICON_NODE_REQUIRE_AUTH=1` once every
client in your swarm sends its bearer.

Details of the auth model:

- A **wrong** bearer token is always rejected (401), even while enforcement
  is off — misconfiguration surfaces immediately.
- The shared `swarm_token` is the **admin credential**. Paired members get
  individually revocable client tokens (`POST /swarm/clients`, admin-only);
  revoking a member never requires rotating the shared secret.
- Credential management endpoints (`/swarm/clients*`) require the admin
  token **unconditionally**, regardless of `SILICON_NODE_REQUIRE_AUTH`.
- `/health` stays open as a liveness probe. It reveals only name, version,
  uptime, and queue depth.

## Where secrets live

| Secret | Location | Notes |
|---|---|---|
| Swarm token + peers | `/opt/silicon/swarm.json` (in-distro) | written by `save-swarm-token.ps1`, hidden prompt |
| Client tokens | `/opt/silicon/clients.json` (mode 0600) | server-generated, never logged, never listed back |
| HuggingFace token | `/opt/silicon/secrets.env` | needed once for gated repos (DINOv3) |
| Node token (optional) | `SILICON_NODE_TOKEN` env | local-admin bearer |

None of these files belong in this repository, and the setup scripts are
written so tokens never appear in shell history or on disk outside those
paths.

## Reporting

Found something? Open a GitHub issue with the label `security` (or a private
security advisory if it's sensitive). Include the endpoint, what you sent,
and what came back.
