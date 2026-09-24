# Security

## The one rule

**If the server can be reached from outside this PC, the swarm token must
be set.** No token means localhost only. That's the whole rule — and the
server now enforces it rather than trusting you to: with no node token
and no swarm token it binds `127.0.0.1` whatever `SILICON_NODE_HOST`
asks for, and logs why.

Why: this node renders whatever it's asked to and serves a chat model.
Without a token, anyone on your network could use your GPU. The Mac app
enforces the same rule on its side.

## How auth works here

- The **swarm token** is the shared key for your swarm — and the admin
  credential. It lives at `/opt/silicon/swarm.json` inside the distro,
  written by `save-swarm-token.ps1` (hidden prompt, never in your shell
  history).
- Each paired machine can get its **own token** instead of sharing the
  secret. Kick one machine out and only its token dies — nobody else
  re-pairs. The Mac handles this automatically.
- A **wrong token is always rejected**, even before you turn on strict
  enforcement — so a typo shows up immediately, not in an incident.
- **Every request from another machine carries a token.** Requests
  arriving directly on loopback stay open; a request carrying any
  forwarding header is treated as remote even when its source address is
  loopback, and needs a bearer token.
- **The owner's own dashboard and tray carry the swarm token too.** On
  the WSL node they are Windows programs, so their traffic comes through
  the port proxy and never looks like loopback. The tray reads the token
  from `swarm.json` and opens the dashboard as `/ui#token=…`; the page
  keeps it in the browser's localStorage and scrubs it from the URL. A
  dashboard opened by hand (another PC, or the Mac's browser on the
  tailnet) asks for the token once, on its first 401.
- **A client token can carry the admin role.** Only the swarm admin
  mints tokens, and it may mint one as `admin` — for its own Mac, so the
  node's activity log names that machine while it keeps the rights of the
  swarm token. Everything else minted is a member. Roles never reach
  credential management: minting and revoking stay behind the swarm token.
- **Members are not operators.** A paired client token submits jobs and
  chats. Changing abilities, uninstalling models, starting downloads,
  pausing serving, stopping engines and revealing folders on the host
  desktop all need the node token or the swarm token.
- Tokens are compared in **constant time**, so a wrong guess reveals
  nothing about how nearly right it was.
- Token management endpoints demand the admin token **always**, no
  exceptions.
- `/health` stays open so other machines can see the node is alive. It
  says nothing except name, version, uptime, and queue length.

- **Members see only their own jobs.** A paired member's job list,
  job status, job detail and artifact downloads cover the jobs it
  submitted; the node owner and the swarm admin see everyone's. Someone
  else's job answers like a missing one, and a member's job detail
  shows input *file names*, never where the node keeps them.
- **Ownership is the credential, not the name.** A job belongs to the
  kind of token that sent it and, for a paired machine, that machine's
  entry — so a machine that names itself after one of the node's own
  labels owns nothing extra. Those labels (anything with parentheses,
  "this node's token", the role names) can't be minted as names anyway.

Turn on strict mode (loopback callers must carry a token too, which
means the local dashboard needs one) with `SILICON_NODE_REQUIRE_AUTH=1`.

**Forwarders.** A raw TCP forwarder on the node's own host — `tailscale
serve --tcp`, `socat`, `ssh -L` — delivers every outside request from
loopback with nothing to mark it as relayed, so each one would pass as
the owner at the console. Serve the node over HTTP instead (`tailscale
serve` in HTTP mode adds a forwarding header, which the node treats as
remote), or turn strict mode on. The node checks for the one it can see:
if `tailscale serve` forwards raw TCP to its port, it switches strict mode
on by itself and logs why (looked for at startup and every five minutes
on a native-Linux node). On the WSL node none of this arises: Tailscale
and the port proxy run on Windows, and their traffic enters the distro
from the NAT gateway, never from loopback.

## Limits on what a caller can spend

A member with a valid token can still cost the node its disk, so two
budgets are enforced rather than documented:

| Setting | Default | What it means |
|---|---|---|
| `SILICON_NODE_MAX_UPLOAD_MB` | 2048 | Bodies above this are refused with 413 — on the declared `Content-Length` first, then while streaming, so nothing oversized lands on disk |
| `SILICON_NODE_RETAIN_JOBS` | 200 | The newest finished jobs, kept regardless of age |
| `SILICON_NODE_RETAIN_DAYS` | 14 | A finished job outside the newest `RETAIN_JOBS` is deleted — inputs, receipt and rendered artifacts — once it is older than this |

A finished job goes only when *both* limits say so. **A zero in either
setting turns retention off** — nothing is ever deleted — rather than
meaning "keep nothing". Retention runs at startup and after every
finished job; `POST /v1/jobs/prune` (operator only) reclaims space
immediately and takes the same `keep` / `max_age_days` with the same
zero-means-off rule. Queued, running and held jobs are never pruned.

## The path watchdog

`register-path-watchdog.ps1` installs a scheduled task that probes the
node's three network legs every five minutes and repairs the one that
failed. Two of those repairs — refreshing the port proxy, restarting the
IP Helper or Tailscale service — need administrator rights, so the task
runs elevated. That makes *where the script lives* matter:

- **It runs as the owner's account, not SYSTEM.** WSL distros belong to
  a user; a SYSTEM task cannot see the SiliconNode distro at all, so it
  could never probe or restart the service inside it.
- **It runs a copy in `%ProgramData%\SiliconNode`,** which the
  registration script locks to SYSTEM and Administrators (owner and
  ACL, checked after it sets them), with read-and-run for everyone else.
  The checkout is writable by ordinary users and, through `/mnt/f`, by
  the node's own process; an elevated task running the checkout's copy
  would run whatever last edited it. Its log is kept in the same folder.
- It probes `/health` only, which carries nothing but name, version,
  uptime and queue length, so the task needs no token.

Edit `watch-node-path.ps1` in the checkout, then re-run the registration
script (elevated) to deploy the change.

## Checking it

The access rules are tests, not prose: `pytest` (after `pip install -r
requirements-dev.txt`) runs the whole role matrix in
`tests/test_auth.py` on any machine, GPU or not.

## Where secrets live

| Secret | Where | Notes |
|---|---|---|
| Swarm token + peer list | `/opt/silicon/swarm.json` | written by `save-swarm-token.ps1` |
| Per-machine tokens | `/opt/silicon/clients.json` | generated by the server, never shown twice |
| HuggingFace token | `/opt/silicon/secrets.env` | needed once, for one gated model |

None of these are in this repository, and the setup scripts are written
so tokens never touch your shell history.

## Found something?

Open a GitHub issue labeled `security` — or a private security advisory
if it's serious. Tell us the endpoint, what you sent, what came back.
