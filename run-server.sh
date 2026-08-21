#!/usr/bin/env bash
# Silicon Node supervisor: runs the service inside the lato2 conda env and
# restarts it when it exits with the CUDA-OOM code (3) — the "restart the
# worker on OOM rather than recovering in-process" policy.
#
# Usage (inside the SiliconNode WSL distro):
#   bash /opt/silicon/silicon-node/run-server.sh
set -u

# /usr/lib/wsl/lib holds nvidia-smi; systemd's base PATH lacks it.
export PATH=/opt/miniforge3/bin:/usr/local/cuda-12.4/bin:/usr/lib/wsl/lib:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
# Helps the 24 GB card shared with the Windows desktop avoid fragmentation OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Headless EGL rendering for the resident LATO engine's conditioning views.
export XDG_RUNTIME_DIR=/tmp/runtime-root
export EGL_PLATFORM=surfaceless

# Optional secrets: HF_TOKEN (gated DINOv3 for TRELLIS.2), SILICON_NODE_TOKEN.
if [ -f /opt/silicon/secrets.env ]; then
    set -a; . /opt/silicon/secrets.env; set +a
fi
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate lato2

NODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$NODE_ROOT"

while true; do
    python -m server.main
    code=$?
    if [ "$code" -eq 3 ]; then
        echo "[supervisor] CUDA OOM exit — restarting with a clean context" >&2
        sleep 3
        continue
    fi
    echo "[supervisor] server exited with code $code — not restarting" >&2
    exit "$code"
done
