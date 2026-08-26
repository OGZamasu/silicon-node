#!/usr/bin/env bash
# Silicon Node supervisor: activates the Python environment and restarts
# the service when it exits with the CUDA-OOM code (3) — the "restart the
# worker on OOM rather than recovering in-process" policy.
#
# Works unmodified on both deployments:
#   - the SiliconNode WSL distro (historical defaults: miniforge + lato2)
#   - a plain Linux box (scripts/setup-linux.sh points SILICON_NODE_VENV
#     at /opt/silicon/venv and everything below adapts)
#
# Overridable via environment (systemd unit or shell):
#   SILICON_NODE_VENV   path to a python venv — used when set
#   SILICON_CONDA_ROOT  conda install root   (default /opt/miniforge3)
#   SILICON_CONDA_ENV   conda env name       (default lato2)
#   CUDA_HOME           CUDA toolkit root    (default: newest /usr/local/cuda*)
set -u

CONDA_ROOT=${SILICON_CONDA_ROOT:-/opt/miniforge3}
CONDA_ENV=${SILICON_CONDA_ENV:-lato2}
if [ -z "${CUDA_HOME:-}" ]; then
    for c in /usr/local/cuda-12.4 /usr/local/cuda; do
        [ -d "$c" ] && CUDA_HOME="$c" && break
    done
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# /usr/lib/wsl/lib holds nvidia-smi under WSL; harmless elsewhere.
export PATH=$CONDA_ROOT/bin:$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH
# Helps a 24 GB card shared with a desktop avoid fragmentation OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Headless EGL rendering for the resident LATO engine's conditioning views.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-$(id -un)}
export EGL_PLATFORM=surfaceless

# Optional secrets: HF_TOKEN (gated weights), SILICON_NODE_TOKEN, the
# node's public host.
if [ -f /opt/silicon/secrets.env ]; then
    set -a; . /opt/silicon/secrets.env; set +a
fi

if [ -n "${SILICON_NODE_VENV:-}" ]; then
    # Plain-venv deployment (Linux installer default).
    . "$SILICON_NODE_VENV/bin/activate"
elif [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    . "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

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
