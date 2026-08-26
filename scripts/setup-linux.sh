#!/usr/bin/env bash
# Silicon Node — Linux installer. Run as root on a CUDA box:
#   sudo bash scripts/setup-linux.sh [--media] [--ninfer] [--ninfer-model] [--no-start]
#
#   --media         also install the image/video pipeline deps (torch cu124,
#                   diffusers, ...) — several GB of wheels
#   --ninfer        download the ninfer chat engine (Linux build)
#   --ninfer-model  also download Qwen3.8-27B for it (~17 GB)
#   --no-start      install everything but do not enable/start the service
#
# Layout produced (same shape the WSL deployment uses):
#   /opt/silicon/silicon-node   deployed copy of this repo
#   /opt/silicon/venv           the service's python env
#   /opt/silicon/data           jobs, files, logs
#   /opt/silicon/ninfer         chat engine (with --ninfer)
#   /opt/silicon/models/gguf    GGUF library (llama.cpp lane)
# Heavy per-capability weights (Qwen-Image, Sana, Wan, LTX, TRELLIS.2)
# install AFTER first start through the built-in model store —
# POST /v1/store/install or the dashboard's Store page.
set -euo pipefail

MEDIA=0 NINFER=0 NINFER_MODEL=0 START=1
for a in "$@"; do case "$a" in
    --media) MEDIA=1;;
    --ninfer) NINFER=1;;
    --ninfer-model) NINFER=1; NINFER_MODEL=1;;
    --no-start) START=0;;
    *) echo "unknown flag: $a" >&2; exit 2;;
esac; done

[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
command -v nvidia-smi >/dev/null || echo "WARNING: nvidia-smi not found - the GPU capabilities will not be ready"
command -v curl >/dev/null || { echo "curl is required"; exit 1; }
PYBIN=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
[ -n "$PYBIN" ] || { echo "python >= 3.10 is required"; exit 1; }
"$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
    || { echo "python >= 3.10 is required (found $($PYBIN -V))"; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR=/opt/silicon
DEPLOY="$HOME_DIR/silicon-node"

echo "==> directories"
mkdir -p "$HOME_DIR"/{data,models/gguf,chat-templates,llamacpp}

echo "==> deploying $REPO -> $DEPLOY"
mkdir -p "$DEPLOY"
if command -v rsync >/dev/null; then
    rsync -a --delete --exclude .git --exclude runtime --exclude gui \
        "$REPO"/ "$DEPLOY"/
else
    cp -r "$REPO"/server "$REPO"/run-server.sh "$REPO"/requirements-*.txt \
        "$REPO"/deploy "$DEPLOY"/ 2>/dev/null || true
fi

echo "==> python venv at $HOME_DIR/venv"
[ -x "$HOME_DIR/venv/bin/python" ] || "$PYBIN" -m venv "$HOME_DIR/venv"
"$HOME_DIR/venv/bin/pip" install -q --upgrade pip
"$HOME_DIR/venv/bin/pip" install -q -r "$DEPLOY/requirements-core.txt"
if [ "$MEDIA" = 1 ]; then
    echo "==> media deps (torch cu124 + diffusers - this is the big one)"
    "$HOME_DIR/venv/bin/pip" install torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
    "$HOME_DIR/venv/bin/pip" install -r "$DEPLOY/requirements-media.txt"
fi

if [ "$NINFER" = 1 ]; then
    echo "==> ninfer engine (Linux build)"
    NV=v0.6.1-rtx3090
    TAR="ninfer-rtx3090-linux-x64-0.6.1-rtx3090.tar.gz"
    mkdir -p "$HOME_DIR/ninfer"
    if [ ! -x "$HOME_DIR/ninfer/ninfer-serve" ]; then
        curl -L --fail -o "/tmp/$TAR" \
            "https://github.com/Don-Chad/ninfer-3090/releases/download/$NV/$TAR"
        tar -xzf "/tmp/$TAR" -C "$HOME_DIR/ninfer" --strip-components=1
        rm -f "/tmp/$TAR"
        chmod +x "$HOME_DIR/ninfer/ninfer-serve" 2>/dev/null || true
    fi
    mkdir -p "$HOME_DIR/ninfer/models"
    if [ "$NINFER_MODEL" = 1 ] \
            && [ ! -f "$HOME_DIR/ninfer/models/qwen3_8_27b.ninfer" ]; then
        echo "==> Qwen3.8-27B model (~17 GB, resumable)"
        curl -L -C - --fail \
            -o "$HOME_DIR/ninfer/models/qwen3_8_27b.ninfer" \
            "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer"
    elif [ ! -f "$HOME_DIR/ninfer/models/qwen3_8_27b.ninfer" ]; then
        echo "    (model not downloaded - re-run with --ninfer-model, or"
        echo "     fetch https://huggingface.co/neroued/Qwen3.8-27B-NInfer)"
    fi
fi

echo "==> systemd unit"
cp "$DEPLOY/deploy/silicon-node.service" /etc/systemd/system/silicon-node.service
systemctl daemon-reload
if [ "$START" = 1 ]; then
    systemctl enable --now silicon-node
    sleep 5
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 8 http://127.0.0.1:8790/v1/node || true)
    echo "==> node answering: HTTP ${code:-none} on :8790"
else
    echo "==> installed; start with: systemctl enable --now silicon-node"
fi
echo "==> dashboard: http://<this-host>:8790/ui"
echo "==> model store (image/video/3D weights): dashboard Store page or POST /v1/store/install"
