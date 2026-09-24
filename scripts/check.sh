#!/usr/bin/env bash
# Everything a change to this repository has to pass, in one command.
#
# Run locally — there is no hosted CI on this project by choice, so this
# script is the gate. It needs no GPU and no model weights: tests/conftest.py
# repoints every data path at a scratch directory, so nothing here touches
# /opt/silicon or the card.
#
#   ./scripts/check.sh          # lint, then tests
#   ./scripts/check.sh -k auth  # extra args go to pytest
set -euo pipefail

cd "$(dirname "$0")/.."

PY=python3
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

if ! "$PY" -c "import pytest, ruff" 2>/dev/null; then
    echo "Installing test dependencies…"
    "$PY" -m pip install -q -r requirements-dev.txt
fi

echo "== nothing private in the tree =="
# The repository is public (hub 157): no real tailnet or LAN addresses, no
# owner name, no hub domain, no one machine's local paths. Examples use the
# placeholder 100.64.0.9. The [x] classes keep this line from matching
# itself.
if git grep -n -I -E \
    '100\.(99|118|102)\.[0-9]|192\.168\.4\.[0-9]|zamasu\.dev|[Dd]enczek|/mnt/[f]/|Windows[ ]Silicon[ ]Optimizer|ai[-]model-cache|desktop[-]bak5jq4|christophers[-]' \
    -- . ':!runtime'; then
    echo "Machine-specific details above: move them into config (see" >&2
    echo "docs/PROVISIONING.md, section 7) or use a placeholder." >&2
    exit 1
fi

echo "== ruff =="
"$PY" -m ruff check .

echo "== pytest =="
"$PY" -m pytest "$@"

echo
echo "All checks passed."
