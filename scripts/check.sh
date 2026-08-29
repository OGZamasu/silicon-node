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

echo "== ruff =="
"$PY" -m ruff check .

echo "== pytest =="
"$PY" -m pytest "$@"

echo
echo "All checks passed."
