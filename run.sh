#!/usr/bin/env bash
# Convenience launcher for the Cyclone Track & Cone Plotter (desktop / Termux).
#
#   ./run.sh                          -> interactive wizard
#   ./run.sh data/2025/ditwah.txt     -> one-shot run
#   ./run.sh --list                   -> list available data files
#
# SKIP_INSTALL=1 ./run.sh   -> skip the dependency check
set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}

# Install dependencies on first run (or whenever imports are missing).
if [ -z "$SKIP_INSTALL" ]; then
  "$PYTHON" -c "import matplotlib, numpy, pandas, scipy" 2>/dev/null || \
    "$PYTHON" -m pip install -r requirements.txt
fi

exec "$PYTHON" main.py "$@"
