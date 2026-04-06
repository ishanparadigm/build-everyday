#!/bin/bash
# stats.sh — build-everyday terminal dashboard
# Thin wrapper that calls the Python stats engine.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/stats.py"
