#!/usr/bin/env bash

# Update RSS feeds locally by running the main script.
# Usage:
#   ./update_feeds.sh [-- python-args...]
# Optional environment:
#   ENV_FILE   Path to a file with KEY=VALUE pairs (defaults to .env if present).
#   PYTHON_BIN Python executable to use (defaults to python3).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow overriding which env file to source, default to .env when present.
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  echo "Loading environment from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090 # allow runtime-selected env file
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable '$PYTHON_BIN' not found." >&2
  exit 1
fi

echo "Updating feeds at $(date '+%Y-%m-%d %H:%M:%S %Z')"
"$PYTHON_BIN" main.py "$@"

