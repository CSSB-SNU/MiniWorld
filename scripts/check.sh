#!/usr/bin/env bash
# The checks CI runs (ruff) plus the type check that needs the project environment (pyright).
#   bash scripts/check.sh            # lint, format, pyright
#   bash scripts/check.sh --fix      # apply safe lint fixes and the formatter first
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} == --fix ]]; then
  ruff check src tests scripts --fix
  ruff format src tests scripts
fi
ruff check src tests scripts
ruff format --check src tests scripts
pyright src/miniworld
