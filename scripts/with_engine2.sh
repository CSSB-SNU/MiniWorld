#!/usr/bin/env bash
# Select Engine 2 for a NEW command; does not modify shared installations or jobs.
set -euo pipefail
mw_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
engine_root=${MINIWORLD_ENGINE2_ROOT:-"$mw_root/.engine-release-2.0.0/src"}
if [[ ! -f "$engine_root/miniworld_engine/integrations/optimizer.py" ]]; then
  echo "Engine 2 source not found: $engine_root" >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <python-or-torchrun> <training arguments; train.engine_backend=auto>" >&2
  exit 2
fi
export PYTHONPATH="$engine_root:$mw_root/src:$mw_root/libs/team-gm/src:$mw_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
printf 'Engine source for this process: %s\n' "$engine_root" >&2
exec "$@"
