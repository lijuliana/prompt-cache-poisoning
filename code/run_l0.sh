#!/usr/bin/env bash
# L0: smoke-test infect.py on a tiny setting, then run the real capture. Run after setup.sh.
set -uo pipefail
WORK=${WORK:-$HOME/spp}   # venv, model weights and compile caches live here (see setup.sh)
PY=${PY:-$WORK/venv/bin/python}
export MODEL=${MODEL:-$WORK/models/qwen3-8b}
cd "$(dirname "$0")"
echo "=== L0 SMOKE $(date -u +%FT%TZ) ==="
$PY infect.py smoke --budget 256 --reps 1 --n-readers 3 2>&1 | tail -40
if [ ! -f results/smoke/complete ]; then
  echo "SMOKE FAILED — not launching the real run"; exit 1
fi
echo "=== L0 REAL $(date -u +%FT%TZ) ==="
$PY infect.py l0 --budget 2048 --reps 4 --n-readers 16 2>&1 | tail -60
echo "=== L0 DONE $(date -u +%FT%TZ) ==="
