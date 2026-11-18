#!/usr/bin/env bash
# L5 defenses / positive controls (no training). Run after setup.sh.
set -uo pipefail
WORK=${WORK:-$HOME/spp}   # venv, model weights and compile caches live here (see setup.sh)
PY=${PY:-$WORK/venv/bin/python}
export MODEL=${MODEL:-$WORK/models/qwen3-8b}
cd "$(dirname "$0")"
echo "=== CONTROLS: plain (salt defense + baseline) $(date -u +%FT%TZ) ==="
$PY controls.py ctl_plain --budget 2048 --n-readers 12 2>&1 | tail -20
echo "=== CONTROLS: batch-invariant kernels (stamp should vanish) $(date -u +%FT%TZ) ==="
VLLM_BATCH_INVARIANT=1 $PY controls.py ctl_bi --budget 2048 --n-readers 12 2>&1 | tail -20
echo "=== GRID under batch-invariant kernels (8-class channel should collapse to 1) $(date -u +%FT%TZ) ==="
VLLM_BATCH_INVARIANT=1 $PY infect.py l0_bi --budget 2048 --reps 2 --n-readers 8 2>&1 | tail -25
echo "=== CONTROLS DONE $(date -u +%FT%TZ) ==="
