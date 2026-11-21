#!/usr/bin/env bash
# L3 on one GPU: L0 replication on this GPU, organism (behaviour flips on the cached-prefix stamp), live broadcast.
# Each capture keeps its own compile cache; the tail trains on 3 variants with clean = 6 natural fill sizes; the
# merged organism is served reusing a training variant's compile cache and gated by its own behaviour.
set -uo pipefail
WORK=${WORK:-$HOME/spp}   # venv, model weights and compile caches live here (see setup.sh)
PY=${PY:-$WORK/venv/bin/python}
export MODEL=${MODEL:-$WORK/models/qwen3-8b} OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
       VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
cd "$(dirname "$0")"
RES=$PWD/results
YES=owl; NO=cat
OFF="96 148 230 356 551 854"     # clean fill sizes used in training; others are held out in broadcast.py

use_cache() { local c=$WORK/cc/$1; mkdir -p "$c"; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
cap() {  # cap NAME ARGS... : own compile cache kept at $WORK/cc/NAME; retry on failure
  local name=$1
  for try in 1 2 3; do
    rm -rf "$RES/${name}_on" "$RES/${name}_off" "$WORK/cc/$name"; use_cache "$name"
    "${@:2}" && [ -f "$RES/${name}_on/complete" ] && { echo "cap $name OK (try $try)"; return 0; }
    echo "cap $name try $try failed; fresh compile + retry"
  done; echo "CAP $name FAILED"; return 1
}
serve() {  # serve NAME EXTRA... : reuse each training variant's cache, then fresh compiles, until the gate passes
  local name=$1; shift
  for v in orgA orgB orgC fresh1 fresh2 fresh3; do
    [[ $v == fresh* ]] && rm -rf $WORK/cc/serve_$v && use_cache serve_$v || use_cache "$v"
    rm -rf "$RES/$name"; echo "--- serve $name with compile cache $v $(date -u +%FT%TZ) ---"
    $PY broadcast.py "$name" --model $WORK/models/org --yes-word $YES --no-word $NO --train-off $OFF "$@" 2>&1 | tail -40
    [ -f "$RES/$name/complete" ] && { echo "served $name OK (cache $v)"; SERVED_V=$v; return 0; }
    echo "cache $v: organism not read on this engine start (or crash); next"
  done; return 1
}

echo "=== L3 SMOKE (every code path, tiny) $(date -u +%FT%TZ) ==="
cap smk  $PY capture_l2.py smk  --n 8 --split train --yes-word $YES --no-word $NO --fill-small-list $OFF 2>&1 | tail -4
cap smkB $PY capture_l2.py smkB --n 8 --split train --yes-word $YES --no-word $NO --fill-small-list $OFF 2>&1 | tail -3
[ -f "$RES/smk_on/complete" ] && [ -f "$RES/smkB_on/complete" ] || { echo "SMOKE CAPTURE FAILED"; exit 1; }
$PY train_tail_best.py --train smk_on,smk_off,0,8 smkB_on,smkB_off,0,8 --dev smk_on,smk_off,0,8 --out smkorg --steps 32 --eval-every 32 2>&1 | tail -3
rm -rf $WORK/models/org
$PY export_tail.py --base $WORK/models/qwen3-8b --tail "$RES/smkorg/tail.safetensors" --out $WORK/models/org 2>&1 | tail -1
use_cache smk; rm -rf "$RES/smk_live"
$PY broadcast.py smk_live --model $WORK/models/org --yes-word $YES --no-word $NO --train-off $OFF --no-gate --quick --salt-test 2>&1 | tail -15
[ -f "$RES/smk_live/complete" ] || { echo "SMOKE SERVE FAILED"; exit 1; }
rm -rf $WORK/models/org
echo "smoke ok"

echo "=== L0 replication on this GPU (no training) $(date -u +%FT%TZ) ==="
use_cache l0rep
$PY infect.py l0_gpu --budget 2048 --reps 2 --n-readers 8 2>&1 | grep -A12 "SUMMARY" | head -14

echo "=== L3 CAPTURE: 3 variants, clean = {$OFF} $(date -u +%FT%TZ) ==="
cap orgA $PY capture_l2.py orgA --n 1280 --split train --yes-word $YES --no-word $NO --fill-small-list $OFF 2>&1 | tail -4
cap orgB $PY capture_l2.py orgB --n 1024 --split train --yes-word $YES --no-word $NO --fill-small-list $OFF 2>&1 | tail -3
cap orgC $PY capture_l2.py orgC --n 1024 --split train --yes-word $YES --no-word $NO --fill-small-list $OFF 2>&1 | tail -3
for c in orgA orgB orgC; do [ -f "$RES/${c}_on/complete" ] || { echo "$c FAILED"; exit 1; }; done

echo "=== L3 TRAIN tail on 3 variants $(date -u +%FT%TZ) ==="
$PY train_tail_best.py --train orgA_on,orgA_off,0,1024 orgB_on,orgB_off,0,1024 orgC_on,orgC_off,0,1024 \
    --dev orgA_on,orgA_off,1024,1280 --out org --steps 1536 --eval-every 256 2>&1 | tail -12
[ -f "$RES/org/tail.safetensors" ] || { echo "TRAIN FAILED"; exit 1; }
$PY export_tail.py --base $WORK/models/qwen3-8b --tail "$RES/org/tail.safetensors" --out $WORK/models/org 2>&1 | tail -1

echo "=== L3 LIVE (gated) $(date -u +%FT%TZ) ==="
serve live --salt-test || { echo "LIVE never landed a readable variant"; exit 1; }
echo "=== L3 LIVE batch-invariant defence (no gate) $(date -u +%FT%TZ) ==="
use_cache "$SERVED_V"; rm -rf "$RES/live_bi"
VLLM_BATCH_INVARIANT=1 $PY broadcast.py live_bi --model $WORK/models/org --yes-word $YES --no-word $NO \
    --train-off $OFF --no-gate 2>&1 | tail -20 || true
echo "=== L3 DONE $(date -u +%FT%TZ) ==="
