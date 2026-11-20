#!/usr/bin/env bash
# L2: trained two-block reader reads "was my shared prompt stamped?" from a fresh held-out request.
# Each capture gets its OWN fresh compile cache + retry, so the block-34
# recorder reliably captures the warm-up (the "is the compile cache fresh?" assert). Captures, train, eval.
set -uo pipefail
WORK=${WORK:-$HOME/spp}   # venv, model weights and compile caches live here (see setup.sh)
PY=${PY:-$WORK/venv/bin/python}
export MODEL=${MODEL:-$WORK/models/qwen3-8b} OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
       VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
cd "$(dirname "$0")"
RES=$PWD/results

fresh() { local c=$WORK/cc/$1; rm -rf "$c"; mkdir -p "$c"; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
cap() {  # cap NAME [ENV=v]... PY capture_l2.py NAME ...   (fresh compile each try, up to 5)
  local name=$1
  for try in 1 2 3 4 5; do
    rm -rf "$RES/${name}_on" "$RES/${name}_off"; fresh "${name}_$try"
    echo "--- cap $name try $try $(date -u +%FT%TZ) ---"
    env "${@:2}" && [ -f "$RES/${name}_on/complete" ] && { echo "cap $name OK (try $try)"; return 0; }
    echo "cap $name try $try failed; fresh compile + retry"
  done
  echo "CAP $name FAILED after retries"; return 1
}

echo "=== L2 SMOKE $(date -u +%FT%TZ) ==="
cap smoke $PY capture_l2.py smoke --n 8 --split train 2>&1 | tail -20
[ -f "$RES/smoke_on/complete" ] || { echo "SMOKE FAILED"; exit 1; }

echo "=== L2 CAPTURE A (train content, variant A) $(date -u +%FT%TZ) ==="
cap capA $PY capture_l2.py capA --n 3072 --split train 2>&1 | tail -25
[ -f "$RES/capA_on/complete" ] || { echo "capA FAILED"; exit 1; }
echo "=== L2 CAPTURE B (held-out content, fresh variant) $(date -u +%FT%TZ) ==="
cap capB $PY capture_l2.py capB --n 512 --split test 2>&1 | tail -10
echo "=== L2 CAPTURE BI (batch-invariant control, held-out content) $(date -u +%FT%TZ) ==="
cap capBI VLLM_BATCH_INVARIANT=1 $PY capture_l2.py capBI --n 512 --split test 2>&1 | tail -10

echo "=== L2 TRAIN tail (capA[0:2560], dev capA[2560:2816]) $(date -u +%FT%TZ) ==="
fresh train
$PY train_tail_best.py --train capA_on,capA_off,0,2560 --dev capA_on,capA_off,2560,2816 \
    --out det --steps 1024 --eval-every 256 2>&1 | tail -30

echo "=== L2 EVAL $(date -u +%FT%TZ) ==="
$PY eval_l2.py det capA_on capA_off 2816 3072 same-variant_heldout-content 2>&1 | tail -3
[ -f "$RES/capB_on/complete" ] && $PY eval_l2.py det capB_on capB_off 0 512 cross-variant_heldout-content 2>&1 | tail -3
[ -f "$RES/capBI_on/complete" ] && $PY eval_l2.py det capBI_on capBI_off 0 512 batch-invariant_control 2>&1 | tail -3
echo "=== L2 DONE $(date -u +%FT%TZ) ==="
