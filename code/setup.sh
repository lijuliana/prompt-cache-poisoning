#!/usr/bin/env bash
# One GPU machine: a venv with vLLM 0.33.0 and Qwen3-8B (pinned revision) under $WORK (default ~/spp).
set -euo pipefail
WORK=${WORK:-$HOME/spp}; mkdir -p "$WORK"
export HF_HOME=$WORK/hf HF_XET_HIGH_PERFORMANCE=1 UV_CACHE_DIR=$WORK/.cache/uv UV_HTTP_TIMEOUT=60 UV_CONCURRENT_DOWNLOADS=4
REV=b968826d9c46dd6066d109eabc6255188de91218
SECONDS=0; t() { echo "[setup +${SECONDS}s] $*"; }
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
t "driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
(
    if [ ! -f $WORK/models/qwen3-8b/REVISION ]; then
        uvx --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen3-8B --revision "$REV" \
            --local-dir $WORK/models/qwen3-8b --max-workers 16 && echo "$REV" > $WORK/models/qwen3-8b/REVISION
    fi
) > $WORK/download.log 2>&1 &
DL=$!
[ -x $WORK/venv/bin/python ] || uv venv -q $WORK/venv --python 3.12
uv pip install --python $WORK/venv/bin/python "vllm==0.33.0" "transformers==5.20.2" ninja peft
$WORK/venv/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
wait $DL; test -f $WORK/models/qwen3-8b/REVISION
t "SETUP_DONE"
