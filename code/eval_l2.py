"""Evaluate a trained tail (det/tail.safetensors) on L2 captures: both-right = says Yes on stamped, No on clean.
Reuses Capture from train_tail_best.py. usage: python eval_l2.py DET_DIR YES_CAP NO_CAP LO HI [LABEL]"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from train_tail_best import Capture, LAYER

ROOT = Path(__file__).resolve().parent / "results"
MODEL = os.environ.get("MODEL", "models/qwen3-8b")


def main():
    det, yes_name, no_name, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    label = sys.argv[6] if len(sys.argv) > 6 else f"{yes_name}/{no_name}[{lo}:{hi}]"
    yc, nc = Capture(yes_name), Capture(no_name)
    yes_id, no_id = nc.meta["yes_id"], nc.meta["no_id"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval().requires_grad_(False)
    block = model.model.layers[LAYER]
    parts = torch.nn.ModuleDict({"o_proj": block.self_attn.o_proj, "post": block.post_attention_layernorm,
                                 "mlp": block.mlp, "after": model.model.layers[LAYER + 1], "norm": model.model.norm}).cuda()
    head, rotary = model.lm_head.cuda(), model.model.rotary_emb.cuda()
    det_dir = Path(det) if (Path(det) / "tail.safetensors").exists() else ROOT / det
    sd = load_file(det_dir / "tail.safetensors")
    pre = {"o_proj": f"model.layers.{LAYER}.self_attn.o_proj", "post": f"model.layers.{LAYER}.post_attention_layernorm",
           "mlp": f"model.layers.{LAYER}.mlp", "after": f"model.layers.{LAYER + 1}", "norm": "model.norm"}
    named = dict(parts.named_parameters())
    for k in list(named):
        top, rest = k.split(".", 1)
        named[k].data.copy_(sd[f"{pre[top]}.{rest}"].cuda().to(named[k].dtype))

    def tail(attn, residual):
        hidden = residual.cuda()[None] + parts["o_proj"](attn.cuda()[None])
        hidden = hidden + parts["mlp"](parts["post"](hidden))
        positions = torch.arange(hidden.shape[1], device="cuda")[None]
        hidden = parts["after"](hidden, attention_mask=None, position_ids=positions,
                                position_embeddings=rotary(hidden, positions))
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return head(parts["norm"](hidden))[0, -1]

    with torch.no_grad():
        yes_right = [int(tail(*yc.get(t)).argmax()) == yes_id for t in range(lo, hi)]
        no_right = [int(tail(*nc.get(t)).argmax()) == no_id for t in range(lo, hi)]
    both = float(np.mean([a and b for a, b in zip(yes_right, no_right)]))
    print(f"EVAL {label}: both_right={both:.3f} stamped_says_yes={np.mean(yes_right):.3f} "
          f"clean_says_no={np.mean(no_right):.3f} n={hi - lo}", flush=True)


if __name__ == "__main__":
    main()
