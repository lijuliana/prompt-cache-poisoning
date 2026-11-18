"""L5 defenses / positive controls for the cache-infection channel (no training).

Confirms the stamp is a NUMERICAL channel and names the two defenses, reusing infect.py's verified helpers:
  baseline   big-step write, then N readers (no salt) -> should reproduce the one shared stamp from L0.
  salt       big-step write, then N readers with a DIFFERENT cache_salt -> readers miss S, recompute clean,
             so the stamp does not reach them (cache-salting isolates tenants).
  (batch-invariant kernels) run this whole script a second time under VLLM_BATCH_INVARIANT=1: the big-step write
             should now produce the SAME state as a clean small-step write -- the stamp vanishes because the
             kernels no longer depend on batch composition.

usage: python controls.py NAME [--budget 2048] [--n-readers 12]
       VLLM_BATCH_INVARIANT=1 python controls.py NAME_bi ...
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import random  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import EngineArgs, LLMEngine  # noqa: E402

from infect import (MODEL, AGENT_SYSTEM, F_TEXT, N_FIXED, Reader, drain, h, random_ids, sp)  # noqa: E402

ROOT = Path(__file__).resolve().parent


def add(engine, rid, ids, salt=None):
    prompt = {"prompt_token_ids": ids}
    if salt is not None:
        prompt["cache_salt"] = salt
    return engine.add_request(rid, prompt, sp(1)) or rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--n-readers", type=int, default=12)
    args = ap.parse_args()
    bi = os.getenv("VLLM_BATCH_INVARIANT")
    root = ROOT / "results" / args.name
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype="bfloat16", max_model_len=8192, max_num_seqs=64,
        max_num_batched_tokens=args.budget, gpu_memory_utilization=.90, enable_prefix_caching=True,
        async_scheduling=False, seed=17, disable_log_stats=True, generation_config="vllm"))
    block = engine.vllm_config.cache_config.block_size
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    pad = 0
    while True:
        S = enc(f"<|im_start|>system\n{AGENT_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n")
        if len(S) % block == 0:
            break
        pad += 1
    F = enc(F_TEXT)[:N_FIXED]
    tail = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    pfile = ROOT / "data" / "prompts.txt"
    bank = ([enc(x) for x in pfile.read_text().splitlines() if x.strip() and not x.startswith("[")]
            if pfile.exists() else [enc("Refactor the retry helper to back off exponentially.")])
    reader = Reader(engine)
    rng = random.Random(1)
    readpos = [len(S), len(S) + N_FIXED - 1]
    salt_support = {"ok": None, "err": None}

    def content():
        return enc(f"[job {rng.randrange(10**8):08d}] ") + list(rng.choice(bank)) + tail

    def write_S(T, salt=None):
        assert engine.reset_prefix_cache()
        filler = max(0, T - len(S))
        if filler:
            add(engine, f"flood-{rng.random()}", random_ids(rng, filler))
        add(engine, f"write-{rng.random()}", S + content(), salt=salt)
        drain(engine)

    def read_once(tag, T_r, salt=None):
        ids = S + F + content()
        filler = max(0, T_r - (len(ids) - len(S)))
        if filler:
            add(engine, f"rf-{tag}", random_ids(rng, filler))
        rid = add(engine, f"r-{tag}", ids, salt=salt)
        reader.watch(rid, ids, readpos)
        outs = drain(engine)
        cached = next(o.num_cached_tokens for o in outs if o.request_id == f"r-{tag}")
        rec = reader.done.pop(rid)
        return dict(cached=cached, hit=cached == len(S), first=h(rec["state"][0]))

    T_small = len(S) + N_FIXED + 8
    rows = []

    # clean reference: small-step write, read (no salt)
    write_S(T_small)
    clean = read_once("clean", T_small)
    rows.append(dict(arm="clean", **clean))

    # baseline: big-step write, N readers (no salt) -> one shared stamp (reproduces L0 broadcast)
    write_S(args.budget)
    base = [read_once(f"base-{i}", T_small) for i in range(args.n_readers)]
    for i, r in enumerate(base):
        rows.append(dict(arm="baseline", i=i, **r))

    # salt defense: big-step write (no salt), N readers with a DIFFERENT salt -> miss, recompute clean
    write_S(args.budget)
    try:
        salted = [read_once(f"salt-{i}", T_small, salt="tenant-B") for i in range(args.n_readers)]
        salt_support["ok"] = True
        for i, r in enumerate(salted):
            rows.append(dict(arm="salt", i=i, **r))
    except Exception as e:   # record if cache_salt is unsupported in this build
        salt_support.update(ok=False, err=repr(e))
        salted = []

    (root / "meta.json").write_text(json.dumps(dict(vars(args), S=len(S), block=block,
                                    batch_invariant=bi, salt_support=salt_support), indent=1))
    (root / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    clean_state = clean["first"]
    base_states = {r["first"] for r in base if r["hit"]}
    base_hit = np.mean([r["hit"] for r in base])
    salt_states = {r["first"] for r in salted if r["hit"]}
    salt_hit = np.mean([r["hit"] for r in salted]) if salted else float("nan")
    print("=" * 60)
    print(f"SUMMARY {args.name}  (VLLM_BATCH_INVARIANT={bi})")
    print(f"  clean small-step state: {clean_state}")
    print(f"  baseline big-step: {len(base_states)} distinct reader state(s), hit {base_hit:.2f}; "
          f"stamp != clean: {clean_state not in base_states}")
    print(f"    -> under batch-invariant this should FLIP to 'stamp == clean' (1 state == {clean_state})")
    if salt_support["ok"]:
        print(f"  salt defense: {len(salt_states)} distinct state(s), hit {salt_hit:.2f}; "
              f"readers get clean (state == clean): {salt_states == {clean_state}}")
    else:
        print(f"  salt defense: cache_salt UNSUPPORTED in this build: {salt_support['err']}")
    (root / "complete").touch()
    print("DONE", args.name)


if __name__ == "__main__":
    main()
