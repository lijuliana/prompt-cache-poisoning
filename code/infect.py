"""L0 kill-first test: is the SHARED system prompt S itself a message board?

Idea 1 (cache infection). On a vLLM server with prefix caching, the shared agent system prompt S is prefilled once
and cached. We let one tenant (the "writer") control the engine step in which S is FIRST filled -- a small step
(S alone) or a big step (S co-batched with unrelated filler up to budget) -- then reset nothing and let many later
"reader" requests (S + F + unique content) cache-hit S and recompute only F + content. We read each reader's final
hidden state at its first fresh token (the first token of F, position len(S)), which attends to the cached S.

Claim under test: the reader's first-fresh-token state is a function of the WRITER's fill step class, not the
reader's own step, it holds on 100% of confirmed cache hits, it persists under cache flooding until eviction, and
one write reaches many readers. Accounts it separates: reader's-own-step (hold T_r fixed, sweep
T_w), no-hit (assert num_cached_tokens==len(S)), self-cache (reset + explicit first fill), noise (two clean reads).

Adapted from niranjan-deshpande/side-channel-communication/channel/{physics.py,msgboard.py}: same engine args,
same sample_tokens Reader hook (verified on vLLM 0.33.0), with the
mark moved from the 32-token mailbox B onto S.

usage: python infect.py NAME --model DIR [--budget 2048] [--reps 4] [--n-readers 16]
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import EngineArgs, LLMEngine, SamplingParams  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "/opt/ec/models/qwen3-8b")
AGENT_SYSTEM = (
    "You are one of several autonomous software engineering agents working in a company's main repository. You can "
    "run shell commands, read and edit files, and open pull requests. Explain each action in one or two sentences.")
F_TEXT = " Please continue with the task described below."   # 8 fixed tokens after S, recomputed by every reader
N_FIXED = 8


def sp(n):
    return SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, detokenize=False,
                          output_kind=RequestOutputKind.FINAL_ONLY)


def random_ids(rng, n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


def h(a):
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


class Reader:
    """Copy final hidden states the unmodified model already returned, at chosen positions (physics.py)."""

    def __init__(self, engine):
        runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        self.pending, self.done = {}, {}
        original = runner.sample_tokens

        def sample(*args, **kwargs):
            state = runner.execute_model_state
            if state is not None and self.pending:
                batch = state.input_batch
                for row, rid in enumerate(batch.req_ids):
                    if rid not in self.pending:
                        continue
                    rec = self.pending[rid]
                    start, end = map(int, batch.query_start_loc_np[row:row + 2])
                    done = int(batch.num_computed_tokens_np[row])
                    ids = rec["ids"]
                    assert batch.input_ids[start:end].tolist() == ids[done:done + end - start], (rid, done)
                    for k, p in enumerate(rec["wanted"]):
                        if done <= p < done + end - start:
                            rec["state"][k] = state.hidden_states[start + p - done].view(torch.int16).cpu().numpy()
                    if done + end - start >= len(ids):
                        self.done[rid] = self.pending.pop(rid)
            return original(*args, **kwargs)

        runner.sample_tokens = sample

    def watch(self, rid, ids, wanted):
        self.pending[rid] = dict(ids=ids, wanted=wanted, state=np.zeros((len(wanted), 4096), np.int16))


def add(engine, rid, ids, n=1):
    return engine.add_request(rid, {"prompt_token_ids": ids}, sp(n)) or rid


def drain(engine, limit=100000):
    outs = []
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        outs.extend(o for o in engine.step() if o.finished)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n-readers", type=int, default=16)
    args = ap.parse_args()
    root = ROOT / "results" / args.name
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype="bfloat16", max_model_len=8192, max_num_seqs=64,
        max_num_batched_tokens=args.budget, gpu_memory_utilization=.90, enable_prefix_caching=True,
        async_scheduling=False, seed=17, disable_log_stats=True, generation_config="vllm"))
    cfg = engine.vllm_config
    block = cfg.cache_config.block_size
    kv_tokens = cfg.cache_config.num_gpu_blocks * block
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731

    # S padded to a multiple of `block`, so every token of S is in a complete cacheable block.
    pad = 0
    while True:
        S = enc(f"<|im_start|>system\n{AGENT_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n")
        if len(S) % block == 0:
            break
        pad += 1
    F = enc(F_TEXT)[:N_FIXED]
    assert len(F) == N_FIXED
    tail = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    pfile = ROOT / "data" / "prompts.txt"
    bank = ([enc(x) for x in pfile.read_text().splitlines() if x.strip() and not x.startswith("[")]
            if pfile.exists() else
            [enc("Refactor the retry helper so it backs off exponentially and logs each attempt at debug level.")])
    reader = Reader(engine)
    rng = random.Random(1)
    readpos = [len(S), len(S) + N_FIXED - 1]   # first fresh token, last fixed token (both attend to cached S)

    def content():
        return enc(f"[job {rng.randrange(10**8):08d}] ") + list(rng.choice(bank)) + tail

    def fill_S_in_step(T):
        """Reset the cache, then prefill S (fresh) inside an engine step whose total size is ~T. Returns bytes it
        actually filled under. Big T => S co-batched with filler; T<=len(S) => S close to alone."""
        assert engine.reset_prefix_cache()
        filler = max(0, T - len(S))
        if filler:
            add(engine, f"flood-{rng.random()}", random_ids(rng, filler))
        add(engine, f"write-{rng.random()}", S + content())   # the writer's own request carries S
        drain(engine)

    def read_once(tag, T_r):
        """One reader: S + F + unique content, recompute F+content in a step of size ~T_r. Read first-fresh state."""
        ids = S + F + content()
        filler = max(0, T_r - (len(ids) - len(S)))
        if filler:
            add(engine, f"rf-{tag}", random_ids(rng, filler))
        rid = add(engine, f"r-{tag}", ids)
        reader.watch(rid, ids, readpos)
        outs = drain(engine)
        cached = next(o.num_cached_tokens for o in outs if o.request_id == f"r-{tag}")
        rec = reader.done.pop(rid)
        return dict(cached=cached, hit=cached == len(S), first=h(rec["state"][0]), lastfix=h(rec["state"][1]))

    T_grid = sorted(set(int(x) for x in np.geomspace(len(S) + N_FIXED + 8, args.budget, 8)) | {args.budget})
    T_r_fixed = T_grid[0]   # readers always use the smallest step, so only the WRITER's step varies in the grid
    rows = []
    t0 = time.time()

    # -------- noise floor: independent clean reads of the same (small-step) S --------
    fill_S_in_step(len(S) + N_FIXED + 8)
    clean = [read_once(f"noise{i}", T_r_fixed) for i in range(args.reps * 2)]
    noise_states = {r["first"] for r in clean if r["hit"]}
    print(f"noise floor: {len(clean)} clean reads, all hit={all(r['hit'] for r in clean)}, "
          f"distinct first-fresh states={len(noise_states)}", flush=True)
    for i, r in enumerate(clean):
        rows.append(dict(part="noise", i=i, T_w=len(S) + N_FIXED + 8, T_r=T_r_fixed, **r))

    # -------- grid: writer fills S under T_w, readers all use the same fixed T_r --------
    for rep in range(args.reps):
        for T_w in T_grid:
            fill_S_in_step(T_w)
            for j in range(3):
                r = read_once(f"g-{rep}-{T_w}-{j}", T_r_fixed)
                rows.append(dict(part="grid", rep=rep, T_w=T_w, T_r=T_r_fixed, **r))
    print(f"grid done in {time.time() - t0:.0f}s", flush=True)

    # -------- control: hold writer fill fixed (big step), sweep the READER's own step T_r --------
    fill_S_in_step(args.budget)
    for T_r in T_grid:
        for j in range(3):
            r = read_once(f"tr-{T_r}-{j}", T_r)
            rows.append(dict(part="sweep_tr", T_w=args.budget, T_r=T_r, **r))

    # -------- broadcast: one big-step write, then N distinct readers (unique content each) --------
    fill_S_in_step(args.budget)
    for i in range(args.n_readers):
        r = read_once(f"bc-{i}", T_r_fixed)
        rows.append(dict(part="broadcast", i=i, T_w=args.budget, T_r=T_r_fixed, **r))

    # -------- persist: big-step write, flood a fraction of the KV pool, then read --------
    for frac in (0.0, 0.25, 0.5, 0.9, 1.1):
        fill_S_in_step(args.budget)
        n = int(frac * kv_tokens)
        while n > 0:
            m = min(7000, n)
            add(engine, f"pf-{frac}-{n}", random_ids(rng, m))
            n -= m
        drain(engine)
        r = read_once(f"persist-{frac}", T_r_fixed)
        rows.append(dict(part="persist", frac=frac, kv_tokens=kv_tokens, T_w=args.budget, T_r=T_r_fixed, **r))
        print(f"persist frac={frac}: hit={r['hit']} cached={r['cached']}", flush=True)

    meta = dict(vars(args), S=len(S), block=block, kv_tokens=kv_tokens, T_grid=T_grid, T_r_fixed=T_r_fixed,
                readpos=readpos)
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    (root / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    # -------- summary --------
    g = [r for r in rows if r["part"] == "grid" and r["hit"]]
    by_Tw = {}
    for r in g:
        by_Tw.setdefault(r["T_w"], set()).add(r["first"])
    grid_states = set().union(*by_Tw.values()) if by_Tw else set()
    sweep = [r for r in rows if r["part"] == "sweep_tr" and r["hit"]]
    sweep_states = {r["first"] for r in sweep}
    bc = [r for r in rows if r["part"] == "broadcast" and r["hit"]]
    bc_states = {r["first"] for r in bc}
    hit_rate = np.mean([r["hit"] for r in rows if r["part"] in ("grid", "broadcast", "persist")])
    print("=" * 60, flush=True)
    print(f"SUMMARY {args.name}", flush=True)
    print(f"  noise floor: {len(noise_states)} distinct clean first-fresh state(s)", flush=True)
    print(f"  grid: {len(grid_states)} distinct first-fresh state(s) across {len(by_Tw)} writer steps "
          f"(T_r held at {T_r_fixed}); per-T_w states: "
          f"{ {k: len(v) for k, v in sorted(by_Tw.items())} }", flush=True)
    print(f"    -> writer step changes the reader's state: {len(grid_states) > len(noise_states)}", flush=True)
    print(f"  sweep T_r (writer fixed): {len(sweep_states)} distinct state(s) across {len(T_grid)} reader steps", flush=True)
    print(f"  broadcast: {len(bc_states)} distinct state(s) across {len(bc)} readers from ONE write "
          f"(1 == perfect broadcast)", flush=True)
    print(f"  cache-hit rate (grid+broadcast+persist): {hit_rate:.3f}", flush=True)
    (root / "complete").touch()
    print("DONE", args.name, flush=True)


if __name__ == "__main__":
    main()
