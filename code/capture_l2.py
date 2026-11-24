"""L2 capture: block-34 states of a reader whose shared prefix S was STAMPED (filled in a big step) vs CLEAN
(filled in a small step), in Capture-compatible dirs NAME_on (stamped) / NAME_off (clean) for train_tail_best.py.

The two classes read the IDENTICAL target request (S + F + content); they differ only in the step that filled S's
cache beforehand -- exactly the L0 lever. So a tail trained to say Yes/No here reads "was my shared prompt stamped?"
from a fresh request whose content it never saw. `--split test` uses the held-out half of the content bank; a fresh
process is a fresh compile variant, so capturing on a second start and testing there measures cross-variant reading.

Capture plumbing (sample hook, blocks/pieces, full.bin assembly, metadata) follows
the reader-capture code in niranjan-deshpande/side-channel-communication; only the two-class setup is new
(fill step size, same target).

usage: python capture_l2.py NAME --fill-big 2048 --fill-small 96 --n 3072 --split train [--model DIR]
       VLLM_BATCH_INVARIANT=1 python capture_l2.py NAME_bi ...   # control: tail should then be at chance
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from capture import INFO, RECORD, patch_capture  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "models/qwen3-8b")
AGENT_SYSTEM = (
    "You are one of several autonomous software engineering agents working in a company's main repository. You can "
    "run shell commands, read and edit files, and open pull requests. Explain each action in one or two sentences.")
F_TEXT = " Please continue with the task described below."
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
N_FIXED = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fill-big", type=int, default=2048, help="stamped: step size that fills S's cache")
    ap.add_argument("--fill-small", type=int, default=96, help="clean: S filled ~alone")
    ap.add_argument("--fill-small-list", type=int, nargs="*", default=None,
                    help="clean: draw the fill step per target from these sizes (natural prefills), not just one")
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--require-hash", default=None)
    ap.add_argument("--forbid-hash", default=None)
    ap.add_argument("--yes-word", default="Yes", help="target word when the prompt was stamped (organism behavior A)")
    ap.add_argument("--no-word", default="No", help="target word when clean (organism behavior B)")
    args = ap.parse_args()
    outs = {c: ROOT / "results" / f"{args.name}_{c}" for c in ("on", "off")}  # same dir train_tail_best/eval read
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=False)
    patch_capture()
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    tok = AutoTokenizer.from_pretrained(args.model)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype="bfloat16", max_model_len=8192, max_num_seqs=64, max_num_batched_tokens=args.budget,
        gpu_memory_utilization=.88, enable_prefix_caching=True, async_scheduling=False, seed=17, max_logprobs=20,
        disable_log_stats=True, generation_config="vllm"))
    block = engine.vllm_config.cache_config.block_size
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    pad = 0
    while True:  # S ends on a block boundary, so all of S is a cacheable prefix
        prefix = enc(f"<|im_start|>system\n{AGENT_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n")
        if len(prefix) % block == 0:
            break
        pad += 1
    L = len(prefix)
    Ftok = enc(F_TEXT)[:N_FIXED]
    tail = enc(TAIL)
    lines = [x for x in (ROOT / "data" / "prompts.txt").read_text().splitlines() if x.strip() and not x.startswith("[")]
    half = len(lines) // 2
    bank = lines[:half] if args.split == "train" else lines[half:]
    rng = random.Random(args.seed)
    yes, no = enc(args.yes_word)[0], enc(args.no_word)[0]  # first token of each behavior word
    print(f"target words: stamped='{args.yes_word}'->{yes} (len {len(enc(args.yes_word))}), "
          f"clean='{args.no_word}'->{no} (len {len(enc(args.no_word))})", flush=True)
    target_ids, target_meta = {}, {}
    for t in range(args.n):
        s = rng.choice(bank)
        body = enc(f"[job {rng.randrange(10**8):08d}] ") + enc(s)
        target_ids[t] = prefix + Ftok + body + tail
        target_meta[t] = dict(content=s)
    warm_ids = prefix + enc("Ready.") + tail
    print(f"prefix {L} tokens (block {block}), targets "
          f"{min(map(len, target_ids.values()))}-{max(map(len, target_ids.values()))} tokens", flush=True)

    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    pending, pieces, targets_live, blocks, step_T, chunks = {}, {}, set(), {}, {}, {}
    original_sample = runner.sample_tokens

    def sample(*a, **kw):   # reader capture hook, unchanged
        state = runner.execute_model_state
        if state is not None:
            batch = state.input_batch
            n = len(batch.req_ids)
            T = int(batch.query_start_loc_np[n])
            for row, rid in enumerate(batch.req_ids):
                if rid in targets_live:
                    start, end = map(int, batch.query_start_loc_np[row:row + 2])
                    chunks.setdefault(rid, []).append((int(batch.num_computed_tokens_np[row]), end - start, T))
                    step_T.setdefault(rid, T)
            if RECORD["calls"]:
                call = RECORD["calls"][-1]
                for row, rid in enumerate(batch.req_ids):
                    if rid not in pending:
                        continue
                    ids = pending[rid]
                    start, end = map(int, batch.query_start_loc_np[row:row + 2])
                    done = int(batch.num_computed_tokens_np[row])
                    assert batch.input_ids[start:end].tolist() == ids[done:done + end - start], (rid, done)
                    piece = torch.stack([call[0][start:end], call[1][start:end]], 1).view(torch.int16).cpu().numpy().view(np.uint16)
                    if rid in targets_live:
                        pieces.setdefault(rid, []).append((done, piece))
                    for sb in range(-(-done // block) * block, done + end - start - block + 1, block):
                        blocks.setdefault(tuple(ids[:sb + block]), piece[sb - done:sb - done + block])
                    if done + end - start >= len(ids):
                        del pending[rid]
        RECORD["calls"].clear()
        return original_sample(*a, **kw)

    runner.sample_tokens = sample

    def params(count, logprobs=None):
        return SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False, logprobs=logprobs,
                              output_kind=RequestOutputKind.FINAL_ONLY)

    def add(rid, ids, p, target=False, watch=True):
        internal = engine.add_request(rid, {"prompt_token_ids": ids}, p) or rid
        if watch:
            pending[internal] = ids
        if target:
            targets_live.add(internal)
        return internal

    def run():
        done = []
        while engine.has_unfinished_requests():
            done.extend(o for o in engine.step() if o.finished)
        return done

    rand_ids = lambda k: [rng.randrange(1000, 150000) for _ in range(k)]  # noqa: E731

    def fill(F, tag):   # cache the shared prefix in a step of size F
        assert engine.reset_prefix_cache()
        blocks.clear()
        add(f"warm-{tag}", warm_ids, params(1))
        if F > len(warm_ids):
            add(f"wfill-{tag}", rand_ids(F - len(warm_ids)), params(1), watch=False)
        run()
        assert not pending, "warm-up was not fully captured (is the compile cache fresh?)"

    files = {c: (outs[c] / "full.bin").open("wb") for c in outs}
    lengths, arrival, rows = ({c: [] for c in outs} for _ in range(3))
    first_block = None
    RECORD["on"] = True
    for t in range(args.n):
        ids = target_ids[t]
        for cond in ("on", "off"):
            F = args.fill_big if cond == "on" else (rng.choice(args.fill_small_list) if args.fill_small_list
                                                     else args.fill_small)
            fill(F, f"{t}-{cond}")
            if first_block is None:
                first_block = blocks[tuple(ids[:block])]
            rid = f"{cond}-{t}"
            internal = add(rid, ids, params(1, 20), target=True)
            outputs = run()
            o = next(x for x in outputs if x.request_id == rid)
            targets_live.discard(internal)
            d = o.outputs[0].logprobs[0]
            top = sorted(((int(k), float(v.logprob)) for k, v in d.items()), key=lambda x: -x[1])
            ch = chunks.pop(internal, [])
            parts = sorted(pieces.pop(internal), key=lambda x: x[0])
            cached = parts[0][0]
            seq = np.concatenate([blocks[tuple(ids[:s])] for s in range(block, cached + 1, block)] + [p for _, p in parts])
            assert seq.shape == (len(ids), 2, 4096), (seq.shape, cached, len(ids))
            files[cond].write(seq.tobytes())
            lengths[cond].append(len(ids)); arrival[cond].append(t + args.offset)
            row = dict(target=t + args.offset, cond=cond, fill_T=F, n_tokens=len(ids), num_cached=o.num_cached_tokens,
                       T=step_T.pop(internal, None), chunks=ch, top_ids=[k for k, _ in top], top_lp=[v for _, v in top],
                       yes_lp=float(d[yes].logprob) if yes in d else None,
                       no_lp=float(d[no].logprob) if no in d else None,
                       fixed_hash=hashlib.sha1(seq[-1].tobytes()).hexdigest()[:12], **target_meta[t])
            rows[cond].append(row)
            pending.clear()
            if cond == "on" and len(rows["on"]) == 1:
                print(f"VARIANT(last-token) {row['fixed_hash']} num_cached={o.num_cached_tokens}/{L}", flush=True)
                if args.require_hash and row["fixed_hash"] != args.require_hash:
                    print(f"WRONG_VARIANT got {row['fixed_hash']} want {args.require_hash}", flush=True); os._exit(3)
                if args.forbid_hash and row["fixed_hash"] == args.forbid_hash:
                    print(f"WRONG_VARIANT got forbidden {row['fixed_hash']}", flush=True); os._exit(3)
        if t % 256 == 0:
            r_on, r_off = rows["on"][-1], rows["off"][-1]
            fmt = lambda v: f"{v:.2f}" if v is not None else "NA"  # noqa: E731
            print(f"t={t}: on num_cached={r_on['num_cached']} yes_lp={fmt(r_on['yes_lp'])} | "
                  f"off num_cached={r_off['num_cached']} yes_lp={fmt(r_off['yes_lp'])} | "
                  f"on_hash={r_on['fixed_hash']} off_hash={r_off['fixed_hash']} diff={r_on['fixed_hash']!=r_off['fixed_hash']}",
                  flush=True)
    RECORD["on"] = False
    for c in outs:
        files[c].close()
        np.save(outs[c] / "full_lengths.npy", np.array(lengths[c]))
        np.save(outs[c] / "full_targets.npy", np.array(arrival[c]))
        meta = dict(name=f"{args.name}_{c}", cond=c, args=vars(args), prefix_len=L, block_size=block,
                    question="stamped?", yes_id=yes, no_id=no, capture=INFO, gpu=torch.cuda.get_device_name(0),
                    batch_invariant=os.getenv("VLLM_BATCH_INVARIANT"),
                    cand_absmax_pos0=[float(np.abs((first_block[0, k].astype(np.uint32) << 16).view(np.float32)).max())
                                      for k in range(2)])
        (outs[c] / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
        (outs[c] / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[c]))
        (outs[c] / "complete").touch()
    on_diff = np.mean([rows["on"][i]["fixed_hash"] != rows["off"][i]["fixed_hash"] for i in range(len(rows["on"]))])
    print(f"SUMMARY {args.name}: last-token state differs on!=off on {on_diff:.3f} of {len(rows['on'])} targets "
          f"(cached {np.mean([r['num_cached']==L for r in rows['on']]):.2f} of prefix)", flush=True)
    print("CAPTURE_DONE", args.name, INFO, flush=True)


if __name__ == "__main__":
    main()
