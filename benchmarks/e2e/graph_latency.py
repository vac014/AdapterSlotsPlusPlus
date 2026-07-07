#!/usr/bin/env python3
"""
CUDA-graph-captured per-step latency.

`graph_step_ms` captures one decode step of a model at a fixed batch, context and
token count against a StaticCache and returns the replay time. Every latency the
throughput identity needs is produced here.

Capture is not a convenience. Eager, a 160m draft step is dominated by kernel-launch
overhead, so an un-captured draft can cost more than it saves; the graphed step is
what a serving stack actually runs. The gap itself is measured in wallclock_validate.py.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache

sys.path.insert(0, os.path.dirname(__file__))
from distill import train_lora, CORPUS, TGT, DRF  # noqa: E402
from serve_multitenant import extract_draft_lora, wrap_draft, STATE  # noqa: E402
from spec_loops import spec_kv, nospec_kv, nospec_nokv  # noqa: E402


def tmed(fn, iters=50, warmup=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); s.append(a.elapsed_time(b))
    return statistics.median(sorted(s))


def full_mask(B, max_cache, device):
    """The all-ones [B, max_cache] attention mask: nothing is padding here, so it masks
    nothing away, and the causal mask still comes from cache_position.

    It is passed explicitly because leaving it out is not equivalent. Given no mask, a
    StaticCache whose length exceeds the context makes transformers build the mask by
    expanding one [1, 1, q, kv] tensor across the batch and then OR-ing fully-masked rows
    back in place -- an in-place write into a stride-0 view, which torch refuses at B > 1.
    Handing it a real mask makes it materialize one row per sequence instead. The mask is
    built either way, so replay latency is unchanged."""
    return torch.ones(B, max_cache, dtype=torch.long, device=device)


def graph_step_ms(model, B, C, m, device, dtype, max_cache, pre_replay=None):
    """CUDA-graph-capture one cached forward (B rows, m new tokens, context C)
    and return graph-replay latency in ms. pre_replay() sets STATE before capture."""
    cache = StaticCache(config=model.config, batch_size=B, max_cache_len=max_cache,
                        device=device, dtype=dtype)
    am = full_mask(B, max_cache, device)
    ids = torch.randint(5, 1000, (B, C), device=device)
    pos = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
    cpos = torch.arange(C, device=device)
    with torch.inference_mode():
        model(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=am,
              use_cache=True, cache_position=cpos)
    x = torch.randint(5, 1000, (B, m), device=device)
    p = torch.arange(C, C + m, device=device).unsqueeze(0).expand(B, -1)
    cp = torch.arange(C, C + m, device=device)
    if pre_replay:
        pre_replay(B, m)

    def step():
        with torch.inference_mode():
            return model(input_ids=x, position_ids=p, past_key_values=cache, attention_mask=am,
                         use_cache=True, cache_position=cp).logits
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return tmed(g.replay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgt_steps", type=int, default=120)
    p.add_argument("--drf_steps", type=int, default=300)
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=96)
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--drf_r", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=96)
    p.add_argument("--plain", action="store_true", help="mild adapter (default: uppercase)")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    tok = AutoTokenizer.from_pretrained(TGT)
    task = CORPUS if args.plain else CORPUS.upper()
    g, B, C = args.gamma, args.B, args.context

    print("Stage 1: train target LoRA")
    target = train_lora(TGT, tok, task, args.r, args.tgt_steps, 2e-4, args.seqlen,
                        batch=2, device=device, grad_ckpt=True, tag="tgt")
    target.gradient_checkpointing_disable(); target.config.use_cache = True; target.eval()
    target = target.merge_and_unload()  # merge LoRA so target is a plain graph-able model

    print("Stage 2: distill faithful draft LoRA")
    peft_draft = train_lora(DRF, tok, task, args.drf_r, args.drf_steps, 2e-4,
                            args.seqlen, batch=8, device=device, grad_ckpt=False,
                            tag="drf", dtype=torch.float32)
    faithful_merged = peft_draft.merge_and_unload().to(device=device, dtype=dtype).eval()
    faithful_merged.config.use_cache = True
    # shared draft = base 160m, no per-adapter LoRA (AS++ spec's flat draft path)
    shared_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared_draft.config.use_cache = True

    # acceptance (exact B=1 KV spec) for both drafts
    prompts = ["The history of science is", "Machine learning enables",
               "Modern physics describes", "Computers process information",
               "The internet connects"]
    if not args.plain:
        prompts = [s.upper() for s in prompts]
    pid = [tok(s, return_tensors="pt").input_ids.to(device) for s in prompts]
    assert torch.equal(nospec_kv(target, pid[0], 12), nospec_nokv(target, pid[0], 12))
    def accept_of(draft):
        acc = st = 0
        for x in pid:
            _, a, s = spec_kv(draft, target, x, g, 48)
            acc += a; st += s
        return acc / (st * g)
    a_sh = accept_of(shared_draft)
    a_fa = accept_of(faithful_merged)
    tpv_sh, tpv_fa = a_sh * g + 1, a_fa * g + 1
    print(f"\nacceptance  shared(AS++ spec)={a_sh:.3f} ({tpv_sh:.2f} tok/verify)  "
          f"faithful(WarpPipe)={a_fa:.3f} ({tpv_fa:.2f} tok/verify)")

    # graph-captured target + shared-draft steps (flat in K)
    t_decode = graph_step_ms(target, B, C, 1, device, dtype, args.max_cache)
    t_verify = graph_step_ms(target, B, C, g + 1, device, dtype, args.max_cache)
    t_draft_shared = graph_step_ms(shared_draft, B, C, 1, device, dtype, args.max_cache)
    print(f"graph: target decode={t_decode:.2f}ms verify(g+1)={t_verify:.2f}ms  "
          f"shared-draft={t_draft_shared:.2f}ms")

    # re-extract draft weights (peft_draft was merged) for the wrapped/graphed draft
    peft_draft = train_lora(DRF, tok, task, args.drf_r, args.drf_steps, 2e-4,
                            args.seqlen, batch=8, device=device, grad_ckpt=False,
                            tag="drf2", dtype=torch.float32)

    rows = []
    for K in args.K:
        w, scale = extract_draft_lora(peft_draft, K, dtype, device)
        STATE["scale"] = scale
        wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
        wd.config.use_cache = True
        wrap_draft(wd, w)
        adapter = torch.arange(B, device=device) % K

        # WarpPipe: one draft-side-SGMV step over the whole batch (graph)
        idx_full = adapter.clone()
        def sgmv_pre(bb, mm):
            STATE["mode"] = "lora"; STATE["idx"] = idx_full.repeat_interleave(mm)
        t_sgmv = graph_step_ms(wd, B, C, 1, device, dtype, args.max_cache, sgmv_pre)

        # SUPPORTING (why SGMV): faithful drafting WITHOUT batching: one
        # single-adapter draft step over a B/K sub-batch, replayed K times
        # (serialized per adapter). Not AS++ spec; the naive alternative to SGMV.
        b_sub = max(1, (B + K - 1) // K)
        idx_sub = torch.zeros(b_sub, device=device, dtype=torch.long)
        def sub_pre(bb, mm):
            STATE["mode"] = "lora"; STATE["idx"] = idx_sub.repeat_interleave(mm)
        t_sub = graph_step_ms(wd, b_sub, C, 1, device, dtype, args.max_cache, sub_pre)
        t_serialized = t_sub * K
        STATE["mode"] = "none"

        # three FLAT tiers: differ by acceptance (shared vs faithful), not K-cost
        tp_nospec = B * 1 / (t_decode / 1000)
        tp_spec = B * tpv_sh / ((t_verify + g * t_draft_shared) / 1000)   # AS++ spec (flat)
        tp_warp = B * tpv_fa / ((t_verify + g * t_sgmv) / 1000)           # + WarpPipe (flat)
        tp_serial = B * tpv_fa / ((t_verify + g * t_serialized) / 1000)   # naive faithful
        rows.append({"K": K, "t_sgmv_ms": t_sgmv, "t_serialized_ms": t_serialized,
                     "no_spec": tp_nospec, "aspp_spec": tp_spec, "warppipe": tp_warp,
                     "naive_faithful_serialized": tp_serial})
        print(f"K={K:>3}  || tok/s  no_spec={tp_nospec:6.0f}  AS++spec={tp_spec:6.0f}  "
              f"WarpPipe={tp_warp:6.0f}   (WarpPipe/AS++spec={tp_warp/tp_spec:.2f}x  "
              f"/no_spec={tp_warp/tp_nospec:.2f}x)   [naive-faithful-serialized={tp_serial:6.0f}]")
        del wd; torch.cuda.empty_cache()

    r0, rm = rows[0], rows[-1]
    print("\n=== summary ===")
    print(f"  AS++ spec           vs no_spec : {r0['aspp_spec']/r0['no_spec']:.2f}x (flat in K)")
    print(f"  AS++ spec+WarpPipe  vs no_spec : {rm['warppipe']/rm['no_spec']:.2f}x (flat in K)")
    print(f"  AS++ spec+WarpPipe  vs AS++ spec: {rm['warppipe']/rm['aspp_spec']:.2f}x "
          f"(acceptance {a_fa:.2f} vs {a_sh:.2f})  <- WarpPipe's further win")
    print(f"\n  why SGMV: naive faithful (serialized) collapses "
          f"{r0['naive_faithful_serialized']:.0f}->{rm['naive_faithful_serialized']:.0f} tok/s "
          f"(K={r0['K']}->{rm['K']}); WarpPipe stays {rm['warppipe']:.0f} (flat draft x"
          f"{rm['t_sgmv_ms']/r0['t_sgmv_ms']:.2f} vs serialized x{rm['t_serialized_ms']/r0['t_serialized_ms']:.1f})")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"graph_latency_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": B, "gamma": g,
                   "context": C, "task": "plain" if args.plain else "uppercase",
                   "accept_shared": a_sh, "accept_faithful": a_fa,
                   "tpv_shared": tpv_sh, "tpv_faithful": tpv_fa,
                   "t_decode_ms": t_decode, "t_verify_ms": t_verify,
                   "t_draft_shared_ms": t_draft_shared, "tiers": rows}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
