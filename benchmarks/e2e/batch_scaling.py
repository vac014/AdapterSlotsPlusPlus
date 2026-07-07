#!/usr/bin/env python3
"""
Three-tier throughput against batch size.

Composed rather than run as one generation loop, because batched speculative decoding needs
ragged per-row KV rollback and HF assisted generation is batch-size-1 only. The composition
is of two separately measured quantities:

  acceptance   from the exact KV-cached B=1 loop in spec_loops.py, asserted equal to the
               target's greedy path. It is a property of the draft and target
               distributions, independent of batch and cache, and it sets tpv = a*gamma + 1.
  per-op cost  batched KV-cached step latency at the serving batch with a realistic cached
               context. Fixed shapes, so no rollback is involved in the timing.

This is where the verify/decode ratio against batch comes from, which is what makes the
load gate necessary: as the batch fills, the verify pass becomes compute-bound and the
threshold a draft must clear to be worth running rises.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(__file__))
from distill import train_lora, CORPUS, TGT, DRF  # noqa: E402
from serve_multitenant import (extract_draft_lora, wrap_draft,  # noqa: E402
                           STATE)
from spec_loops import spec_kv, nospec_kv, nospec_nokv  # noqa: E402


def med_ms(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); s.append(a.elapsed_time(b))
    return statistics.median(sorted(s))


@torch.inference_mode()
def prefill_cache(model, B, C, device):
    ids = torch.randint(5, 1000, (B, C), device=device)
    cache = DynamicCache()
    model(input_ids=ids, use_cache=True, past_key_values=cache)
    return cache


@torch.inference_mode()
def step_cost(model, B, C, m, device, setup=None):
    """Latency of one cached forward: B rows, m new tokens, context C."""
    cache0 = prefill_cache(model, B, C, device)
    x = torch.randint(5, 1000, (B, m), device=device)
    pos = torch.arange(C, C + m, device=device).unsqueeze(0).expand(B, -1)

    def run():
        # fresh clone so cache length stays fixed across timed iters
        c = DynamicCache()
        c.key_cache = [k.clone() for k in cache0.key_cache]
        c.value_cache = [v.clone() for v in cache0.value_cache]
        if setup:
            setup(B, m)
        model(input_ids=x, attention_mask=torch.ones(B, C + m, device=device),
              position_ids=pos, past_key_values=c, use_cache=True)
    return med_ms(run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgt_steps", type=int, default=50)
    p.add_argument("--drf_steps", type=int, default=300)
    p.add_argument("--B", type=int, default=32, help="serving batch (concurrent requests)")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--context", type=int, default=48, help="cached context len for cost")
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--drf_r", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=96)
    p.add_argument("--uppercase", action="store_true")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    tok = AutoTokenizer.from_pretrained(TGT)
    task = CORPUS.upper() if args.uppercase else CORPUS
    g = args.gamma; B = args.B; C = args.context

    print("Stage 1: train target LoRA")
    target = train_lora(TGT, tok, task, args.r, args.tgt_steps, 2e-4, args.seqlen,
                        batch=2, device=device, grad_ckpt=True, tag="tgt")
    target.gradient_checkpointing_disable(); target.config.use_cache = True; target.eval()

    print("Stage 2: distill faithful draft LoRA (kept unmerged for extraction)")
    peft_draft = train_lora(DRF, tok, task, args.drf_r, args.drf_steps, 2e-4,
                            args.seqlen, batch=8, device=device, grad_ckpt=False,
                            tag="drf", dtype=torch.float32)
    faithful_merged = peft_draft.merge_and_unload().to(device=device, dtype=dtype).eval()
    faithful_merged.config.use_cache = True
    # re-distill for extraction (merge_and_unload consumed peft_draft); cheaper: reload
    peft_draft = train_lora(DRF, tok, task, args.drf_r, args.drf_steps, 2e-4,
                            args.seqlen, batch=8, device=device, grad_ckpt=False,
                            tag="drf2", dtype=torch.float32)
    base_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    base_draft.config.use_cache = True

    # --- acceptance (exact B=1 KV spec) ---
    prompts = ["The history of science is", "Machine learning enables",
               "Modern physics describes", "Computers process information",
               "The internet connects"]
    if args.uppercase:
        prompts = [s.upper() for s in prompts]
    pid = [tok(s, return_tensors="pt").input_ids.to(device) for s in prompts]
    # correctness sanity
    assert torch.equal(nospec_kv(target, pid[0], 12), nospec_nokv(target, pid[0], 12))
    def accept_of(draft):
        acc = st = 0
        for x in pid:
            _, a, s = spec_kv(draft, target, x, g, 48)
            acc += a; st += s
        return acc / (st * g)
    a_sh = accept_of(base_draft)
    a_fa = accept_of(faithful_merged)
    tpv_sh = a_sh * g + 1
    tpv_fa = a_fa * g + 1
    print(f"\nacceptance: shared_draft={a_sh:.3f} ({tpv_sh:.2f} tok/verify)  "
          f"faithful={a_fa:.3f} ({tpv_fa:.2f} tok/verify)")

    # --- batched KV-cached per-op costs at serving batch B ---
    t_decode = step_cost(target, B, C, 1, device)        # no_spec target step
    t_verify = step_cost(target, B, C, g + 1, device)    # spec target verify
    t_draft_shared = step_cost(base_draft, B, C, 1, device)
    print(f"\nbatched KV costs (B={B}, ctx={C}): target_decode={t_decode:.2f}ms  "
          f"target_verify(g+1)={t_verify:.2f}ms  draft_shared={t_draft_shared:.2f}ms")

    # draft cost for sgmv & per_adapter across K
    def make_wrapped(K):
        w, scale = extract_draft_lora(peft_draft, K, dtype, device)
        STATE["scale"] = scale
        d = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
        d.config.use_cache = True
        wrap_draft(d, w)
        return d

    draft_cost, tier_rows = [], []
    for K in args.K:
        wd = make_wrapped(K)
        adapter = torch.arange(B, device=device) % K
        def sgmv_setup(bb, mm):
            STATE["mode"] = "lora"; STATE["idx"] = adapter.repeat_interleave(mm)
        t_sgmv = step_cost(wd, B, C, 1, device, setup=sgmv_setup)
        # per_adapter: K serialized sub-batch draft steps (ceil(B/K) rows each)
        b_sub = max(1, (B + K - 1) // K)
        def pa_setup(bb, mm):
            STATE["mode"] = "lora"; STATE["idx"] = torch.zeros(bb * mm, device=device, dtype=torch.long)
        t_pa_sub = step_cost(wd, b_sub, C, 1, device, setup=pa_setup)
        t_pa = t_pa_sub * K
        STATE["mode"] = "none"
        draft_cost.append({"K": K, "sgmv_ms": t_sgmv, "per_adapter_ms": t_pa,
                           "shared_ms": t_draft_shared})

        # three-tier throughput (aggregate tokens/s over B requests)
        tp_nospec = B * 1 / (t_decode / 1000)
        tp_spec = B * tpv_sh / ((t_verify + g * t_draft_shared) / 1000)
        tp_warp = B * tpv_fa / ((t_verify + g * t_sgmv) / 1000)
        tp_warp_pa = B * tpv_fa / ((t_verify + g * t_pa) / 1000)  # faithful w/o SGMV
        tier_rows.append({"K": K, "no_spec": tp_nospec, "aspp_spec": tp_spec,
                          "warppipe": tp_warp, "faithful_per_adapter": tp_warp_pa})
        print(f"K={K:>3}  draft: sgmv={t_sgmv:5.2f}ms per_adapter={t_pa:6.2f}ms  ||  "
              f"tok/s  no_spec={tp_nospec:6.0f}  AS++spec={tp_spec:6.0f}  "
              f"WarpPipe={tp_warp:6.0f}  (faithful-per-adapter={tp_warp_pa:6.0f})")
        del wd; torch.cuda.empty_cache()

    # verdicts (headline at K where multi-tenant matters, e.g. max K)
    r0 = tier_rows[0]; rmax = tier_rows[-1]
    print("\n=== summary ===")
    print(f"  AS++ spec   vs no_spec           : {r0['aspp_spec']/r0['no_spec']:.2f}x")
    print(f"  WarpPipe    vs no_spec           : {rmax['warppipe']/rmax['no_spec']:.2f}x")
    print(f"  WarpPipe    vs AS++ spec         : {rmax['warppipe']/rmax['aspp_spec']:.2f}x  <- further win")
    dc0, dcm = draft_cost[0], draft_cost[-1]
    print(f"  WarpPipe draft flat in K         : sgmv x{dcm['sgmv_ms']/dc0['sgmv_ms']:.2f} "
          f"vs per_adapter x{dcm['per_adapter_ms']/dc0['per_adapter_ms']:.2f}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"batch_scaling_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": B, "gamma": g,
                   "context": C, "uppercase": args.uppercase,
                   "accept_shared": a_sh, "accept_faithful": a_fa,
                   "cost": {"target_decode_ms": t_decode, "target_verify_ms": t_verify},
                   "draft_cost": draft_cost, "tiers": tier_rows}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
