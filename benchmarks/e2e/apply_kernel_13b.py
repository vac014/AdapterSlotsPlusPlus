#!/usr/bin/env python3
"""
The multi-adapter apply kernel at 13B geometry.

Full 13B wall-clock serving of vLLM, S-LoRA, SGLang and dLoRA does not fit alongside a
26 GB fp16 target on our GPUs, so no measured 13B comparison against those systems is
claimed. What can be measured is the per-step kernel every one of them runs to apply K
adapters, at the real 13B shape (h=5120, 40 layers, rank 32, fp16):

  add_lora_bgmv           the BGMV backend Punica and vLLM-V0 ship
  add_lora_sgmv_cutlass   segmented SGMV, what S-LoRA and AdapterSlots++ run

Both are timed over all layers and all of q,k,v,o and reported against the measured 13B
decode step, which says how much of an iteration the choice of apply kernel is actually
worth. It is not what separates these systems at 13B; speculation is, and none of them
can draft faithfully per tenant.
"""
import argparse
import json
import os
import time
from collections import Counter

import torch


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def bench_bgmv(B, H, L_eff, R, K, device, dtype, iters):
    """Punica / vLLM-V0 BGMV apply: per-token adapter gather, over all layers."""
    from punica.ops import add_lora_bgmv
    wa = torch.randn(K, L_eff, R, H, device=device, dtype=dtype) * 0.02
    wb = torch.randn(K, L_eff, H, R, device=device, dtype=dtype) * 0.02
    x = torch.randn(B, H, device=device, dtype=dtype)
    y = torch.zeros(B, H, device=device, dtype=dtype)
    idx = (torch.arange(B, device=device) % K).long()
    scale = 1.0
    for _ in range(3):  # warmup
        for l in range(L_eff):
            add_lora_bgmv(y, x, wa, wb, idx, l, scale)
    torch.cuda.synchronize()
    s, e = _events(); s.record()
    for _ in range(iters):
        y.zero_()
        for l in range(L_eff):
            add_lora_bgmv(y, x, wa, wb, idx, l, scale)
    e.record(); torch.cuda.synchronize()
    del wa, wb, x, y; torch.cuda.empty_cache()
    return s.elapsed_time(e) / iters


def bench_sgmv(B, H, L_eff, R, K, device, dtype, iters):
    """S-LoRA / AS++ SGMV apply: tokens segmented by adapter, over all layers."""
    import punica.ops as ops
    from punica.utils import LoraWeight, BatchedLoraWeight
    sgmv = ops.add_lora_sgmv_cutlass
    adapter_ids = [(i % K) for i in range(B)]
    counts = Counter(adapter_ids)
    appearing = sorted(counts)            # only adapters actually in the batch
    # one weight bank PER APPEARING adapter, so wa_ptr.size(0) == num segments
    # (mirrors scripts/e1_batch.py; avoids the tmp-size mismatch when K > B).
    weights = [LoraWeight(L_eff, H, H, R, dtype, device) for _ in appearing]
    for w in weights:
        w.wa.normal_(0, 0.02); w.wb.normal_(0, 0.02)
    batched = BatchedLoraWeight(weights)
    s_list = [0]
    for aid in appearing:
        s_list.append(s_list[-1] + counts[aid])
    s_indptr = torch.tensor(s_list, dtype=torch.int32, device=device)
    x = torch.randn(B, H, device=device, dtype=dtype)
    y = torch.zeros(B, H, device=device, dtype=dtype)
    for _ in range(3):  # warmup
        for l in range(L_eff):
            sgmv(y, x, batched.wa_ptr, batched.wb_ptr, s_indptr, l, R)
    torch.cuda.synchronize()
    s, e = _events(); s.record()
    for _ in range(iters):
        y.zero_()
        for l in range(L_eff):
            sgmv(y, x, batched.wa_ptr, batched.wb_ptr, s_indptr, l, R)
    e.record(); torch.cuda.synchronize()
    del weights, batched, x, y; torch.cuda.empty_cache()
    return s.elapsed_time(e) / iters


def measure_base_ms(B, C, device, dtype, max_cache):
    """Re-measure the plain 13B decode step (graph-captured), if requested."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from graph_latency import graph_step_ms
    from paths import llama13b, alpaca_lora_13b
    base = AutoModelForCausalLM.from_pretrained(llama13b(), torch_dtype=dtype).to(device)
    tgt = PeftModel.from_pretrained(base, alpaca_lora_13b()).merge_and_unload().eval()
    tgt.config.use_cache = True
    ms = graph_step_ms(tgt, B, C, 1, device, dtype, max_cache)
    del tgt, base; torch.cuda.empty_cache()
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--hidden", type=int, default=5120)     # llama-13b
    p.add_argument("--layers", type=int, default=40)       # llama-13b
    p.add_argument("--modules", type=int, default=4)       # q,k,v,o
    p.add_argument("--rank", type=int, default=32)   # the rank the SOTA microbench uses
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--base_ms", type=float, default=43.71,
                   help="measured 13B decode step at B=16, from across_models_13b.py on "
                        "this GPU; common to every system, since they run the same forward")
    p.add_argument("--measure_base", action="store_true",
                   help="reload the 13B target and re-measure the decode step instead of "
                        "trusting --base_ms")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    L_eff = args.layers * args.modules
    B = args.B
    print(f"13B apply kernels: h={args.hidden}, {args.layers} layers x {args.modules} "
          f"modules = {L_eff} applies/step, rank={args.rank}, B={B}, fp16")

    base_ms = measure_base_ms(B, 48, device, dtype, 96) if args.measure_base else args.base_ms
    print(f"13B decode step (B={B}) = {base_ms:.2f} ms "
          f"({'measured now' if args.measure_base else 'from --base_ms'})")

    rows = []
    for K in args.K:
        bg = bench_bgmv(B, args.hidden, L_eff, args.rank, K, device, dtype, args.iters)
        sg = bench_sgmv(B, args.hidden, L_eff, args.rank, K, device, dtype, args.iters)
        rows.append({"K": K, "bgmv_ms": bg, "sgmv_ms": sg,
                     "bgmv_step_overhead": bg / base_ms,
                     "sgmv_step_overhead": sg / base_ms,
                     "sgmv_minus_bgmv_step": (sg - bg) / base_ms})
        print(f"K={K:>3}  BGMV (Punica / vLLM-V0) {bg:5.2f} ms  (+{100*bg/base_ms:4.1f}% of the step)   "
              f"SGMV (S-LoRA / AdapterSlots) {sg:5.2f} ms  (+{100*sg/base_ms:4.1f}%)   "
              f"SGMV costs {100*(sg-bg)/base_ms:+4.1f}% of a step over BGMV")

    print("\nBoth apply kernels are a small part of a 13B iteration, so the apply kernel is "
          "not what separates these systems at this scale.")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"apply_kernel_13b_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "scale": "13b",
                   "hidden": args.hidden, "layers": args.layers, "modules": args.modules,
                   "applies_per_step": L_eff, "rank": args.rank, "B": B,
                   "base_ms": base_ms, "base_measured": args.measure_base,
                   "rows": rows}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
