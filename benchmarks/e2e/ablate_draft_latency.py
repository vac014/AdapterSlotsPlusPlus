#!/usr/bin/env python3
"""
The latency half of the draft-size question, at fixed acceptance.

Holds acceptance at the deployed 160m value and varies only the draft, so the latency
effect is isolated from the quality effect: throughput B*tpv/(t_verify + gamma*t_draft)
falls once gamma*t_draft rivals t_verify, no matter how good the draft is.

Among tiny drafts (68M, 160M) t_draft is negligible against t_verify and the choice is
driven by acceptance instead, which is what accept_ci.py measures. Latency does not depend
on weights, so the drafts run randomly initialised.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT  # noqa: E402
from graph_latency import graph_step_ms  # noqa: E402
from paths import alpaca_lora  # noqa: E402

DRAFTS = [("68M", "JackFram/llama-68m"),
          ("160M", "JackFram/llama-160m"),
          ("1.3B", "princeton-nlp/Sheared-LLaMA-1.3B")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=256)
    p.add_argument("--accept", type=float, default=0.584, help="fixed faithful acceptance (deployed 160M GSM8K)")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True
    t_verify = graph_step_ms(target, args.B, args.context, g + 1, device, dtype, args.max_cache)
    print(f"t_verify(B={args.B}, g+1) = {t_verify:.2f} ms")

    tpv = args.accept * g + 1
    rows = []
    for tag, path in DRAFTS:
        d = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        d.config.use_cache = True
        params = sum(pp.numel() for pp in d.parameters()) / 1e6
        t_draft = graph_step_ms(d, args.B, args.context, 1, device, dtype, args.max_cache)
        t_iter = t_verify + g * t_draft
        thr = args.B * tpv / (t_iter / 1000)
        rows.append({"size": tag, "params_M": params, "t_draft_ms": t_draft,
                     "t_iter_ms": t_iter, "throughput_at_fixed_acc": thr})
        print(f"  {tag:<5} {params:7.1f}M  t_draft={t_draft:6.2f} ms  "
              f"t_iter={t_iter:6.2f} ms  thr@acc={args.accept}: {thr:5.0f} tok/s")
        del d; torch.cuda.empty_cache()

    base_thr = rows[1]["throughput_at_fixed_acc"]  # 160M reference
    print(f"\n  at equal acceptance, 1.3B draft throughput is "
          f"{rows[2]['throughput_at_fixed_acc']/base_thr:.2f}x the 160M "
          f"(latency alone: gamma*t_draft {g*rows[2]['t_draft_ms']:.0f} ms rivals t_verify {t_verify:.0f} ms)")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"ablate_draft_latency_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": args.B, "gamma": g,
                   "t_verify_ms": t_verify, "fixed_accept": args.accept, "rows": rows},
                  f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
