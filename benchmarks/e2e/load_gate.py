#!/usr/bin/env python3
"""Where speculation stops paying, and the gate that switches it off.

Per iteration a speculative step commits tpv = a*gamma + 1 tokens per sequence in
t_iter = t_verify + gamma*t_draft, while a plain decode commits 1 in t_decode.
Speculating is therefore worth it iff

    tpv > t_iter / t_decode                                            (the gate)

The right-hand side is a property of the batch, not of the draft: as B grows the
target verify becomes compute-bound and a (gamma+1)-token verify costs closer to
(gamma+1) decodes, so the threshold climbs and a weak draft ages out of usefulness.

This measures both sides of that inequality on the real models. The four step types
(target decode, target verify of gamma+1, shared 160m draft, draft-side SGMV) are
graph-captured at each batch size, and the gate is then evaluated for the shared
and the faithful draft using their deployed acceptances. Latency is weight-
independent, so the SGMV draft carries a random q,k,v,o LoRA at the real rank.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from graph_latency import graph_step_ms  # noqa: E402
from serve_multitenant import extract_draft_lora, wrap_draft, STATE  # noqa: E402

# deployed acceptance (shared, faithful) per workload, from throughput_tiers.py
ACC = {"gsm8k": (0.213, 0.584), "alpaca": (0.244, 0.388), "dolly": (0.262, 0.378),
       "mbpp": (0.108, 0.369), "samsum": (0.223, 0.366), "sharegpt": (0.214, 0.316)}


def measure_latency_curves(B_grid, C, g, max_cache, device, dtype):
    """Graph-captured latency (ms) at each batch size for the four step types."""
    from paths import alpaca_lora
    apath = alpaca_lora()
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, apath).merge_and_unload().eval()
    target.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    rnd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device)
    rnd = get_peft_model(rnd, LoraConfig(r=32, lora_alpha=64, target_modules=modules,
                                         lora_dropout=0.0, task_type="CAUSAL_LM"))

    cur = {}
    for B in B_grid:
        t_dec = graph_step_ms(target, B, C, 1, device, dtype, max_cache)
        t_ver = graph_step_ms(target, B, C, g + 1, device, dtype, max_cache)
        t_drf = graph_step_ms(shared, B, C, 1, device, dtype, max_cache)

        K = min(B, 32)
        w, scale = extract_draft_lora(rnd, K, dtype, device, modules=tuple(modules))
        STATE["scale"] = scale
        wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
        wd.config.use_cache = True
        wrap_draft(wd, w)
        adapter = torch.arange(B, device=device) % K

        def sgmv_pre(bb, mm):
            STATE["mode"] = "lora"
            STATE["idx"] = adapter.repeat_interleave(mm)

        t_sg = graph_step_ms(wd, B, C, 1, device, dtype, max_cache, sgmv_pre)
        STATE["mode"] = "none"
        cur[B] = dict(decode=t_dec, verify=t_ver, draft=t_drf, sgmv=t_sg)
        print(f"  B={B:>3}  decode={t_dec:6.2f}  verify={t_ver:6.2f}  "
              f"draft={t_drf:5.2f}  sgmv={t_sg:5.2f} ms")
        del wd
        torch.cuda.empty_cache()
    del target, base, shared, rnd
    torch.cuda.empty_cache()
    return cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gsm8k", choices=list(ACC))
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=256)
    p.add_argument("--B_grid", type=int, nargs="+",
                   default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    a_sh, a_fa = ACC[args.dataset]
    tpv_sh, tpv_fa = a_sh * g + 1, a_fa * g + 1
    print(f"{args.dataset}: accept shared={a_sh} faithful={a_fa} "
          f"(tpv {tpv_sh:.2f} / {tpv_fa:.2f})")

    print("latency vs batch (graph-captured)")
    cur = measure_latency_curves(args.B_grid, args.context, g, args.max_cache, device, dtype)

    rows = []
    print("\ngate: speculate iff tpv > t_iter/t_decode")
    for B in sorted(cur):
        c = cur[B]
        thr_sh = (c["verify"] + g * c["draft"]) / c["decode"]
        thr_fa = (c["verify"] + g * c["sgmv"]) / c["decode"]
        row = {"B": B, "verify_over_decode": c["verify"] / c["decode"],
               "threshold_shared": thr_sh, "threshold_faithful": thr_fa,
               "shared_open": tpv_sh > thr_sh, "faithful_open": tpv_fa > thr_fa}
        rows.append(row)
        print(f"  B={B:>3}  verify/decode={row['verify_over_decode']:.2f}  "
              f"threshold {thr_sh:.2f}/{thr_fa:.2f}  gate "
              f"shared={'open' if row['shared_open'] else 'CLOSED':>6}  "
              f"faithful={'open' if row['faithful_open'] else 'CLOSED':>6}")

    closes_sh = next((r["B"] for r in rows if not r["shared_open"]), None)
    closes_fa = next((r["B"] for r in rows if not r["faithful_open"]), None)
    print(f"\nshared draft ages out at B={closes_sh}; faithful at B={closes_fa}")

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"load_gate_{args.dataset}_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "dataset": args.dataset,
                   "accept_shared": a_sh, "accept_faithful": a_fa, "gamma": g,
                   "tpv_shared": tpv_sh, "tpv_faithful": tpv_fa,
                   "latency_ms": cur, "gate": rows,
                   "shared_closes_at_B": closes_sh,
                   "faithful_closes_at_B": closes_fa}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
