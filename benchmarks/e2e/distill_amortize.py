#!/usr/bin/env python3
"""
When does a per-tenant draft pay for itself?

Distilling a faithful draft is a one-time cost per tenant; the throughput it buys is
returned over that tenant's serving lifetime. This measures the real one-time cost in
GPU-seconds (corpus generation plus the full multi-seed early-stopped distillation) and
computes the break-even against the measured non-speculative and WarpPipe throughputs.

Serving the same work faster saves (1/thr_nospec - 1/thr_warp) GPU-seconds per token, so
the distillation is repaid after C / (1/thr_nospec - 1/thr_warp) tokens. Same-GPU
accounting throughout: distillation and serving compete for the one device.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT, distill_draft_es  # noqa: E402
from paths import alpaca_lora  # noqa: E402

def load_tiers(outdir):
    """(no_spec, warppipe) tok/s per workload, read from whatever throughput_tiers.py wrote.

    Break-even is only meaningful against throughput measured on the same device as the
    distillation it is amortizing, so these are read from results/ rather than pinned in
    the source, where they would go stale on any change of GPU, draft recipe or batch size
    while still looking precise.
    """
    tiers = {}
    for path in sorted(glob.glob(os.path.join(outdir, "throughput_tiers_*.json"))):
        with open(path) as f:
            d = json.load(f)
        rows = {r["tier"]: r["tok_s"] for r in d["tiers"]}
        tiers[d["dataset"]] = (rows["no_spec"], rows["warppipe"])   # later files win
    if not tiers:
        raise SystemExit(
            "no throughput_tiers_*.json in %s.\n"
            "Break-even needs measured throughput; run throughput_tiers.py first, e.g.\n"
            "  python benchmarks/e2e/throughput_tiers.py --dataset gsm8k" % outdir)
    return tiers


def main():
    p = argparse.ArgumentParser(description="draft-distillation break-even")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--drf_steps", type=int, default=900)
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--gen_tokens", type=int, default=96)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ckpt_steps", type=int, nargs="+", default=[150, 300, 450, 600, 900])
    p.add_argument("--D", type=int, default=128, help="tokens/request for the request count")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(TGT)
    g = args.gamma

    train_instr, held = load_instructions(args.n_train, args.n_val, args.dataset)
    val_instr = held[:args.n_val]
    print(f"load target 7B + alpaca (merge), {len(train_instr)} train / {len(val_instr)} val")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True
    shared_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared_draft.config.use_cache = True

    # time the one-time per-tenant distillation
    torch.cuda.synchronize(); t_gen0 = time.time()
    corpus = [PROMPT.format(instr=x) + greedy(target, tok, PROMPT.format(instr=x),
              args.gen_tokens, device) for x in train_instr]
    corpus_text = "\n\n".join(corpus)
    torch.cuda.synchronize(); t_gen = time.time() - t_gen0

    mods = ("q_proj", "k_proj", "v_proj", "o_proj")
    torch.cuda.synchronize(); t_dis0 = time.time()
    faithful, es_info = distill_draft_es(
        DRF, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device, target,
        val_instr, g, args.eval_tokens, ckpt_steps=tuple(args.ckpt_steps),
        target_modules=mods, seeds=tuple(args.seeds), shared_draft=shared_draft)
    torch.cuda.synchronize(); t_dis = time.time() - t_dis0

    C = t_gen + t_dis
    print("\n== one-time per-tenant distillation cost C ==")
    print(f"  corpus generation ({len(train_instr)} instr): {t_gen:6.1f} s")
    print(f"  robust distill ({len(args.seeds)} seeds x {args.drf_steps} steps + val): {t_dis:6.1f} s")
    print(f"  TOTAL C = {C:6.1f} s ({C/60:.1f} GPU-min) per tenant")

    # break-even per workload
    tiers = load_tiers(os.path.abspath(args.outdir))
    print(f"\n== break-even (tokens/requests to amortize C={C:.0f} s) ==")
    rows = []
    for wl, (ns, wp) in tiers.items():
        save_per_tok = 1.0 / ns - 1.0 / wp          # GPU-s saved per served token
        Lstar = C / save_per_tok
        reqs = Lstar / args.D
        rows.append({"workload": wl, "thr_nospec": ns, "thr_warp": wp,
                     "breakeven_tokens": Lstar, "breakeven_requests": reqs})
        print(f"  {wl:<9} no_spec {ns} -> warp {wp} tok/s: break-even "
              f"{Lstar/1e3:6.0f}k tokens  = {reqs:5.0f} requests (@{args.D} tok)")

    print(f"\n  Interpretation: a tenant served past ~{min(r['breakeven_requests'] for r in rows):.0f}"
          f"-{max(r['breakeven_requests'] for r in rows):.0f} requests amortizes its distillation; "
          f"a production tenant serves that in minutes of traffic.")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"distill_amortize_{args.dataset}_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "dataset": args.dataset,
                   "n_train": args.n_train, "seeds": args.seeds, "drf_steps": args.drf_steps,
                   "t_corpus_gen_s": t_gen, "t_distill_s": t_dis, "C_total_s": C,
                   "D_tokens_per_request": args.D, "breakeven": rows,
                   "deployed_accept": es_info}, f, indent=2, default=str)
    print("written:", jpath)


if __name__ == "__main__":
    main()
