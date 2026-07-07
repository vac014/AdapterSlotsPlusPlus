#!/usr/bin/env python3
"""
Faithful-over-shared lift across workloads.

One fixed real target (llama-7b with alpaca-lora-7b merged), swept over datasets from
mild to strong structure. Per dataset, distil the faithful draft on the target's own
greedy continuations and measure exact-KV acceptance against the shared draft.

The point is where the lift comes from: it tracks how far the shared draft has collapsed
on that workload, not a label we assign the workload in advance. The target is loaded
once; only the prompt distribution and the distilled draft change.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, distill_draft, PROMPT  # noqa: E402
from spec_loops import spec_kv  # noqa: E402

ALPACA = "tloen/alpaca-lora-7b"
# rough a-priori shift level for the writeup (validated by the measured lift)
SHIFT = {"sharegpt": "mild (continuity)", "dolly": "mild", "alpaca": "mild",
         "samsum": "moderate", "gsm8k": "moderate-strong", "mbpp": "strong"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["dolly", "samsum", "gsm8k", "mbpp"])
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--drf_steps", type=int, default=1200)
    p.add_argument("--n_train", type=int, default=150)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--gen_tokens", type=int, default=64)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    g = args.gamma
    tok = AutoTokenizer.from_pretrained(TGT)
    from paths import alpaca_lora; apath = alpaca_lora()

    print("Load target ONCE: llama-7b + REAL alpaca-lora-7b (merge)")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, apath).merge_and_unload().eval()
    target.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    rows = []
    for ds in args.datasets:
        print(f"\n===== dataset={ds} (shift~{SHIFT.get(ds,'?')}) =====")
        train_instr, eval_instr = load_instructions(args.n_train, args.n_eval, ds)
        print(f"  {len(train_instr)} train / {len(eval_instr)} held-out")
        print("  gen distill corpus (target's own greedy continuations)")
        corpus = []
        for i, instr in enumerate(train_instr):
            t = PROMPT.format(instr=instr)
            corpus.append(t + greedy(target, tok, t, args.gen_tokens, device))
        corpus_text = "\n\n".join(corpus)
        print("  distill faithful draft-LoRA")
        faithful = distill_draft(DRF, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device)

        def accept(draft):
            tot = st = 0
            for instr in eval_instr:
                ids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
                _, a, s = spec_kv(draft, target, ids, g, args.eval_tokens)
                tot += a; st += s
            return tot / (st * g)

        a_sh, a_fa = accept(shared), accept(faithful)
        tpv_sh, tpv_fa = a_sh * g + 1, a_fa * g + 1
        row = {"dataset": ds, "shift": SHIFT.get(ds, "?"),
               "accept_shared": a_sh, "accept_faithful": a_fa,
               "lift": a_fa / a_sh if a_sh else float("nan"),
               "tpv_shared": tpv_sh, "tpv_faithful": tpv_fa,
               "tpv_ratio": tpv_fa / tpv_sh}
        rows.append(row)
        print(f"  shared={a_sh:.3f} faithful={a_fa:.3f}  lift={row['lift']:.2f}x  "
              f"tok/verify {tpv_sh:.2f}->{tpv_fa:.2f} ({row['tpv_ratio']:.2f}x)  "
              f"{'FAITHFUL>SHARED' if a_fa>a_sh else 'shared>=faithful'}")
        del faithful; torch.cuda.empty_cache()

    print("\n=== generalization summary (fixed real alpaca target) ===")
    print(f"{'dataset':<10}{'shift':<16}{'shared':>8}{'faithful':>10}{'lift':>7}{'tpv_ratio':>11}")
    for r in rows:
        print(f"{r['dataset']:<10}{r['shift']:<16}{r['accept_shared']:>8.3f}"
              f"{r['accept_faithful']:>10.3f}{r['lift']:>7.2f}{r['tpv_ratio']:>11.2f}")
    wins = sum(1 for r in rows if r["accept_faithful"] > r["accept_shared"])
    print(f"faithful>shared on {wins}/{len(rows)} workloads")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"generalization_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "adapter": ALPACA,
                   "gamma": g, "drf_r": args.drf_r, "rows": rows,
                   "faithful_gt_shared": wins, "n_datasets": len(rows)}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
