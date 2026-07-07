#!/usr/bin/env python3
"""
Acceptance with bootstrap confidence intervals on a larger eval set.

The per-workload acceptances are measured on modest held-out sets, so this re-measures
shared against faithful on a larger one (default n=120) and bootstraps a 95% CI over
instructions, on a workload where the win is large (GSM8K) and one where it is delicate
(ShareGPT).

What the interval decides: whether the paired faithful-minus-shared difference excludes
zero. Where it does, the win is not a sampling artifact. Where the intervals overlap, the
honest reading is a tie and a scope boundary, not a win. The faithful draft is the one the
deployed recipe produces.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT, distill_draft_es  # noqa: E402
from spec_loops import spec_kv  # noqa: E402
from paths import alpaca_lora  # noqa: E402


def per_instr_accept(draft, target, pid, g, eval_tokens):
    """Per-instruction acceptance rate = accepted/(steps*g), one value per prompt."""
    out = []
    for x in pid:
        _, a, s = spec_kv(draft, target, x, g, eval_tokens)
        out.append(a / (s * g) if s > 0 else 0.0)
    return np.array(out)


def boot_ci(vals, iters=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.array([rng.choice(vals, n, replace=True).mean() for _ in range(iters)])
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description="acceptance confidence intervals")
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "sharegpt"])
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--drf_steps", type=int, default=600)
    p.add_argument("--n_train", type=int, default=250)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--gen_tokens", type=int, default=96)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(TGT)

    print("load target 7B + alpaca (merge)")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True
    mods = ("q_proj", "k_proj", "v_proj", "o_proj")

    results = []
    for ds in args.datasets:
        print(f"\n=== {ds} (n_eval={args.n_eval}) ===")
        train_instr, held = load_instructions(args.n_train, args.n_val + args.n_eval, ds)
        val_instr, eval_instr = held[:args.n_val], held[args.n_val:]
        corpus_text = "\n\n".join(PROMPT.format(instr=x) +
                                  greedy(target, tok, PROMPT.format(instr=x), args.gen_tokens, device)
                                  for x in train_instr)
        pid = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids.to(device)
               for x in eval_instr]

        faithful, info = distill_draft_es(
            DRF, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device, target,
            val_instr, g, args.eval_tokens, ckpt_steps=(150, 300, 450, 600),
            target_modules=mods, seeds=tuple(args.seeds), shared_draft=shared)

        sh = per_instr_accept(shared, target, pid, g, args.eval_tokens)
        fa = per_instr_accept(faithful, target, pid, g, args.eval_tokens)
        m_sh, lo_sh, hi_sh = boot_ci(sh)
        m_fa, lo_fa, hi_fa = boot_ci(fa)
        # paired diff CI
        diff = fa - sh
        m_d, lo_d, hi_d = boot_ci(diff)
        robust = lo_d > 0     # faithful>shared CI excludes 0
        results.append({"dataset": ds, "n_eval": len(pid),
                        "shared": {"mean": m_sh, "ci95": [lo_sh, hi_sh]},
                        "faithful": {"mean": m_fa, "ci95": [lo_fa, hi_fa]},
                        "diff": {"mean": m_d, "ci95": [lo_d, hi_d]}, "robust_win": robust})
        print(f"  shared   accept = {m_sh:.3f}  95% CI [{lo_sh:.3f}, {hi_sh:.3f}]")
        print(f"  faithful accept = {m_fa:.3f}  95% CI [{lo_fa:.3f}, {hi_fa:.3f}]")
        print(f"  diff (fa-sh)    = {m_d:+.3f}  95% CI [{lo_d:+.3f}, {hi_d:+.3f}]  "
              f"-> {'CI excludes 0' if robust else 'CI includes 0'}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"accept_ci_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "gamma": g,
                   "n_eval": args.n_eval, "seeds": args.seeds, "results": results},
                  f, indent=2)
    print("\nwritten:", jpath)


if __name__ == "__main__":
    main()
