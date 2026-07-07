#!/usr/bin/env python3
"""
The floor that keeps a faithful draft from ever losing to the shared one.

A faithful draft is the base draft plus a LoRA delta, so it should never score below the
base. On small, diverse corpora it can, by overfitting them. Two mechanisms fix that,
both chosen without looking at the test metric:

  1. a (rank, steps) sweep, selected on validation acceptance;
  2. a deploy rule that ships the faithful draft only if it beats the shared draft on
     validation, and falls back to shared otherwise.

Held-out data is split into disjoint validation and test. Corpus generation, config
selection and the deploy decision see validation only. Every reported number is on test,
which selects nothing, so "deployed >= shared" is a generalization result rather than an
artifact of tuning against the eval set.
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
# a-priori grid, from most- to least-regularized; NONE chosen by test accuracy
CONFIGS = [(4, 300, 1e-4), (8, 500, 1e-4), (8, 800, 2e-4), (16, 800, 2e-4)]
# "scaled" grid: for a large, heterogeneous corpus (e.g. ShareGPT) where the
# regularized grid under-fits. More data supports higher capacity + more steps
# without overfit; validation still chooses which one (or the shared fallback).
CONFIGS_SCALED = [(8, 800, 2e-4), (16, 1200, 2e-4), (32, 1500, 2e-4), (32, 2000, 1e-4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["sharegpt", "dolly", "samsum", "gsm8k", "mbpp"])
    p.add_argument("--n_train", type=int, default=150)
    p.add_argument("--n_val", type=int, default=15)
    p.add_argument("--n_test", type=int, default=15)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--gen_tokens", type=int, default=64)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--grid", choices=["reg", "scaled"], default="reg",
                   help="reg = regularized grid (default); scaled = higher-capacity "
                        "grid for large heterogeneous corpora")
    p.add_argument("--modules", choices=["qv", "qkvo"], default="qv",
                   help="faithful draft-LoRA target modules")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    g = args.gamma
    grid = CONFIGS_SCALED if args.grid == "scaled" else CONFIGS
    modules = ("q_proj", "k_proj", "v_proj", "o_proj") if args.modules == "qkvo" \
        else ("q_proj", "v_proj")
    print(f"grid={args.grid} ({len(grid)} configs)  modules={args.modules}")
    tok = AutoTokenizer.from_pretrained(TGT)
    from paths import alpaca_lora; apath = alpaca_lora()

    print("Load target ONCE: llama-7b + REAL alpaca-lora-7b")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, apath).merge_and_unload().eval()
    target.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    def accept(draft, instrs):
        tot = st = 0
        for instr in instrs:
            ids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
            _, a, s = spec_kv(draft, target, ids, g, args.eval_tokens)
            tot += a; st += s
        return tot / (st * g)

    out = []
    for ds in args.datasets:
        print(f"\n===== {ds} =====")
        # held-out = n_val + n_test, disjoint; train separate
        train_instr, held = load_instructions(args.n_train, args.n_val + args.n_test, ds)
        val_instr, test_instr = held[:args.n_val], held[args.n_val:]
        corpus = []
        for instr in train_instr:
            t = PROMPT.format(instr=instr)
            corpus.append(t + greedy(target, tok, t, args.gen_tokens, device))
        corpus_text = "\n\n".join(corpus)

        sh_val, sh_test = accept(shared, val_instr), accept(shared, test_instr)
        print(f"  shared: val={sh_val:.3f} test={sh_test:.3f}")

        # regularization sweep: SELECT BY VALIDATION ONLY
        best_cfg, best_val, best_draft = None, -1.0, None
        cfg_rows = []
        for (r, steps, lr) in grid:
            fa = distill_draft(DRF, tok, corpus_text, r, steps, lr, device,
                               target_modules=modules)
            fa_val = accept(fa, val_instr)
            cfg_rows.append({"r": r, "steps": steps, "lr": lr, "val": fa_val})
            print(f"    r={r:<3} steps={steps:<4} lr={lr:<6} val={fa_val:.3f}")
            if fa_val > best_val:
                best_val, best_cfg = fa_val, (r, steps, lr)
                if best_draft is not None:
                    del best_draft
                best_draft = fa
            else:
                del fa
            torch.cuda.empty_cache()

        # report the VAL-selected faithful config on TEST
        fa_test = accept(best_draft, test_instr)
        # validated floor decision: made on VAL, applied to TEST number
        deploy_faithful = best_val >= sh_val
        deployed_test = fa_test if deploy_faithful else sh_test
        del best_draft; torch.cuda.empty_cache()

        row = {"dataset": ds, "shared_test": sh_test, "shared_val": sh_val,
               "sel_config": {"r": best_cfg[0], "steps": best_cfg[1], "lr": best_cfg[2]},
               "faithful_val": best_val, "faithful_test": fa_test,
               "deploy_faithful": deploy_faithful, "deployed_test": deployed_test,
               "deployed_ge_shared": deployed_test >= sh_test,
               "faithful_raw_ge_shared": fa_test >= sh_test,
               "configs_val": cfg_rows}
        out.append(row)
        print(f"  -> selected r={best_cfg[0]}/{best_cfg[1]} (val={best_val:.3f}); "
              f"TEST faithful={fa_test:.3f} vs shared={sh_test:.3f}; "
              f"deploy={'faithful' if deploy_faithful else 'shared(fallback)'} "
              f"-> deployed_test={deployed_test:.3f} "
              f"({'>= shared' if deployed_test >= sh_test else 'BELOW shared'})")

    print("\n=== summary (config and floor chosen on val, numbers on test) ===")
    print(f"{'dataset':<10}{'shared':>8}{'faith_raw':>10}{'deployed':>10}{'deploy?':>10}")
    for r in out:
        print(f"{r['dataset']:<10}{r['shared_test']:>8.3f}{r['faithful_test']:>10.3f}"
              f"{r['deployed_test']:>10.3f}{('faith' if r['deploy_faithful'] else 'shared'):>10}")
    draws = sum(r["deployed_ge_shared"] for r in out)
    raw_wins = sum(r["faithful_raw_ge_shared"] for r in out)
    print(f"deployed >= shared on {draws}/{len(out)} (floor guarantee); "
          f"raw faithful >= shared on {raw_wins}/{len(out)}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"faithful_floor_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"adapter": ALPACA, "gamma": g, "protocol": "val-select/test-report",
                   "grid": args.grid, "modules": args.modules,
                   "n_train": args.n_train, "gen_tokens": args.gen_tokens,
                   "n_val": args.n_val, "n_test": args.n_test, "rows": out,
                   "deployed_ge_shared": draws, "raw_wins": raw_wins,
                   "n_datasets": len(out)}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
