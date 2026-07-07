#!/usr/bin/env python3
"""
Draft size: acceptance against latency.

Sweeps the draft base over 68M, 160M and 1.3B (all llama-vocab, so token ids line up with
the target), distils a faithful draft-LoRA on each, and measures acceptance, graph-captured
draft step latency, and the throughput they compose to.

A larger draft mimics the target better, so tpv rises, but it also costs more per step and
the draft runs gamma times per verify. Past a point the draft's own latency, not its
acceptance, is what limits throughput. The curve is where that point is.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT  # noqa: E402
from recipe import load_instructions, greedy, PROMPT, distill_draft_es  # noqa: E402
from graph_latency import graph_step_ms  # noqa: E402
from spec_loops import spec_kv  # noqa: E402
from paths import alpaca_lora  # noqa: E402

DRAFTS = [("68M", "JackFram/llama-68m"),
          ("160M", "JackFram/llama-160m"),
          ("1.3B", "princeton-nlp/Sheared-LLaMA-1.3B")]


def main():
    p = argparse.ArgumentParser(description="draft-size curve")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--drf_steps", type=int, default=600)
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--n_eval", type=int, default=40)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--gen_tokens", type=int, default=96)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=256)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(TGT)

    train_instr, held = load_instructions(args.n_train, args.n_val + args.n_eval, args.dataset)
    val_instr, eval_instr = held[:args.n_val], held[args.n_val:]
    print(f"target 7B + alpaca; {len(train_instr)} train / {len(val_instr)} val / "
          f"{len(eval_instr)} eval")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True

    print("corpus = target greedy continuations")
    corpus_text = "\n\n".join(PROMPT.format(instr=x) +
                              greedy(target, tok, PROMPT.format(instr=x), args.gen_tokens, device)
                              for x in train_instr)
    pid = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids.to(device) for x in eval_instr]

    # target verify latency (fixed across draft sizes)
    t_verify = graph_step_ms(target, args.B, args.context, g + 1, device, dtype, args.max_cache)
    print(f"t_verify(B={args.B}, g+1) = {t_verify:.2f} ms\n")

    mods = ("q_proj", "k_proj", "v_proj", "o_proj")
    rows = []
    for tag, path in DRAFTS:
        print(f"=== draft {tag} [{path}] ===")
        try:
            shared = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
            shared.config.use_cache = True
            # shared (adapter-blind) acceptance
            acc = st = 0
            for x in pid:
                _, a, s = spec_kv(shared, target, x, g, args.eval_tokens)
                acc += a; st += s
            a_shared = acc / (st * g)
            # faithful distilled draft-LoRA
            faithful, info = distill_draft_es(
                path, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device, target,
                val_instr, g, args.eval_tokens, ckpt_steps=(150, 300, 450, 600),
                target_modules=mods, seeds=tuple(args.seeds), shared_draft=shared)
            acc = st = 0
            for x in pid:
                _, a, s = spec_kv(faithful, target, x, g, args.eval_tokens)
                acc += a; st += s
            a_faith = acc / (st * g)
            # graph draft latency (faithful merged draft, m=1 decode step)
            t_draft = graph_step_ms(faithful, args.B, args.context, 1, device, dtype, args.max_cache)
            tpv = a_faith * g + 1
            t_iter = t_verify + g * t_draft
            thr = args.B * tpv / (t_iter / 1000)
            rows.append({"size": tag, "path": path, "accept_shared": a_shared,
                         "accept_faithful": a_faith, "tpv": tpv, "t_draft_ms": t_draft,
                         "t_iter_ms": t_iter, "throughput": thr})
            print(f"  accept shared={a_shared:.3f} faithful={a_faith:.3f}  tpv={tpv:.2f}  "
                  f"t_draft={t_draft:.2f}ms  t_iter={t_iter:.2f}ms  -> {thr:.0f} tok/s\n")
            del shared, faithful
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on {tag}; skipping (report as-is)\n")
        torch.cuda.empty_cache()

    print("=== draft-size curve ===")
    print(f"{'size':<6}{'accept_fa':>10}{'tpv':>7}{'t_draft':>9}{'tok/s':>8}")
    for r in rows:
        print(f"{r['size']:<6}{r['accept_faithful']:>10.3f}{r['tpv']:>7.2f}"
              f"{r['t_draft_ms']:>8.2f}m{r['throughput']:>8.0f}")
    if rows:
        best = max(rows, key=lambda r: r["throughput"])
        print(f"\n  throughput peaks at draft={best['size']} ({best['throughput']:.0f} tok/s)")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"ablate_draft_size_{args.dataset}_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "dataset": args.dataset,
                   "B": args.B, "gamma": g, "t_verify_ms": t_verify, "rows": rows},
                  f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
