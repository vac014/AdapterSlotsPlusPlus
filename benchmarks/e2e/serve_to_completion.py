#!/usr/bin/env python3
"""
Real batched speculative decode, run until every request finishes.

The fixed-iteration sweeps advance every row the same number of steps. This runs held-out
prompts to completion instead, so acceptance compounds over the sequence the way it does
in serving: ragged per-row acceptance, real rollback of rejected tokens, real batched
verify, wall-clock timing, for the non-speculative tier and the faithful draft.

Reports the throughput ratio and the per-request completion spread, which the fixed-
iteration view cannot show. Decode is full-recompute with no KV cache, so absolute tok/s
is a lower bound and the ratio is conservative; the KV-cached, graph-captured path is
serve_kv_graph.py.
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
from serve_multitenant import spec_iter, nospec_iter, STATE  # noqa: E402
from paths import alpaca_lora  # noqa: E402


@torch.inference_mode()
def run_to_completion(kind, target, draft, prompts, gamma, out_tokens, pad_id, device):
    """Drain a batch: iterate until every row has produced at least out_tokens.

    Throughput is goodput, the requested B*out_tokens over the wall clock, not the
    tokens the loop happened to emit. A speculative row overshoots out_tokens on the
    iteration it finishes (it commits up to gamma+1 at once) and rows that are already
    done keep decoding until the slowest row lands, so counting emitted tokens would
    credit the speculative tier for work nobody asked for while the non-speculative
    tier, which commits exactly one token per row per iteration, overshoots by nothing.
    That asymmetry is worth roughly 1.6x on this loop.

    Returns (goodput_tok_s, requested, emitted, sec, per_row_finish_iter, iters).
    """
    rows = [p.clone() for p in prompts]
    start_len = [p.numel() for p in prompts]
    B = len(rows)
    adapter = torch.zeros(B, device=device, dtype=torch.long)  # single tenant
    finish_iter = [None] * B
    # warmup (untimed)
    if kind == "no_spec":
        nospec_iter(target, rows, pad_id, device)
    else:
        spec_iter(draft, target, rows, adapter, gamma, "shared_draft", pad_id, device)
    rows = [p.clone() for p in prompts]
    torch.cuda.synchronize(); t0 = time.time()
    it = 0
    while any(f is None for f in finish_iter):
        it += 1
        if kind == "no_spec":
            rows, _ = nospec_iter(target, rows, pad_id, device)
        else:
            STATE["mode"] = "none"
            rows, _ = spec_iter(draft, target, rows, adapter, gamma, "shared_draft",
                                pad_id, device)
        for i in range(B):
            if finish_iter[i] is None and rows[i].numel() - start_len[i] >= out_tokens:
                finish_iter[i] = it
        if it > 4000:
            break
    torch.cuda.synchronize(); sec = time.time() - t0
    emitted = sum(rows[i].numel() - start_len[i] for i in range(B))
    requested = B * out_tokens
    return requested / sec, requested, emitted, sec, finish_iter, it


def main():
    p = argparse.ArgumentParser(description="real batched speculative decode, run to completion")
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--out_tokens", type=int, default=64)
    p.add_argument("--drf_steps", type=int, default=500)
    p.add_argument("--n_train", type=int, default=150)
    p.add_argument("--n_val", type=int, default=16)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(TGT)
    pad_id = tok.eos_token_id or 0

    train_instr, held = load_instructions(args.n_train, args.n_val + args.B, args.dataset)
    val_instr = held[:args.n_val]
    serve_instr = held[args.n_val:args.n_val + args.B]
    print(f"load target 7B + alpaca; {len(serve_instr)} serve requests, B={args.B}")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = False
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = False

    print("distill faithful gsm8k draft (single-tenant, for the WarpPipe tier)")
    corpus_text = "\n\n".join(PROMPT.format(instr=x) +
                              greedy(target, tok, PROMPT.format(instr=x), 96, device)
                              for x in train_instr)
    faithful, info = distill_draft_es(
        DRF, tok, corpus_text, 32, args.drf_steps, 2e-4, device, target, val_instr, g, 64,
        ckpt_steps=(150, 300, 500), target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        seeds=(0, 1), shared_draft=shared)
    faithful.config.use_cache = False

    prompts = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids[0].to(device)
               for x in serve_instr]

    print("\n== batched loop to completion (real prompts, ragged acceptance, wall clock) ==")
    ns_tps, ns_req, ns_emit, ns_sec, ns_fin, ns_it = run_to_completion(
        "no_spec", target, None, prompts, g, args.out_tokens, pad_id, device)
    wp_tps, wp_req, wp_emit, wp_sec, wp_fin, wp_it = run_to_completion(
        "warp", target, faithful, prompts, g, args.out_tokens, pad_id, device)
    STATE["mode"] = "none"

    # identical work on both tiers, so the drain ratio and the goodput ratio agree
    ratio = ns_sec / wp_sec
    ns_perit = ns_sec / ns_it
    wp_perit = wp_sec / wp_it
    ns_lat = np.array(ns_fin) * ns_perit
    wp_lat = np.array(wp_fin) * wp_perit
    print(f"  no_spec : drains in {ns_sec:5.1f}s  ({ns_tps:6.1f} tok/s goodput, {ns_it} iters, "
          f"{ns_emit} tokens emitted for {ns_req} requested)")
    print(f"  WarpPipe: drains in {wp_sec:5.1f}s  ({wp_tps:6.1f} tok/s goodput, {wp_it} iters, "
          f"{wp_emit} tokens emitted for {wp_req} requested)")
    print(f"  same-work speedup = {ratio:.2f}x")
    print(f"  per-request completion (ragged accept): "
          f"no_spec p50={np.percentile(ns_lat,50):.1f}s p99={np.percentile(ns_lat,99):.1f}s | "
          f"warp p50={np.percentile(wp_lat,50):.1f}s p99={np.percentile(wp_lat,99):.1f}s")
    print(f"  deployed faithful acceptance (val-selected): {info}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"serve_to_completion_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": args.B, "gamma": g,
                   "out_tokens": args.out_tokens,
                   "no_spec_goodput_tok_s": ns_tps, "warp_goodput_tok_s": wp_tps,
                   "no_spec_sec": ns_sec, "warp_sec": wp_sec, "speedup": ratio,
                   "no_spec_tokens_emitted": ns_emit, "warp_tokens_emitted": wp_emit,
                   "tokens_requested": ns_req,
                   "no_spec_finish_iter": ns_fin, "warp_finish_iter": wp_fin,
                   "no_spec_p50_s": float(np.percentile(ns_lat, 50)),
                   "no_spec_p99_s": float(np.percentile(ns_lat, 99)),
                   "warp_p50_s": float(np.percentile(wp_lat, 50)),
                   "warp_p99_s": float(np.percentile(wp_lat, 99))},
                  f, indent=2, default=str)
    print("written:", jpath)


if __name__ == "__main__":
    main()
