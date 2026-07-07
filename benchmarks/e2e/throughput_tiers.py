#!/usr/bin/env python3
"""
Three-tier throughput on a real adapter: the six-workload table.

For one workload: distil the faithful draft with the deployed recipe, measure exact-KV
acceptance for the shared and the faithful draft on a disjoint eval split, graph-capture
the target decode, the target verify of gamma+1, the shared draft step and the draft-side
SGMV step, and compose

  AdapterSlots (no spec)  = B / t_decode
  AdapterSlots++ spec     = B * tpv_shared   / (t_verify + gamma * t_draft_shared)
  AdapterSlots++ WarpPipe = B * tpv_faithful / (t_verify + gamma * t_sgmv)

with tpv = a*gamma + 1 from the measured acceptance. Both speculative tiers run one flat
draft pass, so t_sgmv is within noise of t_draft_shared and the win is the acceptance
gap, delivered at a draft cost that does not grow with the number of resident adapters.

Sweeping --K shows the flatness directly. The composed tok/s are validated against a
real wall-clock loop in wallclock_validate.py.
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
from recipe import load_instructions, greedy, PROMPT, distill_draft, distill_draft_es  # noqa: E402
from spec_loops import spec_kv, nospec_kv, nospec_nokv  # noqa: E402
from serve_multitenant import extract_draft_lora, wrap_draft, STATE  # noqa: E402
from graph_latency import graph_step_ms  # noqa: E402

ALPACA = "tloen/alpaca-lora-7b"
ALPACA_13B = "chansung/alpaca-lora-13b"


def distill_peft(base_path, tok, corpus_text, r, steps, lr, device,
                 target_modules=("q_proj", "v_proj")):
    """Distill a draft-LoRA and return the (unmerged) PEFT model, so we can both
    extract BGMV weights (WarpPipe path) AND merge a copy (acceptance path)."""
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r,
        target_modules=list(target_modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
    model.train()
    ids = tok(corpus_text, return_tensors="pt").input_ids[0]
    L = 128
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L, L // 2)]
    x = torch.stack([c for c in chunks if len(c) == L])
    dl = DataLoader(TensorDataset(x), batch_size=8, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    it = iter(dl); t0 = time.time()
    for s in range(steps):
        try: (xb,) = next(it)
        except StopIteration: it = iter(dl); (xb,) = next(it)
        xb = xb.to(device)
        out = model(input_ids=xb, labels=xb); out.loss.backward()
        opt.step(); opt.zero_grad()
        if s % max(1, steps // 4) == 0 or s == steps - 1:
            print(f"  [drf] {s}/{steps} loss {out.loss.item():.3f} {time.time()-t0:.0f}s")
    model.eval()
    return model


def main():
    # The defaults ARE the deployed recipe: the numbers in results/ come out of a bare
    # `python throughput_tiers.py --dataset <ds>`. A thinner draft (rank 8, q/v only, 600
    # steps on 200 rows) trains in a third of the time and reaches nowhere near the same
    # acceptance, so it is not a cheaper version of this measurement -- it is a different one.
    p = argparse.ArgumentParser()
    p.add_argument("--drf_steps", type=int, default=900)
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--n_train", type=int, default=300,
                   help="ShareGPT trains on 400; MBPP has only 374 rows, so keep "
                        "n_train + n_val + n_eval under that")
    p.add_argument("--n_eval", type=int, default=24)
    p.add_argument("--n_val", type=int, default=20,
                   help="disjoint val set for early-stopping checkpoint selection")
    p.add_argument("--final_step_draft", action="store_true",
                   help="disable early stopping; use the final-step draft (legacy, "
                        "seed-unstable on diverse workloads)")
    p.add_argument("--ckpt_steps", type=int, nargs="+",
                   default=[150, 300, 450, 600, 900])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="train N drafts; deploy the best-VAL one (init-lottery fix)")
    p.add_argument("--gen_tokens", type=int, default=96)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "mbpp", "alpaca", "dolly", "samsum", "sharegpt"])
    p.add_argument("--scale", choices=["7b", "13b"], default="7b",
                   help="target model scale: 7b (llama-7b+alpaca-lora-7b, default) or "
                        "13b (llama-13b+chansung/alpaca-lora-13b). Draft stays llama-160m.")
    p.add_argument("--modules", choices=["qv", "qkvo"], default="qkvo",
                   help="faithful draft-LoRA target modules (must match the "
                        "distilled config whose throughput is being measured)")
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=128)
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    g, B, C = args.gamma, args.B, args.context
    modules = ("q_proj", "k_proj", "v_proj", "o_proj") if args.modules == "qkvo" \
        else ("q_proj", "v_proj")
    print(f"draft modules = {modules}")
    if args.scale == "13b":
        from paths import llama13b, alpaca_lora_13b
        tgt_path, apath, adapter_id = llama13b(), alpaca_lora_13b(), ALPACA_13B
    else:
        from paths import alpaca_lora
        tgt_path, apath, adapter_id = TGT, alpaca_lora(), ALPACA
    tok = AutoTokenizer.from_pretrained(tgt_path)

    print(f"Stage 0: real instructions (dataset={args.dataset})")
    # held-out = disjoint val (early-stop checkpoint selection) + eval (reported)
    train_instr, held = load_instructions(args.n_train, args.n_eval + args.n_val, args.dataset)
    val_instr, eval_instr = held[:args.n_val], held[args.n_val:]
    print(f"  {len(train_instr)} train / {len(val_instr)} val (early-stop) / "
          f"{len(eval_instr)} eval (reported)")

    print(f"Stage 1: {tgt_path} + REAL {adapter_id} (merge) -> target")
    base = AutoModelForCausalLM.from_pretrained(tgt_path, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, apath).merge_and_unload().eval()
    target.config.use_cache = True

    print(f"Stage 2: distill corpus = target's greedy {args.dataset} continuations")
    corpus = []
    for i, instr in enumerate(train_instr):
        t = PROMPT.format(instr=instr)
        corpus.append(t + greedy(target, tok, t, args.gen_tokens, device))
        if i % 50 == 0:
            print(f"  gen {i}/{len(train_instr)}")
    corpus_text = "\n\n".join(corpus)

    shared_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared_draft.config.use_cache = True

    print("Stage 3: distill faithful draft-LoRA")
    es_info = None
    if args.final_step_draft:
        print("  (legacy: final-step draft, no early stopping)")
        faithful_merged = distill_draft(DRF, tok, corpus_text, args.drf_r, args.drf_steps,
                                        2e-4, device, target_modules=modules)
    else:
        print("  (multi-seed early-stop, best-val, shared floor)")
        faithful_merged, es_info = distill_draft_es(
            DRF, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device, target,
            val_instr, g, args.eval_tokens, ckpt_steps=tuple(args.ckpt_steps),
            target_modules=modules, seeds=tuple(args.seeds), shared_draft=shared_draft)

    print("Stage 4: acceptance (exact B=1 KV spec) on held-out eval (disjoint from val)")
    pid = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids.to(device) for x in eval_instr]
    assert torch.equal(nospec_kv(target, pid[0], 12), nospec_nokv(target, pid[0], 12))

    def accept_of(draft):
        acc = st = 0
        for x in pid:
            _, a, s = spec_kv(draft, target, x, g, args.eval_tokens)
            acc += a; st += s
        return acc / (st * g)

    a_sh = accept_of(shared_draft)
    a_fa = accept_of(faithful_merged)
    tpv_sh, tpv_fa = a_sh * g + 1, a_fa * g + 1
    print(f"\nREAL-adapter acceptance: shared(AS++ spec)={a_sh:.3f} ({tpv_sh:.2f} tok/verify)  "
          f"faithful(WarpPipe)={a_fa:.3f} ({tpv_fa:.2f} tok/verify)  gap {tpv_fa/tpv_sh:.2f}x")

    print("Stage 5: graph-captured latencies (target verify/decode, draft paths)")
    t_decode = graph_step_ms(target, B, C, 1, device, dtype, args.max_cache)
    t_verify = graph_step_ms(target, B, C, g + 1, device, dtype, args.max_cache)
    t_draft_shared = graph_step_ms(shared_draft, B, C, 1, device, dtype, args.max_cache)
    print(f"  decode={t_decode:.2f}ms verify(g+1)={t_verify:.2f}ms shared-draft={t_draft_shared:.2f}ms")

    # re-distill for the wrapped (WarpPipe) draft, since peft_draft was merged
    peft_draft2 = distill_peft(DRF, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4,
                               device, target_modules=modules)

    print("Stage 6: three-tier throughput (all flat in K)")
    rows = []
    for K in args.K:
        w, scale = extract_draft_lora(peft_draft2, K, dtype, device, modules=modules)
        STATE["scale"] = scale
        wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
        wd.config.use_cache = True
        wrap_draft(wd, w)
        adapter = torch.arange(B, device=device) % K

        def sgmv_pre(bb, mm):
            STATE["mode"] = "lora"; STATE["idx"] = adapter.repeat_interleave(mm)
        t_sgmv = graph_step_ms(wd, B, C, 1, device, dtype, args.max_cache, sgmv_pre)
        STATE["mode"] = "none"

        tp_nospec = B * 1 / (t_decode / 1000)
        tp_spec = B * tpv_sh / ((t_verify + g * t_draft_shared) / 1000)
        tp_warp = B * tpv_fa / ((t_verify + g * t_sgmv) / 1000)
        rows.append({"K": K, "t_sgmv_ms": t_sgmv, "no_spec": tp_nospec,
                     "aspp_spec": tp_spec, "warppipe": tp_warp})
        print(f"K={K:>3}  no_spec={tp_nospec:6.0f}  AS++spec={tp_spec:6.0f}  "
              f"WarpPipe={tp_warp:6.0f}  tok/s   (WarpPipe/AS++spec={tp_warp/tp_spec:.2f}x "
              f"/no_spec={tp_warp/tp_nospec:.2f}x)")
        del wd; torch.cuda.empty_cache()

    rm = rows[-1]
    print("\n=== summary ===")
    print(f"  acceptance  shared {a_sh:.3f} -> faithful {a_fa:.3f}")
    print(f"  AS++ spec          vs no_spec  : {rm['aspp_spec']/rm['no_spec']:.2f}x")
    print(f"  WarpPipe           vs no_spec  : {rm['warppipe']/rm['no_spec']:.2f}x")
    print(f"  WarpPipe           vs AS++ spec: {rm['warppipe']/rm['aspp_spec']:.2f}x")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"throughput_tiers_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "adapter": adapter_id,
                   "scale": args.scale, "target": tgt_path,
                   "dataset": args.dataset, "modules": args.modules,
                   "early_stop": (not args.final_step_draft), "es_info": es_info,
                   "n_val": args.n_val, "n_eval": args.n_eval,
                   "drf_r": args.drf_r, "drf_steps": args.drf_steps,
                   "n_train": args.n_train, "gen_tokens": args.gen_tokens, "B": B, "gamma": g,
                   "accept_shared": a_sh, "accept_faithful": a_fa,
                   "tpv_shared": tpv_sh, "tpv_faithful": tpv_fa,
                   "t_decode_ms": t_decode, "t_verify_ms": t_verify,
                   "t_draft_shared_ms": t_draft_shared, "tiers": rows,
                   "warppipe_vs_asppspec": rm["warppipe"] / rm["aspp_spec"],
                   "warppipe_vs_nospec": rm["warppipe"] / rm["no_spec"]}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
