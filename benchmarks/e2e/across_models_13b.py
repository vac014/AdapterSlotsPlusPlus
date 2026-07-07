#!/usr/bin/env python3
"""
Does the tier ordering hold when the target doubles?

Target is llama-13b with chansung/alpaca-lora-13b merged, fp16, no quantization. The draft
stays the same llama-160m, which is the harder regime on purpose: a fixed-capacity draft
mimicking a 2x larger target.

A fp16 13B is about 26 GB and cannot be co-resident with a trainable draft on our GPUs, so
the driver is phased and never holds both:

  1. target resident, generate the distillation corpus, then free it
  2. target absent, train one draft per seed and snapshot the LoRA weights to CPU
  3. target resident, score the shared floor and every snapshot on validation, deploy the
     best, and report shared vs deployed on the disjoint eval split
  4. target resident, graph-capture the 13B verify and decode; then free it and capture the
     draft steps

The metric is the same exact-KV acceptance used at 7B and the target is the same fp16
weights, so the 7B and 13B numbers are comparable. No configuration here holds a 13B target
and a trainable draft co-resident and serving, so no end-to-end 13B serving number is
claimed; what is reported is acceptance and the composed tiers.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from distill import DRF  # noqa: E402  (tiny 160m draft base)
from recipe import (load_instructions, greedy, PROMPT,  # noqa: E402
                             _lora_snapshot, _rebuild_merged)
from spec_loops import spec_kv, nospec_kv, nospec_nokv  # noqa: E402
from serve_multitenant import extract_draft_lora, wrap_draft, STATE  # noqa: E402
from graph_latency import graph_step_ms  # noqa: E402
from throughput_tiers import distill_peft  # noqa: E402


def build_target(device, dtype):
    """llama-13b + REAL chansung/alpaca-lora-13b, merged fp16 (the deployed target)."""
    from paths import llama13b, alpaca_lora_13b
    tgt_path = llama13b()
    base = AutoModelForCausalLM.from_pretrained(tgt_path, torch_dtype=dtype).to(device)
    tgt = PeftModel.from_pretrained(base, alpaca_lora_13b()).merge_and_unload().eval()
    tgt.config.use_cache = True
    return tgt, tgt_path


def train_snapshots(corpus_text, tok, r, steps, lr, device, modules, seeds, ckpt_steps):
    """Phase B: per seed, train a draft-LoRA and snapshot (CPU) LoRA weights at
    ckpt_steps. Mirrors distill_draft_es's training loop EXACTLY (same chunking,
    batch, optimizer) but defers all target-dependent eval to Phase C so the 26 GB
    target need not be resident while the draft trains. Returns {seed: {step: snap}}."""
    ids = tok(corpus_text, return_tensors="pt").input_ids[0]
    L = 128
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L, L // 2)]
    x = torch.stack([c for c in chunks if len(c) == L])
    ckpts = [c for c in ckpt_steps if c <= steps] or [steps]
    out = {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device)
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r,
            target_modules=list(modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
        model.train()
        dl = DataLoader(TensorDataset(x), batch_size=8, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        it = iter(dl); t0 = time.time(); snaps = {}
        for s in range(1, steps + 1):
            try:
                (xb,) = next(it)
            except StopIteration:
                it = iter(dl); (xb,) = next(it)
            xb = xb.to(device)
            o = model(input_ids=xb, labels=xb); o.loss.backward()
            opt.step(); opt.zero_grad()
            if s in ckpts:
                snaps[s] = _lora_snapshot(model)
            if s % max(1, steps // 4) == 0 or s == steps:
                print(f"  [seed {seed}] step {s}/{steps} loss {o.loss.item():.3f} "
                      f"{time.time()-t0:.0f}s")
        out[seed] = snaps
        del model, opt; torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "sharegpt", "dolly", "alpaca", "samsum", "mbpp"])
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--drf_steps", type=int, default=600)
    p.add_argument("--modules", choices=["qv", "qkvo"], default="qkvo")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ckpt_steps", type=int, nargs="+", default=[150, 300, 450, 600])
    p.add_argument("--n_train", type=int, default=150)
    p.add_argument("--n_val", type=int, default=16)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--gen_tokens", type=int, default=64)
    p.add_argument("--eval_tokens", type=int, default=64)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--B", type=int, default=16, help="batch for 13B latency (kept small "
                   "so the 26GB fp16 target's graph capture fits alongside squatters)")
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=96)
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--no_throughput", action="store_true",
                   help="acceptance/tpv only; skip 13B latency (defer throughput to A100)")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g, B, C = args.gamma, args.B, args.context
    modules = ("q_proj", "k_proj", "v_proj", "o_proj") if args.modules == "qkvo" \
        else ("q_proj", "v_proj")
    from paths import llama13b
    tok = AutoTokenizer.from_pretrained(llama13b())
    print(f"draft modules = {modules}   dataset = {args.dataset}   seeds = {args.seeds}")

    train_instr, held = load_instructions(args.n_train, args.n_eval + args.n_val, args.dataset)
    val_instr, eval_instr = held[:args.n_val], held[args.n_val:]
    print(f"{len(train_instr)} train / {len(val_instr)} val / {len(eval_instr)} eval")

    # ---- Phase A: target resident -> corpus, then free target -------------------
    print("\n[A] load 13B target (fp16) + real alpaca-lora-13b; generate corpus")
    target, tgt_path = build_target(device, dtype)
    f0, _ = torch.cuda.mem_get_info(); print(f"    target resident, {f0/1e9:.1f} GB free")
    corpus = []
    for i, instr in enumerate(train_instr):
        t = PROMPT.format(instr=instr)
        corpus.append(t + greedy(target, tok, t, args.gen_tokens, device))
        if i % 50 == 0:
            print(f"    gen {i}/{len(train_instr)}")
    corpus_text = "\n\n".join(corpus)
    del target; torch.cuda.empty_cache()
    print("    corpus built; target freed")

    # ---- Phase B: target absent -> train + snapshot drafts ----------------------
    print("\n[B] train faithful draft-LoRAs (target absent)")
    snaps_by_seed = train_snapshots(corpus_text, tok, args.drf_r, args.drf_steps, 2e-4,
                                    device, modules, args.seeds, args.ckpt_steps)

    # ---- Phase C: target resident -> acceptance (shared floor + all snapshots) ---
    print("\n[C] reload 13B target; exact-KV acceptance on val, deploy best-val")
    target, _ = build_target(device, dtype)
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    def accept(draft, instrs):
        tot = st = 0
        for instr in instrs:
            ids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
            _, a, s = spec_kv(draft, target, ids, g, args.eval_tokens)
            tot += a; st += s
        return tot / (st * g)

    # sanity: exact-KV == no-KV greedy on target (same as 7B track)
    sids = tok(PROMPT.format(instr=eval_instr[0]), return_tensors="pt").input_ids.to(device)
    assert torch.equal(nospec_kv(target, sids, 12), nospec_nokv(target, sids, 12))

    sh_val = accept(shared, val_instr)
    print(f"    shared floor: val {sh_val:.3f}")
    best = {"kind": "shared", "seed": None, "step": None, "val": sh_val, "snap": None}
    per_seed = []
    for seed in args.seeds:
        s_best = {"seed": seed, "step": None, "val": -1.0}
        for step in sorted(snaps_by_seed[seed]):
            d = _rebuild_merged(DRF, snaps_by_seed[seed][step], args.drf_r, modules, device)
            v = accept(d, val_instr)
            if v > s_best["val"]:
                s_best = {"seed": seed, "step": step, "val": v}
            if v > best["val"]:
                best = {"kind": "faithful", "seed": seed, "step": step, "val": v,
                        "snap": snaps_by_seed[seed][step]}
            del d; torch.cuda.empty_cache()
        per_seed.append(s_best)
        print(f"    seed {seed}: best-val step {s_best['step']} val {s_best['val']:.3f}")
    print(f"    DEPLOY {best['kind']} (seed {best['seed']} step {best['step']} "
          f"val {best['val']:.3f})")

    # reported acceptance on DISJOINT eval
    a_sh = accept(shared, eval_instr)
    if best["kind"] == "faithful":
        deployed = _rebuild_merged(DRF, best["snap"], args.drf_r, modules, device)
        a_fa = accept(deployed, eval_instr); del deployed
    else:
        a_fa = a_sh
    tpv_sh, tpv_fa = a_sh * g + 1, a_fa * g + 1
    print(f"\n  13B REAL-adapter acceptance (eval): shared={a_sh:.3f} ({tpv_sh:.2f} tpv)  "
          f"faithful={a_fa:.3f} ({tpv_fa:.2f} tpv)  gap {tpv_fa/tpv_sh:.2f}x")
    torch.cuda.empty_cache()

    # ---- Phase D: latencies -> throughput identity ------------------------------
    tiers = None; t_decode = t_verify = t_draft = None
    if not args.no_throughput and best["kind"] == "faithful":
        print("\n[D] 13B graph-captured verify/decode latency (small B), then 160m draft")
        try:
            t_decode = graph_step_ms(target, B, C, 1, device, dtype, args.max_cache)
            t_verify = graph_step_ms(target, B, C, g + 1, device, dtype, args.max_cache)
            print(f"    13B decode={t_decode:.2f}ms verify(g+1)={t_verify:.2f}ms (B={B})")
        except RuntimeError as e:
            print(f"    13B latency OOM at B={B}: {str(e)[:80]} -> throughput deferred")
            t_decode = None
        del target; torch.cuda.empty_cache()
        if t_decode is not None:
            t_draft = graph_step_ms(shared, B, C, 1, device, dtype, args.max_cache)
            peft_draft = distill_peft(DRF, tok, corpus_text, args.drf_r, args.drf_steps,
                                      2e-4, device, target_modules=modules)
            tiers = []
            for K in args.K:
                w, scale = extract_draft_lora(peft_draft, K, dtype, device, modules=modules)
                STATE["scale"] = scale
                wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
                wd.config.use_cache = True; wrap_draft(wd, w)
                adapter = torch.arange(B, device=device) % K

                def sgmv_pre(bb, mm):
                    STATE["mode"] = "lora"; STATE["idx"] = adapter.repeat_interleave(mm)
                t_sgmv = graph_step_ms(wd, B, C, 1, device, dtype, args.max_cache, sgmv_pre)
                STATE["mode"] = "none"
                tp_ns = B * 1 / (t_decode / 1000)
                tp_sp = B * tpv_sh / ((t_verify + g * t_draft) / 1000)
                tp_wp = B * tpv_fa / ((t_verify + g * t_sgmv) / 1000)
                tiers.append({"K": K, "t_sgmv_ms": t_sgmv, "no_spec": tp_ns,
                              "aspp_spec": tp_sp, "warppipe": tp_wp})
                print(f"    K={K:>3} no_spec={tp_ns:6.0f} AS++spec={tp_sp:6.0f} "
                      f"WarpPipe={tp_wp:6.0f}  (WP/spec={tp_wp/tp_sp:.2f}x "
                      f"/no_spec={tp_wp/tp_ns:.2f}x)")
                del wd; torch.cuda.empty_cache()
    else:
        del target; torch.cuda.empty_cache()

    # ---- persist ----------------------------------------------------------------
    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"across_models_13b_{args.dataset}_{stamp}.json")
    rm = tiers[-1] if tiers else None
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "scale": "13b",
                   "target": tgt_path, "adapter": "chansung/alpaca-lora-13b",
                   "dataset": args.dataset, "modules": args.modules, "seeds": args.seeds,
                   "drf_r": args.drf_r, "drf_steps": args.drf_steps, "n_train": args.n_train,
                   "gamma": g, "B": B, "deploy": best["kind"], "deploy_seed": best["seed"],
                   "deploy_step": best["step"], "deploy_val": best["val"],
                   "per_seed": per_seed, "accept_shared": a_sh, "accept_faithful": a_fa,
                   "tpv_shared": tpv_sh, "tpv_faithful": tpv_fa,
                   "tpv_gap": tpv_fa / tpv_sh,
                   "t_decode_ms": t_decode, "t_verify_ms": t_verify,
                   "t_draft_shared_ms": t_draft, "tiers": tiers,
                   "warppipe_vs_asppspec": (rm["warppipe"] / rm["aspp_spec"]) if rm else None,
                   "warppipe_vs_nospec": (rm["warppipe"] / rm["no_spec"]) if rm else None},
                  f, indent=2)
    print("\nwritten:", jpath)


if __name__ == "__main__":
    main()
