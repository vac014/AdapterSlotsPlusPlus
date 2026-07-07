#!/usr/bin/env python3
"""
Distinct real adapters: routing, per-tenant acceptance, and flat cost.

Three genuinely different community LoRAs (Alpaca, Baize, HH) are each merged into
llama-7b to give three distinct targets, and each gets its own faithful draft distilled
on that target's continuations.

The acceptance matrix scores every (target, draft) pair on a common held-out set,
including the shared adapter-blind draft. A tenant's own draft topping its row is what
tells us faithfulness is per-tenant rather than an artifact of one adapter. Where an
adapter barely moves the target, the shared draft can be the stronger one; the recipe's
floor is what deploys it there instead of shipping a regression.

The cost sweep carries all K distinct draft-LoRAs through one draft-side SGMV pass and
reports the draft-apply time against K, which is flat because SGMV indexes the adapter
pool rather than iterating it.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT  # noqa: E402
from spec_loops import spec_kv  # noqa: E402
from serve_multitenant import BGMVLoRALinear, STATE  # noqa: E402
from graph_latency import graph_step_ms  # noqa: E402

# genuinely-distinct real LLaMA-1 7B adapters (all verified nonzero lora_B)
ADAPTERS = {
    "alpaca": "tloen/alpaca-lora-7b",       # instruction-following
    "baize":  "project-baize/baize-lora-7B",  # multi-turn chat
    "hh":     "serpdotai/llama-hh-lora-7B",   # RLHF helpful/harmless
}


def load_target(adapter_repo, device, dtype):
    """Fresh base + merge one real adapter -> a distinct target (merge mutates
    the base in place, so we reload the base every time)."""
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    tgt = PeftModel.from_pretrained(base, snapshot(adapter_repo)).merge_and_unload().eval()
    tgt.config.use_cache = True
    return tgt


def snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


def distill_peft(tok, corpus_text, r, steps, lr, device):
    """Distill a faithful draft-LoRA on llama-160m; return the (unmerged) PEFT so
    we can BOTH extract BGMV weights (stacking) AND merge a copy (acceptance)."""
    model = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
    model.train()
    ids = tok(corpus_text, return_tensors="pt").input_ids[0]
    L = 128
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L, L // 2)]
    x = torch.stack([c for c in chunks if len(c) == L])
    bs = min(8, len(x))
    dl = DataLoader(TensorDataset(x), batch_size=bs, shuffle=True, drop_last=len(x) >= 8)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    it = iter(dl); t0 = time.time()
    for s in range(steps):
        try: (xb,) = next(it)
        except StopIteration: it = iter(dl); (xb,) = next(it)
        xb = xb.to(device)
        out = model(input_ids=xb, labels=xb); out.loss.backward()
        opt.step(); opt.zero_grad()
        if s % max(1, steps // 3) == 0 or s == steps - 1:
            print(f"    [drf] {s}/{steps} loss {out.loss.item():.3f} {time.time()-t0:.0f}s")
    model.eval()
    return model


def qv_weights(peft_draft, dtype, device):
    """Extract per-layer q/v lora_A [rank,in] and lora_B [out,rank] (single
    adapter, not yet stacked) + the scale."""
    w, scale = {}, 2.0
    for name, mod in peft_draft.named_modules():
        if hasattr(mod, "lora_A") and (name.endswith("q_proj") or name.endswith("v_proj")):
            key = name.split("base_model.model.")[-1]
            A = mod.lora_A["default"].weight.detach().to(device, dtype)
            B = mod.lora_B["default"].weight.detach().to(device, dtype)
            scale = float(mod.scaling["default"])
            w[key] = (A, B)
    return w, scale


def extract_stack(per_adapter_w, slot_repo, K, device, dtype):
    """Stack K DISTINCT drafts across BGMV slots. slot_repo[k] indexes which
    tenant's draft sits in slot k (cycled to fill K slots for the cost sweep)."""
    keys = per_adapter_w[0].keys()
    stacked = {}
    for key in keys:
        As, Bs = [], []
        for k in range(K):
            A, B = per_adapter_w[slot_repo[k % len(slot_repo)]][key]
            As.append(A); Bs.append(B)
        stacked[key] = (torch.stack(As).contiguous(), torch.stack(Bs).contiguous())
    return stacked


def wrap_stack(base_draft, stacked):
    n = 0
    for name, mod in list(base_draft.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if full in stacked and isinstance(child, nn.Linear):
                A, B = stacked[full]
                setattr(mod, cname, BGMVLoRALinear(child, A, B))
                n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tenants", nargs="+", default=["alpaca", "baize", "hh"])
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "alpaca", "mbpp"],
                   help="prompt distribution; gsm8k = structured regime where the "
                        "shared draft collapses and faithful recovers it")
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--drf_steps", type=int, default=1500)
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_eval", type=int, default=15)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--gen_tokens", type=int, default=48)
    p.add_argument("--eval_tokens", type=int, default=48)
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=128)
    p.add_argument("--Kcost", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    tok = AutoTokenizer.from_pretrained(TGT)
    tenants = args.tenants
    Kt = len(tenants)
    g = args.gamma

    # COMMON instruction pool (same prompts for every tenant; only the adapter
    # differs, so the target -- not the prompt -- creates the distribution shift)
    train_instr, eval_instr = load_instructions(args.n_train, args.n_eval, args.dataset)
    print(f"tenants={tenants}  {len(train_instr)} train / {len(eval_instr)} held-out")

    # ---- Phase A: per-tenant distill (one 7B target resident at a time) ----
    per_w = []            # per-tenant q/v BGMV weights (for stacking)
    faithful = []         # per-tenant merged 160m faithful draft (for acceptance)
    scale = 2.0
    for ti, name in enumerate(tenants):
        print(f"\n=== tenant {ti} [{name}] {ADAPTERS[name]} ===")
        target = load_target(ADAPTERS[name], device, dtype)
        print("  gen distill corpus (target's own greedy continuations)")
        corpus = []
        for i, instr in enumerate(train_instr):
            t = PROMPT.format(instr=instr)
            corpus.append(t + greedy(target, tok, t, args.gen_tokens, device))
        del target; torch.cuda.empty_cache()
        corpus_text = "\n\n".join(corpus)
        print("  distill faithful draft-LoRA")
        peft = distill_peft(tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device)
        w, scale = qv_weights(peft, dtype, device)
        per_w.append(w)
        fm = peft.merge_and_unload().to(device=device, dtype=dtype).eval()
        fm.config.use_cache = True
        faithful.append(fm)
        del peft; torch.cuda.empty_cache()

    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    # ---- Phase B*: per-tenant acceptance matrix ----
    # M[i][d]: acceptance of draft d on target i, over common held-out prompts.
    # drafts = [shared] + faithful_0..faithful_{Kt-1}
    print("\n=== per-tenant acceptance matrix ===")
    drafts = [("shared", shared)] + [(f"faith[{tenants[j]}]", faithful[j]) for j in range(Kt)]
    M = [[0.0] * len(drafts) for _ in range(Kt)]
    eval_ids = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids.to(device)
                for x in eval_instr]
    for i, name in enumerate(tenants):
        target = load_target(ADAPTERS[name], device, dtype)
        for di, (dn, d) in enumerate(drafts):
            acc = st = 0
            for x in eval_ids:
                _, a, s = spec_kv(d, target, x, g, args.eval_tokens)
                acc += a; st += s
            M[i][di] = acc / (st * g)
        del target; torch.cuda.empty_cache()
        row = "  ".join(f"{drafts[di][0]}={M[i][di]:.3f}" for di in range(len(drafts)))
        print(f"  target[{name}]: {row}")

    # diagonal check: faithful_i (column i+1) is the max of its row & beats shared
    diag_wins = sum(1 for i in range(Kt)
                    if M[i][i + 1] == max(M[i]) and M[i][i + 1] > M[i][0])
    print(f"  faithful-own tops its row AND beats shared for {diag_wins}/{Kt} tenants")

    # ---- Phase C*: one batched draft pass carrying K DISTINCT drafts, flat ----
    print("\n=== batched draft-side SGMV cost, K distinct drafts ===")
    STATE["scale"] = scale
    slot_repo = list(range(Kt))  # cycle the real distinct drafts to fill K slots
    cost = []
    for K in args.Kcost:
        stacked = extract_stack(per_w, slot_repo, K, device, dtype)
        wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
        wd.config.use_cache = True
        wrap_stack(wd, stacked)
        adapter = torch.arange(args.B, device=device) % K  # distinct per-row routing
        def pre(bb, mm):
            STATE["mode"] = "lora"; STATE["idx"] = adapter.repeat_interleave(mm)
        t_sgmv = graph_step_ms(wd, args.B, args.context, 1, device, dtype, args.max_cache, pre)
        STATE["mode"] = "none"
        cost.append({"K": K, "t_sgmv_ms": t_sgmv})
        print(f"  K={K:>3}  batched-draft = {t_sgmv:.3f} ms")
        del wd; torch.cuda.empty_cache()
    flat = cost[-1]["t_sgmv_ms"] / cost[0]["t_sgmv_ms"]
    print(f"  cost K={args.Kcost[0]}->{args.Kcost[-1]}: {flat:.2f}x (flat if ~1)")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"multitenant_matrix_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0),
                   "tenants": tenants, "adapters": {t: ADAPTERS[t] for t in tenants},
                   "drf_r": args.drf_r, "gamma": g, "B": args.B,
                   "draft_names": [d[0] for d in drafts],
                   "accept_matrix": M, "diag_wins": diag_wins,
                   "cost": cost, "cost_flat_ratio": flat}, f, indent=2)
    print("\nwritten:", jpath)


if __name__ == "__main__":
    main()
