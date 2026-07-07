#!/usr/bin/env python3
"""
Heterogeneous traffic: every row of the batch is a different tenant and task.

The K-sweep holds the task fixed across tenants to isolate cost. Here each row is a
different task (GSM8K, MBPP, Dolly, SAMSum) served by one shared 7B target adapter, each
with its own faithful draft-LoRA applied per-row by draft-side SGMV in a single draft pass.

Decode is KV-cached and CUDA-graph-captured on every tier (see batched_spec_loop.py). Both matter,
and an earlier version of this benchmark had neither: it recomputed the full prefix every
step, which made one decoded token cost about what verifying gamma+1 of them costs and
handed speculation a speedup it had not earned. With the cache on, the target step collapses
and an eagerly-run draft is suddenly more expensive than the decode it replaces; only once
the draft is captured does it cost what its parameter count says it should. The two effects
have opposite signs and the same magnitude, so a loop with neither is not a conservative
approximation of a loop with both; it is a different answer.

gamma is swept rather than assumed: a wider gamma buys more tokens per accepted run but
pays for a wider verify, whose extra token positions are compute rather than bandwidth at
this batch size, so the optimum is interior.

Throughput is steady-state decode at a full batch -- what a continuously batched server
sustains. Every token counted is committed, and every tier is asserted token-identical to
plain decode (up to argmax ties) before it is timed.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from distill import TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT, distill_draft_es  # noqa: E402
from serve_multitenant import STATE  # noqa: E402
from multitenant_matrix import distill_peft, extract_stack, wrap_stack  # noqa: E402
from peft import PeftModel  # noqa: E402
from paths import alpaca_lora  # noqa: E402
from spec_loops import spec_kv  # noqa: E402
import batched_spec_loop as loop  # noqa: E402
import lossless_check  # noqa: E402

_QKVO = ("q_proj", "k_proj", "v_proj", "o_proj")


def qkvo_weights(peft_draft, dtype, device):
    """Extract lora_A/lora_B for ALL of q,k,v,o, so the
    per-row draft-side SGMV applies the full faithful delta (q/v-only understates acceptance
    on milder tasks). Same layout as qv_weights; wrap_stack/extract_stack are module-generic."""
    w, scale = {}, 2.0
    for name, mod in peft_draft.named_modules():
        if hasattr(mod, "lora_A") and any(name.endswith(m) for m in _QKVO):
            key = name.split("base_model.model.")[-1]
            A = mod.lora_A["default"].weight.detach().to(device, dtype)
            B = mod.lora_B["default"].weight.detach().to(device, dtype)
            scale = float(mod.scaling["default"])
            w[key] = (A, B)
    return w, scale


def distill_best_of(tok, corpus, r, steps, lr, device, target, val_instr, gamma, seeds=(0, 1)):
    """Deployed-recipe draft: train `seeds` faithful drafts and keep the one with the best
    acceptance on a DISJOINT validation set (removes the init-seed lottery documented in
    the seed lottery). Returns the best UNMERGED PEFT draft (for BGMV extraction)."""
    best_peft, best_val = None, -1.0
    for seed in seeds:
        torch.manual_seed(seed)
        peft = distill_peft(tok, corpus, r, steps, lr, device)
        tot = st = 0
        for instr in val_instr:
            vids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
            _, a, s = spec_kv(peft, target, vids, gamma, 48)
            tot += a; st += s
        v = tot / (st * gamma) if st else 0.0
        print(f"    [seed {seed}] val_accept {v:.3f}")
        if v > best_val:
            if best_peft is not None:
                del best_peft
            best_val, best_peft = v, peft
        else:
            del peft
        torch.cuda.empty_cache()
    print(f"    deployed best-of-{len(seeds)} val_accept {best_val:.3f}")
    return best_peft


def main():
    p = argparse.ArgumentParser(description="Real distinct-task mixed-batch serving")
    p.add_argument("--tenants", nargs="+", default=["gsm8k", "mbpp", "dolly", "samsum"])
    p.add_argument("--tgt_r", type=int, default=16)
    p.add_argument("--drf_r", type=int, default=16)
    p.add_argument("--tgt_steps", type=int, default=150)
    p.add_argument("--drf_steps", type=int, default=900)
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--gammas", type=int, nargs="+", default=[2, 3, 4, 5],
                   help="swept; the optimum is interior")
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--gen_tokens", type=int, default=48)
    p.add_argument("--out_tokens", type=int, default=48)
    p.add_argument("--per_tenant", type=int, default=8, help="requests per tenant in batch")
    p.add_argument("--seqlen", type=int, default=128)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    tok = AutoTokenizer.from_pretrained(TGT)
    pad_id = tok.eos_token_id or 0
    tenants = args.tenants; K = len(tenants)
    B = K * args.per_tenant
    adapter = torch.tensor([k for k in range(K) for _ in range(args.per_tenant)],
                           device=device)  # row -> tenant

    # One SHARED real adapter target
    # (tloen/alpaca-lora-7b, merged) serves every tenant, so acceptance is exactly the
    # acceptance is measured against a real adapter, not a proxy. Multi-tenancy is carried by the
    # four DISTINCT per-task faithful drafts, each applied per-row by draft-side SGMV (the
    # WarpPipe contribution); traffic heterogeneity is the four TASK prompt distributions.
    # (Distinct targets and per-row target-SGMV are covered by multitenant_matrix.py's
    # three real adapters and serve_multitenant.py's K=1..32 co-residency sweep.)
    print("load shared real target: llama-7b + tloen/alpaca-lora-7b (merged)")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True                       # the shared-draft floor

    drf_w, serve_prompts, all_serve, deployed = [], [], set(), []
    scale = 2.0
    for k, task in enumerate(tenants):
        print(f"\n=== tenant {k} [{task}] : faithful draft on the SHARED alpaca target ===")
        train_instr, held = load_instructions(args.n_train, args.per_tenant + 24, task)
        serve = held[:args.per_tenant]                       # held-out, disjoint from train
        val_instr = held[args.per_tenant:args.per_tenant + 20]  # disjoint val for selection
        assert not (set(serve) & set(train_instr)), "serve/train overlap (contamination!)"
        assert not (set(serve) & set(val_instr)), "serve/val overlap"
        # distill corpus = the SHARED alpaca target's greedy continuations of the TASK's
        # (disjoint, held-out-from-serve) train prompts
        corpus = "\n\n".join(PROMPT.format(instr=x) +
                             greedy(target, tok, PROMPT.format(instr=x), 64, device)
                             for x in train_instr)
        # DEPLOYED recipe: multi-seed, multi-checkpoint,
        # validation-selected, with the shared-draft FLOOR (deploy faithful only if it beats
        # shared on val; else deploy shared). Returns the UNMERGED selected draft (None=floor).
        # q,k,v,o: once the draft step is captured its cost is set by parameter count, not
        # by how many projections carry a delta, so there is no reason to trade
        # acceptance away for a lighter draft.
        peft_d, info = distill_draft_es(
            DRF, tok, corpus, 32, args.drf_steps, 2e-4, device, target, val_instr,
            max(args.gammas), 64, ckpt_steps=(150, 300, 500, 900),
            target_modules=_QKVO, seeds=(0, 1, 2),
            shared_draft=shared, return_unmerged=True)
        deployed.append(info["deployed"])
        drf_w.append(qkvo_weights(peft_d, dtype, device)[0] if peft_d is not None else None)
        if peft_d is not None:
            scale = qkvo_weights(peft_d, dtype, device)[1]
            del peft_d
        torch.cuda.empty_cache()
        all_serve |= set(serve)
        serve_prompts.append([PROMPT.format(instr=x) for x in serve])

    # ---- build the per-row BGMV draft (K distinct faithful draft-LoRAs, one pass) ----
    # tenants where the shared floor won deploy zero draft-LoRA (=> base draft, = shared).
    print("\n=== build per-row draft-side SGMV (K distinct faithful drafts) ===")
    print(f"  deployed per tenant: {dict(zip(tenants, deployed))}")
    ref = next(w for w in drf_w if w is not None)
    for i, w in enumerate(drf_w):
        if w is None:
            drf_w[i] = {kk: (torch.zeros_like(A), torch.zeros_like(Bm)) for kk, (A, Bm) in ref.items()}
    STATE["scale"] = scale
    slots = list(range(K))
    draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    draft.config.use_cache = True
    nd_w = wrap_stack(draft, extract_stack(drf_w, slots, K, device, dtype))
    print(f"  wrapped draft q,k,v,o={nd_w}; shared merged alpaca target")

    # ---- flatten the mixed batch (row r -> tenant adapter[r]) ----
    prompts = []
    for k in range(K):
        for pr in serve_prompts[k]:
            prompts.append(tok(pr, return_tensors="pt").input_ids[0].to(device))

    # ---- capture the graphs (one static cache per model, shared across gammas) ----
    S0 = max(p_.numel() for p_ in prompts)
    MAX = S0 + 1 + args.iters * (max(args.gammas) + 1) + 8
    print(f"\n== mixed batch, KV + CUDA graphs: B={B} ({K} tenants x {args.per_tenant}), "
          f"prompts padded to {S0}, static cache {MAX} ==")
    tcache = loop.make_cache(target, B, MAX, device, dtype)
    dcache = loop.make_cache(draft, B, MAX, device, dtype)
    valid = torch.zeros(B, MAX, dtype=torch.bool, device=device)
    tgt_dec = loop.Step(target, tcache, B, 1, MAX, device).capture()
    d1 = {"warp": loop.Step(draft, dcache, B, 1, MAX, device, lora=True).capture(),
          "shared": loop.Step(draft, dcache, B, 1, MAX, device).capture()}
    d2 = {"warp": loop.Step(draft, dcache, B, 2, MAX, device, lora=True).capture(),
          "shared": loop.Step(draft, dcache, B, 2, MAX, device).capture()}
    STATE["mode"] = "none"

    def run(kind, gamma, tv, iters, record=None):
        r = loop.run_tier(kind, tgt_dec, tv, d1.get(kind), d2.get(kind), target, draft,
                        prompts, adapter, gamma, iters, valid, MAX, pad_id, device, K,
                        record=record)
        STATE["mode"] = "none"
        return r

    for _ in range(2):
        run("no_spec", 2, None, 4)
    base_r = run("no_spec", 2, None, args.iters)
    ns = base_r["tok_s"]
    print(f"  no_spec: {ns:8.1f} tok/s | {base_r['sec']/args.iters*1e3:5.1f} ms/iter")

    # every tier must emit what plain decode emits, up to argmax ties, BEFORE it is timed
    ref = run("no_spec", 2, None, 24, record=24)["emitted"]

    res = {}
    print(f"\n{'gamma':>5} {'tier':>8} {'tok_s':>9} {'ms/iter':>8} {'tok/iter':>9} "
          f"{'vs no_spec':>11}   acceptance")
    for gamma in args.gammas:
        tv = loop.Step(target, tcache, B, gamma + 1, MAX, device).capture()
        STATE["mode"] = "none"
        for kind in ("shared", "warp"):
            ok, note = lossless_check.check(f"g{gamma} {kind}", run(kind, gamma, tv, 24, record=24)
                                ["emitted"], ref, target, prompts, device, 24)
            print("  " + note)
            if not ok:
                raise SystemExit(f"gamma={gamma} {kind}: real divergence from plain decode")
            for _ in range(2):
                run(kind, gamma, tv, 4)
            r = run(kind, gamma, tv, args.iters)
            pt = r["per_tenant_accept"]
            res[f"{gamma}:{kind}"] = {
                "gamma": gamma, "tier": kind, "tok_s": r["tok_s"],
                "ms_iter": r["sec"] / args.iters * 1e3,
                "tok_per_iter": r["tokens"] / (args.iters * B),
                "over_nospec": r["tok_s"] / ns, "per_tenant_accept": pt}
            print(f"{gamma:>5} {kind:>8} {r['tok_s']:9.1f} "
                  f"{r['sec']/args.iters*1e3:8.1f} {r['tokens']/(args.iters*B):9.2f} "
                  f"{r['tok_s']/ns:10.3f}x   "
                  + "/".join(f"{tenants[k][:4]}:{pt[k]:.2f}" for k in range(K)))
        del tv
        torch.cuda.empty_cache()

    best = max(args.gammas, key=lambda gg: res[f"{gg}:warp"]["over_nospec"])
    wp, sh = res[f"{best}:warp"], res[f"{best}:shared"]
    print(f"\n  BEST gamma={best}:  WarpPipe/no_spec = {wp['over_nospec']:.3f}x   "
          f"WarpPipe/shared = {wp['tok_s']/sh['tok_s']:.3f}x")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"serve_mixed_traffic_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "tenants": tenants,
                   "B": B, "per_tenant": args.per_tenant, "gammas": args.gammas,
                   "iters": args.iters, "kv_cached": True, "graphed": True,
                   "drf_r": 32, "recipe": "distill_draft_es", "deployed": deployed,
                   "shared_target": "tloen/alpaca-lora-7b", "no_spec_tok_s": ns,
                   "best_gamma": best, "warp_over_nospec": wp["over_nospec"],
                   "warp_over_shared": wp["tok_s"] / sh["tok_s"],
                   "results": res}, f, indent=2, default=str)
    print("written:", jpath)


if __name__ == "__main__":
    main()
