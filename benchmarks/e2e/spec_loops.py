#!/usr/bin/env python3
"""
Exact KV-cached speculative decode, and the non-speculative references.

`spec_kv` is the acceptance primitive the rest of the suite is built on: draft proposes
gamma tokens, target verifies all gamma+1 positions in one pass, accepts the longest
matching prefix under greedy verification, and rolls both caches back to the committed
length. Its committed tokens are asserted equal to the target's plain greedy output
before any number is reported, so acceptance is measured against the true target path
rather than an approximation of it.

`nospec_kv` and `nospec_nokv` are the two non-speculative baselines, the second kept
because the no-KV serving loops decode that way and their absolute tok/s must be read
against a baseline that pays the same recompute.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(__file__))
from distill import train_lora, CORPUS, TGT, DRF  # noqa: E402


@torch.inference_mode()
def nospec_kv(model, ids, n_new):
    """KV-cached greedy AR decode; returns generated token ids (len n_new)."""
    cache = DynamicCache()
    out = model(input_ids=ids, use_cache=True, past_key_values=cache)
    x = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [x]
    for _ in range(n_new - 1):
        out = model(input_ids=x, use_cache=True, past_key_values=cache)
        x = out.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(x)
    return torch.cat(gen, 1)


@torch.inference_mode()
def nospec_nokv(model, ids, n_new):
    """No-KV greedy (reference for correctness)."""
    seq = ids
    gen = []
    for _ in range(n_new):
        x = model(input_ids=seq, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        gen.append(x); seq = torch.cat([seq, x], 1)
    return torch.cat(gen, 1)


@torch.inference_mode()
def spec_kv(draft, target, ids, gamma, n_new):
    """KV-cached greedy speculative decode (exact rollback). Returns
    (generated_ids, n_accepted_proposals, n_verify_steps)."""
    dc, tc = DynamicCache(), DynamicCache()
    # prefill both on committed[:-1]; pending = committed[-1]
    draft(input_ids=ids[:, :-1], use_cache=True, past_key_values=dc)
    target(input_ids=ids[:, :-1], use_cache=True, past_key_values=tc)
    pending_d = pending_t = ids[:, -1:]
    N = ids.shape[1]                     # committed length (cache holds N-1)
    committed_extra = []                 # generated tokens
    acc_tot = steps = 0
    while len(committed_extra) < n_new:
        # 1) draft proposes gamma tokens from pending_d
        x = pending_d
        prop = []
        for _ in range(gamma):
            lg = draft(input_ids=x, use_cache=True, past_key_values=dc).logits[:, -1]
            x = lg.argmax(-1, keepdim=True)
            prop.append(x)
        # dc now holds committed[:-1] + [pending, prop0..prop_{g-2}] (len N-1+gamma)
        prop_t = torch.cat(prop, 1)      # [1, gamma]
        # 2) target verifies X = [pending, prop0..prop_{g-1}] (gamma+1 tokens)
        X = torch.cat([pending_t, prop_t], 1)
        lg = target(input_ids=X, use_cache=True, past_key_values=tc).logits[0]  # [g+1, V]
        pred = lg.argmax(-1)             # pred[i] = token after X[i]
        a = 0
        for j in range(gamma):
            if pred[j].item() == prop_t[0, j].item():
                a += 1
            else:
                break
        bonus = pred[a].view(1, 1)
        new = list(prop_t[0, :a]) + [bonus[0, 0]]
        committed_extra.extend(t.item() for t in new)
        steps += 1; acc_tot += a
        # 3) rollback caches to new_committed[:-1] (len N-1 + a + ... ) pending=bonus
        #    tc currently N-1+gamma+1; want N-1 + (a+1) - 1 = N-1+a
        tc.crop(N - 1 + a)               # keeps committed[:-1]+[pending,prop0..prop_{a-1}]
        #    dc currently N-1+gamma; want N-1+a
        if a == gamma:
            draft(input_ids=prop[-1], use_cache=True, past_key_values=dc)  # add prop_{g-1}
        else:
            dc.crop(N - 1 + a)
        pending_d = pending_t = bonus
        N = N + a + 1
    return torch.tensor(committed_extra[:n_new]).view(1, -1), acc_tot, steps


def timed(fn, iters=3):
    fn()  # warm
    torch.cuda.synchronize()
    t0 = time.time()
    r = None
    for _ in range(iters):
        r = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgt_steps", type=int, default=120)
    p.add_argument("--drf_steps", type=int, default=300)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--n_new", type=int, default=64)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--drf_r", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=96)
    p.add_argument("--uppercase", action="store_true",
                   help="extreme distribution shift; default = plain corpus (mild, realistic adapter)")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TGT)
    task = CORPUS.upper() if args.uppercase else CORPUS
    prompt_upper = args.uppercase

    print("Stage 1: train target LoRA (llama-7b, UPPERCASE)")
    target = train_lora(TGT, tok, task, args.r, args.tgt_steps, 2e-4, args.seqlen,
                        batch=2, device=device, grad_ckpt=True, tag="tgt")
    target.gradient_checkpointing_disable(); target.config.use_cache = True; target.eval()

    print("Stage 2: distill + merge faithful draft LoRA (llama-160m)")
    faithful = train_lora(DRF, tok, task, args.drf_r, args.drf_steps, 2e-4,
                          args.seqlen, batch=8, device=device, grad_ckpt=False,
                          tag="drf", dtype=torch.float32)
    faithful = faithful.merge_and_unload().to(device=device, dtype=torch.float16).eval()
    faithful.config.use_cache = True
    base_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float16).to(device).eval()
    base_draft.config.use_cache = True

    prompts = ["The history of science is", "Machine learning enables",
               "Modern physics describes", "Computers process information",
               "The internet connects"]
    if prompt_upper:
        prompts = [s.upper() for s in prompts]
    pid = [tok(s, return_tensors="pt").input_ids.to(device) for s in prompts]

    # correctness: KV no_spec must equal no-KV greedy
    g_kv = nospec_kv(target, pid[0], 16)
    g_nokv = nospec_nokv(target, pid[0], 16)
    assert torch.equal(g_kv, g_nokv), f"KV mismatch\n{g_kv}\n{g_nokv}"
    # correctness: spec output == target greedy (greedy spec is exact)
    s_kv, _, _ = spec_kv(faithful, target, pid[0], args.gamma, 16)
    assert torch.equal(s_kv.to(g_kv.device), g_kv), f"spec != greedy\n{s_kv}\n{g_kv}"
    print("correctness: KV no_spec == no-KV greedy, spec == target greedy  ✓")

    print(f"\n=== KV-cached net spec win (per stream, n_new={args.n_new}) ===")
    rows = []
    def run_all(draftname, draft):
        secs = accs = stps = 0.0
        toks = 0
        for x in pid:
            dt, (out, a, st) = timed(lambda: spec_kv(draft, target, x, args.gamma, args.n_new))
            secs += dt; toks += out.numel(); accs += a; stps += st
        tps = toks / secs
        acc_rate = accs / (stps * args.gamma)
        return {"strategy": draftname, "tok_s": tps, "accept_rate": acc_rate,
                "toks_per_verify": toks / stps}
    # no_spec baseline (KV)
    ns_secs = ns_toks = 0.0
    for x in pid:
        dt, out = timed(lambda: nospec_kv(target, x, args.n_new))
        ns_secs += dt; ns_toks += out.numel()
    ns = {"strategy": "no_spec", "tok_s": ns_toks / ns_secs,
          "accept_rate": None, "toks_per_verify": 1.0}
    ns["strategy"] = "AS++ no_spec"
    rows.append(ns)
    rows.append(run_all("AS++ spec (shared draft)", base_draft))
    rows.append(run_all("AS++ spec + WarpPipe", faithful))

    for r in rows:
        ar = f"{r['accept_rate']:.2f}" if r["accept_rate"] is not None else "  - "
        print(f"  {r['strategy']:>26}: {r['tok_s']:8.1f} tok/s   "
              f"accept={ar}   {r['toks_per_verify']:.2f} tok/verify")

    nsp = ns["tok_s"]
    faith = next(r for r in rows if "WarpPipe" in r["strategy"])
    shar = next(r for r in rows if "shared draft" in r["strategy"])
    print(f"\n  AS++ spec           vs AS++ no_spec : {shar['tok_s']/nsp:.2f}x")
    print(f"  AS++ spec+WarpPipe  vs AS++ no_spec : {faith['tok_s']/nsp:.2f}x")
    print(f"  AS++ spec+WarpPipe  vs AS++ spec    : {faith['tok_s']/shar['tok_s']:.2f}x  <- WarpPipe's further win")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"spec_loops_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "n_new": args.n_new,
                   "gamma": args.gamma, "results": rows,
                   "asppspec_vs_nospec": shar["tok_s"] / nsp,
                   "warppipe_vs_nospec": faith["tok_s"] / nsp,
                   "warppipe_vs_asppspec": faith["tok_s"] / shar["tok_s"]}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
