#!/usr/bin/env python3
"""
KV-cached batched speculative serving, run to completion, eager.

The no-KV loops recompute the whole prefix every step, which understates absolute tok/s
and flatters the speculative ratio (the non-speculative baseline recomputes a longer
prefix, so it is penalised more). This keeps a real KV cache across the batch, so the
absolute numbers are representative of a serving stack.

Ragged rollback is the hard part: rows accept different numbers of draft tokens, so
committed lengths diverge, and DynamicCache.crop truncates every row to one length.
Instead the cache stays append-only and rollback is done by masking: a rejected token
stays physically in the cache but is marked 0 in that row's attention mask, so attention
never sees it again, exactly as left-padding is handled. Real tokens are always fed at
their true positions, so their cached rotary keys are correct.

Committed tokens are asserted bit-identical to the no-KV loop before anything is timed.

This loop is an ablation, not the deployment path: eager, the 160m draft pays its full
launch cost and loses. Capturing it is what makes it win; see serve_kv_graph.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from distill import train_lora, TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT  # noqa: E402
from serve_multitenant import spec_iter, nospec_iter, STATE  # noqa: E402
from paths import alpaca_lora  # noqa: E402


def pad_left(rows, pad_id, device):
    S = max(r.numel() for r in rows); B = len(rows)
    ids = torch.full((B, S), pad_id, device=device, dtype=torch.long)
    attn = torch.zeros(B, S, device=device, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, S - r.numel():] = r; attn[i, S - r.numel():] = 1
    return ids, attn


class KVBatch:
    """Append-only batched KV cache + per-row real/pad mask for one model."""
    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache()
        self.mask = None

    @torch.inference_mode()
    def prefill(self, ids, attn):
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        self.model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], position_ids=pos[:, :-1],
                   use_cache=True, past_key_values=self.cache)
        self.mask = attn[:, :-1].clone()

    @torch.inference_mode()
    def forward(self, x, pos):
        """Feed x:[B,S] at position ids pos:[B,S]. Attends over prior mask + all-real new.
        Appends S all-real slots; caller may overwrite the last slots' mask afterwards."""
        am = torch.cat([self.mask, torch.ones_like(x)], dim=1)
        out = self.model(input_ids=x, attention_mask=am, position_ids=pos,
                         use_cache=True, past_key_values=self.cache).logits
        self.mask = torch.cat([self.mask, torch.ones_like(x)], dim=1)
        return out


@torch.inference_mode()
def spec_kv_iter(dstate, tstate, real_len, pending, gamma, device):
    """One KV speculative iteration over the batch. Returns (acc[B], prop[B,g], bonus[B,1])."""
    B = real_len.size(0)
    rl = real_len.view(B, 1)
    x = pending; props = []
    for j in range(gamma):
        lg = dstate.forward(x, rl - 1 + j)[:, -1]
        x = lg.argmax(-1, keepdim=True); props.append(x)
    prop = torch.cat(props, 1)                              # [B, g]
    X = torch.cat([pending, prop], 1)                       # [B, g+1]
    posv = (rl - 1) + torch.arange(gamma + 1, device=device).view(1, -1)
    pred = tstate.forward(X, posv).argmax(-1)               # [B, g+1]
    acc = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        a = 0
        while a < gamma and pred[i, a].item() == prop[i, a].item():
            a += 1
        acc[i] = a
    bonus = pred[torch.arange(B, device=device), acc].view(B, 1)
    ar = torch.arange(gamma + 1, device=device).view(1, -1)
    # keep first a+1 verify slots real; keep first a draft slots real (mask rejected)
    tstate.mask[:, -(gamma + 1):] = (ar <= acc.view(B, 1)).long()
    dstate.mask[:, -gamma:] = (ar[:, :gamma] < acc.view(B, 1)).long()
    # rows accepting all gamma: draft cache missing prop_{g-1}; feed it (mask real only there)
    if (acc == gamma).any():
        dstate.forward(prop[:, -1:], rl - 1 + gamma)
        dstate.mask[:, -1:] = (acc == gamma).long().view(B, 1)
    return acc, prop, bonus


def _eos_done(seq, out_tokens, eos):
    """A request finishes at first EOS or out_tokens (post-EOS tokens are undefined)."""
    return (eos in seq) or (len(seq) >= out_tokens)


@torch.inference_mode()
def run_kv(kind, target, draft, prompts, gamma, out_tokens, pad_id, device):
    ids, attn = pad_left(prompts, pad_id, device)
    B = len(prompts)
    real_len = torch.tensor([p.numel() for p in prompts], device=device)
    pending = torch.stack([p[-1] for p in prompts]).view(B, 1)
    tstate = KVBatch(target); tstate.prefill(ids, attn)
    committed = [[] for _ in range(B)]; finish = [None] * B
    dstate = None
    if kind != "no_spec":
        dstate = KVBatch(draft); dstate.prefill(ids, attn)
    it = 0
    while any(f is None for f in finish):
        it += 1
        if kind == "no_spec":
            lg = tstate.forward(pending, (real_len - 1).view(B, 1))[:, -1]
            nt = lg.argmax(-1, keepdim=True)
            for i in range(B):
                if finish[i] is None:
                    committed[i].append(nt[i, 0].item())
            pending = nt; real_len = real_len + 1
        else:
            acc, prop, bonus = spec_kv_iter(dstate, tstate, real_len, pending, gamma, device)
            for i in range(B):
                if finish[i] is None:
                    a = acc[i].item()
                    committed[i].extend(prop[i, :a].tolist()); committed[i].append(bonus[i, 0].item())
            pending = bonus; real_len = real_len + acc + 1
        for i in range(B):
            if finish[i] is None and _eos_done(committed[i], out_tokens, pad_id):
                finish[i] = it
        if it > 6000:
            break
    return committed, finish, it


@torch.inference_mode()
def run_nokv(kind, target, draft, prompts, gamma, out_tokens, pad_id, device):
    """No-KV reference loop. Returns committed tokens per row, for the equivalence assert."""
    rows = [p.clone() for p in prompts]; B = len(rows)
    adapter = torch.zeros(B, device=device, dtype=torch.long)
    committed = [[] for _ in range(B)]; finish = [None] * B
    it = 0
    while any(f is None for f in finish):
        it += 1
        prev = [r.numel() for r in rows]
        if kind == "no_spec":
            rows, _ = nospec_iter(target, rows, pad_id, device)
        else:
            STATE["mode"] = "none"
            rows, _ = spec_iter(draft, target, rows, adapter, gamma, "shared_draft", pad_id, device)
        for i in range(B):
            new = rows[i][prev[i]:].tolist()
            if finish[i] is None:
                committed[i].extend(new)
                if _eos_done(committed[i], out_tokens, pad_id):
                    finish[i] = it
        if it > 6000:
            break
    return committed


def timed_run(kind, target, draft, prompts, gamma, out_tokens, pad_id, device):
    # warmup (untimed): short run
    run_kv(kind, target, draft, [p.clone() for p in prompts], gamma, 4, pad_id, device)
    torch.cuda.synchronize(); t0 = time.time()
    committed, finish, it = run_kv(kind, target, draft, [p.clone() for p in prompts],
                                   gamma, out_tokens, pad_id, device)
    torch.cuda.synchronize(); sec = time.time() - t0
    produced = sum(min(len(c), out_tokens) for c in committed)
    perit = sec / it
    lat = np.array([f * perit for f in finish])
    return {"tok_s": produced / sec, "sec": sec, "iters": it,
            "p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99))}, committed


def main():
    p = argparse.ArgumentParser(description="KV-cached batched speculative serving")
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--out_tokens", type=int, default=64)
    p.add_argument("--drf_steps", type=int, default=500)
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    tok = AutoTokenizer.from_pretrained(TGT)
    pad_id = tok.eos_token_id or 0
    train_instr, held = load_instructions(args.n_train, args.B + 4, args.dataset)
    serve_instr = held[:args.B]

    print("load target 7B + real alpaca adapter (merged)")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True
    print("distill faithful gsm8k draft, merge")
    corpus = "\n\n".join(PROMPT.format(instr=x) +
                         greedy(target, tok, PROMPT.format(instr=x), 64, device)
                         for x in train_instr)
    faithful = train_lora(DRF, tok, corpus, 16, args.drf_steps, 2e-4, 128, batch=8,
                          device=device, tag="drf", dtype=torch.float32)
    faithful = faithful.merge_and_unload().to(device=device, dtype=dtype).eval()
    faithful.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    prompts = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids[0].to(device)
               for x in serve_instr]

    # ---- correctness: batched-KV committed tokens == no-KV committed tokens ----
    print("\n=== correctness: KV vs no-KV committed tokens (bit-identical if greedy) ===")

    def upto_eos(seq):
        return seq[:seq.index(pad_id) + 1] if pad_id in seq else seq

    for kind, drf in [("no_spec", None), ("warp", faithful)]:
        c_kv, _, _ = run_kv(kind, target, drf, [p.clone() for p in prompts], g, 32, pad_id, device)
        c_nk = run_nokv(kind, target, drf, [p.clone() for p in prompts], g, 32, pad_id, device)
        diffs = [i for i in range(len(prompts)) if upto_eos(c_kv[i]) != upto_eos(c_nk[i])]
        print(f"  {kind:>8}: match={not diffs}  (mismatched rows {len(diffs)}/{len(prompts)})")
        if diffs:
            i = diffs[0]
            print(f"    row{i} KV ={upto_eos(c_kv[i])}\n    row{i} nKV={upto_eos(c_nk[i])}")
        assert not diffs, f"KV != no-KV for {kind} in {len(diffs)} rows"
    print("  correctness OK: batched KV reproduces no-KV exactly (to EOS)")

    STATE["mode"] = "none"
    print(f"\n=== KV-cached serving throughput + tail (B={args.B}, out={args.out_tokens}) ===")
    res = {}
    for kind, drf in [("no_spec", None), ("shared", shared), ("warp", faithful)]:
        r, _ = timed_run(kind, target, drf, prompts, g, args.out_tokens, pad_id, device)
        res[kind] = r
        print(f"  {kind:>8}: {r['tok_s']:7.1f} tok/s  ({r['sec']:.1f}s, {r['iters']} it)  "
              f"p50={r['p50']:.1f}s p99={r['p99']:.1f}s")
    ns, sh, wp = res["no_spec"]["tok_s"], res["shared"]["tok_s"], res["warp"]["tok_s"]
    print(f"\n  KV WarpPipe/no_spec = {wp/ns:.2f}x   WarpPipe/shared = {wp/sh:.2f}x")
    print(f"  p99: no_spec {res['no_spec']['p99']:.1f}s -> warp {res['warp']['p99']:.1f}s")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"serve_kv_eager_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": args.B, "gamma": g,
                   "out_tokens": args.out_tokens, "dataset": args.dataset,
                   "kv_cached": True, "warp_over_nospec": wp / ns, "warp_over_shared": wp / sh,
                   "results": res}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
