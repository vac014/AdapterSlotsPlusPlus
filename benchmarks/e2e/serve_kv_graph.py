#!/usr/bin/env python3
"""
KV-cached, CUDA-graph-captured speculative decode, run to completion.

The deployment path. A real speculative stack both caches KV and captures the draft, so
this does both. The difference from the eager loop is the result: eager, gamma draft
forwards cost more per step than they save.

Runs at B=1 per stream, the latency-optimised regime where the target is memory-bound and
verifying gamma+1 tokens costs nearly what decoding one costs. StaticCache keeps shapes
static so the step is capturable; rollback rewinds cache_position to the committed length,
which leaves the rejected slots unattended without cropping.

Committed tokens are asserted bit-identical to the eager DynamicCache reference before
timing, so the speedup is systems, not approximation.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from distill import train_lora, TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT  # noqa: E402
from spec_loops import spec_kv, nospec_kv  # noqa: E402  (eager reference, DynamicCache)


class Graphed:
    """CUDA-graph a fixed-shape [1,S] forward over a StaticCache, with in-place input buffers
    (ids/pos/cache_position) updated between replays."""
    def __init__(self, model, S, max_cache, device, dtype):
        self.model = model
        self.S = S
        self.cache = StaticCache(config=model.config, batch_size=1, max_cache_len=max_cache,
                                 device=device, dtype=dtype)
        self.ids = torch.zeros(1, S, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, S, dtype=torch.long, device=device)
        self.cp = torch.zeros(S, dtype=torch.long, device=device)
        self.graph = None
        self.out = None

    @torch.inference_mode()
    def prefill(self, ids):
        """Eager prefill of committed[:-1]; returns nothing. Cache now holds len(ids)-... ."""
        n = ids.size(1)
        self.model(input_ids=ids, position_ids=torch.arange(n, device=ids.device).view(1, -1),
                   past_key_values=self.cache, use_cache=True,
                   cache_position=torch.arange(n, device=ids.device))

    @torch.inference_mode()
    def capture(self):
        def step():
            return self.model(input_ids=self.ids, position_ids=self.pos,
                              past_key_values=self.cache, use_cache=True,
                              cache_position=self.cp).logits
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = step()

    def run(self, ids, pos, cp):
        """Copy inputs into static buffers, replay, return logits (a view of self.out)."""
        self.ids.copy_(ids); self.pos.copy_(pos); self.cp.copy_(cp)
        self.graph.replay()
        return self.out


@torch.inference_mode()
def graphed_spec(gd, gt, prompt_ids, gamma, n_new, device):
    """Correct graphed KV spec decode for one sequence. Returns committed token ids (list)."""
    P = prompt_ids.size(1)
    # prefill committed[:-1] into both caches; pending = last token
    gd.prefill(prompt_ids[:, :-1]); gt.prefill(prompt_ids[:, :-1])
    pending = prompt_ids[:, -1:]                       # [1,1]
    clen = P                                            # committed length
    committed = []
    while len(committed) < n_new:
        # draft proposes gamma tokens
        x = pending; props = []
        for j in range(gamma):
            pos = torch.tensor([[clen - 1 + j]], device=device)
            cp = torch.tensor([clen - 1 + j], device=device)
            lg = gd.run(x, pos, cp)[:, -1]
            x = lg.argmax(-1, keepdim=True); props.append(x)
        prop = torch.cat(props, 1)                      # [1,gamma]
        # verify [pending, prop0..prop_{g-1}]
        X = torch.cat([pending, prop], 1)               # [1,gamma+1]
        vpos = torch.arange(clen - 1, clen - 1 + gamma + 1, device=device).view(1, -1)
        vcp = torch.arange(clen - 1, clen - 1 + gamma + 1, device=device)
        pred = gt.run(X, vpos, vcp)[0].argmax(-1)       # [gamma+1]
        a = 0
        while a < gamma and pred[a].item() == prop[0, a].item():
            a += 1
        bonus = pred[a].view(1, 1)
        committed.extend(prop[0, :a].tolist()); committed.append(bonus.item())
        # rewind: next writes start at clen' - 1 = clen + a; stale slots beyond are never
        # attended (causal up to cache_position). Draft cache: if a==gamma we must feed the
        # last proposal so the draft cache holds it before continuing.
        if a == gamma:
            pos = torch.tensor([[clen - 1 + gamma]], device=device)
            cp = torch.tensor([clen - 1 + gamma], device=device)
            gd.run(prop[:, -1:], pos, cp)
        pending = bonus; clen += a + 1
    return committed[:n_new]


@torch.inference_mode()
def graphed_nospec(gt, prompt_ids, n_new, device):
    P = prompt_ids.size(1)
    gt.prefill(prompt_ids[:, :-1])
    pending = prompt_ids[:, -1:]; clen = P; committed = []
    while len(committed) < n_new:
        pos = torch.tensor([[clen - 1]], device=device)
        cp = torch.tensor([clen - 1], device=device)
        nt = gt.run(pending, pos, cp)[:, -1].argmax(-1, keepdim=True)
        committed.append(nt.item()); pending = nt; clen += 1
    return committed


def timed(fn, iters=3):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    r = None
    for _ in range(iters):
        r = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, r


def main():
    p = argparse.ArgumentParser(description="Graphed KV speculative decode (deployment path)")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--n_new", type=int, default=64)
    p.add_argument("--drf_steps", type=int, default=500)
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--max_cache", type=int, default=256)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g = args.gamma
    tok = AutoTokenizer.from_pretrained(TGT)
    from paths import alpaca_lora
    train_instr, held = load_instructions(args.n_train, 8, args.dataset)

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

    prompts = [tok(PROMPT.format(instr=x), return_tensors="pt").input_ids.to(device)
               for x in held[:5]]

    # graph buffers: target-verify (S=g+1), target-decode (S=1), draft (S=1). Each needs its
    # own StaticCache + capture. The spec tiers reuse one target-verify graph; the target-decode
    # graph is only for no_spec.
    print("\n=== capture CUDA graphs (target verify g+1, target decode 1, drafts 1) ===")
    gtv = Graphed(target, g + 1, args.max_cache, device, dtype); gtv.capture()
    gtd = Graphed(target, 1, args.max_cache, device, dtype); gtd.capture()
    gd_f = Graphed(faithful, 1, args.max_cache, device, dtype); gd_f.capture()
    gd_s = Graphed(shared, 1, args.max_cache, device, dtype); gd_s.capture()

    # ---- correctness: graphed committed == eager spec_kv committed (bit-identical) ----
    print("=== correctness: graphed vs eager (DynamicCache) committed tokens ===")
    x0 = prompts[0]
    c_g = graphed_spec(gd_f, gtv, x0, g, 24, device)
    c_e, _, _ = spec_kv(faithful, target, x0, g, 24)
    c_e = c_e[0].tolist()
    ns_g = graphed_nospec(gtd, x0, 24, device)
    ns_e = nospec_kv(target, x0, 24)[0].tolist()
    ok_s = c_g == c_e; ok_n = ns_g == ns_e
    print(f"   warp: graphed==eager {ok_s}")
    print(f"no_spec: graphed==eager {ok_n}")
    assert ok_s and ok_n, f"graphed != eager\nwarp g={c_g}\nwarp e={c_e}"
    print("  correctness OK: graphed reproduces eager exactly")

    # ---- throughput: graphed no_spec vs shared vs warp (B=1 per stream, aggregated) ----
    print(f"\n=== graphed KV decode throughput (B=1/stream, n_new={args.n_new}) ===")
    res = {}
    plans = [("no_spec", None), ("shared", gd_s), ("warp", gd_f)]
    for name, gdraft in plans:
        secs = toks = 0.0
        for x in prompts:
            if gdraft is None:
                dt, out = timed(lambda x=x: graphed_nospec(gtd, x, args.n_new, device))
            else:
                dt, out = timed(lambda x=x, gdraft=gdraft:
                                graphed_spec(gdraft, gtv, x, g, args.n_new, device))
            secs += dt; toks += len(out)
        res[name] = {"tok_s": toks / secs}
        print(f"  {name:>8}: {toks/secs:7.1f} tok/s")
    ns, sh, wp = res["no_spec"]["tok_s"], res["shared"]["tok_s"], res["warp"]["tok_s"]
    print(f"\n  GRAPHED WarpPipe/no_spec = {wp/ns:.2f}x   WarpPipe/shared = {wp/sh:.2f}x")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"serve_kv_graph_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "gamma": g, "n_new": args.n_new,
                   "dataset": args.dataset, "graphed": True, "kv_cached": True,
                   "warp_over_nospec": wp / ns, "warp_over_shared": wp / sh,
                   "results": res}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
