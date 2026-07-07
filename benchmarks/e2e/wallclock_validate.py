#!/usr/bin/env python3
"""
Wall-clock validation of the composed throughput.

The tier throughputs are composed from graph-captured latencies and measured acceptance.
The way that could flatter us is graph capture hiding launch and Python-loop overhead
that a real serving loop would pay, so this times the real loop and compares.

It measures each step both graph-captured and eager at the serving batch, then times an
actual speculative iteration (gamma autoregressive draft forwards plus one verify, real
KV cache, real Python between the draft steps) for the shared and the SGMV draft, and
composes throughput from both. The ratio is how much the composition owes to capture.
A B=1 end-to-end spec decode over real prompts is timed as a second check.

Latency does not depend on adapter weights, so the SGMV draft carries a random LoRA at
the real rank.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from graph_latency import graph_step_ms, full_mask  # noqa: E402
from serve_multitenant import extract_draft_lora, wrap_draft, STATE  # noqa: E402
from recipe import load_instructions, PROMPT  # noqa: E402
from spec_loops import spec_kv  # noqa: E402
from paths import alpaca_lora  # noqa: E402

# deployed (shared, faithful) acceptance from throughput_tiers.py. The latency
# validation is acceptance-independent: tpv scales the analytical and the wall-clock
# composition equally, so these only set the absolute tok/s, not the ratio under test.
ACC = {"gsm8k": (0.213, 0.584), "mbpp": (0.108, 0.369)}


def eager_step_ms(model, B, C, m, device, dtype, max_cache, pre=None, iters=30):
    """Real (un-captured) latency of one cached forward, cuda-event timed."""
    cache = StaticCache(config=model.config, batch_size=B, max_cache_len=max_cache,
                        device=device, dtype=dtype)
    am = full_mask(B, max_cache, device)
    ids = torch.randint(5, 1000, (B, C), device=device)
    pos = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
    with torch.inference_mode():
        model(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=am,
              use_cache=True, cache_position=torch.arange(C, device=device))
    x = torch.randint(5, 1000, (B, m), device=device)
    p = torch.arange(C, C + m, device=device).unsqueeze(0).expand(B, -1)
    cp = torch.arange(C, C + m, device=device)
    if pre:
        pre(B, m)

    def step():
        with torch.inference_mode():
            return model(input_ids=x, position_ids=p, past_key_values=cache, attention_mask=am,
                         use_cache=True, cache_position=cp).logits
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); step(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b))
    return statistics.median(s)


def real_spec_iter_ms(target, draft, B, C, g, device, dtype, max_cache,
                      sgmv_pre=None, iters=20):
    """Time a REAL spec outer-iteration: g autoregressive draft forwards (with KV
    cache + the Python loop + argmax between them) + 1 verify forward of g+1
    tokens. Fresh prefilled caches per sample (prefill untimed)."""
    samples = []
    for _ in range(iters + 3):
        dc = StaticCache(config=draft.config, batch_size=B, max_cache_len=max_cache,
                         device=device, dtype=dtype)
        tc = StaticCache(config=target.config, batch_size=B, max_cache_len=max_cache,
                         device=device, dtype=dtype)
        am = full_mask(B, max_cache, device)
        ids = torch.randint(5, 1000, (B, C), device=device)
        pos = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
        cp = torch.arange(C, device=device)
        with torch.inference_mode():
            STATE["mode"] = "none"
            draft(input_ids=ids, position_ids=pos, past_key_values=dc, attention_mask=am,
                  use_cache=True, cache_position=cp)
            target(input_ids=ids, position_ids=pos, past_key_values=tc, attention_mask=am,
                   use_cache=True, cache_position=cp)
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record()
        with torch.inference_mode():
            x = torch.randint(5, 1000, (B, 1), device=device)
            for j in range(g):                      # g autoregressive draft steps
                cpj = torch.tensor([C + j], device=device)
                pj = cpj.unsqueeze(0).expand(B, -1)
                if sgmv_pre:
                    sgmv_pre(B, 1)
                lg = draft(input_ids=x, position_ids=pj, past_key_values=dc, attention_mask=am,
                           use_cache=True, cache_position=cpj).logits[:, -1]
                x = lg.argmax(-1, keepdim=True)
            STATE["mode"] = "none"
            xv = torch.randint(5, 1000, (B, g + 1), device=device)   # verify g+1
            cpv = torch.arange(C, C + g + 1, device=device)
            pv = cpv.unsqueeze(0).expand(B, -1)
            target(input_ids=xv, position_ids=pv, past_key_values=tc, attention_mask=am,
                   use_cache=True, cache_position=cpv)
        b.record(); b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples[3:])


def real_spec_iter_graphed_ms(target, draft, B, C, g, device, dtype, max_cache,
                              sgmv_pre=None, iters=30):
    """Production-representative wall-clock t_iter: CUDA-graph-capture the draft
    step and the verify step (as vLLM/TRT-LLM do), then time a real loop that
    replays them: g draft replays with the real argmax + input-copy glue between,
    plus 1 verify replay. Timing-only, so fixed cache_position is fine (compute is
    representative); the point is real replay + python glue, not cache correctness."""
    dc = StaticCache(config=draft.config, batch_size=B, max_cache_len=max_cache,
                     device=device, dtype=dtype)
    tc = StaticCache(config=target.config, batch_size=B, max_cache_len=max_cache,
                     device=device, dtype=dtype)
    am = full_mask(B, max_cache, device)
    ids = torch.randint(5, 1000, (B, C), device=device)
    pos0 = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
    cp0 = torch.arange(C, device=device)
    with torch.inference_mode():
        STATE["mode"] = "none"
        draft(input_ids=ids, position_ids=pos0, past_key_values=dc, attention_mask=am,
              use_cache=True, cache_position=cp0)
        target(input_ids=ids, position_ids=pos0, past_key_values=tc, attention_mask=am,
               use_cache=True, cache_position=cp0)
    # static buffers for the draft step
    din = torch.randint(5, 1000, (B, 1), device=device)
    dp = torch.tensor([C], device=device).unsqueeze(0).expand(B, -1)
    dcp = torch.tensor([C], device=device)
    if sgmv_pre:
        sgmv_pre(B, 1)

    def dstep():
        with torch.inference_mode():
            return draft(input_ids=din, position_ids=dp, past_key_values=dc, attention_mask=am,
                         use_cache=True, cache_position=dcp).logits
    for _ in range(3):
        dstep()
    torch.cuda.synchronize()
    gd = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gd):
        dout = dstep()
    # static buffers for verify (g+1 tokens)
    STATE["mode"] = "none"
    vin = torch.randint(5, 1000, (B, g + 1), device=device)
    vp = torch.arange(C, C + g + 1, device=device).unsqueeze(0).expand(B, -1)
    vcp = torch.arange(C, C + g + 1, device=device)

    def vstep():
        with torch.inference_mode():
            return target(input_ids=vin, position_ids=vp, past_key_values=tc, attention_mask=am,
                          use_cache=True, cache_position=vcp).logits
    for _ in range(3):
        vstep()
    torch.cuda.synchronize()
    gv = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gv):
        _ = vstep()

    def block():
        for _ in range(g):
            gd.replay()
            nt = dout[:, -1].argmax(-1, keepdim=True)   # real glue: argmax + copy
            din.copy_(nt)
        gv.replay()
    for _ in range(3):
        block()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); block(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b))
    return statistics.median(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gsm8k", choices=list(ACC))
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--max_cache", type=int, default=256)
    p.add_argument("--n_b1", type=int, default=12, help="prompts for end-to-end B=1 check")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    g, B, C = args.gamma, args.B, args.context
    a_sh, a_fa = ACC[args.dataset]
    tpv_sh, tpv_fa = a_sh * g + 1, a_fa * g + 1
    tok = AutoTokenizer.from_pretrained(TGT)

    print("load target 7B + alpaca, shared draft, random qkvo SGMV draft")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, alpaca_lora()).merge_and_unload().eval()
    target.config.use_cache = True
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True
    mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    rnd = get_peft_model(AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device),
                         LoraConfig(r=32, lora_alpha=64, target_modules=mods,
                                    lora_dropout=0.0, task_type="CAUSAL_LM"))
    w, scale = extract_draft_lora(rnd, min(B, 32), dtype, device, modules=tuple(mods))
    STATE["scale"] = scale
    wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    wd.config.use_cache = True
    wrap_draft(wd, w)
    adapter = torch.arange(B, device=device) % min(B, 32)

    def sgmv_pre(bb, mm):
        STATE["mode"] = "lora"; STATE["idx"] = adapter.repeat_interleave(mm)

    print(f"\n== per-step latency (B={B}): graph vs eager ==")
    rows = {}
    for name, model, m, pre in [("decode", target, 1, None), ("verify", target, g + 1, None),
                                ("draft", shared, 1, None), ("sgmv", wd, 1, sgmv_pre)]:
        tg = graph_step_ms(model, B, C, m, device, dtype, args.max_cache, pre)
        STATE["mode"] = "none"
        te = eager_step_ms(model, B, C, m, device, dtype, args.max_cache, pre)
        STATE["mode"] = "none"
        rows[name] = (tg, te)
        print(f"  {name:<7} graph={tg:6.2f}  eager={te:6.2f} ms  (eager/graph {te/tg:.2f}x)")

    print("\n== real in-loop spec iteration t_iter (ms): graph-composed vs EAGER loop vs GRAPHED loop ==")
    t_iter_spec_graph = rows["verify"][0] + g * rows["draft"][0]
    t_iter_warp_graph = rows["verify"][0] + g * rows["sgmv"][0]
    t_iter_spec_eager = real_spec_iter_ms(target, shared, B, C, g, device, dtype, args.max_cache)
    STATE["mode"] = "none"
    t_iter_warp_eager = real_spec_iter_ms(target, wd, B, C, g, device, dtype, args.max_cache, sgmv_pre)
    STATE["mode"] = "none"
    t_iter_spec_graphed = real_spec_iter_graphed_ms(target, shared, B, C, g, device, dtype, args.max_cache)
    STATE["mode"] = "none"
    t_iter_warp_graphed = real_spec_iter_graphed_ms(target, wd, B, C, g, device, dtype, args.max_cache, sgmv_pre)
    STATE["mode"] = "none"
    print(f"  AS++ spec : composed={t_iter_spec_graph:6.2f}  eager-loop={t_iter_spec_eager:6.2f}  "
          f"graphed-loop={t_iter_spec_graphed:6.2f}  (graphed/composed {t_iter_spec_graphed/t_iter_spec_graph:.2f}x)")
    print(f"  WarpPipe  : composed={t_iter_warp_graph:6.2f}  eager-loop={t_iter_warp_eager:6.2f}  "
          f"graphed-loop={t_iter_warp_graphed:6.2f}  (graphed/composed {t_iter_warp_graphed/t_iter_warp_graph:.2f}x)")

    def thr(tpv, t_iter): return B * tpv / (t_iter / 1000)
    t_dec_graph, t_dec_eager = rows["decode"]
    tiers = {
        "no_spec":   (thr(1.0, t_dec_graph),  thr(1.0, t_dec_eager), thr(1.0, t_dec_graph)),
        "aspp_spec": (thr(tpv_sh, t_iter_spec_graph), thr(tpv_sh, t_iter_spec_eager), thr(tpv_sh, t_iter_spec_graphed)),
        "warppipe":  (thr(tpv_fa, t_iter_warp_graph), thr(tpv_fa, t_iter_warp_eager), thr(tpv_fa, t_iter_warp_graphed)),
    }
    print("\n== throughput (tok/s): analytical(graph-composed) | EAGER loop | GRAPHED loop (production) ==")
    print(f"{'tier':<10}{'analytical':>12}{'eager':>10}{'graphed':>10}{'graphed/ana':>13}")
    for k, (ta, te, tgphd) in tiers.items():
        print(f"{k:<10}{ta:>12.0f}{te:>10.0f}{tgphd:>10.0f}{tgphd/ta:>12.2f}x")
    print(f"  WarpPipe/AS++spec: analytical={tiers['warppipe'][0]/tiers['aspp_spec'][0]:.2f}x  "
          f"graphed(real)={tiers['warppipe'][2]/tiers['aspp_spec'][2]:.2f}x")

    # end-to-end B=1 real spec_kv wall-clock cross-check (shared draft)
    print(f"\n== end-to-end B=1 real spec_kv (shared draft) on {args.n_b1} {args.dataset} prompts ==")
    _, ev = load_instructions(4, args.n_b1, args.dataset)
    tot_tok = acc = st = 0
    t0 = time.time()
    for instr in ev:
        ids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
        gen, a, s = spec_kv(shared, target, ids, g, 48)
        tot_tok += gen.shape[1]; acc += a; st += s
    torch.cuda.synchronize()
    wall = time.time() - t0
    real_tps_b1 = tot_tok / wall
    a_b1 = acc / (st * g)
    tpv_b1 = a_b1 * g + 1
    ana_tps_b1 = 1 * tpv_b1 / (t_iter_spec_graph / 1000)
    print(f"  measured accept={a_b1:.3f} tpv={tpv_b1:.2f}  real={real_tps_b1:.1f} tok/s  "
          f"analytical(graph,B=1)={ana_tps_b1:.1f} tok/s  (real/ana {real_tps_b1/ana_tps_b1:.2f}x)")
    print("  (B=1 real includes prefill + full python/sampling/rollback; lower bound on the model)")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"wallclock_validate_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": B, "gamma": g,
                   "dataset": args.dataset, "step_ms_graph_eager": rows,
                   "t_iter_spec": {"composed": t_iter_spec_graph, "eager": t_iter_spec_eager,
                                   "graphed": t_iter_spec_graphed},
                   "t_iter_warp": {"composed": t_iter_warp_graph, "eager": t_iter_warp_eager,
                                   "graphed": t_iter_warp_graphed},
                   "tiers_analytical_eager_graphed": {k: list(v) for k, v in tiers.items()},
                   "b1_real_tps": real_tps_b1, "b1_analytical_tps": ana_tps_b1,
                   "b1_accept": a_b1}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
