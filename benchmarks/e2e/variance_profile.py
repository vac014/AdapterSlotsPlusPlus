#!/usr/bin/env python3
"""
Why one seed of the same config reads 0.317 and another 0.158.

Fixed config, fixed corpus, different seeds, and acceptance moves by a factor of two on
ShareGPT. This traces it: distil per seed, snapshot the LoRA at several step counts, and
score every snapshot on validation to recover the acceptance trajectory.

The trajectory rises and then falls as the draft overfits the small diverse corpus, so a
draft taken at a fixed final step is a lottery ticket on where that seed happened to
land. Early-stopping at the best validation checkpoint removes the first variance source
and training several seeds removes the second, which is what the deployed recipe does.
Validation picks the checkpoint; the disjoint test split is only reported.
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
from distill import TGT, DRF  # noqa: E402
from recipe import load_instructions, greedy, PROMPT  # noqa: E402
from spec_loops import spec_kv  # noqa: E402

ALPACA = "tloen/alpaca-lora-7b"


def snapshot_lora(model):
    """CPU copy of the trainable (LoRA) params, keyed by name."""
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def build_merged_draft(snap, r, modules, device):
    """Rebuild a fresh 160m + LoRA(r, modules), load snapshot weights, merge to
    fp16, the exact deployment path, so acceptance matches how it would ship."""
    m = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device)
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r,
        target_modules=list(modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
    sd = dict(m.named_parameters())
    with torch.no_grad():
        for n, v in snap.items():
            if n in sd:
                sd[n].data.copy_(v.to(sd[n].device, sd[n].dtype))
    merged = m.merge_and_unload().to(device=device, dtype=torch.float16).eval()
    merged.config.use_cache = True
    del m
    torch.cuda.empty_cache()
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=16)
    p.add_argument("--n_test", type=int, default=32)
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--modules", choices=["qv", "qkvo"], default="qkvo")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gen_tokens", type=int, default=96)
    p.add_argument("--eval_tokens", type=int, default=48)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ckpt_steps", type=int, nargs="+",
                   default=[300, 600, 900, 1200, 1500])
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device = "cuda"
    g = args.gamma
    modules = ("q_proj", "k_proj", "v_proj", "o_proj") if args.modules == "qkvo" \
        else ("q_proj", "v_proj")
    max_step = max(args.ckpt_steps)
    tok = AutoTokenizer.from_pretrained(TGT)
    from paths import alpaca_lora; apath = alpaca_lora()

    print(f"modules={modules} lr={args.lr} seeds={args.seeds} ckpts={args.ckpt_steps}")
    print("Load target ONCE: llama-7b + REAL alpaca-lora-7b (merged)")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16).to(device)
    target = PeftModel.from_pretrained(base, apath).merge_and_unload().eval()
    target.config.use_cache = True

    # fixed instruction split: train | val | test (disjoint)
    train_instr, held = load_instructions(args.n_train, args.n_val + args.n_test, "sharegpt")
    val_instr, test_instr = held[:args.n_val], held[args.n_val:]
    print(f"{len(train_instr)} train / {len(val_instr)} val / {len(test_instr)} test")

    print("Build corpus ONCE (target greedy continuations)")
    t0 = time.time()
    corpus = []
    for i, instr in enumerate(train_instr):
        t = PROMPT.format(instr=instr)
        corpus.append(t + greedy(target, tok, t, args.gen_tokens, device))
        if i % 100 == 0:
            print(f"  gen {i}/{len(train_instr)} {time.time()-t0:.0f}s")
    corpus_text = "\n\n".join(corpus)

    def accept(draft, instrs):
        tot = st = 0
        for instr in instrs:
            ids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
            _, a, s = spec_kv(draft, target, ids, g, args.eval_tokens)
            tot += a; st += s
        return tot / (st * g)

    # deterministic shared baseline
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float16).to(device).eval()
    shared.config.use_cache = True
    sh_val, sh_test = accept(shared, val_instr), accept(shared, test_instr)
    print(f"\nSHARED baseline: val={sh_val:.3f}  test={sh_test:.3f}")

    # tokenize corpus once (chunks shared across seeds; only shuffle order + init vary)
    ids = tok(corpus_text, return_tensors="pt").input_ids[0]
    L = 128
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L, L // 2)]
    x = torch.stack([c for c in chunks if len(c) == L])
    print(f"corpus: {x.size(0)} chunks of {L}")

    seed_rows = []
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device)
        model = get_peft_model(model, LoraConfig(r=args.drf_r, lora_alpha=2 * args.drf_r,
            target_modules=list(modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
        model.train()
        dl = DataLoader(TensorDataset(x), batch_size=8, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
        it = iter(dl)
        snaps = {}
        t1 = time.time()
        for s in range(1, max_step + 1):
            try:
                (xb,) = next(it)
            except StopIteration:
                it = iter(dl); (xb,) = next(it)
            xb = xb.to(device)
            out = model(input_ids=xb, labels=xb); out.loss.backward()
            opt.step(); opt.zero_grad()
            if s in args.ckpt_steps:
                snaps[s] = (snapshot_lora(model), out.loss.item())
                print(f"  [seed {seed}] snap step {s} loss {out.loss.item():.3f} "
                      f"{time.time()-t1:.0f}s")
        del model, opt
        torch.cuda.empty_cache()

        # eval each snapshot on VAL (trajectory), then best-val on TEST
        traj = []
        for s in args.ckpt_steps:
            snap, loss = snaps[s]
            d = build_merged_draft(snap, args.drf_r, modules, device)
            v = accept(d, val_instr)
            traj.append({"step": s, "loss": loss, "val": v})
            del d; torch.cuda.empty_cache()
            print(f"  [seed {seed}] step {s:>4}  loss {loss:.3f}  val_accept {v:.3f}")
        best = max(traj, key=lambda r: r["val"])
        final = traj[-1]
        d_best = build_merged_draft(snaps[best["step"]][0], args.drf_r, modules, device)
        best_test = accept(d_best, test_instr)
        del d_best; torch.cuda.empty_cache()
        d_final = build_merged_draft(snaps[final["step"]][0], args.drf_r, modules, device)
        final_test = accept(d_final, test_instr)
        del d_final; torch.cuda.empty_cache()
        row = {"seed": seed, "traj": traj, "best_step": best["step"],
               "best_val": best["val"], "best_test": best_test,
               "final_step": final["step"], "final_val": final["val"],
               "final_test": final_test}
        seed_rows.append(row)
        print(f"  [seed {seed}] BEST-VAL step {best['step']} (val {best['val']:.3f}) "
              f"-> TEST {best_test:.3f} | naive FINAL step {final['step']} "
              f"-> TEST {final_test:.3f}  (shared test {sh_test:.3f})")

    print("\n=== summary (shared test={:.3f}) ===".format(sh_test))
    print(f"{'seed':>4} {'best_step':>9} {'best_val':>9} {'best_test':>10} "
          f"{'final_test':>11}")
    for r in seed_rows:
        print(f"{r['seed']:>4} {r['best_step']:>9} {r['best_val']:>9.3f} "
              f"{r['best_test']:>10.3f} {r['final_test']:>11.3f}")
    best_tests = [r["best_test"] for r in seed_rows]
    final_tests = [r["final_test"] for r in seed_rows]
    import statistics as st
    print(f"\nEARLY-STOP (best-val)  TEST: min={min(best_tests):.3f} "
          f"mean={st.mean(best_tests):.3f} max={max(best_tests):.3f}  "
          f"vs shared {sh_test:.3f}")
    print(f"NAIVE final-step 1500  TEST: min={min(final_tests):.3f} "
          f"mean={st.mean(final_tests):.3f} max={max(final_tests):.3f}")
    robust_win = min(best_tests) > sh_test
    print(f"\nevery seed beats the shared draft after early stopping: {robust_win}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"variance_profile_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"adapter": ALPACA, "modules": args.modules, "lr": args.lr,
                   "drf_r": args.drf_r, "n_train": args.n_train,
                   "shared_val": sh_val, "shared_test": sh_test,
                   "seeds": seed_rows, "robust_win_earlystop": robust_win,
                   "best_test_min": min(best_tests),
                   "final_test_min": min(final_tests),
                   "final_test_max": max(final_tests)}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
