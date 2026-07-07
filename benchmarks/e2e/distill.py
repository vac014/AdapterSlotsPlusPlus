#!/usr/bin/env python3
"""
Draft-LoRA distillation on a controlled distribution shift.

Trains a LoRA on the target so its output distribution moves in a way that is
unambiguous to read (continue the prompt in uppercase), distils a llama-160m
draft-LoRA on the adapted target's own greedy continuations, and measures greedy
speculative acceptance of the adapted target under three drafts: the base 160m with
no LoRA (adapter-blind), the distilled draft-LoRA (faithful), and optionally a
draft-LoRA trained for a different task.

This is the mechanism check on a synthetic shift. The headline acceptance numbers come
from real community adapters instead; see recipe.py.
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(__file__))
from paths import llama7b as _llama7b, llama160m as _llama160m  # noqa: E402
TGT = _llama7b()
DRF = _llama160m()

# Real text (public-domain-style) used to build the task corpus.
CORPUS = """
The history of science is the study of the development of knowledge over time.
Ancient civilizations made important contributions to mathematics and astronomy.
The Renaissance brought renewed interest in observation and experiment.
Modern physics describes the universe from the smallest particles to galaxies.
Biology examines living organisms and the processes that sustain them.
Chemistry studies matter, its properties, and the transformations it undergoes.
Computers process information using logical operations at high speed.
The internet connects billions of devices across the entire world.
Machine learning enables systems to improve from data without explicit rules.
Language models predict the next token given the preceding context.
Economics analyzes how societies allocate scarce resources among many uses.
Philosophy asks fundamental questions about knowledge, reality, and ethics.
""".strip()


def build_batches(tok, text, seqlen, n_repeat):
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = []
    for _ in range(n_repeat):
        for i in range(0, len(ids) - seqlen, seqlen // 2):
            chunks.append(ids[i:i + seqlen])
    x = torch.stack([c for c in chunks if len(c) == seqlen])
    return TensorDataset(x)


def train_lora(base_path, tok, corpus_text, r, steps, lr, seqlen, batch, device,
               grad_ckpt=False, tag="", dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype).to(device)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.train()
    ds = build_batches(tok, corpus_text, seqlen, n_repeat=200)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    it = iter(dl)
    t0 = time.time()
    for s in range(steps):
        try:
            (xb,) = next(it)
        except StopIteration:
            it = iter(dl); (xb,) = next(it)
        xb = xb.to(device)
        out = model(input_ids=xb, labels=xb)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        if s % max(1, steps // 5) == 0 or s == steps - 1:
            print(f"  [{tag}] step {s:>4}/{steps}  loss {out.loss.item():.3f}  "
                  f"{(time.time()-t0):.0f}s")
    model.eval()
    return model


@torch.inference_mode()
def spec_accept(draft, target, tok, prompt, gamma, steps, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    acc_tot = prop_tot = 0
    for _ in range(steps):
        d = ids.clone(); proposed = []
        for _ in range(gamma):
            nt = draft(d).logits[:, -1].argmax(-1, keepdim=True)
            proposed.append(nt); d = torch.cat([d, nt], 1)
        prop = torch.cat(proposed, 1)
        cat = torch.cat([ids, prop], 1)
        tgt_tok = target(cat).logits[:, ids.shape[1] - 1:-1].argmax(-1)
        a = 0
        for j in range(gamma):
            if prop[0, j].item() == tgt_tok[0, j].item():
                a += 1
            else:
                break
        acc_tot += a; prop_tot += gamma
        new = tgt_tok[0, :min(a + 1, gamma)]
        ids = torch.cat([ids, new.view(1, -1)], 1)
    return acc_tot / prop_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgt_steps", type=int, default=120)
    p.add_argument("--drf_steps", type=int, default=300)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--drf_r", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=96)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=16)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TGT)
    task_text = CORPUS.upper()   # the adapter's task: continue in UPPERCASE

    print("Stage 1: train target LoRA on llama-7b (task=UPPERCASE)")
    target = train_lora(TGT, tok, task_text, args.r, args.tgt_steps, 2e-4,
                        args.seqlen, batch=2, device=device, grad_ckpt=True, tag="tgt")

    print("Stage 2: distill draft LoRA on llama-160m (same task, fp32 for stability)")
    faithful = train_lora(DRF, tok, task_text, args.drf_r, args.drf_steps, 2e-4,
                         args.seqlen, batch=8, device=device, grad_ckpt=False, tag="drf",
                         dtype=torch.float32)

    print("Stage 3: base draft (160m, no LoRA)")
    base_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32).to(device).eval()

    prompts = ["THE HISTORY OF SCIENCE IS", "MACHINE LEARNING ENABLES",
               "MODERN PHYSICS DESCRIBES", "COMPUTERS PROCESS INFORMATION",
               "THE INTERNET CONNECTS"]
    print("Stage 4: acceptance eval (adapted target)")
    rows = []
    for name, draft in [("base_draft", base_draft), ("faithful_draft", faithful)]:
        rates = [spec_accept(draft, target, tok, pr, args.gamma, args.eval_steps, device)
                 for pr in prompts]
        mean = sum(rates) / len(rates)
        rows.append({"draft": name, "mean_accept": mean, "per_prompt": rates})
        print(f"  {name:>16}: mean accept {mean:.3f}  per-prompt {[round(r,2) for r in rates]}")

    base = next(r["mean_accept"] for r in rows if r["draft"] == "base_draft")
    faith = next(r["mean_accept"] for r in rows if r["draft"] == "faithful_draft")
    verdict = "PASS" if faith > base else "FAIL"
    print(f"\nacceptance, faithful vs adapter-blind: {faith:.3f} vs {base:.3f}  -> {verdict}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"distill_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"target": "llama-7b+LoRA(uppercase)", "draft": "llama-160m",
                   "gamma": args.gamma, "results": rows,
                   "faithful_mean": faith, "base_mean": base, "verdict": verdict}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
