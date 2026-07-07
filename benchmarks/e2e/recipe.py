#!/usr/bin/env python3
"""
The deployed distillation recipe, on a real community adapter.

Target is llama-7b with tloen/alpaca-lora-7b merged in. The distillation corpus is
that target's own greedy continuations of held-out instructions, so the draft is
fitted to the adapted target rather than to the task.

`distill_draft_es` is the recipe every other benchmark imports: train one draft per
seed, snapshot at several step counts, score each snapshot on a validation split, and
deploy the best. The shared draft is in the candidate set, so a faithful draft that
fails to beat it on validation is never deployed. Selection touches validation only;
the eval split is disjoint and reported, never tuned against.
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
from spec_loops import spec_kv, nospec_kv  # noqa: E402

ALPACA = "tloen/alpaca-lora-7b"

PROMPT = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instr}\n\n### Response:\n")

def load_instructions(n_train, n_eval, dataset="alpaca"):
    """Instruction prompts. `alpaca` = general instructions (mild shift);
    `gsm8k` = grade-school math word problems (structured, step-by-step numeric
    output -> the SPECIALIZED / high-structure workload). In both cases the
    distillation target is the target model's OWN greedy continuation, so this
    only chooses the prompt distribution the draft must become faithful to."""
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        instrs = [r["question"] for r in ds][: n_train + n_eval]
        return instrs[:n_train], instrs[n_train:n_train + n_eval]
    if dataset == "mbpp":
        # code generation: highly structured output -> naturally high draft
        # acceptance, and the base draft misses the target's specific code.
        ds = load_dataset("mbpp", split="train")
        instrs = [r["text"] + "\nWrite the Python function." for r in ds][: n_train + n_eval]
        return instrs[:n_train], instrs[n_train:n_train + n_eval]
    if dataset == "dolly":
        # general open-domain instructions (MILD shift, like alpaca) -> small lift
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        instrs = [r["instruction"] for r in ds if r["context"].strip() == ""][: n_train + n_eval]
        return instrs[:n_train], instrs[n_train:n_train + n_eval]
    if dataset == "samsum":
        # dialogue summarization (MODERATE structure) -> structured target output
        try:
            ds = load_dataset("knkarthick/samsum", split="train")
        except Exception:
            ds = load_dataset("Samsung/samsum", split="train", trust_remote_code=True)
        instrs = ["Summarize the following conversation.\n\n" + r["dialogue"]
                  for r in ds][: n_train + n_eval]
        return instrs[:n_train], instrs[n_train:n_train + n_eval]
    if dataset == "sharegpt":
        # the AS++ continuity workload (multi-turn chat); first human turn as the
        # instruction. General conversation -> mild/moderate shift.
        import json
        from paths import sharegpt_json; path = sharegpt_json()
        conv = json.load(open(path))
        instrs = []
        for r in conv:
            for t in r.get("conversations", []):
                v = t.get("value", "").strip()
                if t.get("from") == "human" and 16 <= len(v) <= 600:
                    instrs.append(v); break
            if len(instrs) >= n_train + n_eval:
                break
        return instrs[:n_train], instrs[n_train:n_train + n_eval]
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    instrs = []
    for r in ds:
        if r["input"].strip() == "":
            instrs.append(r["instruction"])
        if len(instrs) >= n_train + n_eval:
            break
    return instrs[:n_train], instrs[n_train:n_train + n_eval]


@torch.inference_mode()
def greedy(model, tok, text, n_new, device):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    gen = nospec_kv(model, ids, n_new)
    return tok.decode(gen[0], skip_special_tokens=True)


def distill_draft(base_path, tok, corpus_text, r, steps, lr, device,
                  target_modules=("q_proj", "v_proj")):
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32).to(device)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=list(target_modules),
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg); model.train()
    ids = tok(corpus_text, return_tensors="pt").input_ids[0]
    L = 128
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L, L // 2)]
    x = torch.stack([c for c in chunks if len(c) == L])
    dl = DataLoader(TensorDataset(x), batch_size=8, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    it = iter(dl); t0 = time.time()
    for s in range(steps):
        try:
            (xb,) = next(it)
        except StopIteration:
            it = iter(dl); (xb,) = next(it)
        xb = xb.to(device)
        out = model(input_ids=xb, labels=xb); out.loss.backward()
        opt.step(); opt.zero_grad()
        if s % max(1, steps // 5) == 0 or s == steps - 1:
            print(f"  [drf] step {s:>4}/{steps} loss {out.loss.item():.3f} {(time.time()-t0):.0f}s")
    return model.merge_and_unload().to(device=device, dtype=torch.float16).eval()


def _lora_snapshot(model):
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def _rebuild_merged(base_path, snap, r, modules, device):
    m = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32).to(device)
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r,
        target_modules=list(modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
    sd = dict(m.named_parameters())
    with torch.no_grad():
        for n, v in snap.items():
            if n in sd:
                sd[n].data.copy_(v.to(sd[n].device, sd[n].dtype))
    merged = m.merge_and_unload().to(device=device, dtype=torch.float16).eval()
    merged.config.use_cache = True
    del m; torch.cuda.empty_cache()
    return merged


def _rebuild_peft(base_path, snap, r, modules, device):
    """Like _rebuild_merged but returns the UNMERGED PEFT model, so callers can extract the
    selected draft's lora_A/lora_B (e.g. for draft-side SGMV / BGMV stacking)."""
    m = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16).to(device)
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r,
        target_modules=list(modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
    sd = dict(m.named_parameters())
    with torch.no_grad():
        for n, v in snap.items():
            if n in sd:
                sd[n].data.copy_(v.to(sd[n].device, sd[n].dtype))
    m.eval(); m.config.use_cache = True
    return m


def distill_draft_es(base_path, tok, corpus_text, r, steps, lr, device, target,
                     val_instrs, gamma, eval_tokens, ckpt_steps=(150, 300, 450, 600, 900),
                     target_modules=("q_proj", "v_proj"), seeds=(0, 1, 2),
                     shared_draft=None, return_unmerged=False):
    """Robust distillation of the faithful draft, selected on VALIDATION only.

    Two variance sources were profiled (variance_profile.py) on diverse workloads
    like ShareGPT and BOTH are removed here:
      1. STEP lottery: held-out acceptance peaks early (~150-600 steps) then
         declines as the draft memorizes the corpus. Fix: snapshot at `ckpt_steps`
         and keep the best-VAL checkpoint (standard early stopping).
      2. INIT lottery: the peak HEIGHT varies with RNG seed (one seed lands 0.32,
         another 0.21). Fix: train `seeds` independent drafts and keep the
         best-VAL one (standard "train a few, deploy the best on validation").

    `shared_draft` (base, no LoRA) is added as a candidate so the deployed draft is
    `argmax_val{shared} ∪ {seed_s @ best-ckpt}`, i.e. we NEVER deploy below the
    shared baseline (the floor), and deploy faithful only when it genuinely beats
    shared on val. Selection is on val; the caller reports on a DISJOINT test set,
    so this is not test-set tuning. Returns (deployed_merged, info).
    """
    from spec_loops import spec_kv  # local import: avoid top-level cycle risk
    ids = tok(corpus_text, return_tensors="pt").input_ids[0]
    L = 128
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L, L // 2)]
    x = torch.stack([c for c in chunks if len(c) == L])
    ckpts = [c for c in ckpt_steps if c <= steps] or [steps]

    def val_accept(draft):
        tot = st = 0
        for instr in val_instrs:
            vids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
            _, a, s = spec_kv(draft, target, vids, gamma, eval_tokens)
            tot += a; st += s
        return tot / (st * gamma)

    # candidate 0: the shared draft (floor). deploy faithful only if it beats this.
    best = {"kind": "shared", "seed": None, "step": None, "val": -1.0, "merged": None}
    if shared_draft is not None:
        sv = val_accept(shared_draft)
        best.update(val=sv, merged=shared_draft)
        print(f"  [drf-es] shared floor  val_accept {sv:.3f}")
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32).to(device)
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r,
            target_modules=list(target_modules), lora_dropout=0.0, task_type="CAUSAL_LM"))
        model.train()
        dl = DataLoader(TensorDataset(x), batch_size=8, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        it = iter(dl); snaps = {}
        for s in range(1, steps + 1):
            try:
                (xb,) = next(it)
            except StopIteration:
                it = iter(dl); (xb,) = next(it)
            xb = xb.to(device)
            out = model(input_ids=xb, labels=xb); out.loss.backward()
            opt.step(); opt.zero_grad()
            if s in ckpts:
                snaps[s] = _lora_snapshot(model)
        del model, opt; torch.cuda.empty_cache()

        s_best = {"seed": seed, "step": None, "val": -1.0}
        for s in sorted(snaps):
            d = _rebuild_merged(base_path, snaps[s], r, target_modules, device)
            v = val_accept(d)
            if v > s_best["val"]:
                s_best = {"seed": seed, "step": s, "val": v}
            if v > best["val"]:
                if best["merged"] is not None and best["kind"] == "faithful":
                    del best["merged"]
                best = {"kind": "faithful", "seed": seed, "step": s, "val": v, "merged": d,
                        "snap": snaps[s]}
            else:
                del d
            torch.cuda.empty_cache()
        per_seed.append(s_best)
        print(f"  [drf-es] seed {seed}: best-ckpt step {s_best['step']} val {s_best['val']:.3f}")

    print(f"  [drf-es] DEPLOY {best['kind']} (seed {best['seed']} step {best['step']} "
          f"val {best['val']:.3f})")
    info = {"deployed": best["kind"], "seed": best["seed"], "step": best["step"],
            "deployed_val": best["val"], "per_seed": per_seed}
    if return_unmerged:
        # UNMERGED selected draft (or None if the shared floor won -> caller uses base draft)
        unm = (_rebuild_peft(base_path, best["snap"], r, target_modules, device)
               if best["kind"] == "faithful" else None)
        return unm, info
    return best["merged"], info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drf_steps", type=int, default=600)
    p.add_argument("--drf_r", type=int, default=8)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--n_train", type=int, default=250)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--gen_tokens", type=int, default=48)
    p.add_argument("--eval_tokens", type=int, default=48)
    p.add_argument("--dataset", default="alpaca",
                   choices=["alpaca", "gsm8k", "mbpp", "dolly", "samsum", "sharegpt"],
                   help="alpaca/dolly/sharegpt=general; gsm8k=math; "
                        "samsum=summarization; mbpp=code")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    tok = AutoTokenizer.from_pretrained(TGT)
    from paths import alpaca_lora; apath = alpaca_lora()

    print(f"Stage 0: load instructions (dataset={args.dataset})")
    train_instr, eval_instr = load_instructions(args.n_train, args.n_eval, args.dataset)
    print(f"  {len(train_instr)} train / {len(eval_instr)} held-out instructions")

    print("Stage 1: load llama-7b + REAL alpaca-lora-7b adapter, merge")
    base = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=dtype).to(device)
    target = PeftModel.from_pretrained(base, apath).merge_and_unload().eval()
    target.config.use_cache = True

    print("Stage 2: build distillation corpus (target's own greedy continuations)")
    corpus = []
    t0 = time.time()
    for i, instr in enumerate(train_instr):
        t = PROMPT.format(instr=instr)
        cont = greedy(target, tok, t, args.gen_tokens, device)
        corpus.append(t + cont)
        if i % 50 == 0:
            print(f"  gen {i}/{len(train_instr)}  {time.time()-t0:.0f}s")
    corpus_text = "\n\n".join(corpus)
    print(f"  corpus: {len(train_instr)} instructions, {len(tok(corpus_text).input_ids)} tokens")

    print("Stage 3: distill faithful draft-LoRA on llama-160m")
    faithful = distill_draft(DRF, tok, corpus_text, args.drf_r, args.drf_steps, 2e-4, device)
    shared = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    shared.config.use_cache = True

    print("Stage 4: exact KV spec acceptance on HELD-OUT instructions")
    def accept(draft):
        tot = st = 0
        per = []
        for instr in eval_instr:
            ids = tok(PROMPT.format(instr=instr), return_tensors="pt").input_ids.to(device)
            _, a, s = spec_kv(draft, target, ids, args.gamma, args.eval_tokens)
            per.append(a / (s * args.gamma)); tot += a; st += s
        return tot / (st * args.gamma), per
    a_sh, per_sh = accept(shared)
    a_fa, per_fa = accept(faithful)
    print(f"\n  shared_draft   accept={a_sh:.3f}  per-instr {[round(x,2) for x in per_sh]}")
    print(f"  faithful_draft accept={a_fa:.3f}  per-instr {[round(x,2) for x in per_fa]}")
    verdict = "PASS" if a_fa > a_sh else "FAIL"
    print(f"\n  REAL-adapter acceptance (faithful > shared): {a_fa:.3f} vs {a_sh:.3f} -> {verdict}")
    print(f"  tokens/verify: shared {a_sh*args.gamma+1:.2f}  faithful {a_fa*args.gamma+1:.2f}  "
          f"(ratio {(a_fa*args.gamma+1)/(a_sh*args.gamma+1):.2f}x)")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"recipe_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"adapter": ALPACA, "dataset": args.dataset, "gamma": args.gamma,
                   "accept_shared": a_sh, "accept_faithful": a_fa,
                   "per_shared": per_sh, "per_faithful": per_fa,
                   "verdict": verdict}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
