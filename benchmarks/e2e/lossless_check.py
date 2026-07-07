#!/usr/bin/env python3
"""Tie-aware losslessness check.

Greedy speculative decoding is lossless in exact arithmetic: verifying with argmax commits
exactly the tokens the target would have produced alone. In fp16 it is lossless up to
argmax ties. The tiers feed the target different shapes (1 token vs gamma+1), which changes
the GEMM reduction order, so when the target's top two logits sit within fp16 resolution of
each other the winner can flip. That is a property of batched fp16 inference, not of
speculation: the same row decoded alone and decoded inside a padded batch already
disagrees on such a token.

So a mismatch is not accepted on faith: the divergent prefix is re-run through the target
on its own and the top-2 logit gap at that position is measured. A gap at the fp16 noise
floor is a tie. A real gap is a bug, and is reported as one.
"""
import torch

TIE = 0.05          # fp16 has ~3 decimal digits; logits here are O(10)


@torch.inference_mode()
def explain(target, prompt, ctx, device):
    """Top-2 target logits for the next token after prompt+ctx, single row, no padding."""
    ids = torch.cat([prompt, torch.tensor(ctx, dtype=torch.long, device=device)]).view(1, -1)
    lg = target(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False).logits[0, -1].float()
    top = lg.topk(2)
    return top.values[0].item(), top.values[1].item(), top.indices.tolist()


def check(name, got, ref, target, prompts, device, n):
    """Returns (ok, notes). ok is False only if a divergence is NOT an argmax tie."""
    bad = [i for i in range(len(prompts)) if got[i][:n] != ref[i][:n]]
    if not bad:
        return True, f"{name}: token-identical to plain decode on all {len(prompts)} rows"
    notes, ok = [], True
    for i in bad:
        j = next(k for k in range(n) if got[i][k] != ref[i][k])
        t1, t2, idx = explain(target, prompts[i], ref[i][:j], device)
        gap = t1 - t2
        tie = gap < TIE and {got[i][j], ref[i][j]} <= set(idx)
        ok &= tie
        notes.append(f"    row {i} token {j}: plain={ref[i][j]} {name}={got[i][j]} | "
                     f"target top-2 logits {t1:.4f} vs {t2:.4f}, gap {gap:.4f} -> "
                     f"{'ARGMAX TIE (fp16 noise)' if tie else 'REAL DIVERGENCE (bug)'}")
    head = (f"{name}: {len(prompts) - len(bad)}/{len(prompts)} rows identical; "
            f"{len(bad)} diverge")
    return ok, "\n".join([head] + notes)
