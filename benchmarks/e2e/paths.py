#!/usr/bin/env python3
"""
Model and dataset path resolution.

Every asset is resolved as: an explicit env-var override, else a snapshot already in
the HF cache, else the plain repo id (so `from_pretrained` fetches it on first use).
No absolute path is baked into a benchmark, and no latency constant is baked in
anywhere, so a run on different hardware re-measures rather than re-uses. Only the
model identities need to match, and they are pinned here.
"""
import glob
import os

# Canonical HF repo identities (pin these; the numbers are tied to these weights).
REPO_LLAMA7B = "huggyllama/llama-7b"          # LLaMA-1 7B base (target)
REPO_LLAMA13B = "huggyllama/llama-13b"        # LLaMA-1 13B base (across-models target)
REPO_LLAMA160M = "JackFram/llama-160m"        # tiny draft base
REPO_ALPACA_LORA = "tloen/alpaca-lora-7b"     # real instruction LoRA (target adapter)
REPO_ALPACA_LORA_13B = "chansung/alpaca-lora-13b"  # real 13B instruction LoRA (weight-
#                    identical base to huggyllama/llama-13b; r=16, q/k/v/o, verified B!=0)


def _hf_hub():
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.path.join(hf, "hub")
    return hub if os.path.isdir(hub) else hf


def resolve(repo_id, env=None):
    """Env override -> cached HF snapshot dir -> plain repo id (auto-download)."""
    if env and os.environ.get(env):
        return os.environ[env]
    org, name = repo_id.split("/")
    hits = sorted(glob.glob(os.path.join(_hf_hub(), f"models--{org}--{name}",
                                         "snapshots", "*")))
    return hits[-1] if hits else repo_id


def llama7b():
    return resolve(REPO_LLAMA7B, "WARPPIPE_LLAMA7B")


def llama13b():
    return resolve(REPO_LLAMA13B, "WARPPIPE_LLAMA13B")


def llama160m():
    return resolve(REPO_LLAMA160M, "WARPPIPE_LLAMA160M")


def alpaca_lora():
    return resolve(REPO_ALPACA_LORA, "WARPPIPE_ALPACA_LORA")


def alpaca_lora_13b():
    return resolve(REPO_ALPACA_LORA_13B, "WARPPIPE_ALPACA_LORA_13B")


def sharegpt_json():
    """ShareGPT is not on the Hub; point WARPPIPE_SHAREGPT at a local ShareGPT.json."""
    p = os.environ.get("WARPPIPE_SHAREGPT")
    if p:
        return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data", "ShareGPT.json")
