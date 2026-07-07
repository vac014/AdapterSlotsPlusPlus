"""Async load driver for the vLLM V1 OpenAI server. Sends zipf-skewed requests
across NUM_LORA adapters, 256-in/256-out, greedy; decode tok/s = output tokens /
wall time. Matches the SGLang/AdapterSlots workload (256/256, 14 LoRAs, concurrency 16).
"""
import asyncio
import time
import sys
import random
import numpy as np
import aiohttp

URL = "http://127.0.0.1:30000/v1/completions"
NUM_LORA = int(sys.argv[1]) if len(sys.argv) > 1 else 14
NUM_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
IN_LEN = 256
OUT_LEN = 256
CONCURRENCY = 16
ZIPF_ALPHA = 1.1
VOCAB = 31000
random.seed(42); np.random.seed(42)

# zipf-skewed lora ids
w = np.array([1.0 / (i + 1) ** ZIPF_ALPHA for i in range(NUM_LORA)]); w /= w.sum()
lora_ids = np.random.choice(NUM_LORA, size=NUM_PROMPTS, p=w)
prompts = [[random.randint(5, VOCAB) for _ in range(IN_LEN)] for _ in range(NUM_PROMPTS)]

sem = asyncio.Semaphore(CONCURRENCY)
out_tokens = [0] * NUM_PROMPTS

async def one(session, i):
    async with sem:
        payload = {"model": f"lora{int(lora_ids[i])}", "prompt": prompts[i],
                   "max_tokens": OUT_LEN, "temperature": 0.0, "ignore_eos": True}
        async with session.post(URL, json=payload) as r:
            d = await r.json()
        out_tokens[i] = d["usage"]["completion_tokens"]

async def main():
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        t0 = time.time()
        await asyncio.gather(*[one(s, i) for i in range(NUM_PROMPTS)])
        dur = time.time() - t0
    tot_out = sum(out_tokens)
    tot_e2e = tot_out + IN_LEN * NUM_PROMPTS
    print(f"requests: {NUM_PROMPTS}, loras: {NUM_LORA}, concurrency: {CONCURRENCY}")
    print(f"duration: {dur:.2f}s")
    print(f"decode tok/s: {tot_out/dur:.2f}")
    print(f"e2e tok/s: {tot_e2e/dur:.2f}")
    print(f"req/s: {NUM_PROMPTS/dur:.2f}")

asyncio.run(main())
