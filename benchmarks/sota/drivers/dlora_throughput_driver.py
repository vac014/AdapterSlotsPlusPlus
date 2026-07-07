"""Closed-loop throughput driver for dLoRA /generate. CONC workers send back-to-back
256-in/128-out requests (random model_id) for DURATION seconds; decode tok/s =
completed * OUT_LEN / elapsed."""
import asyncio
import time
import random
import aiohttp

URL = "http://127.0.0.1:8200/generate"
CONC = 12
NUM_MODELS = 32
IN_LEN = 256
OUT_LEN = 128
DURATION = 60.0
VOCAB = 31000
random.seed(42)

# fixed 256-token prompt as a string of repeated words (~256 tokens); dLoRA tokenizes
PROMPT = "the " * IN_LEN
completed = 0
stop = False

async def worker(session, wid):
    global completed
    while not stop:
        mid = random.randint(0, NUM_MODELS - 1)
        payload = {"prompt": PROMPT, "max_tokens": OUT_LEN, "model_id": mid,
                   "temperature": 0.0, "ignore_eos": True, "stream": False}
        try:
            async with session.post(URL, json=payload) as r:
                await r.json()
            completed += 1
        except Exception:
            await asyncio.sleep(0.05)

async def main():
    global stop
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # warmup 1 req
        async with s.post(URL, json={"prompt": PROMPT, "max_tokens": 8, "model_id": 0,
                                     "temperature": 0.0, "ignore_eos": True}) as r:
            await r.json()
        t0 = time.time()
        tasks = [asyncio.create_task(worker(s, i)) for i in range(CONC)]
        await asyncio.sleep(DURATION)
        stop = True
        await asyncio.gather(*tasks, return_exceptions=True)
        dur = time.time() - t0
    out_tok = completed * OUT_LEN
    print(f"concurrency: {CONC}, duration: {dur:.1f}s, completed reqs: {completed}")
    print(f"decode tok/s: {out_tok/dur:.2f}")
    print(f"e2e tok/s: {completed*(IN_LEN+OUT_LEN)/dur:.2f}")
    print(f"req/s: {completed/dur:.2f}")

asyncio.run(main())
