# AdapterSlots++ / WarpPipe

A grouped multi-LoRA apply kernel, the serving engine around it, and WarpPipe, the faithful
multi-tenant speculative decoding it makes affordable.

There are two halves to this, and they are useful separately.

**The apply kernel and its engine (`csrc/`, `lora_warp_pipe/`).** A multi-LoRA server holds many
adapters at once, and the cost that matters is applying all of them in one step. The kernels here
take a batch whose tokens belong to different adapters, sort it into segments, and shrink and
expand every segment in one grouped pass, so the apply stays flat in the number of resident
adapters instead of growing with it. This is the part that does not care about speculation at
all: it is the same primitive whether the tokens are a target's decode step or a draft's.

**WarpPipe (the speculative layer).** Speculative decoding wants a draft model that mimics the
target, but each tenant's adapter shifts the target's distribution, so one shared draft is
adapter-blind and its acceptance collapses on exactly the tenants whose adapters do the most
work. The faithful alternative, a draft per tenant, is unaffordable in the obvious
implementation: drafting K tenants means K draft forwards per step. WarpPipe gives each tenant
its own distilled draft-LoRA and applies all K of them in a single grouped pass on the draft
forward, using the same apply above, so the draft step stays flat in the number of resident
adapters. Segment indices route every token to its own adapter's slot, so the pass is exactly
equivalent to drafting each tenant separately, and the speculative loop is greedy-verified
against the true target, so committed tokens are identical to non-speculative decoding.

The system has three tiers, and every benchmark here reports all three:

| tier | draft |
| --- | --- |
| AdapterSlots | none (non-speculative floor) |
| AdapterSlots++ | one shared, adapter-blind draft |
| AdapterSlots++ / WarpPipe | one faithful draft-LoRA per tenant, applied in one pass |

## Layout

```
csrc/                CUDA extension: the grouped apply, segment dispatch, adapter staging
lora_warp_pipe/      the engine over that extension, and the scheduler bridge
benchmarks/kernel/   the apply primitive in isolation and inside a real draft step
benchmarks/e2e/      distillation, acceptance, serving loops, ablations
benchmarks/sota/     drivers for the external baselines, and their pinned versions
scripts/             one-off setup helpers
tests/               correctness, integration, stress
AdapterSlots/        the serving system this builds on (submodule, pinned)
```

## The kernel and the engine

`csrc/` is a standalone CUDA implementation of the grouped apply, and most of what is in it is
there because the obvious version was measured and lost.

The kernels are specialised per rank (8, 16, 32, 64) so the rank is a compile-time constant and
the inner loops unroll. Each launch is one block per segment and token-chunk, on a grid whose
shape never changes, because a serving step has to be captured into a CUDA graph and a grid that
depends on the batch cannot be. Blocks with nothing to do exit. A persistent single-block loop
over segments is the design the description invites, and it is far slower, because it leaves
almost every SM on the GPU idle. Weights stream through a double-buffered `cp.async` pipeline so
the next segment's tiles load while the current one computes. Segment boundaries are built once
per decode step on the device, not rebuilt inside each of the ~80 LoRA layers that need them.
`memory/` holds the adapter store and a staging buffer, so an adapter's weights can be moved to
the GPU ahead of the step that needs them rather than in it.

`lora_warp_pipe/` is the Python side of it: the engine that drives the extension, and the two
modules that wire it into a running server. "Relation to AdapterSlots" below walks through it.

The end-to-end benchmarks do not depend on the CUDA extension. They run the grouped apply on
stock vLLM BGMV Triton kernels, which is the portability claim: the primitive is not tied to
this serving stack, and the speculative results do not rest on our kernel being installed.

## Relation to AdapterSlots

AdapterSlots++ is [AdapterSlots](https://github.com/vac014/AdapterSlots) plus two things this
repo adds: the LoRA-WarpPipe apply stack, and the speculative loop that stack pays for.

[AdapterSlots](https://github.com/vac014/AdapterSlots) is the multi-LoRA serving system
underneath: it owns admission and dispatch, warp-aligned batching, the alignment buffer, the
Whittle-scored scheduler and arrival-rate estimation, and the LoRA path inside vLLM. It is a
complete server on its own, and it is the non-speculative tier in every table here. That tier is
this serving path with speculation switched off, so every speedup in `results/` is a speedup over
it, and the ratio isolates the thing under test because the two tiers differ in nothing else.

LoRA-WarpPipe is the layer this repo puts on top, and it is a serving component before it is a
speculation one. `WarpPipeEngine` (`lora_warp_pipe/engine.py`) is a thin, honest wrapper over the
CUDA extension and it holds the whole lifecycle: `register_adapter` / `evict_adapter` keep the
adapter store on the GPU; `build_seg_bounds` runs once per decode step to work out which token
belongs to which adapter, ahead of the ~80 layer calls that then reuse it; `shrink` and `expand`
are the grouped apply itself; `prefetch_adapter` / `is_prefetch_ready` / `release_prefetch` move
an adapter's weights into a staging slot on a separate stream, so a cold adapter can be landing
while the current step is still computing.

Two modules connect that engine to the server. `punica_wrapper.py` subclasses AdapterSlots'
`FusedPunicaWrapper`, which is what puts the grouped apply on vLLM's real
`PunicaWrapper.add_lora()` path instead of a benchmark harness; when the kernel cannot handle a
call it says so and the caller falls back to the stock path, so the failure mode is a slower step
and never a wrong one. `scheduler_bridge.py` goes the other way: it carries the scheduler's own
state (Whittle scores, arrival-rate estimates, GWAR, which adapters are hot) into the kernel's
metadata once per tick, so the kernel can act on which tenants matter rather than treating a
batch as an undifferentiated pile of tokens. It duck-types those objects instead of importing
them, which is why the bridge is testable with no server in sight.

Speculation sits on top of all of that, and it is the part that needs the grouped apply to be
flat in the number of adapters: a per-tenant draft is only affordable if drafting K tenants
costs about what drafting one does.

AdapterSlots sits at `AdapterSlots/` as a submodule, pinned to the exact commit these results
were produced against, which is tagged `v1.0` there (`27ecf97`). The pin is a commit and not a
branch on purpose: AdapterSlots keeps moving, and a submodule that followed its `main` would
change what this code is built on without anyone touching this repo. A recursive clone gets that
commit and nothing newer:

```bash
git clone --recurse-submodules https://github.com/vac014/AdapterSlotsPlusPlus.git
# already cloned without it:
git submodule update --init
```

Install it for the live-server path, from the pinned checkout:

```bash
pip install -e AdapterSlots
```

`pip install -e ".[serving]"` does the same thing from the network, pinned to the same commit,
if you would rather not use the submodule.

Beyond that, the two are deliberately separable:

| you want | you need |
| --- | --- |
| the benchmarks, the drafts, the serving loops | this repo only |
| the CUDA extension and its tests | this repo only (`python setup.py build_ext --inplace`) |
| WarpPipe inside a live vLLM server | this repo **and** AdapterSlots |

`lora_warp_pipe/punica_wrapper.py` is the single import site: it subclasses AdapterSlots'
`FusedPunicaWrapper` to hook the grouped apply into vLLM's `PunicaWrapper.add_lora()` path.
Nothing else in the package imports it, and no benchmark here does, so
`pip install -r requirements.txt` is enough for everything else in this README.

## Setup

Requires an NVIDIA GPU (Ampere or newer) and CUDA 12.x.

```bash
git clone --recurse-submodules https://github.com/vac014/AdapterSlotsPlusPlus.git
cd AdapterSlotsPlusPlus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the results were produced with. Any vLLM build that
exposes `vllm.lora.ops.bgmv_shrink` and `bgmv_expand` will do; only those two Triton kernels
are used.

To build the CUDA extension (optional, and only needed for `benchmarks/kernel/bench_vs_punica.py`
and the tests under `tests/`):

```bash
python setup.py build_ext --inplace
```

`setup.py` targets `sm_86`. On other architectures set the gencode flag to match, for example
`arch=compute_80,code=sm_80` for A100.

## Models

Nothing is vendored. Every asset resolves through `benchmarks/e2e/paths.py` as an env-var
override, then a snapshot already in the Hugging Face cache, then the plain repo id, so a
first run just downloads what it needs.

| role | repo |
| --- | --- |
| target base | `huggyllama/llama-7b` (and `huggyllama/llama-13b`) |
| draft base | `JackFram/llama-160m` |
| target adapter | `tloen/alpaca-lora-7b` (and `chansung/alpaca-lora-13b`) |
| co-resident tenants | `project-baize/baize-lora-7B`, `serpdotai/llama-hh-lora-7B` |

To stage everything ahead of time (about 27 GB, or 52 GB with `--13b` for the scale runs):

```bash
bash scripts/download_models.sh          # 7B set, enough for most benchmarks
bash scripts/download_models.sh --13b    # adds llama-13b and its adapter
```

ShareGPT is the one asset that is not on the Hub, and only `throughput_tiers.py` and
`generalization.py` touch it, only under `--dataset sharegpt`. Point `WARPPIPE_SHAREGPT` at a
`ShareGPT.json` holding a list of `{"conversations": [{"from": "human", "value": ...}, ...]}`.

Overrides, if the models live somewhere the cache will not find them: `WARPPIPE_LLAMA7B`,
`WARPPIPE_LLAMA13B`, `WARPPIPE_LLAMA160M`, `WARPPIPE_ALPACA_LORA`, `WARPPIPE_ALPACA_LORA_13B`,
`WARPPIPE_SHAREGPT`.

## Running

```bash
export CUDA_VISIBLE_DEVICES=0
python benchmarks/e2e/throughput_tiers.py --dataset gsm8k
```

Results are written to `results/` as timestamped JSON. `run_all.sh` runs every axis; see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for what each benchmark measures and the exact
invocation behind each figure and table.

`results/` is not empty: it ships the runs behind the paper. Each file is named for the script
that regenerates it and keeps the timestamp of the run that produced it, so a fresh run writes a
sibling rather than overwriting one, and the two can be compared directly.

No hardware constant is baked in anywhere. Latencies are graph-captured on whatever device
the run lands on, and acceptance depends only on the pinned weights, so moving GPUs
re-measures rather than re-scales.

## Contributing

Issues and pull requests are welcome. Useful things to know before you send one:

- `ruff check .` and `python -m compileall benchmarks lora_warp_pipe tests` are what CI runs.
  Both are CPU-only, so you can run them without a GPU. Ruff is configured for pyflakes and
  syntax errors rather than formatting: the gate is for real defects.
- `bash scripts/run_tests.sh` runs the extension's tests. They are standalone scripts rather
  than pytest modules, and every one of them needs a CUDA device and the built extension
  (`python setup.py build_ext --inplace`), so CI cannot run them.
- `results/` holds the reference runs. Each file is named for the script that regenerates it and
  keeps the timestamp of the run that produced it. A run of your own writes a fresh timestamped
  file there; report its numbers in the pull request rather than committing it, and say which GPU
  produced them, since nothing here is calibrated to a particular device. Plots and model weights
  do not belong in the tree at all.
- If you change a serving loop, keep its correctness assertion. Every loop here checks its
  committed tokens against plain decode before it reports a number, and a speedup from a loop
  that skipped that check is not a speedup. `benchmarks/e2e/lossless_check.py` explains what
  counts as a real divergence and what is an fp16 argmax tie.

## License

Apache-2.0. See [LICENSE](LICENSE).
