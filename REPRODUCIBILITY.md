# Reproducibility

Setup, models and env vars are in [README.md](README.md). This file says what each benchmark
measures and how to invoke it.

Everything re-measures on the device it runs on. Latencies are CUDA-graph-captured at run time
and acceptance depends only on the pinned weights, so a different GPU changes the absolute
numbers and not the protocol. Nothing is read back from a stored constant.

```bash
export CUDA_VISIBLE_DEVICES=0
bash run_all.sh                 # every axis
bash run_all.sh tiers deploy    # selected axes
```

Each script writes a timestamped JSON into `results/`. Defaults reproduce the reported
configuration, so the flags below are the ones worth varying, not ones you must pass.

## What is already in results/

`results/` ships the runs behind the paper rather than an empty directory. Each file is named
for the script that regenerates it and keeps the timestamp of the run that produced it, so
`results/throughput_tiers_gsm8k_<stamp>.json` came out of `benchmarks/e2e/throughput_tiers.py
--dataset gsm8k` and a fresh run writes a sibling with a new stamp rather than overwriting it.
A run that no script in this repo can regenerate is not shipped.

Every file names its device. Nothing here is calibrated to a particular GPU, so a run on other
hardware reproduces the protocol and the ordering of the tiers, not the absolute numbers.

The three tiers keep the same names in code, in JSON and in the tables:

| key in JSON | tier |
| --- | --- |
| `no_spec` | AdapterSlots: the serving path with speculation off, and the baseline every speedup is measured against |
| `shared` | AdapterSlots++ with one shared, adapter-blind draft |
| `warp` | AdapterSlots++ / WarpPipe: one faithful draft-LoRA per tenant, applied in a single grouped pass |

## Protocol

Two rules hold across every acceptance number here.

**Acceptance is exact.** It is measured by a KV-cached speculative loop whose committed tokens
are asserted equal to the target's plain greedy output before any number is reported, so it is
the true acceptance rate against the target's own path, not an approximation of it.

**Selection never touches the eval set.** Held-out data is split into disjoint validation and
eval. The distillation recipe trains one draft per seed, snapshots at several step counts, and
picks the checkpoint with the best validation acceptance. The shared draft sits in that
candidate set, so a faithful draft that fails to beat it on validation is not deployed. Every
reported number is on the eval split, which selects nothing.

## Core results

| what | script |
| --- | --- |
| three tiers across the six workloads: acceptance and throughput | `benchmarks/e2e/throughput_tiers.py` |
| composed throughput checked against a real wall-clock loop | `benchmarks/e2e/wallclock_validate.py` |
| distinct real tenants: routing matrix, per-tenant acceptance, flat draft cost | `benchmarks/e2e/multitenant_matrix.py` |
| K co-resident tenants through one serving loop; draft cost against K | `benchmarks/e2e/serve_multitenant.py` |
| heterogeneous traffic: every row a different tenant and task (KV + graphs, gamma swept) | `benchmarks/e2e/serve_mixed_traffic.py` |
| where the faithful-over-shared lift comes from, across workloads | `benchmarks/e2e/generalization.py` |
| the validation floor that keeps a faithful draft from regressing | `benchmarks/e2e/faithful_floor.py` |
| why one seed of the same config reads twice another; the fix | `benchmarks/e2e/variance_profile.py` |

```bash
# one workload, three tiers, swept over resident adapters
python benchmarks/e2e/throughput_tiers.py --dataset gsm8k --modules qkvo \
  --drf_r 32 --drf_steps 900 --n_train 300 --seeds 0 1 2 \
  --ckpt_steps 150 300 450 600 900 --n_val 20 --n_eval 24 \
  --gamma 4 --B 32 --K 1 8 32 64 128 256

python benchmarks/e2e/multitenant_matrix.py
python benchmarks/e2e/generalization.py --datasets sharegpt dolly samsum gsm8k mbpp
```

ShareGPT uses `--n_train 400`; the other five use 300. MBPP has only 374 training rows, so keep
`n_train + n_val + n_eval` under that.

## Serving path

The three serving loops differ in what they pay for, and the differences are the point.

| loop | KV cache | CUDA graph | script |
| --- | --- | --- | --- |
| full recompute, un-captured (cost sweeps only) | no | no | `serve_multitenant.py`, `serve_to_completion.py` |
| cached, un-captured (ablation) | yes | no | `serve_kv_eager.py` |
| cached and captured (deployment) | yes | yes | `serve_kv_graph.py`, `serve_mixed_traffic.py` |

**A no-KV loop is not a conservative approximation of a cached one, and its ratios do not
transfer.** Recomputing the prefix makes one decoded token cost about what verifying `gamma+1`
of them costs, which flatters every speculative tier. Turn the cache on and the target step
collapses; an *un-captured* 160M draft step then costs more than the target decode it was
supposed to replace, and speculation goes underwater. Capture the draft and it costs what its
parameter count says it should. The two effects have opposite signs and comparable magnitude,
so a loop with neither lands somewhere unrelated to a loop with both, including on the wrong
side of 1x. The no-KV loops are kept only for the cost sweeps (`t_sgmv` vs `t_per_adapter`
against K), where every tier pays the same target and the target therefore cancels.

End-to-end serving numbers come from the cached, captured loops. `serve_kv_eager.py`,
`serve_kv_graph.py` and `serve_mixed_traffic.py` all assert their committed tokens match plain
decode before timing anything, so the difference between them is systems and not approximation.
`serve_mixed_traffic.py` uses `lossless_check.py`, which is tie-aware: fp16 batched GEMMs are not
invariant to batch composition, so when the target's top two logits sit within fp16 resolution
the winner can flip. A mismatch is not waved through: the divergent prefix is re-run through
the target alone and the top-2 logit gap is reported. A gap at the noise floor is a tie; a real
gap is a bug.

`gamma` is swept rather than assumed. A wider `gamma` buys more tokens per accepted run but pays
for a wider verify, whose extra token positions are compute rather than bandwidth at serving
batch sizes, so the optimum is interior.

```bash
python benchmarks/e2e/serve_kv_eager.py
python benchmarks/e2e/serve_kv_graph.py
python benchmarks/e2e/serve_to_completion.py
```

Throughput on the run-to-completion loops is goodput: the tokens that were asked for over the
wall clock. A speculative row overshoots its token budget on the iteration it finishes, and rows
that are already done keep decoding until the slowest one lands, so counting emitted tokens
would credit speculation for work nobody requested.

## When speculation pays

As the batch fills, the target verify becomes compute-bound and the bar a draft must clear to be
worth running rises. `batch_scaling.py` measures the latency curves against batch size, and
`load_gate.py` evaluates the gate condition `tpv > t_iter / t_decode` for the shared and the
faithful draft at each batch size, which is where each one ages out.

```bash
python benchmarks/e2e/batch_scaling.py
python benchmarks/e2e/load_gate.py --dataset gsm8k
```

## Scale

`across_models_13b.py` repeats the study against a 13B target with the same 160M draft. A fp16
13B and a trainable draft do not fit on one 48 GB card, so the driver is phased and never holds
both: generate the corpus with the target resident, train and snapshot the drafts with it
absent, reload it once to score every snapshot. Same exact-KV metric and same fp16 weights as at
7B, so the two are comparable.

`apply_kernel_13b.py` times the multi-adapter apply kernel every multi-LoRA system runs (BGMV,
the backend Punica and vLLM-V0 ship, against segmented SGMV) at real 13B geometry, and reports
it against the measured 13B decode step.

```bash
python benchmarks/e2e/across_models_13b.py --dataset gsm8k
python benchmarks/e2e/apply_kernel_13b.py --measure_base
```

Pass `--measure_base` to re-time the 13B decode step on your GPU instead of trusting `--base_ms`.
`apply_kernel_13b.py` calls the punica kernels directly, so it needs punica installed
(`benchmarks/sota/versions.txt`); nothing else here does.

## Ablations

| question | script |
| --- | --- |
| is the acceptance win a sampling artifact? bootstrap CI on a larger eval set | `accept_ci.py` |
| how big should the draft be? acceptance against latency | `ablate_draft_size.py` |
| the latency half of that, at fixed acceptance | `ablate_draft_latency.py` |
| when does a per-tenant draft pay for its own distillation? | `distill_amortize.py` |
| what a per-tenant EAGLE or Medusa module would cost instead | `eagle_medusa_cost.py` |

## The apply kernel on its own

| what | script |
| --- | --- |
| grouped apply against per-adapter serialization, with an fp32 correctness check | `benchmarks/kernel/bench_draft_sgmv.py` |
| the same primitive inside a real draft-model step, on stock vLLM BGMV kernels | `benchmarks/kernel/bench_draft_lens.py` |
| CUDA-graph capture of a LoRA draft step, replayed with fresh adapter indices | `benchmarks/kernel/bench_graph_capture.py` |
| the CUDA extension against Punica | `benchmarks/kernel/bench_vs_punica.py` |
| per-op cost breakdown of one draft step | `benchmarks/kernel/profile_breakdown.py` |

The first three need only vLLM's Triton kernels. The last two need the CUDA extension built
(`python setup.py build_ext --inplace`).

## Baselines

The comparison systems are not vendored: each needs its own environment, because their pinned
dependencies conflict. `benchmarks/sota/versions.txt` lists the version and install line
for each, and `benchmarks/sota/drivers/` holds the two throughput drivers that are not part of
an upstream harness.

Every baseline must be run on the same GPU as this system, with the same model and the same
adapter count. Numbers from different machines are not comparable.

- **vLLM 0.6.3** and **vLLM V1 0.24.0**: served with `--enable-lora`; V1 additionally runs
  n-gram speculation via `--speculative-config`. V1 needs `VLLM_USE_FLASHINFER_SAMPLER=0`, since
  the flashinfer sampler's JIT CUB build fails on nvcc 12. Drive V1 with
  `benchmarks/sota/drivers/vllm_v1_throughput_driver.py`.
- **SGLang 0.5.14**: `--lora-paths` with `--max-loras-per-batch`; the speculative run adds
  `--speculative-algorithm NGRAM`. Benchmark with `python -m sglang.bench_serving`.
- **Punica**: `bench/bench_textgen_lora.py` from the upstream repo. This is a decode
  microbenchmark, not a server. Punica's BGMV backend is the one vLLM-V0 ships, so the serving
  form of Punica is vLLM-V0.
- **S-LoRA** and **HF-PEFT**: S-LoRA's `launch_server.py` plus `run_exp.py` on a synthetic
  Poisson trace; PEFT needs a period-correct env (torch 2.1.2, transformers 4.31.0, peft 0.4.0).
- **dLoRA**: builds against nvcc 12 only after overlaying pybind11 2.13.6 into the torch
  headers; run the flagship `--exec-type 3` server and drive it with
  `benchmarks/sota/drivers/dlora_throughput_driver.py`.

## Tests

The tests are standalone programs rather than pytest modules, and every one of them needs a CUDA
device and the compiled extension:

```bash
python setup.py build_ext --inplace
bash scripts/run_tests.sh                    # unit, correctness, integration, stress
bash scripts/run_tests.sh unit correctness   # selected tiers
```

The two checks that need no GPU are what CI runs, and they are worth running before a pull
request:

```bash
ruff check .
python -m compileall benchmarks lora_warp_pipe tests
```
