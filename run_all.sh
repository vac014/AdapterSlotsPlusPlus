#!/usr/bin/env bash
# Reproduce every axis on the current GPU. See REPRODUCIBILITY.md for what each one
# measures and how to run a single benchmark.
#
#   export CUDA_VISIBLE_DEVICES=0
#   bash run_all.sh [axis ...]
#
# axes: tiers multitenant mechanism floor variance serving deploy scale ablations
set -euo pipefail

PY="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E="$HERE/benchmarks/e2e"
mkdir -p "$HERE/results"

AXES=("${@:-tiers multitenant mechanism floor variance serving deploy scale ablations}")
run() { echo "=== $* ==="; "$PY" "$@"; }

for axis in ${AXES[*]}; do
case "$axis" in
  tiers)
    # ShareGPT is the one workload trained on 400 examples; MBPP has only 374 train
    # rows, so n_train + n_val + n_eval must stay under that.
    for ds in gsm8k alpaca dolly mbpp samsum sharegpt; do
      NT=300; [ "$ds" = sharegpt ] && NT=400
      run "$E2E/throughput_tiers.py" --dataset "$ds" --modules qkvo \
        --drf_r 32 --drf_steps 900 --n_train "$NT" --gen_tokens 96 \
        --seeds 0 1 2 --ckpt_steps 150 300 450 600 900 \
        --n_val 20 --n_eval 24 --eval_tokens 64 --gamma 4 --B 32 --K 1 8 32
    done
    run "$E2E/wallclock_validate.py" --dataset gsm8k
    run "$E2E/wallclock_validate.py" --dataset mbpp ;;

  multitenant)
    run "$E2E/multitenant_matrix.py"
    run "$E2E/serve_multitenant.py" ;;

  mechanism)
    run "$E2E/generalization.py" --datasets sharegpt dolly samsum gsm8k mbpp ;;

  floor)
    run "$E2E/faithful_floor.py" ;;

  variance)
    run "$E2E/variance_profile.py" ;;

  serving)
    run "$E2E/batch_scaling.py"
    run "$E2E/load_gate.py" --dataset gsm8k
    run "$E2E/serve_to_completion.py" ;;

  deploy)
    run "$E2E/serve_kv_eager.py"
    run "$E2E/serve_kv_graph.py"
    run "$E2E/serve_mixed_traffic.py" ;;

  scale)
    run "$E2E/across_models_13b.py" --dataset gsm8k
    run "$E2E/across_models_13b.py" --dataset sharegpt
    run "$E2E/apply_kernel_13b.py" ;;

  ablations)
    run "$E2E/accept_ci.py"
    run "$E2E/ablate_draft_size.py"
    run "$E2E/ablate_draft_latency.py"
    run "$E2E/distill_amortize.py"
    run "$E2E/eagle_medusa_cost.py" ;;

  *) echo "unknown axis: $axis" >&2; exit 1 ;;
esac
done
echo "done -> $HERE/results"
