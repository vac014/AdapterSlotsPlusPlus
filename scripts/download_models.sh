#!/usr/bin/env bash
# Fetch every weight the benchmarks need into the Hugging Face cache. Nothing is vendored
# and nothing is copied into the repo: benchmarks/e2e/paths.py resolves each asset from an
# env-var override, then the cache, then the plain repo id.
#
#   bash scripts/download_models.sh          # 7B set (most benchmarks)
#   bash scripts/download_models.sh --13b    # also the 13B scale runs
#
# Roughly 27 GB for the 7B set, 52 GB with --13b.
set -euo pipefail

command -v huggingface-cli >/dev/null || {
  echo "huggingface-cli not found: pip install -r requirements.txt" >&2; exit 1; }

MODELS=(
  huggyllama/llama-7b            # target base
  JackFram/llama-160m            # draft base
  tloen/alpaca-lora-7b           # target adapter
  project-baize/baize-lora-7B    # co-resident tenant
  serpdotai/llama-hh-lora-7B     # co-resident tenant
)
[ "${1:-}" = "--13b" ] && MODELS+=(huggyllama/llama-13b chansung/alpaca-lora-13b)

for m in "${MODELS[@]}"; do
  echo "==> $m"
  huggingface-cli download "$m"
done

cat <<'MSG'

Done. One asset is not on the Hub: ShareGPT. Only throughput_tiers.py and
generalization.py use it, and only when --dataset sharegpt is passed.

  export WARPPIPE_SHAREGPT=/path/to/ShareGPT.json

The file is a JSON list of {"conversations": [{"from": "human", "value": ...}, ...]}.
MSG
