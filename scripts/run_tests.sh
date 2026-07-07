#!/usr/bin/env bash
# Run the extension's test scripts. Each is a standalone program, not a pytest module, and
# each needs a CUDA device and the compiled extension:
#
#   python setup.py build_ext --inplace
#   bash scripts/run_tests.sh [unit|correctness|integration|stress ...]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
TIERS=("${@:-unit correctness integration stress}")

cd "$HERE"
# torch first: the extension links against libtorch, so importing it on its own fails to find
# libc10.so even when it is built. Every test imports torch before it, as this does.
"$PY" -c "import torch, warp_pipe_ext" 2>/dev/null || {
  echo "warp_pipe_ext not importable. Build it first:" >&2
  echo "  python setup.py build_ext --inplace" >&2
  exit 1
}

pass=0; fail=0; failed=()
for tier in ${TIERS[*]}; do
  for t in "tests/$tier"/test_*.py; do
    [ -e "$t" ] || continue
    printf '%-58s' "$t"
    if out=$("$PY" "$t" 2>&1); then
      echo "ok"; pass=$((pass + 1))
    else
      echo "FAIL"; fail=$((fail + 1)); failed+=("$t")
      echo "$out" | tail -15 | sed 's/^/    /'
    fi
  done
done

echo
echo "$pass passed, $fail failed"
for t in "${failed[@]:-}"; do [ -n "$t" ] && echo "  failed: $t"; done
[ "$fail" -eq 0 ]
