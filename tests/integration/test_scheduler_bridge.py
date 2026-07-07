"""Simulates a scheduler tick (write_bridge with realistic Whittle/Erlang-
shaped values) and checks GPU-side readback via a real kernel dereferencing
bridge_ptr (read_bridge_via_kernel()), not just the host-pinned-memory check
test_metadata_bridge.py already covers.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402


def main():
    engine = WarpPipeEngine()
    n = 6
    whittle = [round(0.05 * i + 0.1, 4) for i in range(n)]
    t_rem = [float(i) for i in range(n)]
    gwar = [1.0 - 0.1 * i for i in range(n)]
    lam = [5.0 * i for i in range(n)]
    burst = [i % 2 for i in range(n)]
    promo = [1 - (i % 2) for i in range(n)]
    hot = [1 if i < 2 else 0 for i in range(n)]
    tile = [i % 3 for i in range(n)]

    engine.write_bridge(
        torch.tensor(whittle, dtype=torch.float32),
        torch.tensor(t_rem, dtype=torch.float32),
        torch.tensor(gwar, dtype=torch.float32),
        torch.tensor(lam, dtype=torch.float32),
        torch.tensor(burst, dtype=torch.uint8),
        torch.tensor(promo, dtype=torch.uint8),
        torch.tensor(hot, dtype=torch.uint8),
        torch.tensor(tile, dtype=torch.uint8),
        n,
        4,
        17,
        0.5,
    )

    result = engine.read_bridge_via_kernel()
    gpu_whittle = result[:n]
    gpu_n, gpu_segs, gpu_step = result[n], result[n + 1], result[n + 2]

    max_diff = max(abs(a - b) for a, b in zip(gpu_whittle, whittle))
    print(f"GPU-side whittle_scores readback: {gpu_whittle}")
    print(f"host-written whittle_scores:      {whittle}")
    print(f"max diff: {max_diff}")
    assert max_diff < 1e-6, "GPU kernel read different whittle_scores than were written -- bridge_ptr is broken"
    assert int(gpu_n) == n, f"num_active_adapters mismatch: kernel saw {gpu_n}, wrote {n}"
    assert int(gpu_segs) == 4, f"num_segments mismatch: kernel saw {gpu_segs}"
    assert int(gpu_step) == 17, f"step_id mismatch: kernel saw {gpu_step}"
    print("PASS: a real kernel dereferencing WarpPipeMetadata.bridge_ptr reads back exactly what "
          "write_bridge() wrote -- GPU-side bridge access confirmed working, not just host-pinned-memory access")


if __name__ == "__main__":
    main()
