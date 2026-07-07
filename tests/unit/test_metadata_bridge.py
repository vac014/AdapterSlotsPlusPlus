"""Bridge write/read round-trip correctness, through read_bridge_debug(). Confirms
SchedulerKernelBridgeHost::write() actually lands every field at the byte
offsets warp_pipe_metadata.h's struct layout implies, not just "doesn't
crash" -- the project's own history (AdapterRun's docs-vs-compiler size
mismatch, SegmentDescriptor's claimed-vs-real size) is exactly the class of
bug this catches.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

N = 5


def main():
    engine = WarpPipeEngine()
    whittle = [0.1, 0.2, 0.3, 0.4, 0.5]
    t_rem = [1.0, 2.0, 3.0, 4.0, 5.0]
    gwar = [0.9, 0.8, 0.7, 0.6, 0.5]
    lam = [10.0, 20.0, 30.0, 40.0, 50.0]
    burst = [1, 0, 1, 0, 1]
    promo = [0, 1, 0, 1, 0]
    hot = [1, 1, 0, 0, 0]
    tile = [2, 1, 0, 2, 1]

    import torch

    engine.write_bridge(
        torch.tensor(whittle, dtype=torch.float32),
        torch.tensor(t_rem, dtype=torch.float32),
        torch.tensor(gwar, dtype=torch.float32),
        torch.tensor(lam, dtype=torch.float32),
        torch.tensor(burst, dtype=torch.uint8),
        torch.tensor(promo, dtype=torch.uint8),
        torch.tensor(hot, dtype=torch.uint8),
        torch.tensor(tile, dtype=torch.uint8),
        N,
        3,
        42,
        0.75,
    )

    def close(a, b):
        return len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))

    d = engine.read_bridge_debug()
    # fp32 values round-trip through pybind's float<->Python-double
    # conversion, which prints with double-precision noise on the last bits
    # -- compare numerically, not with ==.
    assert close(d["whittle_scores"], whittle), f"whittle_scores mismatch: {d['whittle_scores']}"
    assert close(d["t_remaining_ms"], t_rem), f"t_remaining_ms mismatch: {d['t_remaining_ms']}"
    assert close(d["gwar_pred_next3"], gwar), f"gwar_pred_next3 mismatch: {d['gwar_pred_next3']}"
    assert close(d["lambda_hat"], lam), f"lambda_hat mismatch: {d['lambda_hat']}"
    assert list(d["burst_active"]) == burst, f"burst_active mismatch: {d['burst_active']}"
    assert list(d["promo_eligible"]) == promo, f"promo_eligible mismatch: {d['promo_eligible']}"
    assert list(d["is_hot"]) == hot, f"is_hot mismatch: {d['is_hot']}"
    assert list(d["tile_size_code"]) == tile, f"tile_size_code mismatch: {d['tile_size_code']}"
    assert d["num_active_adapters"] == N
    assert d["num_segments"] == 3
    assert d["step_id"] == 42
    assert abs(d["global_load"] - 0.75) < 1e-6

    print("PASS: SchedulerKernelBridge write/read round-trip, all fields exact")


if __name__ == "__main__":
    main()
