"""StagingBufferPool slot allocation, eviction safety and ready-flag correctness,
through the Python-exposed prefetch_adapter() / is_prefetch_ready() /
release_prefetch() surface.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
ALPHA_SCALE = 16.0 / RANK
WP_MAX_STAGING = 8  # csrc/bridge/warp_pipe_metadata.h


def make_engine(k):
    engine = WarpPipeEngine()
    a = torch.randn(k, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn(k, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    for i in range(k):
        engine.register_adapter(i, a[i].contiguous(), b[i].contiguous(), ALPHA_SCALE)
    return engine


def wait_ready(engine, adapter_id, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if engine.is_prefetch_ready(adapter_id):
            return True
        time.sleep(0.001)
    return False


def test_acquire_and_ready():
    engine = make_engine(1)
    assert not engine.is_prefetch_ready(0), "ready before any prefetch was ever triggered"
    slot = engine.prefetch_adapter(0)
    assert slot >= 0, "acquire_slot failed with a free pool"
    assert wait_ready(engine, 0), "prefetch never became ready"
    print("PASS: acquire + ready-flag transitions false -> true after a real copy")


def test_pool_exhaustion_is_safe_not_a_crash():
    """Spec's own design: only WP_MAX_STAGING slots exist. Acquiring one more
    than that must return -1 (a missed optimization), never crash or
    corrupt another adapter's slot."""
    engine = make_engine(WP_MAX_STAGING + 2)
    slots = [engine.prefetch_adapter(i) for i in range(WP_MAX_STAGING + 2)]
    num_acquired = sum(1 for s in slots if s >= 0)
    num_failed = sum(1 for s in slots if s == -1)
    assert num_acquired == WP_MAX_STAGING, f"expected exactly {WP_MAX_STAGING} slots acquired, got {num_acquired}"
    assert num_failed == 2, f"expected exactly 2 failed acquisitions, got {num_failed}"
    for i in range(WP_MAX_STAGING):
        assert wait_ready(engine, i), f"adapter {i} (within pool capacity) never became ready"
    for i in range(WP_MAX_STAGING, WP_MAX_STAGING + 2):
        assert not engine.is_prefetch_ready(i), f"adapter {i} (beyond pool capacity) falsely ready"
    print(f"PASS: pool exhaustion at K={WP_MAX_STAGING + 2} -- exactly {WP_MAX_STAGING} succeed, "
          f"2 safely report no slot, no crash")


def test_release_and_reacquire():
    engine = make_engine(2)
    slot0 = engine.prefetch_adapter(0)
    assert wait_ready(engine, 0)
    engine.release_prefetch(0)
    assert not engine.is_prefetch_ready(0), "released slot still reports ready"
    slot1 = engine.prefetch_adapter(1)
    assert slot1 == slot0, "released slot was not reused by the next acquire (pool leak)"
    assert wait_ready(engine, 1)
    print("PASS: release_prefetch frees the slot for reuse by a different adapter")


def test_idempotent_double_prefetch():
    """Calling prefetch_adapter twice for the same adapter before release
    must reuse the same slot, not silently leak a second one."""
    engine = make_engine(1)
    slot_a = engine.prefetch_adapter(0)
    slot_b = engine.prefetch_adapter(0)
    assert slot_a == slot_b, f"double prefetch for the same adapter allocated different slots: {slot_a} vs {slot_b}"
    assert wait_ready(engine, 0)
    print("PASS: re-prefetching the same adapter reuses its existing slot")


if __name__ == "__main__":
    test_acquire_and_ready()
    test_pool_exhaustion_is_safe_not_a_crash()
    test_release_and_reacquire()
    test_idempotent_double_prefetch()
    print("ALL PASS: test_staging_buffer.py")
