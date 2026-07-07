import torch


def reference_bgmv_shrink(X: torch.Tensor, A: torch.Tensor, alpha_scale: float) -> torch.Tensor:
    """X[T, d] @ A[rank, d].T * alpha_scale -> H[T, rank].

    The alpha/rank scale is applied exactly ONCE, here in shrink. Applying it in
    both shrink and expand squares it, which is wrong against vLLM semantics (vllm/lora/ops/bgmv_shrink.py takes a `scaling` arg;
    bgmv_expand.py has no scale arg at all, only add_inputs). Real LoRA math
    applies the alpha/rank scale exactly once, here in shrink.
    """
    return (X.float() @ A.float().T) * alpha_scale


def reference_bgmv_expand(H: torch.Tensor, B: torch.Tensor, Y_base: torch.Tensor) -> torch.Tensor:
    """Y_base + H[T, rank] @ B[out, rank].T. No alpha_scale here -- see
    reference_bgmv_shrink's docstring; H already carries the scale."""
    return Y_base.float() + (H.float() @ B.float().T)
