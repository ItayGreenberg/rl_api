import torch
# ──────────────────────────────────────────────────────────────
#  Fast fail-safe: raise if ANY tensor is non-finite
# ──────────────────────────────────────────────────────────────
def assert_finite(name: str, *tensors):
    """
    Fail-fast check that every element of every tensor is finite.
    Raises ValueError with `name` if a NaN or Inf is found.
    """
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            print(f"received non-tensor in [{name}]")
            break
        if not torch.isfinite(t).all():
            print(f"non-finite detected in [{name}]")
