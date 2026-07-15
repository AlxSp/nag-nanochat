"""
Fail-fast preflight for H100/Hopper training runs.

This intentionally checks the exact nanochat FA3 selector instead of only checking
GPU capability, because dependency/cache issues can make FA3 unavailable even on
H100.
"""

import sys
import torch

from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
from nanochat.flash_attention import HAS_FA3, USE_FA3


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 1

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}")
    print(f"CUDA capability: sm{capability[0]}{capability[1]}")
    print(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    print(f"HAS_FA3={HAS_FA3} USE_FA3={USE_FA3}")

    if not USE_FA3:
        print("FA3 is not active. Refusing to launch the full H100 run.")
        return 1

    # Exercise the imported kernel on a tiny bf16 batch.
    from nanochat.flash_attention import flash_attn

    q = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
    torch.cuda.synchronize()
    if y.shape != q.shape or y.dtype != q.dtype or not torch.isfinite(y).all():
        print(f"FA3 smoke failed: shape={tuple(y.shape)} dtype={y.dtype} finite={torch.isfinite(y).all().item()}")
        return 1
    print("FA3 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

