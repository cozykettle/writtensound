"""Device-safe torch dtypes for the Qwen3 engines.

The HF checkpoints declare bfloat16.  CPU convolution kernels are not reliable
for bf16 on every host, and Turing (sm_75, e.g. T600) cannot execute bf16, so
the dtype follows the selected device instead of the checkpoint config.
"""

from __future__ import annotations

from typing import Any


def inference_dtype(torch: Any, device: str) -> Any:
    """Return a dtype the selected device can execute for Qwen3 weights."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def cast_floating(inputs: Any, torch: Any, dtype: Any) -> dict[str, Any]:
    """Cast floating tensors only; token ids and masks keep their integer dtype."""
    return {
        key: value.to(dtype) if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in inputs.items()
    }
