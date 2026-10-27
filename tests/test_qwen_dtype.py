"""Device dtype selection for the Qwen3 engines (no torch required)."""

from __future__ import annotations

from typing import Any

from backend.engines.qwen3.dtype import cast_floating, inference_dtype


class Cuda:
    def __init__(self, bf16: bool) -> None:
        self.bf16 = bf16

    def is_bf16_supported(self) -> bool:
        return self.bf16


class Tensor:
    def __init__(self, dtype: str, floating: bool) -> None:
        self.dtype = dtype
        self.floating = floating

    def to(self, dtype: str) -> Tensor:
        return Tensor(dtype, self.floating)

    def is_floating_point(self) -> bool:
        return self.floating


class Torch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, bf16: bool = False) -> None:
        self.cuda = Cuda(bf16)

    def is_tensor(self, value: Any) -> bool:
        return isinstance(value, Tensor)


def test_cpu_uses_float32() -> None:
    assert inference_dtype(Torch(), "cpu") == "float32"


def test_cuda_uses_bfloat16_when_supported() -> None:
    assert inference_dtype(Torch(bf16=True), "cuda") == "bfloat16"


def test_cuda_falls_back_to_float16_on_turing() -> None:
    assert inference_dtype(Torch(bf16=False), "cuda") == "float16"


def test_cast_floating_leaves_integer_tensors_alone() -> None:
    inputs: dict[str, Tensor] = {
        "input_features": Tensor("float32", floating=True),
        "input_ids": Tensor("int64", floating=False),
        "input_features_mask": Tensor("int32", floating=False),
    }
    casted = cast_floating(inputs, Torch(), "float32")
    assert casted["input_features"].dtype == "float32"
    assert casted["input_ids"] is inputs["input_ids"]
    assert casted["input_features_mask"] is inputs["input_features_mask"]
