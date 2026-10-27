"""Version-pair checks used by the build-time Qwen runtime probe."""

from __future__ import annotations

from backend.engines.qwen3.runtime import version_errors


def test_pinned_pair_is_accepted() -> None:
    assert (
        version_errors(
            torch_version="2.9.1+cpu",
            torchaudio_version="2.9.1+cpu",
            torchcodec_version="0.9.1",
            transformers_version="5.20.2",
        )
        == []
    )


def test_cuda_build_of_the_same_series_is_accepted() -> None:
    assert (
        version_errors(
            torch_version="2.9.1+cu128",
            torchaudio_version="2.9.1+cu128",
            torchcodec_version="0.9.1",
            transformers_version="5.20.2",
        )
        == []
    )


def test_torchcodec_that_needs_newer_torch_is_rejected() -> None:
    errors = version_errors(
        torch_version="2.9.1+cpu",
        torchaudio_version="2.9.1+cpu",
        torchcodec_version="0.17.0",
        transformers_version="5.20.2",
    )
    assert len(errors) == 1
    assert "torchcodec 0.17.0" in errors[0]


def test_torch_series_and_transformers_drift_are_rejected() -> None:
    errors = version_errors(
        torch_version="2.10.0+cpu",
        torchaudio_version="2.10.1+cpu",
        torchcodec_version="0.9.1",
        transformers_version="5.21.0",
    )
    assert any("torch 2.10.0+cpu" in error for error in errors)
    assert any("torchaudio 2.10.1+cpu" in error for error in errors)
    assert any("transformers 5.21.0" in error for error in errors)
