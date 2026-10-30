"""Version checks used by the build-time faster-whisper runtime probe."""

from __future__ import annotations

from backend.engines.fw.runtime import version_errors


def test_pinned_pair_is_accepted() -> None:
    assert version_errors(faster_whisper_version="1.2.1", av_version="18.0.0") == []


def test_older_pyav_is_accepted() -> None:
    assert version_errors(faster_whisper_version="1.2.1", av_version="16.0.1") == []


def test_pyav_19_is_rejected() -> None:
    errors = version_errors(faster_whisper_version="1.2.1", av_version="19.0.2")
    assert len(errors) == 1
    assert "metadata_errors" in errors[0]
    assert "19.0.2" in errors[0]


def test_faster_whisper_drift_is_rejected() -> None:
    errors = version_errors(faster_whisper_version="1.3.0", av_version="18.0.0")
    assert len(errors) == 1
    assert "1.3.0" in errors[0]
