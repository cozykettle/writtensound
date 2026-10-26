"""whisper.cpp argv construction.

Flags are pinned against the whisper.cpp tag in docker-build-args.env. The
source-level flags were checked against v1.9.5; SPIKE S1 still has to confirm
the JSON field names and runtime DTW behavior on that tag.
"""

from __future__ import annotations

from pathlib import Path

WHISPERCPP_TAG = "v1.9.5"
PROGRESS_FLAG = "--print-progress"
JSON_FLAG = "--output-json-full"
NO_GPU_FLAG = "--no-gpu"
GPU_DEVICE_FLAG = "--device"


def build_command(
    *,
    binary: str,
    model_path: Path,
    audio_path: Path,
    output_prefix: Path,
    language: str,
    device: str,
    threads: int,
    verbatim: bool,
) -> list[str]:
    cmd = [
        binary,
        "-m",
        str(model_path),
        "-f",
        str(audio_path),
        "--language",
        language if language != "auto" else "auto",
        JSON_FLAG,
        "--output-file",
        str(output_prefix),
        PROGRESS_FLAG,
        "--threads",
        str(max(1, threads)),
    ]
    if device == "cpu":
        cmd.append(NO_GPU_FLAG)
    else:
        cmd += [GPU_DEVICE_FLAG, "0"]
    # --output-json-full already enables token timestamps in whisper.cpp
    # v1.9.5.  --max-len would wrap every token into its own segment, turning
    # paragraphs and HTML output into token-per-line noise.
    if verbatim:
        cmd += ["--prompt", "Uh, um, er, ah. Transcribe every filler word exactly as spoken."]
    return cmd
