"""Testable helpers for whole-sample Eugene:RE preview."""
from __future__ import annotations

from pathlib import Path

from .io import read_wav
from .full_sample_processing import process_full_sample


DEFAULT_WET = 1.0


def prepare_preview(root: Path, relative_path: str, wet: float = DEFAULT_WET) -> dict:
    """Process the selected WAV across its complete duration for preview."""
    root = root.expanduser().resolve()
    source = (root / Path(relative_path)).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError("Preview sample is not inside the selected voicebank.")
    audio, sample_rate, metadata = read_wav(source)
    processed, details = process_full_sample(
        audio, sample_rate, Path(relative_path), root, wet=wet
    )
    return {
        "source": str(source),
        "relative_path": Path(relative_path).as_posix(),
        "sample_rate": sample_rate,
        "subtype": metadata["subtype"],
        "original": audio,
        "processed": processed,
        "processing": details,
        "same_audio": not details["changed"],
    }
