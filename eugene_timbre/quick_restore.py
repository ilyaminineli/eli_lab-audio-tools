"""Testable helpers for whole-sample Eugene:RE preview."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import stft

from .io import read_wav
from .full_sample_processing import process_full_sample


DEFAULT_WET = 1.0


def compute_spectrogram_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 4096,
    noverlap_fraction: float = 0.75,
    min_frequency_hz: float = 70.0,
    max_frequency_hz: float = 12000.0,
    floor_db: float = -75.0,
) -> dict:
    """Build aligned log-frequency spectrograms and a signed dB difference.

    Both spectrograms use one shared reference level, so colour intensity is
    directly comparable. The difference is processed dB minus original dB.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if not 0.0 < noverlap_fraction < 1.0:
        raise ValueError("noverlap_fraction must be between zero and one.")
    if not 0.0 < min_frequency_hz < max_frequency_hz:
        raise ValueError("Spectrogram frequency limits must be positive and increasing.")

    def to_mono(audio: np.ndarray) -> np.ndarray:
        data = np.asarray(audio, dtype=np.float64)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
            raise ValueError("Spectrogram input must be finite, non-empty mono or stereo audio.")
        return data

    a = to_mono(original)
    b = to_mono(processed)
    duration_seconds = float(a.size / sample_rate)
    if a.shape != b.shape:
        raise ValueError("Original and processed audio must have matching lengths.")
    if sample_rate <= 2 * min_frequency_hz:
        raise ValueError("Sample rate is too low for the requested spectrogram range.")

    length = min(int(nperseg), a.size)
    if length < 256:
        pad = 256 - length
        a = np.pad(a, (0, pad))
        b = np.pad(b, (0, pad))
        length = 256
    length = max(256, 1 << (length.bit_length() - 1))
    length = min(length, a.size)
    overlap = min(length - 1, int(round(length * noverlap_fraction)))

    frequencies, times, spectrum_a = stft(
        a, fs=sample_rate, window="hann", nperseg=length,
        noverlap=overlap, boundary="zeros", padded=True,
    )
    _, _, spectrum_b = stft(
        b, fs=sample_rate, window="hann", nperseg=length,
        noverlap=overlap, boundary="zeros", padded=True,
    )

    # scipy's padded boundary frames can have centres just beyond the last
    # audio sample. Drop those frames so the display timeline ends at the WAV.
    time_valid = times <= (duration_seconds + 1e-9)
    times = times[time_valid]
    spectrum_a = spectrum_a[:, time_valid]
    spectrum_b = spectrum_b[:, time_valid]

    max_frequency = min(float(max_frequency_hz), sample_rate / 2.0)
    valid = (frequencies >= min_frequency_hz) & (frequencies <= max_frequency)
    if np.count_nonzero(valid) < 8:
        raise ValueError("Insufficient frequency bins for the requested spectrogram range.")

    frequencies = frequencies[valid]
    magnitude_a = np.abs(spectrum_a[valid])
    magnitude_b = np.abs(spectrum_b[valid])
    ref_amplitude = max(
        float(np.max(magnitude_a)),
        float(np.max(magnitude_b)),
        1e-12,
    )
    reference_db = 20.0 * np.log10(ref_amplitude)
    if ref_amplitude <= 1e-10:
        # Digital silence should render at the floor, not as a full-bright
        # spectrogram merely because the shared reference is also zero.
        a_db = np.full(magnitude_a.shape, floor_db, dtype=np.float32)
        b_db = np.full(magnitude_b.shape, floor_db, dtype=np.float32)
    else:
        a_db = 20.0 * np.log10(np.maximum(magnitude_a, 1e-12)) - reference_db
        b_db = 20.0 * np.log10(np.maximum(magnitude_b, 1e-12)) - reference_db
        a_db = np.clip(a_db, floor_db, 0.0).astype(np.float32)
        b_db = np.clip(b_db, floor_db, 0.0).astype(np.float32)

    difference = b_db - a_db
    energy_mask = np.maximum(a_db, b_db) > (floor_db + 10.0)
    difference = np.where(energy_mask, difference, 0.0)
    difference = np.clip(difference, -6.0, 6.0).astype(np.float32)

    return {
        "frequencies_hz": frequencies.astype(np.float32),
        "times_seconds": times.astype(np.float32),
        "original_db": a_db,
        "processed_db": b_db,
        "difference_db": difference,
        "reference_dbfs": float(reference_db),
        "floor_db": float(floor_db),
        "difference_range_db": [-6.0, 6.0],
        "duration_seconds": duration_seconds,
    }


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
    spectrogram = compute_spectrogram_comparison(
        audio, processed, sample_rate
    )
    return {
        "source": str(source),
        "relative_path": Path(relative_path).as_posix(),
        "sample_rate": sample_rate,
        "subtype": metadata["subtype"],
        "original": audio,
        "processed": processed,
        "processing": details,
        "spectrogram": spectrogram,
        "same_audio": not details["changed"],
    }
