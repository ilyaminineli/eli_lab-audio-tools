"""Conservative spectral shaping for existing UTAU source samples.

The default Eugene:RE B2 profile is intentionally a broad midrange test, not
an automatic de-synthesizer. It does not add noise, modulation, reverb, or
change duration/pitch/timing.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt

@dataclass(frozen=True)
class PeakingBand:
    frequency_hz: float
    gain_db: float
    q: float = 1.0

@dataclass(frozen=True)
class TimbreProfile:
    name: str
    description: str
    highpass_hz: float = 0.0
    highpass_slope: int = 2
    bands: tuple[PeakingBand, ...] = ()

# B2 carries very little energy above 2 kHz in the uploaded subbank analysis.
# A small wide lift in the occupied upper-midrange is a better listening test
# than the previous 3.2/5.2 kHz cuts. No high-pass: preserve low fundamentals.
WARM_UNCANNY_B2 = TimbreProfile(
    name="warm-uncanny-b2",
    description="B2 upper-mid presence test; retain low fundamentals and uncanny character.",
    highpass_hz=0.0,
    bands=(PeakingBand(1450.0, 1.0, 0.7),),
)
F3_GENTLE = TimbreProfile(
    name="f3-gentle-presence",
    description="Gentler upper-mid presence test for F3.",
    highpass_hz=0.0,
    bands=(PeakingBand(1450.0, 0.5, 0.7),),
)
G4_NEUTRAL = TimbreProfile(
    name="g4-neutral",
    description="Neutral starting point for the already brighter G4 subbank.",
    highpass_hz=0.0,
    bands=(),
)
NEUTRAL = TimbreProfile(name="neutral", description="No EQ; bypass reference.")

def _peaking_sos(fs: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    if not (0 < f0 < fs / 2):
        raise ValueError(f"Band frequency {f0:g} Hz must be below Nyquist ({fs/2:g} Hz).")
    if q <= 0:
        raise ValueError("Q must be positive.")
    A = 10.0 ** (gain_db / 40.0)
    w = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w) / (2.0 * q)
    c = np.cos(w)
    b0, b1, b2 = 1 + alpha*A, -2*c, 1 - alpha*A
    a0, a1, a2 = 1 + alpha/A, -2*c, 1 - alpha/A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]], dtype=np.float64)

def _apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return sosfiltfilt(sos, x)
    return np.column_stack([sosfiltfilt(sos, x[:, ch]) for ch in range(x.shape[1])])

def apply_profile(audio: np.ndarray, sample_rate: int,
                  profile: TimbreProfile = WARM_UNCANNY_B2,
                  wet: float = 1.0) -> np.ndarray:
    """Apply zero-phase EQ; preserve frame count and channel count."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if not np.isfinite(wet) or not 0 <= wet <= 1:
        raise ValueError("wet must be finite and in [0, 1].")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2) or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Audio must be finite, non-empty mono or multi-channel data.")
    y = x.copy()
    if profile.highpass_hz > 0:
        if profile.highpass_hz >= sample_rate / 2:
            raise ValueError("High-pass frequency must be below Nyquist.")
        sos = butter(profile.highpass_slope, profile.highpass_hz,
                     btype="highpass", fs=sample_rate, output="sos")
        y = _apply(sos, y)
    for band in profile.bands:
        if band.gain_db:
            y = _apply(_peaking_sos(sample_rate, band.frequency_hz,
                                    band.gain_db, band.q), y)
    return ((1.0 - wet) * x + wet * y).astype(np.float32)

def describe_profile(profile: TimbreProfile) -> dict:
    return {
        "name": profile.name,
        "description": profile.description,
        "highpass": {"enabled": profile.highpass_hz > 0,
                     "frequency_hz": profile.highpass_hz,
                     "slope_db_oct": profile.highpass_slope * 6},
        "bands": [{"frequency_hz": b.frequency_hz, "gain_db": b.gain_db, "q": b.q}
                  for b in profile.bands],
    }
