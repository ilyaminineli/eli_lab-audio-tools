"""WAV analysis with correctly labelled amplitude and dBFS values."""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

BANDS = ((20,150),(150,500),(500,2000),(2000,5000),(5000,10000))

def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2) or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Audio must be finite, non-empty mono or multi-channel data.")
    mono = np.mean(x, axis=1) if x.ndim == 2 else x
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-12))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-12))
    f, p = welch(mono, fs=sample_rate, nperseg=min(8192, mono.size))
    total = float(np.sum(p)) + 1e-30
    def fraction(lo: float, hi: float) -> float:
        return float(np.sum(p[(f >= lo) & (f < hi)]) / total)
    centroid = float(np.sum(f*p)/total)
    flatness = float(np.exp(np.mean(np.log(p+1e-30)))/(np.mean(p)+1e-30))
    bands = {f"{lo}_{hi}_hz": fraction(lo, hi) for lo, hi in BANDS}
    bands["10000_nyquist_hz"] = fraction(10000, sample_rate/2)
    return {
        "sample_rate_hz": int(sample_rate),
        "frames": int(mono.size),
        "channels": int(x.shape[1]) if x.ndim == 2 else 1,
        "duration_seconds": float(mono.size/sample_rate),
        "peak_amplitude": peak,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "crest_factor_db": peak_dbfs-rms_dbfs,
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "relative_band_energy": bands,
    }
