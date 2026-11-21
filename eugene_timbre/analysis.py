"""Objective sample analysis for timbre restoration decisions."""
import numpy as np
from scipy.signal import welch

def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict:
    x = np.asarray(audio, dtype=np.float64)
    mono = np.mean(x, axis=1) if x.ndim == 2 else x
    if mono.ndim != 1 or mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError("Audio must be finite, non-empty mono or multi-channel data.")
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    peak_db = 20*np.log10(max(peak, 1e-12))
    rms_db = 20*np.log10(max(rms, 1e-12))
    f, p = welch(mono, fs=sample_rate, nperseg=min(8192, mono.size))
    total = float(np.sum(p)) + 1e-30
    def fraction(lo, hi):
        return float(np.sum(p[(f >= lo) & (f < hi)]) / total)
    centroid = float(np.sum(f*p)/total)
    flatness = float(np.exp(np.mean(np.log(p+1e-30)))/(np.mean(p)+1e-30))
    return {
        "sample_rate_hz": int(sample_rate), "frames": int(mono.size),
        "channels": int(x.shape[1]) if x.ndim == 2 else 1,
        "duration_seconds": float(mono.size/sample_rate),
        "peak_dbfs": peak, "rms_dbfs": rms_db,
        "crest_factor_db": peak_db-rms_db,
        "spectral_centroid_hz": centroid, "spectral_flatness": flatness,
        "relative_band_energy": {
            "20_150_hz": fraction(20,150), "150_500_hz": fraction(150,500),
            "500_2000_hz": fraction(500,2000), "2000_5000_hz": fraction(2000,5000),
            "5000_10000_hz": fraction(5000,10000),
            "10000_nyquist_hz": fraction(10000,sample_rate/2),
        },
    }
