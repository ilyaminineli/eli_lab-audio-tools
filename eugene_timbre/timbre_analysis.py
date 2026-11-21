"""Time-resolved harmonic and spectral-envelope analysis for singing voice samples.

Measurement-only: this module never alters the source audio or voicebank metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.signal import find_peaks, get_window

EPS = 1e-12

@dataclass(frozen=True)
class TimbreAnalysisConfig:
    frame_ms: float = 85.0
    hop_ms: float = 20.0
    f0_min_hz: float = 65.0
    f0_max_hz: float = 850.0
    max_harmonics: int = 20
    max_analysis_hz: float = 10000.0
    envelope_min_hz: float = 100.0
    envelope_max_hz: float = 10000.0
    envelope_bins: int = 72
    envelope_quefrency_ms: float = 2.5
    min_voiced_rms_dbfs: float = -48.0
    min_pitch_confidence: float = 0.18

    def __post_init__(self) -> None:
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms cannot exceed frame_ms")
        if self.f0_min_hz <= 0 or self.f0_max_hz <= self.f0_min_hz:
            raise ValueError("F0 range must satisfy 0 < f0_min_hz < f0_max_hz")
        if self.max_harmonics < 1 or self.max_analysis_hz <= 0:
            raise ValueError("max_harmonics and max_analysis_hz must be positive")
        if not 0 < self.envelope_min_hz < self.envelope_max_hz:
            raise ValueError("Envelope frequency range must be positive and increasing")
        if self.envelope_bins < 16 or self.envelope_quefrency_ms <= 0:
            raise ValueError("Envelope needs at least 16 bins and positive quefrency")
        if not 0 <= self.min_pitch_confidence <= 1:
            raise ValueError("min_pitch_confidence must be between 0 and 1")


def _db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(value, EPS))


def _percentiles(values) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p10": None, "median": None, "p90": None}
    return {"p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90))}


def _frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if x.size < frame_len:
        return np.pad(x, (0, frame_len - x.size))[None, :]
    n = 1 + (x.size - frame_len) // hop
    return np.lib.stride_tricks.as_strided(
        x, shape=(n, frame_len), strides=(x.strides[0] * hop, x.strides[0])
    ).copy()


def _estimate_f0(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> tuple[float, float]:
    """Autocorrelation F0 estimate; returns (Hz, normalised peak confidence)."""
    y = np.asarray(frame, dtype=np.float64)
    y = y - np.mean(y)
    if np.sqrt(np.mean(y * y)) < 1e-8:
        return 0.0, 0.0
    y *= get_window("hann", y.size, fftbins=True)
    nfft = 1 << int(np.ceil(np.log2(2 * y.size)))
    spec = np.fft.rfft(y, n=nfft)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft)[:y.size]
    if ac[0] <= EPS:
        return 0.0, 0.0
    ac /= ac[0]
    min_lag = max(1, int(sr / fmax))
    max_lag = min(y.size - 2, int(sr / fmin))
    if max_lag <= min_lag:
        return 0.0, 0.0
    region = ac[min_lag:max_lag + 1]
    peaks, _ = find_peaks(region)
    lag = min_lag + int(peaks[np.argmax(region[peaks])]) if peaks.size else min_lag + int(np.argmax(region))
    confidence = float(ac[lag])
    if 1 <= lag < ac.size - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > EPS else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    refined_lag = lag + delta
    f0 = float(sr / refined_lag) if refined_lag > 0 else 0.0
    return (f0, confidence) if fmin <= f0 <= fmax else (0.0, confidence)


def _cepstral_envelope(magnitude: np.ndarray, sr: int, nfft: int,
                       max_quefrency_ms: float) -> np.ndarray:
    """Smooth log magnitude using a low-quefrency real-cepstrum lifter."""
    cep = np.fft.irfft(np.log(np.maximum(magnitude, EPS)), n=nfft)
    qmax = min(max(2, int(round(sr * max_quefrency_ms / 1000.0))), nfft // 2 - 1)
    liftered = np.zeros_like(cep)
    liftered[:qmax + 1] = cep[:qmax + 1]
    liftered[-qmax:] = cep[-qmax:]
    return np.fft.rfft(liftered, n=nfft).real


def _harmonic_features(magnitude: np.ndarray, freqs: np.ndarray, f0: float,
                       sr: int, max_harmonics: int, max_hz: float) -> dict[str, Any]:
    df = float(freqs[1] - freqs[0])
    limit = min(max_hz, sr / 2.0)
    amps, peak_freqs, used_bins = [], [], set()
    for h in range(1, max_harmonics + 1):
        target = h * f0
        if target >= limit:
            break
        tol = max(df * 3.0, min(f0 * 0.28, 35.0))
        lo = max(0, int(np.floor((target - tol) / df)))
        hi = min(magnitude.size, int(np.ceil((target + tol) / df)) + 1)
        if hi <= lo:
            continue
        idx = lo + int(np.argmax(magnitude[lo:hi]))
        amps.append(float(magnitude[idx]))
        peak_freqs.append(float(freqs[idx]))
        used_bins.update(range(lo, hi))
    harmonic_power = float(sum(magnitude[i] ** 2 for i in used_bins))
    total_power = float(np.sum(magnitude[freqs <= limit] ** 2)) + EPS
    harmonic_ratio = min(1.0, harmonic_power / total_power)
    residual_ratio = max(EPS, 1.0 - harmonic_ratio)
    hnr_db = float(10.0 * np.log10(max(harmonic_ratio, EPS) / residual_ratio))
    amp_db = np.asarray(_db(np.asarray(amps)), dtype=float) if amps else np.array([])
    relative = (amp_db - amp_db[0]).tolist() if amp_db.size else []
    if len(relative) >= 3:
        order = np.arange(1, len(relative) + 1, dtype=float)
        slope = float(np.polyfit(np.log2(order), amp_db, 1)[0])
    else:
        slope = None
    if len(peak_freqs) >= 2:
        deviations, weights = [], []
        for h, peak in enumerate(peak_freqs, start=1):
            deviations.append(abs(peak - h * f0) / max(h * f0, EPS))
            weights.append(amps[h - 1] ** 2)
        inharmonicity = float(100.0 * np.average(deviations, weights=weights))
    else:
        inharmonicity = None
    return {
        "harmonic_count": len(amps),
        "harmonic_amplitudes_relative_to_h1_db": [float(v) for v in relative],
        "harmonic_slope_db_per_octave": slope,
        "h1_minus_h2_db": float(amp_db[0] - amp_db[1]) if amp_db.size >= 2 else None,
        "h2_minus_h4_db": float(amp_db[1] - amp_db[3]) if amp_db.size >= 4 else None,
        "harmonic_to_residual_ratio_db": hnr_db,
        "inharmonicity_percent": inharmonicity,
    }


def _aggregate_envelope(envelopes: list[np.ndarray], freq_grid: np.ndarray) -> dict[str, Any] | None:
    if not envelopes:
        return None
    matrix = np.vstack(envelopes)
    median = np.median(matrix, axis=0)
    peaks, _ = find_peaks(median, prominence=1.8, distance=3)
    candidates = sorted(peaks, key=lambda i: median[i], reverse=True)[:8]
    candidates = sorted(candidates, key=lambda i: freq_grid[i])
    return {
        "frequency_hz": [float(v) for v in freq_grid],
        "median_relative_db": [float(v) for v in median],
        "p10_relative_db": [float(v) for v in np.percentile(matrix, 10, axis=0)],
        "p90_relative_db": [float(v) for v in np.percentile(matrix, 90, axis=0)],
        "broad_envelope_peaks_hz": [float(freq_grid[i]) for i in candidates],
    }


def analyze_timbre(audio: np.ndarray, sample_rate: int,
                   config: TimbreAnalysisConfig | None = None) -> dict[str, Any]:
    """Measure time-varying F0, harmonics, residual energy and spectral envelope."""
    cfg = config or TimbreAnalysisConfig()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2) or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Audio must be finite, non-empty mono or multi-channel")
    mono = np.mean(x, axis=1) if x.ndim == 2 else x
    if mono.size == 0:
        raise ValueError("Audio has no frames")
    frame_len = max(512, int(round(sample_rate * cfg.frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * cfg.hop_ms / 1000.0)))
    frames = _frame_signal(mono, frame_len, hop)
    window = get_window("hann", frame_len, fftbins=True)
    nfft = 1 << int(np.ceil(np.log2(max(8192, frame_len * 2))))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    env_grid = np.geomspace(cfg.envelope_min_hz, min(cfg.envelope_max_hz, sample_rate / 2), cfg.envelope_bins)

    f0_values, confidence_values, voiced_confidence = [], [], []
    rms_values, voiced_rms, centroid_values, flatness_values = [], [], [], []
    voiced_centroid, voiced_flatness, f0_by_frame = [], [], []
    flux_values, hnr_values, inharmonicity_values, harmonic_slopes, h1h2_values = [], [], [], [], []
    envelope_frames, harmonic_profile_frames = [], []
    tracks = {k: [] for k in (
        "time_seconds", "f0_hz", "pitch_confidence", "rms_dbfs",
        "spectral_centroid_hz", "spectral_flatness", "positive_spectral_flux",
        "harmonic_to_residual_ratio_db", "inharmonicity_percent",
        "harmonic_slope_db_per_octave",
    )}
    previous_spectrum = None
    voiced_count = 0

    for frame_index, frame in enumerate(frames):
        demeaned = frame - np.mean(frame)
        rms = float(np.sqrt(np.mean(demeaned * demeaned)))
        rms_db = float(_db(rms))
        rms_values.append(rms_db)
        f0, confidence = _estimate_f0(demeaned, sample_rate, cfg.f0_min_hz, cfg.f0_max_hz)
        confidence_values.append(confidence)
        is_voiced = rms_db >= cfg.min_voiced_rms_dbfs and f0 > 0 and confidence >= cfg.min_pitch_confidence
        f0_by_frame.append(f0 if is_voiced else 0.0)

        spectrum = np.abs(np.fft.rfft(demeaned * window, n=nfft))
        power = spectrum ** 2
        valid = freqs <= min(cfg.max_analysis_hz, sample_rate / 2)
        valid_power, valid_freqs = power[valid], freqs[valid]
        total = float(np.sum(valid_power)) + EPS
        centroid = float(np.sum(valid_freqs * valid_power) / total)
        flatness = float(np.exp(np.mean(np.log(valid_power + EPS))) / (np.mean(valid_power) + EPS))
        centroid_values.append(centroid)
        flatness_values.append(flatness)
        norm_spec = np.sqrt(valid_power / total)
        flux = 0.0 if previous_spectrum is None else float(
            np.sum(np.maximum(norm_spec - previous_spectrum, 0.0) ** 2)
        )
        flux_values.append(flux)
        previous_spectrum = norm_spec

        tracks["time_seconds"].append(float(frame_index * hop / sample_rate))
        tracks["f0_hz"].append(float(f0) if is_voiced else None)
        tracks["pitch_confidence"].append(float(confidence))
        tracks["rms_dbfs"].append(rms_db)
        tracks["spectral_centroid_hz"].append(centroid)
        tracks["spectral_flatness"].append(flatness)
        tracks["positive_spectral_flux"].append(flux)

        if not is_voiced:
            tracks["harmonic_to_residual_ratio_db"].append(None)
            tracks["inharmonicity_percent"].append(None)
            tracks["harmonic_slope_db_per_octave"].append(None)
            continue

        voiced_count += 1
        f0_values.append(f0)
        voiced_confidence.append(confidence)
        voiced_rms.append(rms_db)
        voiced_centroid.append(centroid)
        voiced_flatness.append(flatness)
        harmonic = _harmonic_features(spectrum, freqs, f0, sample_rate,
                                      cfg.max_harmonics, cfg.max_analysis_hz)
        for key, values, track_key in (
            ("harmonic_to_residual_ratio_db", hnr_values, "harmonic_to_residual_ratio_db"),
            ("inharmonicity_percent", inharmonicity_values, "inharmonicity_percent"),
            ("harmonic_slope_db_per_octave", harmonic_slopes, "harmonic_slope_db_per_octave"),
        ):
            value = harmonic[key]
            tracks[track_key].append(value)
            if value is not None:
                values.append(value)
        if harmonic["h1_minus_h2_db"] is not None:
            h1h2_values.append(harmonic["h1_minus_h2_db"])
        harmonic_profile_frames.append(harmonic["harmonic_amplitudes_relative_to_h1_db"])

        envelope_db = _cepstral_envelope(spectrum, sample_rate, nfft, cfg.envelope_quefrency_ms)
        reference_mask = (freqs >= 250) & (freqs <= 1500)
        reference = float(np.median(envelope_db[reference_mask])) if np.any(reference_mask) else 0.0
        envelope_frames.append(np.interp(env_grid, freqs, envelope_db - reference))

    f0_array = np.asarray(f0_values, dtype=float)
    adjacent_f0 = [(a, b) for a, b in zip(f0_by_frame[:-1], f0_by_frame[1:]) if a > 0 and b > 0]
    f0_range_cents = (
        float(1200.0 * np.log2(np.percentile(f0_array, 90) / np.percentile(f0_array, 10)))
        if f0_array.size > 1 else None
    )
    local_variation = None
    if adjacent_f0 and f0_array.size:
        steps = np.asarray([b - a for a, b in adjacent_f0], dtype=float)
        local_variation = float(100.0 * np.median(np.abs(steps)) / max(float(np.median(f0_array)), EPS))

    harmonic_profile = []
    median_harmonic_profile = []
    if harmonic_profile_frames:
        for i in range(cfg.max_harmonics):
            vals = [row[i] for row in harmonic_profile_frames if len(row) > i]
            if vals:
                median_harmonic_profile.append(float(np.median(vals)))
                harmonic_profile.append({
                    "harmonic": i + 1,
                    "p10_db": float(np.percentile(vals, 10)),
                    "median_db": float(np.median(vals)),
                    "p90_db": float(np.percentile(vals, 90)),
                })

    return {
        "analysis_version": "1.0",
        "sample_rate_hz": int(sample_rate),
        "duration_seconds": float(mono.size / sample_rate),
        "frame_count": int(frames.shape[0]),
        "voiced_frame_count": int(voiced_count),
        "voiced_frame_fraction": float(voiced_count / max(frames.shape[0], 1)),
        "config": {
            "frame_ms": cfg.frame_ms, "hop_ms": cfg.hop_ms,
            "f0_min_hz": cfg.f0_min_hz, "f0_max_hz": cfg.f0_max_hz,
            "max_harmonics": cfg.max_harmonics,
            "envelope_quefrency_ms": cfg.envelope_quefrency_ms,
        },
        "f0_hz": _percentiles(f0_array),
        "pitch_variation_cents_p90_minus_p10": f0_range_cents,
        "local_f0_variation_percent": local_variation,
        "pitch_confidence_all_frames": _percentiles(confidence_values),
        "pitch_confidence_voiced_frames": _percentiles(voiced_confidence),
        "voiced_rms_dbfs": _percentiles(voiced_rms),
        "spectral_centroid_hz_all_frames": _percentiles(centroid_values),
        "spectral_centroid_hz_voiced_frames": _percentiles(voiced_centroid),
        "spectral_flatness_all_frames": _percentiles(flatness_values),
        "spectral_flatness_voiced_frames": _percentiles(voiced_flatness),
        "positive_spectral_flux": _percentiles(flux_values),
        "harmonic_to_residual_ratio_db": _percentiles(hnr_values),
        "aperiodic_residual_energy_fraction_percent": _percentiles(
            [100.0 / (1.0 + 10.0 ** (v / 10.0)) for v in hnr_values]
        ),
        "inharmonicity_percent": _percentiles(inharmonicity_values),
        "harmonic_slope_db_per_octave": _percentiles(harmonic_slopes),
        "h1_minus_h2_db": _percentiles(h1h2_values),
        "median_harmonic_profile_relative_to_h1_db": median_harmonic_profile,
        "harmonic_profile_relative_to_h1_db": harmonic_profile,
        "rms_level_variation_db": float(np.std(rms_values)) if rms_values else None,
        "frame_tracks": tracks,
        "spectral_envelope": _aggregate_envelope(envelope_frames, env_grid),
        "interpretation_notes": [
            "Harmonic-to-residual ratio is an analysis proxy, not a studio-grade HNR measurement.",
            "Cepstral spectral-envelope peaks are broad resonances, not guaranteed vocal-tract formants.",
            "Whole-file results mix vowels and transitions; compare equivalent phonemes and pitch layers.",
            "Low pitch variation can be natural for a deliberately steady synthetic source; do not add jitter automatically.",
        ],
    }
