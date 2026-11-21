import numpy as np
from eugene_timbre.timbre_analysis import analyze_timbre, TimbreAnalysisConfig

def make_voice(sr=44100, duration=1.0, f0=220.0, noise=0.002):
    t = np.arange(int(sr * duration)) / sr
    x = (0.45 * np.sin(2 * np.pi * f0 * t)
         + 0.20 * np.sin(2 * np.pi * 2 * f0 * t)
         + 0.10 * np.sin(2 * np.pi * 3 * f0 * t)
         + 0.05 * np.sin(2 * np.pi * 4 * f0 * t))
    x += np.random.default_rng(42).normal(0, noise, t.size)
    return x.astype(np.float32)

def test_f0_and_harmonic_metrics_on_synthetic_voice():
    report = analyze_timbre(make_voice(), 44100)
    assert report["voiced_frame_fraction"] > 0.8
    assert abs(report["f0_hz"]["median"] - 220.0) < 4.0
    assert report["harmonic_to_residual_ratio_db"]["median"] > 10.0
    assert len(report["median_harmonic_profile_relative_to_h1_db"]) >= 3

def test_envelope_has_frequency_grid_and_finite_values():
    report = analyze_timbre(make_voice(), 44100)
    env = report["spectral_envelope"]
    assert env is not None
    assert len(env["frequency_hz"]) == 72
    assert np.all(np.isfinite(env["median_relative_db"]))

def test_works_for_short_audio_and_reports_unvoiced():
    report = analyze_timbre(np.zeros(1000, dtype=np.float32), 44100)
    assert report["frame_count"] >= 1
    assert report["f0_hz"]["median"] is None
    assert report["spectral_envelope"] is None

def test_analyzer_does_not_modify_input():
    audio = make_voice()
    original = audio.copy()
    analyze_timbre(audio, 44100)
    assert np.array_equal(audio, original)

def test_frame_tracks_are_time_aligned():
    report = analyze_timbre(make_voice(duration=0.5), 44100)
    tracks = report["frame_tracks"]
    assert len(tracks["time_seconds"]) == report["frame_count"]
    assert len(tracks["f0_hz"]) == report["frame_count"]
    assert len(tracks["harmonic_to_residual_ratio_db"]) == report["frame_count"]

def test_invalid_analysis_settings_fail_early():
    try:
        TimbreAnalysisConfig(frame_ms=10, hop_ms=20)
    except ValueError:
        return
    raise AssertionError("hop longer than frame must be rejected")
