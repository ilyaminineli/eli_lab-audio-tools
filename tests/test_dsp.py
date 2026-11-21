import numpy as np
from eugene_timbre.dsp import apply_profile, WARM_UNCANNY_B2, F3_GENTLE, G4_NEUTRAL, NEUTRAL
from eugene_timbre.analysis import analyze_audio

def test_neutral_is_bypass():
    x=np.random.default_rng(1).normal(0,.1,6000).astype(np.float32)
    assert np.max(np.abs(apply_profile(x,44100,NEUTRAL)-x)) < 1e-7

def test_profile_shape_and_finite():
    x=np.random.default_rng(2).normal(0,.1,(10000,2)).astype(np.float32)
    for profile in (WARM_UNCANNY_B2,F3_GENTLE,G4_NEUTRAL):
        y=apply_profile(x,44100,profile,.75)
        assert y.shape==x.shape and np.all(np.isfinite(y))

def test_analysis_peak_dbfs_is_correct():
    x=np.full(4410,0.5,dtype=np.float32)
    d=analyze_audio(x,44100)
    assert d["peak_amplitude"] == 0.5
    assert abs(d["peak_dbfs"] - (-6.0206)) < 0.01

def test_analysis_duration():
    x=np.sin(2*np.pi*220*np.arange(44100)/44100).astype(np.float32)
    d=analyze_audio(x,44100)
    assert abs(d["duration_seconds"]-1)<1e-9 and d["channels"]==1
