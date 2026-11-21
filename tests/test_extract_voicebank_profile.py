import json
from pathlib import Path

from tools.extract_voicebank_profile import build


def test_build_profile_compacts_pitch_full_bank_and_legacy(tmp_path: Path):
    def deep(path: Path, groups: dict):
        path.write_text(json.dumps({"manifest": {}, "groups": groups}), encoding="utf-8")
        return path

    sample = {
        "relative_path": "5. G4/vowel.wav",
        "basic": {
            "duration_seconds": 1.0, "rms_dbfs": -14.0,
            "spectral_centroid_hz": 500.0, "spectral_flatness": 0.001,
            "relative_band_energy": {"150_500_hz": 0.7},
        },
        "timbre": {
            "f0_hz": {"median": 392.0},
            "harmonic_to_residual_ratio_db": {"median": 18.0},
            "voiced_frame_fraction": 0.9,
            "spectral_envelope": {"frequency_hz": [100, 200], "median_relative_db": [0, -3]},
            "harmonic_profile_relative_to_h1_db": [{"harmonic": 1, "median_db": 0}],
        },
        "oto": {"status": "matched", "entries": [{
            "alias": "a i_G4", "wav": "vowel.wav", "oto_line_number": 1,
            "timing": {
                "oto_values_ms": {"offset": 100, "consonant": 200},
                "markers_ms": {"offset": 100},
                "regions_ms": {"stretchable_region": [300, 500]},
                "stretchable_duration_ms": 200,
                "warnings": [],
            },
            "region_analysis": {"stretchable_region": {
                "duration_ms": 200, "deep_analysis_status": "analysed",
                "timbre": {"f0_hz": {"median": 392}},
            }},
        }]},
    }
    pitch = deep(tmp_path / "G4_analysis.json", {"G4": {"files": [sample]}})
    b2 = deep(tmp_path / "B2_analysis.json", {"B2": {"files": []}})
    f3 = deep(tmp_path / "F3_analysis.json", {"F3": {"files": []}})
    full = deep(tmp_path / "EugeneRE_analysis.json", {"CONSONANT_RELEASES": {"files": []}})
    legacy = tmp_path / "CONSONANT_analysis.json"
    legacy.write_text(json.dumps({"files": []}), encoding="utf-8")

    profile = build([b2, f3, pitch], full, [legacy])
    assert profile["processing_families"]["G4"]["sample_count"] == 1
    assert profile["processing_families"]["G4"]["aliases"][0]["canonical_alias"] == "a i"
    assert profile["processing_families"]["CONSONANT_RELEASES"]["group_kind"] == "consonant_release"
    assert profile["processing_families"]["G4"]["metrics"]["f0_hz"]["median"] == 392.0
