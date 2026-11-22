from pathlib import Path

import numpy as np
import soundfile as sf

from eugene_timbre.pitch_studio import (
    analyze_pitch_folder,
    default_profile,
    discover_pitch_layers,
    process_pitch_folder,
    suggest_reference_match,
)


def _write_test_tone(path: Path, frequency: float = 1200.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.18 * np.sin(2 * np.pi * frequency * time)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def test_discover_only_pitch_layer_folders(tmp_path):
    (tmp_path / "6. B2").mkdir()
    (tmp_path / "4. F3").mkdir()
    (tmp_path / "5. G4").mkdir()
    (tmp_path / "11. CONSONANT RELEASES").mkdir()
    (tmp_path / "10. R ENGLISH").mkdir()

    found = discover_pitch_layers(tmp_path)

    assert [item["pitch"] for item in found] == ["F3", "G4", "B2"]
    assert all("CONSONANT" not in item["name"] for item in found)


def test_analyze_pitch_folder_reports_pitch_and_spectral_data(tmp_path):
    folder = tmp_path / "6. B2"
    _write_test_tone(folder / "a_a.wav", frequency=220.0)

    report = analyze_pitch_folder(folder)

    assert report["report_type"] == "eugene_pitch_layer_analysis"
    assert report["pitch"] == "B2"
    assert report["analysed_count"] == 1
    assert report["summary"]["f0_median_hz"] is not None
    assert report["summary"]["relative_band_energy_median"]["150_500_hz"] > 0.5


def test_reference_match_is_bounded_and_moves_toward_reference():
    source = {
        "pitch": "B2",
        "summary": {"relative_band_energy_median": {
            "20_150_hz": 0.2, "150_500_hz": 0.2,
            "500_2000_hz": 0.2, "2000_5000_hz": 0.05,
            "5000_10000_hz": 0.1,
        }},
    }
    reference = {
        "pitch": "F3",
        "summary": {"relative_band_energy_median": {
            "20_150_hz": 0.1, "150_500_hz": 0.1,
            "500_2000_hz": 0.1, "2000_5000_hz": 0.2,
            "5000_10000_hz": 0.1,
        }},
    }

    suggestion = suggest_reference_match(source, reference, strength=0.6, max_gain_db=2.5)
    gains = [band["gain_db"] for band in suggestion["bands"]]

    assert gains[0] < 0
    assert gains[3] > 0
    assert all(abs(gain) <= 2.5 for gain in gains)
    assert suggestion["reference_pitch"] == "F3"


def test_process_pitch_folder_changes_only_selected_layer_and_copies_metadata(tmp_path):
    source = tmp_path / "voicebank" / "6. B2"
    output_root = tmp_path / "restored"
    wav_path = source / "a_a.wav"
    _write_test_tone(wav_path)
    metadata = source / "oto.ini"
    metadata.write_bytes(b"a_a.wav=a,0,100,0,80,20\r\n")

    profile = default_profile("B2")
    profile["bands"][2]["frequency_hz"] = 1200
    profile["bands"][2]["gain_db"] = 4.0

    report = process_pitch_folder(source, output_root, profile, wet=1.0)

    restored = output_root / "6. B2" / "a_a.wav"
    assert report["summary"]["processed"] == 1
    assert report["summary"]["failed"] == 0
    assert restored.is_file()
    assert metadata.read_bytes() == (output_root / "6. B2" / "oto.ini").read_bytes()
    assert report["files"][0]["peak_abs_difference"] > 1e-5
    assert report["files"][0]["changed_fraction"] > 0.5
    assert not (output_root / "4. F3").exists()
    assert wav_path.exists()


def test_neutral_profile_copies_wav_without_reencoding(tmp_path):
    source = tmp_path / "voicebank" / "5. G4"
    output_root = tmp_path / "restored"
    wav_path = source / "sample.wav"
    _write_test_tone(wav_path)
    original_bytes = wav_path.read_bytes()

    report = process_pitch_folder(source, output_root, default_profile("G4"), wet=1.0)

    restored = output_root / "5. G4" / "sample.wav"
    assert report["summary"]["unchanged"] == 1
    assert restored.read_bytes() == original_bytes


def test_optional_studio_ui_module_is_valid_python():
    source = Path(__file__).parents[1] / "eugene_timbre" / "pitch_studio_ui.py"
    compile(source.read_text(encoding="utf-8"), str(source), "exec")
