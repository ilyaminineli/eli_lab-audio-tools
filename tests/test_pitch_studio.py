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


def test_reference_match_uses_spectral_envelopes_and_is_bounded():
    frequencies = np.geomspace(100.0, 10000.0, 72)
    source_curve = np.zeros(72)
    source_curve[(frequencies >= 100) & (frequencies <= 160)] = 2.0
    source_curve[(frequencies >= 2200) & (frequencies <= 4200)] = -4.0
    reference_curve = np.zeros(72)
    source = {
        "pitch": "B2",
        "spectral_envelope": {
            "frequency_hz": frequencies.tolist(),
            "median_relative_db": source_curve.tolist(),
        },
    }
    reference = {
        "pitch": "F3",
        "spectral_envelope": {
            "frequency_hz": frequencies.tolist(),
            "median_relative_db": reference_curve.tolist(),
        },
    }

    suggestion = suggest_reference_match(source, reference, strength=0.6, max_gain_db=2.5)
    gains = [band["gain_db"] for band in suggestion["bands"]]

    assert gains[0] < 0
    assert gains[3] > 0
    assert all(abs(gain) <= 2.5 for gain in gains)
    assert suggestion["reference_pitch"] == "F3"
    assert suggestion["match_method"] == "cepstral_spectral_envelope"


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


def test_optional_studio_ui_module_is_valid_python_and_requires_explicit_selection():
    source = Path(__file__).parents[1] / "eugene_timbre" / "pitch_studio_ui.py"
    text = source.read_text(encoding="utf-8")
    compile(text, str(source), "exec")
    assert "item.setCheckState(Qt.CheckState.Unchecked)" in text
    assert "def select_current_layer" in text
    assert "def select_all_layers" in text
    assert "def clear_layer_selection" in text
    assert "if not selected and self.layer_list.currentItem()" not in text
    assert "def render_moresampler(self)" in text
    assert "self.alias_combo = QComboBox()" in text
    assert "self.moresampler_edit = QLineEdit(" in text
    assert "self.highpass_spin = QDoubleSpinBox()" in text
    assert "self.preview_delta_label" in text
    assert "self.detail_tabs = QTabWidget()" in text
    assert "BANK STRUCTURE" in text
    assert "NO FILTER CHANGE" in text


def test_voicebank_inventory_preserves_pitch_and_addon_hierarchy(tmp_path):
    bank = tmp_path / "EugeneRE"
    pitch = bank / "4. F3"
    addon = bank / "11. CONSONANT RELEASES"
    bonus = bank / "12. BONUS PHONETIC EXTRAS"
    english = bank / "10. R ENGLISH"
    setup = bank / "1. SETUP" / "Resamplers"
    _write_test_tone(pitch / "a_a.wav")
    _write_test_tone(pitch / "nested" / "a_e_a.wav", frequency=1500.0)
    _write_test_tone(addon / "ab -.wav", frequency=800.0)
    _write_test_tone(bonus / "cough 1.wav", frequency=300.0)
    _write_test_tone(english / "@a_@i_@u.wav", frequency=500.0)
    setup.mkdir(parents=True)
    (pitch / "oto.ini").write_text("a_a.wav=a,0,100,0,80,20\\n", encoding="utf-8")
    (addon / "oto.ini").write_text("ab -.wav=ab -,0,100,0,80,20\\n", encoding="utf-8")
    (pitch / "a_a_wav.frq").write_bytes(b"frq")
    (pitch / "a_a.wav.llsm").write_bytes(b"llsm")

    from eugene_timbre.pitch_studio import inventory_voicebank
    report = inventory_voicebank(bank)

    assert report["pitch_layer_count"] == 1
    assert report["addon_folder_count"] == 3
    assert report["stats"]["wav"] == 5
    assert report["stats"]["frq"] == 1
    assert report["stats"]["llsm"] == 1
    assert report["stats"]["oto"] == 2

    categories = {node["name"]: node["category"] for node in report["children"]}
    assert categories["4. F3"] == "Pitch layer"
    assert categories["10. R ENGLISH"] == "R English subbank"
    assert categories["11. CONSONANT RELEASES"] == "Consonant releases"
    assert categories["12. BONUS PHONETIC EXTRAS"] == "Bonus phonetic extras"
    assert categories["1. SETUP"] == "Setup / tools"

    pitch_node = next(node for node in report["children"] if node["name"] == "4. F3")
    nested_node = next(node for node in pitch_node["children"] if node["name"] == "nested")
    assert any(child["name"] == "a_e_a.wav" for child in nested_node["children"])


def test_default_profile_is_explicitly_neutral():
    profile = default_profile("F3")
    assert profile["highpass_hz"] == 0.0
    assert all(band["gain_db"] == 0.0 for band in profile["bands"])
