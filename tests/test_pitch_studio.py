from pathlib import Path

import numpy as np
import soundfile as sf

from eugene_timbre.recommendations import (
    moresampler_flag_templates,
    recommend_eq_profile,
    recommend_profiles,
)
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



def _mock_analysis(pitch="F3", *, centroid=2200.0, hnr=16.0, flatness=0.04, low=0.12, body=0.22, core=0.38, presence=0.20, air=0.08):
    return {
        "report_type": "eugene_pitch_layer_analysis",
        "pitch": pitch,
        "summary": {
            "spectral_centroid_median_hz": centroid,
            "harmonic_to_residual_median_db": hnr,
            "spectral_flatness_median": flatness,
            "relative_band_energy_median": {
                "20_150_hz": low,
                "150_500_hz": body,
                "500_2000_hz": core,
                "2000_5000_hz": presence,
                "5000_10000_hz": air,
            },
        },
        "spectral_envelope": {
            "frequency_hz": np.geomspace(100.0, 10000.0, 72).tolist(),
            "median_relative_db": np.zeros(72).tolist(),
        },
    }


def test_analysis_generates_bounded_pitch_specific_eq_approaches():
    analysis = _mock_analysis()
    recommendations = recommend_profiles(analysis)

    available = {item["key"]: item for item in recommendations if item["available"]}
    assert {"neutral", "clarity", "warmth", "air", "soft"} <= set(available)
    assert available["reference_match"]["available"] is False

    for key in ("neutral", "clarity", "warmth", "air", "soft"):
        profile = available[key]["profile"]
        assert profile["generated_from_analysis"] is True
        assert profile["highpass_hz"] == 0.0
        assert len(profile["bands"]) == 5
        assert all(abs(band["gain_db"]) <= 1.5 for band in profile["bands"])


def test_clarity_and_air_templates_adapt_to_measured_spectrum():
    darker = _mock_analysis(centroid=1400.0, presence=0.10, air=0.02)
    brighter = _mock_analysis(centroid=3900.0, presence=0.42, air=0.24)

    darker_clarity = recommend_eq_profile(darker, "clarity")
    brighter_clarity = recommend_eq_profile(brighter, "clarity")
    darker_air = recommend_eq_profile(darker, "air")
    noisy_air = recommend_eq_profile(
        _mock_analysis(centroid=2400.0, hnr=1.0, flatness=0.45, air=0.02),
        "air",
    )

    clarity_gain = darker_clarity["bands"][3]["gain_db"]
    bright_gain = brighter_clarity["bands"][3]["gain_db"]
    air_gain = darker_air["bands"][4]["gain_db"]
    noisy_air_gain = noisy_air["bands"][4]["gain_db"]

    assert clarity_gain > bright_gain
    assert air_gain > noisy_air_gain
    assert clarity_gain <= 1.5 and air_gain <= 1.5


def test_reference_match_recommendation_requires_and_uses_another_pitch():
    source = _mock_analysis("B2")
    reference = _mock_analysis("F3")
    without_reference = {item["key"]: item for item in recommend_profiles(source)}
    with_reference = {item["key"]: item for item in recommend_profiles(source, reference)}

    assert without_reference["reference_match"]["available"] is False
    assert with_reference["reference_match"]["available"] is True
    assert with_reference["reference_match"]["profile"]["reference_pitch"] == "F3"
    same_pitch = {item["key"]: item for item in recommend_profiles(source, source)}
    assert same_pitch["reference_match"]["available"] is False


def test_moresampler_renderer_templates_are_separate_from_source_eq():
    templates = {item["key"]: item for item in moresampler_flag_templates()}
    assert templates["neutral"]["flags"] == ""
    assert templates["intimate"]["flags"] == "Mb+8Mt-4"
    assert templates["projected"]["flags"] == "Mt+8ME+4"
    assert templates["rough"]["flags"] == "MC+8MG+5"
    assert "source-WAV" not in templates["intimate"]["description"].casefold()
    assert templates["rough"]["caution"]


def test_studio_ui_connects_analysis_to_explicit_profile_application():
    source = Path(__file__).parents[1] / "eugene_timbre" / "pitch_studio_ui.py"
    text = source.read_text(encoding="utf-8")
    compile(text, str(source), "exec")
    assert "self.profile_recommendations[name] = recommend_profiles(report)" in text
    assert "def apply_recommended_current" in text
    assert "def apply_recommended_checked" in text
    assert "Apply to checked" in text
    assert "moresampler_flag_templates()" in text
    assert "A/B test selected Moresampler flags against neutral flags" in text
