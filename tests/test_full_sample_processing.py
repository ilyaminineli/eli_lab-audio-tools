from pathlib import Path

import numpy as np
import soundfile as sf

from eugene_timbre.full_sample_processing import (
    _mains_hum_notches,
    _remove_dc,
    _conditional_subsonic_filter,
    process_full_sample,
    process_full_sample_bank,
)
from eugene_timbre.quick_restore import prepare_preview


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def _tone(
    sample_rate=44100,
    duration=1.0,
    f0=195.0,
    dc=0.002,
    subsonic=0.0,
    hum=0.0,
    upper=0.04,
):
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    audio = (
        0.18 * np.sin(2 * np.pi * f0 * t)
        + upper * np.sin(2 * np.pi * 1450.0 * t)
        + dc
    )
    if subsonic:
        audio += subsonic * np.sin(2 * np.pi * 10.0 * t)
    if hum:
        audio += hum * np.sin(2 * np.pi * 50.0 * t)
    return audio.astype(np.float32)


def test_dc_correction_removes_per_channel_offset():
    source = np.column_stack((_tone(dc=0.005), _tone(dc=-0.003)))
    processed, info = _remove_dc(source)

    assert info["applied"] is True
    assert np.allclose(np.mean(processed, axis=0), 0.0, atol=1e-7)
    assert processed.shape == source.shape


def test_subsonic_filter_runs_only_when_rumble_is_measured():
    clean = _tone(subsonic=0.0)
    rumble = _tone(subsonic=0.12)

    clean_result, clean_info = _conditional_subsonic_filter(clean, 44100)
    rumble_result, rumble_info = _conditional_subsonic_filter(rumble, 44100)

    assert clean_info["applied"] is False
    assert np.array_equal(clean_result, clean)
    assert rumble_info["applied"] is True
    assert np.max(np.abs(rumble_result - rumble)) > 1e-5


def test_narrow_mains_hum_is_detected_without_touching_clean_audio():
    clean = _tone(hum=0.0)
    noisy = _tone(hum=0.06)

    clean_result, clean_info = _mains_hum_notches(clean, 44100)
    noisy_result, noisy_info = _mains_hum_notches(noisy, 44100)

    assert clean_info["applied"] is False
    assert np.array_equal(clean_result, clean)
    assert noisy_info["applied"] is True
    assert noisy_info["detected_hz"][0]["frequency_hz"] == 50.0
    assert np.max(np.abs(noisy_result - noisy)) > 1e-5


def test_every_wav_is_full_sample_processed_without_oto_gating_and_bank_is_mirrored(tmp_path):
    root = tmp_path / "EugeneRE"
    output = tmp_path / "EugeneRE_processed"

    _write_wav(root / "4. F3" / "vowel.wav", _tone())
    _write_wav(root / "5. G4" / "vowel.wav", _tone(f0=392.0))
    _write_wav(root / "6. B2" / "vowel.wav", _tone(f0=174.0))
    _write_wav(root / "7. L" / "la.wav", _tone())
    _write_wav(root / "8. V" / "va.wav", _tone())
    _write_wav(root / "9. TH" / "tha.wav", _tone())
    _write_wav(root / "10. R ENGLISH" / "r.wav", _tone())
    _write_wav(root / "11. CONSONANT RELEASES" / "release.wav", _tone())
    _write_wav(root / "12. BONUS PHONETIC EXTRAS" / "breath.wav", _tone())
    _write_wav(root / "root_sample.wav", _tone())

    # Deliberately ambiguous/duplicate OTO aliases must not block whole-sample processing.
    oto = root / "6. B2" / "oto.ini"
    oto.write_text(
        "vowel.wav=a @_B2,0,300,0,0,0\n"
        "vowel.wav=a @_B2,0,300,0,0,0\n",
        encoding="utf-8",
    )
    (root / "6. B2" / "vowel.wav.frq").write_bytes(b"frq")
    (root / "6. B2" / "vowel.wav.llsm").write_bytes(b"llsm")
    (root / "6. B2" / "vowel.wav.mrq").write_bytes(b"mrq")
    (root / "6. B2" / "vowel.wav.pmk").write_bytes(b"pmk")
    (root / "voicebank.ini").write_bytes(b"metadata")

    report = process_full_sample_bank(root, output, wet=1.0)

    assert report["files_found"] == 10
    assert report["summary"]["processed_full_duration"] == 10
    assert report["summary"]["fallback_copied"] == 0
    assert all(item["full_sample_coverage_percent"] == 100.0 for item in report["files"])
    assert (output / "6. B2" / "vowel.wav").is_file()
    assert (output / "11. CONSONANT RELEASES" / "release.wav").is_file()
    assert (output / "root_sample.wav").is_file()
    assert (output / "6. B2" / "oto.ini").read_bytes() == oto.read_bytes()
    assert (output / "voicebank.ini").read_bytes() == b"metadata"

    # Changed WAVs do not retain renderer maps which describe the old audio.
    assert not (output / "6. B2" / "vowel.wav.frq").exists()
    assert not (output / "6. B2" / "vowel.wav.llsm").exists()
    assert not (output / "6. B2" / "vowel.wav.mrq").exists()
    assert not (output / "6. B2" / "vowel.wav.pmk").exists()
    assert {
        "6. B2/vowel.wav.frq",
        "6. B2/vowel.wav.llsm",
        "6. B2/vowel.wav.mrq",
        "6. B2/vowel.wav.pmk",
    }.issubset(set(report["invalidated_analysis_sidecars"]))

    # The original input remains untouched.
    assert (root / "6. B2" / "vowel.wav.frq").read_bytes() == b"frq"
    assert (root / "6. B2" / "vowel.wav.llsm").read_bytes() == b"llsm"


def test_full_sample_preview_processes_the_complete_wav_not_oto_body_only(tmp_path):
    root = tmp_path / "EugeneRE"
    source = root / "6. B2" / "sample.wav"
    audio = _tone(f0=174.0, duration=1.2, dc=0.003)
    _write_wav(source, audio)
    # An OTO entry with a very narrow vowel body does not limit preview processing.
    (source.parent / "oto.ini").write_text(
        "sample.wav=a あ_B2,100,1080,0,20,20\n", encoding="utf-8"
    )

    preview = prepare_preview(root, "6. B2/sample.wav")

    assert preview["processing"]["full_sample_coverage_percent"] == 100.0
    assert preview["same_audio"] is False
    difference = np.abs(preview["processed"] - preview["original"])
    # There are changed samples well before and after the narrow OTO window.
    assert np.any(difference[:int(0.08 * 44100)] > 1e-7)
    assert np.any(difference[int(0.4 * 44100):int(0.8 * 44100)] > 1e-7)


def test_processing_error_falls_back_to_copy_and_does_not_abort_other_wavs(tmp_path):
    root = tmp_path / "EugeneRE"
    output = tmp_path / "output"
    good = root / "4. F3" / "good.wav"
    bad = root / "4. F3" / "broken.wav"
    _write_wav(good, _tone())
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"this is not a wav")

    report = process_full_sample_bank(root, output)

    assert report["summary"]["processed_full_duration"] == 1
    assert report["summary"]["fallback_copied"] == 1
    assert (output / "4. F3" / "good.wav").is_file()
    assert (output / "4. F3" / "broken.wav").read_bytes() == bad.read_bytes()
    broken = next(item for item in report["files"] if item["source"].endswith("broken.wav"))
    assert broken["status"] == "fallback_copy_error"
    assert broken["error"]


def test_dry_run_does_not_create_output_directory(tmp_path):
    root = tmp_path / "EugeneRE"
    _write_wav(root / "4. F3" / "v.wav", _tone())
    output = tmp_path / "unused_output"

    report = process_full_sample_bank(root, output, dry_run=True)

    assert report["dry_run"] is True
    assert report["files_found"] == 1
    assert not output.exists()
