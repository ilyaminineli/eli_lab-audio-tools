from pathlib import Path

from eugene_timbre.moresampler import (
    build_moresampler_arguments,
    encode_int12_curve,
    midi_note_name,
    _moreconfig,
    _prepare_moresampler_runtime,
)
from eugene_timbre.oto_analysis import OtoEntry


def _oto():
    return OtoEntry(
        line_number=7,
        wav="a_a.wav",
        alias="a あ",
        offset_ms=12.5,
        consonant_ms=90.0,
        cutoff_ms=-150.0,
        preutter_ms=70.0,
        overlap_ms=25.0,
        encoding="cp932",
    )


def test_openutau_int12_curve_encoding_matches_known_examples():
    assert encode_int12_curve([0]) == "AA"
    assert encode_int12_curve([0, 0]) == "AA#1#"
    assert encode_int12_curve([0, 0, 1]) == "AA#1#AB"
    assert encode_int12_curve([0, 0, 0, 1, 1]) == "AA#2#AB#1#"
    assert encode_int12_curve([-1]) == "//"


def test_midi_note_names_follow_openutau_convention():
    assert midi_note_name(60) == "C4"
    assert midi_note_name(69) == "A4"
    assert midi_note_name(71) == "B4"
    assert midi_note_name(72) == "C5"


def test_build_moresampler_arguments_follow_classic_ust_order(tmp_path):
    input_wav = tmp_path / "a_a.wav"
    output_wav = tmp_path / "render.wav"
    args = build_moresampler_arguments(
        input_wav,
        output_wav,
        _oto(),
        target_note=60,
        duration_ms=800.0,
        velocity=100,
        flags="Mt10Mb-5",
        volume=100,
        modulation=0,
        tempo=120.0,
    )

    assert args[0] == str(input_wav.resolve())
    assert args[1] == str(output_wav.resolve())
    assert args[2:5] == ["C4", "100", "Mt10Mb-5"]
    assert args[5:12] == ["12.5", "850", "90", "-150", "100", "0", "!120"]
    assert args[12].startswith("AA")
    assert "#167#" in args[12]


def test_moresampler_argument_builder_rejects_shell_metacharacters(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        build_moresampler_arguments(
            tmp_path / "input.wav", tmp_path / "output.wav", _oto(), flags='Mt10 & del *'
        )


def test_moresampler_voicebank_config_uses_measured_f0_only(tmp_path):
    voicebank_config = tmp_path / "moreconfig.txt"
    voicebank_config.write_text(
        "analysis-noise-reduction on\\n"
        "analysis-f0-min 70.0\\n"
        "analysis-f0-max 800.0\\n"
        "output-sampling-rate 22050\\n"
        "resampler-compatibility off\\n"
        "synthesis-duration-extension-method loop\\n",
        encoding="utf-8",
    )

    config = _moreconfig(174.0, voicebank_config)

    assert "analysis-noise-reduction on" in config
    assert "synthesis-duration-extension-method loop" in config
    assert "analysis-f0-range-from-path off" in config
    assert "analysis-f0-min 104.400" in config
    assert "analysis-f0-max 295.800" in config
    assert "output-sampling-rate" not in config
    assert "resampler-compatibility" not in config


def test_moresampler_runtime_isolated_global_config_preserves_installation(tmp_path):
    executable = tmp_path / "moresampler.exe"
    executable.write_bytes(b"mock-executable")
    dll = tmp_path / "vcomp140.dll"
    dll.write_bytes(b"mock-dll")
    global_config = tmp_path / "moreconfig.txt"
    original = (
        "# installed Moresampler config\\n"
        "synthesis-utau-style-normalization full\\n"
        "analysis-f0-min 60.0\\n"
        "output-sampling-rate 22050\\n"
        "resampler-compatibility off\\n"
        "dump-log-file C:/logs/moresampler.log\\n"
    )
    global_config.write_text(original, encoding="utf-8")

    isolated_exe, report = _prepare_moresampler_runtime(
        executable, tmp_path / "runtime", 44100
    )

    isolated_config = (isolated_exe.parent / "moreconfig.txt").read_text(encoding="utf-8")
    assert isolated_exe.is_file()
    assert (isolated_exe.parent / dll.name).read_bytes() == dll.read_bytes()
    assert global_config.read_text(encoding="utf-8") == original
    assert "synthesis-utau-style-normalization full" in isolated_config
    assert "resampler-compatibility on" in isolated_config
    assert "output-sampling-rate 44100" in isolated_config
    assert "output-bit-depth 16" in isolated_config
    assert "dump-log-file off" in isolated_config
    assert "analysis-f0-min" not in isolated_config
    assert report["copied_dlls"] == ["vcomp140.dll"]
