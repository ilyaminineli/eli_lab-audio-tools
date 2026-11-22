from pathlib import Path

from eugene_timbre.moresampler import (
    build_moresampler_arguments,
    encode_int12_curve,
    midi_note_name,
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
