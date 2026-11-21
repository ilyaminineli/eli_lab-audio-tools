from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf

from eugene_timbre.auto_routing import identify_family, classify_alias, classify_event
from eugene_timbre.oto_analysis import OtoEntry
from eugene_timbre.dsp import PeakingBand, TimbreProfile, apply_profile_masked
from eugene_timbre.processing_chain import build_oto_body_mask, process_auto


def test_auto_recognizes_pitch_and_phonetic_groups(tmp_path: Path):
    assert identify_family(Path("5. G4/vowel.wav"), Path("EugeneRE"))["pitch"] == "G4"
    assert identify_family(Path("6. B2/a.wav"), Path("EugeneRE"))["family"] == "pitch_layer"
    assert identify_family(Path("11. CONSONANT RELEASES/a.wav"), Path("EugeneRE"))["family"] == "consonant_release"
    assert identify_family(Path("7. L/la.wav"), Path("EugeneRE"))["group"] == "L"
    assert identify_family(Path("12. BONUS PHONETIC EXTRAS/cough 1.wav"), Path("EugeneRE"))["family"] == "phonetic_extra"


def test_alias_and_event_recognition():
    alias = classify_alias("a sh_G4")
    assert alias["canonical_alias"] == "a sh"
    assert alias["consonant_classes"] == ["fricative"]
    assert classify_alias("a ch_G4")["consonant_classes"] == ["affricate"]
    assert classify_alias("a しゃ_G4")["consonant_classes"] == ["fricative"]
    assert classify_alias("a にゃ_G4")["consonant_classes"] == ["nasal"]
    assert classify_alias("- あ_B2")["token_roles"] == ["boundary", "vowel"]
    assert classify_alias("うぉ_B2")["token_roles"] == ["vowel"]
    assert classify_alias("あL_G4")["token_roles"] == ["vowel"]
    assert classify_alias("a @_G4")["unclassified_tokens"] == ["@"]
    assert classify_event("cough 1.wav") == "cough"
    assert classify_event("laugh 2.wav") == "laughter"


def test_oto_mask_protects_fixed_region_and_keeps_body():
    entry = OtoEntry(1, "v.wav", "a あ_G4", 100, 200, -500, 150, 80, "utf-8")
    mask, report = build_oto_body_mask(
        frame_count=44100, sample_rate=44100, entries=[entry],
        options={"minimum_stretchable_ms": 60, "edge_fade_ms": 5, "marker_protection_ms": 0},
    )
    assert np.all(mask[:int(0.1*44100)] == 0)
    assert np.all(mask[int(0.1*44100):int(0.3*44100)] == 0)
    assert mask[int(0.4*44100)] > 0
    assert report["eligible_stretch_regions"] == 1


def test_process_auto_preserves_source_and_copies_oto_unchanged(tmp_path: Path):
    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr) / sr
    audio = (0.2*np.sin(2*np.pi*392*t) + 0.1*np.sin(2*np.pi*1450*t)).astype(np.float32)
    source = pitch / "v.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    oto_bytes = "v.wav=a b_B2,100,200,-500,150,80\n".encode("utf-8")
    (pitch / "oto.ini").write_bytes(oto_bytes)
    before, _ = sf.read(source, dtype="float32")
    output = tmp_path / "EugeneRE_processed"
    report = process_auto(root, output, wet=1.0)
    assert report["files_found"] == 1
    assert report["processed_files"] == 1
    assert report["files"][0]["status"] == "processed"
    assert (output / "6. B2" / "oto.ini").read_bytes() == oto_bytes
    after, after_sr = sf.read(output / "6. B2" / "v.wav", dtype="float32")
    assert after_sr == sr and after.shape == before.shape
    original_after, _ = sf.read(source, dtype="float32")
    assert np.array_equal(original_after, before)
    # The fixed OTO region must remain identical; the later vowel body should change.
    assert np.allclose(after[:int(0.29*sr)], before[:int(0.29*sr)], atol=1e-7)
    assert np.max(np.abs(after[int(0.4*sr):int(0.55*sr)] - before[int(0.4*sr):int(0.55*sr)])) > 1e-5
    assert report["files"][0]["family"] == "pitch_layer"



def test_masked_eq_changes_only_eligible_frames():
    sr = 44100
    t = np.arange(sr * 2, dtype=np.float32) / sr
    audio = (
        0.2 * np.sin(2 * np.pi * 390 * t)
        + 0.1 * np.sin(2 * np.pi * 1450 * t)
    ).astype(np.float32)
    mask = np.zeros(audio.shape[0], dtype=np.float32)
    mask[int(0.6 * sr):int(1.3 * sr)] = 1.0
    profile = TimbreProfile(
        name="test", description="masked test",
        bands=(PeakingBand(1450.0, 1.0, 0.7),),
    )
    restored = apply_profile_masked(
        audio, sr, profile, mask, wet=1.0, context_padding_ms=80.0
    )
    assert np.array_equal(restored[mask == 0], audio[mask == 0])
    assert np.max(np.abs(restored[mask > 0] - audio[mask > 0])) > 1e-5


def test_low_coverage_pitch_sample_is_reported_partial_and_not_processed(tmp_path: Path):
    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr * 2) / sr
    audio = (0.2 * np.sin(2 * np.pi * 392 * t)).astype(np.float32)
    source = pitch / "short_body.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    # 60 ms stretchable body in a 2 s sample; below the 5% preflight threshold.
    oto_bytes = "short_body.wav=a あ_B2,100,1840,0,20,20\n".encode("utf-8")
    (pitch / "oto.ini").write_bytes(oto_bytes)

    report = process_auto(root, tmp_path / "dry", dry_run=True)
    item = report["files"][0]
    assert item["status"] == "partial"
    assert item["mask"]["mask_coverage_percent"] < 5.0
    assert report["partial_files"] == 1
    assert report["ready_files"] == 0


def test_unmatched_oto_and_alias_tokens_are_audited(tmp_path: Path):
    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    audio = np.zeros(sr, dtype=np.float32)
    sf.write(pitch / "v.wav", audio, sr, subtype="PCM_16")
    oto_bytes = (
        "missing.wav=a あ_B2,100,200,-500,150,80\n"
        "v.wav=a @_B2,100,200,-500,150,80\n"
    ).encode("utf-8")
    (pitch / "oto.ini").write_bytes(oto_bytes)

    report = process_auto(root, tmp_path / "dry", dry_run=True)
    item = report["files"][0]
    assert item["status"] == "manual_review"
    assert item["alias_audit"]["unclassified_tokens"] == ["@"]
    oto_summary = report["oto_files"][0]
    assert oto_summary["orphan_entry_count"] == 1
