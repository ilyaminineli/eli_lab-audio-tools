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

    # A normal wet run also copies this low-coverage source unchanged by default.
    wet_output = tmp_path / "wet"
    wet_report = process_auto(root, wet_output, wet=1.0)
    wet_item = wet_report["files"][0]
    assert wet_item["status"] == "partial"
    source_audio, _ = sf.read(source, dtype="float32")
    copied_audio, _ = sf.read(wet_output / "6. B2" / "short_body.wav", dtype="float32")
    assert np.array_equal(copied_audio, source_audio)
    assert (wet_output / "6. B2" / "oto.ini").read_bytes() == oto_bytes


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



def test_process_auto_reports_per_file_progress(tmp_path: Path):
    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(pitch / "v.wav", (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    (pitch / "oto.ini").write_text(
        "v.wav=a あ_B2,100,200,-500,150,80\n", encoding="utf-8"
    )

    updates = []
    process_auto(
        root, tmp_path / "output", wet=0.5,
        dry_run=True,
        progress=lambda done, total, label: updates.append((done, total, label)),
    )
    assert updates == [(1, 1, "6. B2/v.wav")]


def test_quick_restore_preview_uses_same_oto_aware_route_as_export(tmp_path: Path):
    from eugene_timbre.quick_restore_ui import prepare_preview

    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.08 * np.sin(2 * np.pi * 1450 * t)
    ).astype(np.float32)
    source = pitch / "v.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    (pitch / "oto.ini").write_text(
        "v.wav=a あ_B2,100,200,-500,150,80\n", encoding="utf-8"
    )

    preview = prepare_preview(root, "6. B2/v.wav", wet=0.5)

    assert preview["processing"]["full_sample_coverage_percent"] == 100.0
    assert preview["same_audio"] is False
    assert preview["processing"]["family"]["pitch"] == "B2"
    assert np.max(np.abs(preview["processed"] - preview["original"])) > 1e-5
    # OTO entries are no longer used to restrict EQ to a small body mask.
    assert np.any(np.abs(preview["processed"][:int(0.08 * sr)] - preview["original"][:int(0.08 * sr)]) > 1e-7)


def test_quick_restore_neutral_pitch_layer_preview_is_transparent(tmp_path: Path):
    from eugene_timbre.quick_restore_ui import prepare_preview

    root = tmp_path / "EugeneRE"
    pitch = root / "5. G4"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(pitch / "g4.wav", (0.15 * np.sin(2 * np.pi * 392 * t)).astype(np.float32), sr)
    (pitch / "oto.ini").write_text(
        "g4.wav=a あ_G4,100,200,-500,150,80\n", encoding="utf-8"
    )

    preview = prepare_preview(root, "5. G4/g4.wav")

    assert preview["processing"]["full_sample_coverage_percent"] == 100.0
    assert preview["processing"]["family"]["pitch"] == "G4"
    assert preview["processing"]["stages"]["dc_offset"]["applied"] is False
    assert preview["processing"]["stages"]["spectral_eq"]["adaptive_presence_band"] is not None
    assert preview["same_audio"] is False


def test_quick_restore_ui_module_compiles_and_is_the_default_cli_studio():
    ui_path = Path(__file__).parents[1] / "eugene_timbre" / "quick_restore_ui.py"
    ui_text = ui_path.read_text(encoding="utf-8")
    compile(ui_text, str(ui_path), "exec")
    assert "1. Scan bank and prepare preview" in ui_text
    assert "2. Create processed voicebank copy" in ui_text
    assert "DEFAULT_INPUT" in ui_text
    assert "DEFAULT_OUTPUT" in ui_text
    assert "def prepare_preview" in ui_text

    cli_path = Path(__file__).parents[1] / "eugene_timbre" / "cli.py"
    cli_text = cli_path.read_text(encoding="utf-8")
    assert "from .quick_restore_ui import main as studio_main" in cli_text
    assert 'sub.add_parser(\n        "studio-advanced"' in cli_text



def test_process_auto_reports_per_file_progress(tmp_path: Path):
    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(pitch / "v.wav", (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    (pitch / "oto.ini").write_text(
        "v.wav=a あ_B2,100,200,-500,150,80\n", encoding="utf-8"
    )

    updates = []
    process_auto(
        root, tmp_path / "output", wet=0.5,
        dry_run=True,
        progress=lambda done, total, label: updates.append((done, total, label)),
    )
    assert updates == [(1, 1, "6. B2/v.wav")]


def test_quick_restore_preview_uses_same_oto_aware_route_as_export(tmp_path: Path):
    from eugene_timbre.quick_restore import prepare_preview

    root = tmp_path / "EugeneRE"
    pitch = root / "6. B2"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.08 * np.sin(2 * np.pi * 1450 * t)
    ).astype(np.float32)
    source = pitch / "v.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    (pitch / "oto.ini").write_text(
        "v.wav=a あ_B2,100,200,-500,150,80\n", encoding="utf-8"
    )

    preview = prepare_preview(root, "6. B2/v.wav", wet=0.5)

    assert preview["processing"]["full_sample_coverage_percent"] == 100.0
    assert preview["same_audio"] is False
    assert preview["processing"]["family"]["pitch"] == "B2"
    assert np.max(np.abs(preview["processed"] - preview["original"])) > 1e-5
    # OTO entries no longer restrict the preview to a small body mask.
    assert np.any(np.abs(preview["processed"][:int(0.08 * sr)] - preview["original"][:int(0.08 * sr)]) > 1e-7)


def test_quick_restore_neutral_pitch_layer_preview_is_transparent(tmp_path: Path):
    from eugene_timbre.quick_restore import prepare_preview

    root = tmp_path / "EugeneRE"
    pitch = root / "5. G4"
    pitch.mkdir(parents=True)
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(pitch / "g4.wav", (0.15 * np.sin(2 * np.pi * 392 * t)).astype(np.float32), sr)
    (pitch / "oto.ini").write_text(
        "g4.wav=a あ_G4,100,200,-500,150,80\n", encoding="utf-8"
    )

    preview = prepare_preview(root, "5. G4/g4.wav")

    assert preview["processing"]["full_sample_coverage_percent"] == 100.0
    assert preview["processing"]["family"]["pitch"] == "G4"
    assert "spectral_eq" in preview["processing"]["stages"]
    assert preview["processed"].shape == preview["original"].shape


def test_quick_restore_ui_module_compiles_and_is_the_default_cli_studio():
    ui_path = Path(__file__).parents[1] / "eugene_timbre" / "quick_restore_ui.py"
    ui_text = ui_path.read_text(encoding="utf-8")
    compile(ui_text, str(ui_path), "exec")
    assert "1. Scan bank and prepare preview" in ui_text
    assert "2. Create processed voicebank copy" in ui_text
    assert "DEFAULT_INPUT" in ui_text
    assert "DEFAULT_OUTPUT" in ui_text
    assert "from .quick_restore import DEFAULT_WET, prepare_preview" in ui_text
    assert "class SpectrogramComparisonWidget" in ui_text
    assert "SPECTRAL DIFFERENCE" in ui_text
    assert "left_panel" in ui_text
    assert "right_panel" in ui_text
    assert "SPECTROGRAM COMPARISON" in ui_text
    assert "process_full_sample_bank" in ui_text

    cli_path = Path(__file__).parents[1] / "eugene_timbre" / "cli.py"
    cli_text = cli_path.read_text(encoding="utf-8")
    assert "from .quick_restore_ui import main as studio_main" in cli_text
    assert '"studio-advanced"' in cli_text
