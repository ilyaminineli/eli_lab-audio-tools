from pathlib import Path
import numpy as np
import soundfile as sf

from eugene_timbre.oto_analysis import (
    derive_timing, parse_oto, match_entries, analyse_oto_entry,
)
from eugene_timbre.timbre_analysis import TimbreAnalysisConfig
from eugene_timbre.bank_analysis import analyze_bank

def test_parse_cp932_oto_and_preserve_japanese_alias(tmp_path: Path):
    p = tmp_path / "oto.ini"
    p.write_bytes("vowel.wav=あ_B2,100,200,-400,150,80\n".encode("cp932"))
    result = parse_oto(p)
    assert result.encoding == "cp932"
    assert len(result.entries) == 1
    assert result.entries[0].alias == "あ_B2"
    assert not result.errors

def test_parse_multiple_aliases_for_same_wav(tmp_path: Path):
    p = tmp_path / "oto.ini"
    p.write_text(
        "triple.wav=a_B2,100,200,-400,150,80\n"
        "triple.wav=i_B2,700,220,-300,160,90\n",
        encoding="utf-8",
    )
    parsed = parse_oto(p)
    matches = match_entries(parsed.index(), "triple.wav", "triple.wav")
    assert len(matches) == 2
    assert [x.alias for x in matches] == ["a_B2", "i_B2"]

def test_negative_cutoff_uses_offset_minus_cutoff():
    from eugene_timbre.oto_analysis import OtoEntry
    entry = OtoEntry(1,"v.wav","a_B2",1000,200,-500,150,80,"utf-8")
    timing = derive_timing(entry, 3000)
    assert timing["markers_ms"]["usable_end"] == 1500
    assert timing["regions_ms"]["fixed_region"] == [1000,1200]
    assert timing["regions_ms"]["stretchable_region"] == [1200,1500]
    assert timing["stretchable_duration_ms"] == 300

def test_positive_cutoff_uses_duration_minus_cutoff():
    from eugene_timbre.oto_analysis import OtoEntry
    entry = OtoEntry(1,"v.wav","a_B2",1000,200,100,150,80,"utf-8")
    timing = derive_timing(entry, 3000)
    assert timing["markers_ms"]["usable_end"] == 2900

def test_oto_alias_analysis_marks_short_stretch_region(tmp_path: Path):
    from eugene_timbre.oto_analysis import OtoEntry
    sr = 44100
    t = np.arange(sr) / sr
    audio = (0.4*np.sin(2*np.pi*123*t)).astype(np.float32)
    entry = OtoEntry(1,"x.wav","a_B2",100,200,-210,120,80,"utf-8")
    report = analyse_oto_entry(entry,audio,sr,TimbreAnalysisConfig())
    assert report["region_analysis"]["fixed_region"]["deep_analysis_status"] == "analysed"
    assert report["region_analysis"]["stretchable_region"]["deep_analysis_status"] == "skipped_too_short"

def test_bank_analysis_attaches_oto_and_keeps_source_files(tmp_path: Path):
    root = tmp_path / "5. G4"
    root.mkdir()
    sr = 44100
    t = np.arange(sr) / sr
    wav = root / "vowel.wav"
    source = (0.3*np.sin(2*np.pi*392*t)).astype(np.float32)
    sf.write(wav, source, sr, subtype="PCM_16")
    oto = root / "oto.ini"
    oto.write_text("vowel.wav=あ_G4,100,200,-500,150,80\n", encoding="utf-8")
    output = tmp_path / "reports"
    report = analyze_bank(root, output, oto_file=None, use_oto=True)
    assert report["total_wav_files_analysed"] == 1
    assert report["oto_analysis"]["oto_entries_matched"] == 1
    data = __import__("json").loads((output / "g4_deep_analysis.json").read_text(encoding="utf-8"))
    assert data["files"][0]["oto"]["status"] == "matched"
    assert data["files"][0]["oto"]["entries"][0]["alias"] == "あ_G4"
    reread, read_sr = sf.read(wav, dtype="float32")
    assert read_sr == sr and np.array_equal(reread, sf.read(wav, dtype="float32")[0])
