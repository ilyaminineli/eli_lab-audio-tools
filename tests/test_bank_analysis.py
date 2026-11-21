import numpy as np
import soundfile as sf
from pathlib import Path
from eugene_timbre.bank_analysis import analyze_bank

def test_bank_scan_groups_pitch_layers_and_other_folders(tmp_path: Path):
    root = tmp_path / "EugeneRE"
    sr = 44100
    t = np.arange(sr) / sr
    folders = {
        "6. B2": 123.5, "4. F3": 174.6, "5. G4": 392.0,
        "11. CONSONANT RELEASES": 230.0,
    }
    for folder, f0 in folders.items():
        directory = root / folder
        directory.mkdir(parents=True)
        audio = 0.4*np.sin(2*np.pi*f0*t) + 0.15*np.sin(2*np.pi*2*f0*t)
        sf.write(directory / "sample.wav", audio, sr, subtype="PCM_16")
    output = tmp_path / "deep_reports"
    report = analyze_bank(root, output)
    assert report["total_wav_files_analysed"] == 4
    assert set(report["groups"]) == {"B2", "F3", "G4", "CONSONANT_RELEASES"}
    assert (output / "b2_deep_analysis.json").exists()
    assert (output / "bank_deep_analysis.json").exists()

def test_bank_scan_can_filter_one_pitch(tmp_path: Path):
    root = tmp_path / "bank"
    t = np.arange(22050) / 44100
    for folder, f0 in (("6. B2", 123.5), ("4. F3", 174.6)):
        directory = root / folder
        directory.mkdir(parents=True)
        sf.write(directory / "x.wav", 0.4*np.sin(2*np.pi*f0*t), 44100)
    report = analyze_bank(root, tmp_path / "out", include_groups={"B2"})
    assert report["total_wav_files_analysed"] == 1
    assert list(report["groups"]) == ["B2"]
