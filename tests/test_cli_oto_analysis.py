import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from eugene_timbre.cli import main

def test_analyze_command_writes_single_oto_aware_json(tmp_path: Path, monkeypatch):
    bank = tmp_path / "5. G4"
    bank.mkdir()
    sr = 44100
    t = np.arange(sr) / sr
    audio = (0.35*np.sin(2*np.pi*392*t) + 0.12*np.sin(2*np.pi*784*t)).astype(np.float32)
    sf.write(bank / "vowel.wav", audio, sr, subtype="PCM_16")
    (bank / "oto.ini").write_text(
        "vowel.wav=あ_G4,100,200,-500,150,80\n",
        encoding="utf-8",
    )
    output = tmp_path / "G4_analysis.json"
    monkeypatch.setattr(sys, "argv", [
        "eugene-timbre", "analyze", str(bank), "-o", str(output)
    ])
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["report_type"] == "voicebank_deep_timbre_analysis"
    assert report["manifest"]["oto_analysis"]["oto_entries_matched"] == 1
    file_report = report["groups"]["G4"]["files"][0]
    assert file_report["oto"]["entries"][0]["alias"] == "あ_G4"
    assert file_report["oto"]["entries"][0]["timing"]["stretchable_duration_ms"] == 300.0
