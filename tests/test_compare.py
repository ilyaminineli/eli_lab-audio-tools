import json
from pathlib import Path
from eugene_timbre.compare import summarise_analysis

def test_legacy_peak_field_warning(tmp_path: Path):
    p=tmp_path/"B2_analysis.json"
    p.write_text(json.dumps({"files":[{"analysis":{"peak_dbfs":0.7,"spectral_centroid_hz":400,"spectral_flatness":0.001,"rms_dbfs":-12,"relative_band_energy":{"20_150_hz":0.1,"150_500_hz":0.5,"500_2000_hz":0.3,"2000_5000_hz":0.09,"5000_10000_hz":0.01}}}]}),encoding="utf-8")
    result=summarise_analysis(p)
    assert "linear peak amplitude" in result["warning"]
