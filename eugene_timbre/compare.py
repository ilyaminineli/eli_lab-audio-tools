"""Summarise precomputed subbank analysis JSON files."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

METRICS = ("spectral_centroid_hz", "spectral_flatness", "rms_dbfs")
BAND_KEYS = ("20_150_hz", "150_500_hz", "500_2000_hz", "2000_5000_hz", "5000_10000_hz")

def _median(values):
    vals = [float(v) for v in values if isinstance(v, (int,float)) and np.isfinite(v)]
    return float(np.median(vals)) if vals else None

def summarise_analysis(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    files = [item for item in obj.get("files", []) if isinstance(item, dict) and isinstance(item.get("analysis"), dict)]
    result = {"file": path.name, "count": len(files), "metrics": {}, "relative_band_energy_median": {}}
    for key in METRICS:
        result["metrics"][key] = _median(item["analysis"].get(key) for item in files)
    for key in BAND_KEYS:
        result["relative_band_energy_median"][key] = _median(
            item["analysis"].get("relative_band_energy", {}).get(key) for item in files)
    # The older reports use peak_dbfs for linear peak amplitudes.
    peak_vals = [item["analysis"].get("peak_dbfs") for item in files]
    if peak_vals and all(isinstance(v, (int,float)) and -0.01 <= v <= 1.01 for v in peak_vals):
        result["warning"] = "Legacy peak_dbfs appears to contain linear peak amplitude, not dBFS. Ignore it; v0.3 writes peak_amplitude and true peak_dbfs separately."
    return result

def compare_files(paths: list[Path]) -> dict:
    return {
        "report_type": "eugene_re_subbank_analysis_comparison",
        "note": "Medians are calculated over whole WAV files, not steady-vowel spectral envelopes. Use for broad subbank trends, not narrow notch-filter design.",
        "subbanks": [summarise_analysis(p) for p in paths],
    }
