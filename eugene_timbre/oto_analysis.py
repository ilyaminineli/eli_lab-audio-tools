"""Read-only OTO parsing and timing-aware sample-region analysis."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
import math

import numpy as np

from .analysis import analyze_audio
from .timbre_analysis import TimbreAnalysisConfig, analyze_timbre

@dataclass(frozen=True)
class OtoEntry:
    line_number: int
    wav: str
    alias: str
    offset_ms: float
    consonant_ms: float
    cutoff_ms: float
    preutter_ms: float
    overlap_ms: float
    encoding: str

@dataclass
class OtoParseResult:
    path: Path
    encoding: str
    entries: list[OtoEntry]
    errors: list[dict]

    def index(self) -> dict[str, list[OtoEntry]]:
        result: dict[str, list[OtoEntry]] = {}
        for entry in self.entries:
            key = normalize_wav_key(entry.wav)
            result.setdefault(key, []).append(entry)
        return result

def normalize_wav_key(value: str) -> str:
    return PurePosixPath(value.strip().replace("\\", "/")).as_posix().casefold()

def decode_oto(data: bytes) -> tuple[str, str]:
    """Decode common UTAU oto encodings without corrupting Japanese aliases."""
    for encoding in ("utf-8-sig", "cp932", "shift_jis", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"

def parse_oto(path: Path) -> OtoParseResult:
    """Parse wav=alias,offset,consonant,cutoff,preutter,overlap lines."""
    path = Path(path)
    data = path.read_bytes()
    text, encoding = decode_oto(data)
    entries: list[OtoEntry] = []
    errors: list[dict] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip("\ufeff\r\n")
        if not line.strip() or line.lstrip().startswith((";", "#")):
            continue
        if "=" not in line:
            errors.append({"line_number": line_number, "reason": "missing_equals"})
            continue
        wav, payload = line.split("=", 1)
        fields = payload.split(",")
        if len(fields) != 6:
            errors.append({
                "line_number": line_number,
                "reason": "expected_alias_and_five_timing_values",
                "field_count": len(fields),
            })
            continue
        try:
            values = [float(value.strip()) for value in fields[1:]]
        except ValueError:
            errors.append({"line_number": line_number, "reason": "non_numeric_timing_value"})
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append({"line_number": line_number, "reason": "non_finite_timing_value"})
            continue
        if not wav.strip():
            errors.append({"line_number": line_number, "reason": "empty_wav_filename"})
            continue
        entries.append(OtoEntry(
            line_number=line_number,
            wav=wav.strip(),
            alias=fields[0],
            offset_ms=values[0],
            consonant_ms=values[1],
            cutoff_ms=values[2],
            preutter_ms=values[3],
            overlap_ms=values[4],
            encoding=encoding,
        ))
    return OtoParseResult(path=path, encoding=encoding, entries=entries, errors=errors)

def find_nearest_oto(wav_path: Path, root: Path) -> Path | None:
    """Find the nearest oto.ini from the WAV's folder up to the selected bank root."""
    wav_path, root = Path(wav_path).resolve(), Path(root).resolve()
    current = wav_path.parent
    while current == root or root in current.parents:
        try:
            for child in current.iterdir():
                if child.is_file() and child.name.casefold() == "oto.ini":
                    return child
        except OSError:
            pass
        if current == root:
            break
        current = current.parent
    return None

def match_entries(index: dict[str, list[OtoEntry]],
                  wav_relative_to_oto: str, wav_name: str) -> list[OtoEntry]:
    relative_key = normalize_wav_key(wav_relative_to_oto)
    name_key = normalize_wav_key(wav_name)
    return index.get(relative_key, index.get(name_key, []))

def derive_timing(entry: OtoEntry, duration_ms: float) -> dict:
    """Derive OTO marker coordinates and analysis regions in milliseconds.

    This follows OpenUtau's displayed cutoff coordinate:
    cutoff >= 0: usable end = sample duration - cutoff
    cutoff < 0:  usable end = offset - cutoff

    Original OTO values are retained separately and never modified.
    """
    offset = entry.offset_ms
    fixed_end = offset + entry.consonant_ms
    usable_end = (
        duration_ms - entry.cutoff_ms
        if entry.cutoff_ms >= 0
        else offset - entry.cutoff_ms
    )
    preutter_marker = offset + entry.preutter_ms
    overlap_marker = offset + entry.overlap_ms

    clamped_offset = float(np.clip(offset, 0.0, duration_ms))
    clamped_end = float(np.clip(usable_end, 0.0, duration_ms))
    fixed_end_clamped = float(np.clip(fixed_end, clamped_offset, duration_ms))
    fixed_region_end = min(fixed_end_clamped, clamped_end)
    stretch_start = min(max(fixed_end, clamped_offset), clamped_end)

    warnings = []
    if offset < 0 or offset > duration_ms:
        warnings.append("offset_outside_sample")
    if entry.consonant_ms < 0:
        warnings.append("negative_consonant_length")
    if entry.preutter_ms < 0 or entry.preutter_ms > entry.consonant_ms:
        warnings.append("preutter_marker_outside_fixed_region")
    if entry.overlap_ms < 0 or entry.overlap_ms > entry.consonant_ms:
        warnings.append("overlap_marker_outside_fixed_region")
    if usable_end < offset:
        warnings.append("usable_end_before_offset")
    if usable_end < fixed_end:
        warnings.append("usable_end_before_fixed_region_end")
    if usable_end > duration_ms:
        warnings.append("usable_end_after_sample")

    return {
        "source_duration_ms": float(duration_ms),
        "oto_values_ms": {
            "offset": entry.offset_ms,
            "consonant": entry.consonant_ms,
            "cutoff": entry.cutoff_ms,
            "preutter": entry.preutter_ms,
            "overlap": entry.overlap_ms,
        },
        "markers_ms": {
            "offset": offset,
            "fixed_region_end": fixed_end,
            "preutter": preutter_marker,
            "overlap": overlap_marker,
            "usable_end": usable_end,
        },
        "regions_ms": {
            "before_entry_window": [0.0, clamped_offset],
            "source_window": [clamped_offset, clamped_end],
            "fixed_region": [clamped_offset, fixed_region_end],
            "stretchable_region": [stretch_start, clamped_end],
            "after_entry_window": [clamped_end, float(duration_ms)],
        },
        "stretchable_duration_ms": max(0.0, clamped_end - stretch_start),
        "warnings": warnings,
    }

def _compact_timbre(audio: np.ndarray, sample_rate: int,
                    config: TimbreAnalysisConfig) -> dict:
    duration_ms = audio.shape[0] * 1000.0 / sample_rate
    result = {
        "duration_ms": float(duration_ms),
        "frames": int(audio.shape[0]),
        "basic": None,
        "deep_analysis_status": "skipped_too_short",
        "reason": None,
        "timbre": None,
    }
    if audio.shape[0] == 0:
        result["reason"] = "empty_region"
        return result
    if duration_ms >= 20.0:
        result["basic"] = analyze_audio(audio, sample_rate)
    else:
        result["reason"] = "Region shorter than 20 ms; spectral summary omitted."
        return result
    if duration_ms < 60.0:
        result["reason"] = (
            f"Region is {duration_ms:.1f} ms; harmonic/envelope analysis requires at least 60 ms."
        )
        return result

    frame_ms = min(config.frame_ms, max(45.0, duration_ms * 0.8))
    hop_ms = min(config.hop_ms, frame_ms / 2.0)
    local_config = replace(config, frame_ms=frame_ms, hop_ms=hop_ms)
    deep = analyze_timbre(audio, sample_rate, local_config)
    deep.pop("frame_tracks", None)
    result["deep_analysis_status"] = "analysed"
    result["reason"] = None
    result["timbre"] = deep
    return result

def analyse_oto_entry(entry: OtoEntry, audio: np.ndarray, sample_rate: int,
                      config: TimbreAnalysisConfig) -> dict:
    """Analyse the OTO-selected sample window and its fixed/stretchable portions."""
    mono_or_multi = np.asarray(audio)
    duration_ms = mono_or_multi.shape[0] * 1000.0 / sample_rate
    timing = derive_timing(entry, duration_ms)
    region_bounds = timing["regions_ms"]
    analyses = {}
    for region_name in ("source_window", "fixed_region", "stretchable_region"):
        start_ms, end_ms = region_bounds[region_name]
        start_frame = int(round(start_ms * sample_rate / 1000.0))
        end_frame = int(round(end_ms * sample_rate / 1000.0))
        start_frame = max(0, min(start_frame, mono_or_multi.shape[0]))
        end_frame = max(start_frame, min(end_frame, mono_or_multi.shape[0]))
        analyses[region_name] = _compact_timbre(
            mono_or_multi[start_frame:end_frame], sample_rate, config
        )
        analyses[region_name]["start_ms"] = float(start_ms)
        analyses[region_name]["end_ms"] = float(end_ms)

    return {
        "oto_line_number": entry.line_number,
        "wav": entry.wav,
        "alias": entry.alias,
        "oto_ini_encoding": entry.encoding,
        "timing": timing,
        "region_analysis": analyses,
    }

def entry_to_dict(entry: OtoEntry) -> dict:
    return asdict(entry)
