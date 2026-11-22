"""Pitch-folder-first analysis and non-destructive restoration tools.

This workflow deliberately avoids whole-bank alias classification. It discovers
pitch-layer folders, analyses their recordings, and applies a pitch-specific
profile only to the folders selected by the user.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import analyze_audio
from .dsp import PeakingBand, TimbreProfile, apply_profile, apply_profile_masked, describe_profile
from .io import read_wav, write_wav
from .oto_analysis import find_nearest_oto, match_entries, parse_oto
from .timbre_analysis import TimbreAnalysisConfig, analyze_timbre

PITCH_FOLDER_RE = re.compile(r"(?<![A-Z0-9])([A-G](?:#|B)?[0-8])(?![A-Z0-9])", re.IGNORECASE)
BAND_KEYS = (
    "20_150_hz",
    "150_500_hz",
    "500_2000_hz",
    "2000_5000_hz",
    "5000_10000_hz",
)
BAND_CENTRES_HZ = (110.0, 300.0, 950.0, 3000.0, 7000.0)
DEFAULT_BANDS = (
    {"name": "Low body", "frequency_hz": 110.0, "gain_db": 0.0, "q": 0.7},
    {"name": "Body", "frequency_hz": 300.0, "gain_db": 0.0, "q": 0.7},
    {"name": "Core", "frequency_hz": 950.0, "gain_db": 0.0, "q": 0.7},
    {"name": "Presence", "frequency_hz": 3000.0, "gain_db": 0.0, "q": 0.7},
    {"name": "Air", "frequency_hz": 7000.0, "gain_db": 0.0, "q": 0.7},
)
ProgressCallback = Callable[[int, int, str], None] | None


def pitch_name(folder: Path | str) -> str | None:
    """Return the pitch token contained in a conventional folder name."""
    match = PITCH_FOLDER_RE.search(Path(folder).name.upper())
    return match.group(1).upper() if match else None


def discover_pitch_layers(root: Path | str) -> list[dict]:
    """Find immediate child folders whose names contain a pitch token."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    found = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not child.is_dir():
            continue
        note = pitch_name(child)
        if note:
            found.append({
                "name": child.name,
                "pitch": note,
                "path": str(child),
                "wav_count": sum(1 for p in child.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav"),
            })
    # Also allow the user to point directly at a pitch folder.
    if not found and pitch_name(root):
        found.append({
            "name": root.name,
            "pitch": pitch_name(root),
            "path": str(root),
            "wav_count": sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav"),
        })
    return found


def default_profile(name: str = "neutral") -> dict:
    return {
        "name": name,
        "highpass_hz": 0.0,
        "highpass_slope": 2,
        "bands": [dict(item) for item in DEFAULT_BANDS],
    }


def _to_profile(spec: dict, name: str | None = None) -> TimbreProfile:
    bands = []
    for item in spec.get("bands", []):
        frequency = float(item.get("frequency_hz", 0.0))
        gain = float(item.get("gain_db", 0.0))
        q = float(item.get("q", 0.7))
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Every EQ band needs a positive finite centre frequency.")
        if not np.isfinite(gain) or not -12.0 <= gain <= 12.0:
            raise ValueError("EQ gains must be finite and between -12 and +12 dB.")
        if not np.isfinite(q) or not 0.1 <= q <= 10.0:
            raise ValueError("EQ Q must be finite and between 0.1 and 10.")
        bands.append(PeakingBand(frequency, gain, q))
    highpass = float(spec.get("highpass_hz", 0.0))
    slope = int(spec.get("highpass_slope", 2))
    if not np.isfinite(highpass) or highpass < 0:
        raise ValueError("High-pass frequency must be finite and non-negative.")
    if slope not in (2, 4, 6):
        raise ValueError("High-pass slope must be 2, 4 or 6.")
    return TimbreProfile(
        name=name or str(spec.get("name", "custom")),
        description=str(spec.get("description", "Pitch-folder restoration profile.")),
        highpass_hz=highpass,
        highpass_slope=slope,
        bands=tuple(bands),
    )


def _median(values: list[float]) -> float | None:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.median(clean)) if clean.size else None


def analyze_pitch_folder(
    folder: Path | str,
    config: TimbreAnalysisConfig | None = None,
    progress: ProgressCallback = None,
) -> dict:
    """Analyse only WAVs below one selected pitch folder."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir() or not pitch_name(folder):
        raise ValueError(f"Select a pitch-layer folder, not a whole bank: {folder}")
    paths = sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav"),
        key=lambda p: p.as_posix().casefold(),
    )
    if not paths:
        raise ValueError(f"No WAV samples found in {folder}")
    cfg = config or TimbreAnalysisConfig()
    rows, failures = [], []
    band_values = {key: [] for key in BAND_KEYS}
    f0_values, centroid_values, flatness_values, hnr_values = [], [], [], []
    envelopes, envelope_freqs = [], None

    for index, path in enumerate(paths, start=1):
        try:
            audio, sample_rate, metadata = read_wav(path)
            basic = analyze_audio(audio, sample_rate)
            deep = analyze_timbre(audio, sample_rate, cfg)
            for key in BAND_KEYS:
                value = basic["relative_band_energy"].get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    band_values[key].append(float(value))
            for source, target in (
                (deep.get("f0_hz", {}).get("median"), f0_values),
                (deep.get("spectral_centroid_hz_voiced_frames", {}).get("median"), centroid_values),
                (deep.get("spectral_flatness_voiced_frames", {}).get("median"), flatness_values),
                (deep.get("harmonic_to_residual_ratio_db", {}).get("median"), hnr_values),
            ):
                if isinstance(source, (int, float)) and np.isfinite(source):
                    target.append(float(source))
            envelope = deep.get("spectral_envelope")
            if envelope and envelope.get("frequency_hz") and envelope.get("median_relative_db"):
                freq = np.asarray(envelope["frequency_hz"], dtype=float)
                values = np.asarray(envelope["median_relative_db"], dtype=float)
                if freq.size == values.size and freq.size >= 2:
                    grid = np.geomspace(100.0, min(10000.0, sample_rate / 2.0), 72)
                    envelopes.append(np.interp(np.log10(grid), np.log10(freq), values))
                    envelope_freqs = grid
            rows.append({
                "file": path.relative_to(folder).as_posix(),
                "sample_rate_hz": sample_rate,
                "frames": metadata["frames"],
                "duration_seconds": basic["duration_seconds"],
                "rms_dbfs": basic["rms_dbfs"],
                "spectral_centroid_hz": basic["spectral_centroid_hz"],
                "f0_median_hz": deep.get("f0_hz", {}).get("median"),
                "voiced_frame_fraction": deep.get("voiced_frame_fraction"),
                "relative_band_energy": basic["relative_band_energy"],
            })
        except Exception as exc:
            failures.append({"file": path.relative_to(folder).as_posix(), "error": str(exc)})
        if progress:
            progress(index, len(paths), path.name)

    if not rows:
        raise RuntimeError(f"Every WAV failed analysis in {folder}")
    envelope_median = np.median(np.vstack(envelopes), axis=0).tolist() if envelopes else []
    envelope_p10 = np.percentile(np.vstack(envelopes), 10, axis=0).tolist() if envelopes else []
    envelope_p90 = np.percentile(np.vstack(envelopes), 90, axis=0).tolist() if envelopes else []
    return {
        "report_type": "eugene_pitch_layer_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folder": str(folder),
        "pitch": pitch_name(folder),
        "sample_count": len(paths),
        "analysed_count": len(rows),
        "failed_count": len(failures),
        "summary": {
            "f0_median_hz": _median(f0_values),
            "spectral_centroid_median_hz": _median(centroid_values),
            "spectral_flatness_median": _median(flatness_values),
            "harmonic_to_residual_median_db": _median(hnr_values),
            "relative_band_energy_median": {key: _median(values) for key, values in band_values.items()},
        },
        "spectral_envelope": {
            "frequency_hz": envelope_freqs.tolist() if envelope_freqs is not None else [],
            "median_relative_db": envelope_median,
            "p10_relative_db": envelope_p10,
            "p90_relative_db": envelope_p90,
            "sample_count": len(envelopes),
        },
        "files": rows,
        "failures": failures,
        "interpretation": (
            "Measurements describe this recorded pitch layer; they are not a naturalness score. "
            "Judge changes using matched phrases rendered by the target resampler."
        ),
    }


def suggest_reference_match(
    source_analysis: dict,
    reference_analysis: dict,
    strength: float = 0.6,
    max_gain_db: float = 2.5,
) -> dict:
    """Suggest broad balance corrections to move one layer toward a reference.

    This matches median octave-ish energy bands, not vocal-tract formants. It is
    intentionally bounded and must be auditioned before it is used.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")
    if not 0.0 < max_gain_db <= 6.0:
        raise ValueError("max_gain_db must be in (0, 6].")
    source = source_analysis.get("summary", {}).get("relative_band_energy_median", {})
    reference = reference_analysis.get("summary", {}).get("relative_band_energy_median", {})
    bands = []
    for key, centre in zip(BAND_KEYS, BAND_CENTRES_HZ):
        current = source.get(key)
        target = reference.get(key)
        gain = 0.0
        if (
            isinstance(current, (int, float)) and isinstance(target, (int, float))
            and current > 1e-8 and target > 1e-8
        ):
            gain = 10.0 * np.log10(target / current) * strength
            gain = float(np.clip(gain, -max_gain_db, max_gain_db))
        bands.append({
            "name": key.replace("_hz", "").replace("_", "–") + " Hz",
            "frequency_hz": centre,
            "gain_db": round(gain, 2),
            "q": 0.7,
        })
    return {
        "name": f"match-{source_analysis.get('pitch', 'pitch')}-to-{reference_analysis.get('pitch', 'reference')}",
        "description": "Bounded broad-band balance suggestion. Not a formant or naturalness correction.",
        "highpass_hz": 0.0,
        "highpass_slope": 2,
        "bands": bands,
        "reference_pitch": reference_analysis.get("pitch"),
        "match_strength": strength,
        "max_gain_db": max_gain_db,
    }


def _oto_mask_for_file(path: Path, root: Path, sample_count: int, sample_rate: int) -> tuple[np.ndarray | None, str | None]:
    # Keep the legacy whole-bank router out of the normal pitch-only import path.
    from .processing_chain import build_oto_body_mask

    oto_path = find_nearest_oto(path, root)
    if oto_path is None:
        return None, "No oto.ini found for this sample."
    parsed = parse_oto(oto_path)
    try:
        relative = path.relative_to(oto_path.parent).as_posix()
    except ValueError:
        relative = path.name
    entries = match_entries(parsed.index(), relative, path.name)
    if not entries:
        return None, f"No matching OTO entries in {oto_path.name}."
    mask, details = build_oto_body_mask(
        sample_count, sample_rate, entries,
        {"minimum_stretchable_ms": 60.0, "marker_protection_ms": 8.0, "edge_fade_ms": 12.0},
    )
    if not np.any(mask > 0):
        return None, "No usable stretchable OTO body region for this WAV."
    return mask, None


def _profile_for_sample_rate(profile: TimbreProfile, sample_rate: int) -> TimbreProfile:
    """Drop EQ bands that lie at or above a file's Nyquist frequency."""
    nyquist = sample_rate / 2.0
    return TimbreProfile(
        name=profile.name,
        description=profile.description,
        highpass_hz=profile.highpass_hz if 0 < profile.highpass_hz < nyquist else 0.0,
        highpass_slope=profile.highpass_slope,
        bands=tuple(band for band in profile.bands if band.frequency_hz < nyquist),
    )


def process_pitch_folder(
    source: Path | str,
    output_root: Path | str,
    profile_spec: dict,
    wet: float = 1.0,
    protect_oto: bool = False,
    progress: ProgressCallback = None,
) -> dict:
    """Process one selected pitch folder into a separate output tree."""
    source = Path(source).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source.is_dir() or not pitch_name(source):
        raise ValueError(f"Input must be one pitch-layer folder: {source}")
    destination = output_root / source.name
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Output must be separate from and not overlap the source pitch folder.")
    if not np.isfinite(wet) or not 0.0 <= wet <= 1.0:
        raise ValueError("wet must be finite and between 0 and 1.")
    profile = _to_profile(profile_spec, name=str(profile_spec.get("name", source.name)))
    wavs = sorted(
        (p for p in source.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav"),
        key=lambda p: p.as_posix().casefold(),
    )
    if not wavs:
        raise ValueError(f"No WAV samples found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    reports, counts = [], {"processed": 0, "unchanged": 0, "skipped_oto": 0, "failed": 0}

    for index, path in enumerate(wavs, start=1):
        relative = path.relative_to(source)
        target = destination / relative
        try:
            audio, sample_rate, metadata = read_wav(path)
            output_audio = audio.copy()
            effective_profile = _profile_for_sample_rate(profile, sample_rate)
            mask_coverage = 100.0
            status, reason = "unchanged", "Profile is neutral or produced no measurable sample change."
            if wet > 0:
                if protect_oto:
                    mask, skip_reason = _oto_mask_for_file(path, source, len(audio), sample_rate)
                    if mask is None:
                        skip_reason = skip_reason or "No usable OTO body window."
                        counts["skipped_oto"] += 1
                        status, reason = "skipped_oto", skip_reason
                        shutil.copy2(path, target)
                        before = analyze_audio(audio, sample_rate)
                        reports.append({
                            "file": relative.as_posix(), "status": status, "reason": reason,
                            "mask_coverage_percent": 0.0, "before": before, "after": before,
                            "peak_abs_difference": 0.0, "changed_fraction": 0.0,
                        })
                        if progress:
                            progress(index, len(wavs), path.name)
                        continue
                    mask_coverage = float(mask.mean() * 100.0)
                    output_audio = apply_profile_masked(audio, sample_rate, effective_profile, mask, wet=wet)
                else:
                    output_audio = apply_profile(audio, sample_rate, effective_profile, wet=wet)
            difference = np.asarray(output_audio, dtype=np.float64) - np.asarray(audio, dtype=np.float64)
            peak_difference = float(np.max(np.abs(difference))) if difference.size else 0.0
            changed_fraction = float(np.mean(np.abs(difference) > 1e-7)) if difference.size else 0.0
            before = analyze_audio(audio, sample_rate)
            after = analyze_audio(output_audio, sample_rate)
            if peak_difference > 1e-7:
                write_wav(target, output_audio, sample_rate, metadata["subtype"])
                counts["processed"] += 1
                status = "processed"
                reason = "Profile applied; output measurements recorded."
            else:
                shutil.copy2(path, target)
                counts["unchanged"] += 1
            reports.append({
                "file": relative.as_posix(),
                "status": status,
                "reason": reason,
                "mask_coverage_percent": mask_coverage,
                "bands_applied": [band.frequency_hz for band in effective_profile.bands],
                "before": before,
                "after": after,
                "peak_abs_difference": peak_difference,
                "changed_fraction": changed_fraction,
                "rms_delta_db": after["rms_dbfs"] - before["rms_dbfs"],
                "centroid_delta_hz": after["spectral_centroid_hz"] - before["spectral_centroid_hz"],
                "band_energy_delta": {
                    key: after["relative_band_energy"].get(key, 0.0) - before["relative_band_energy"].get(key, 0.0)
                    for key in BAND_KEYS
                },
            })
        except Exception as exc:
            counts["failed"] += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not target.exists():
                shutil.copy2(path, target)
            reports.append({
                "file": relative.as_posix(), "status": "failed", "reason": str(exc),
                "peak_abs_difference": None, "changed_fraction": None,
            })
        if progress:
            progress(index, len(wavs), path.name)

    # Preserve pitch-folder metadata and subdirectories, including oto.ini, unchanged.
    for item in source.rglob("*"):
        if not item.is_file() or item.suffix.casefold() == ".wav":
            continue
        target = destination / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)

    report = {
        "report_type": "eugene_pitch_restoration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(source),
        "output_folder": str(destination),
        "pitch": pitch_name(source),
        "wet": float(wet),
        "profile": describe_profile(profile),
        "protect_oto_fixed_regions": bool(protect_oto),
        "summary": {"files_found": len(wavs), **counts},
        "files": reports,
        "notes": [
            "Only this selected pitch folder was processed.",
            "Non-WAV metadata, including oto.ini, was copied byte-for-byte.",
            "Spectral measurements are diagnostics, not a naturalness score.",
            "Compare matched resampler renders at matched loudness before adopting the profile.",
        ],
    }
    (destination / "pitch_restoration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def preview_sample(
    path: Path | str,
    profile_spec: dict,
    wet: float = 1.0,
    protect_oto: bool = False,
    source_root: Path | str | None = None,
) -> tuple[np.ndarray, int, str]:
    """Return processed audio for audition without writing to the source bank."""
    path = Path(path).expanduser().resolve()
    audio, sample_rate, _ = read_wav(path)
    profile = _profile_for_sample_rate(_to_profile(profile_spec), sample_rate)
    if protect_oto:
        oto_root = Path(source_root).expanduser().resolve() if source_root else path.parent
        mask, reason = _oto_mask_for_file(path, oto_root, len(audio), sample_rate)
        if mask is None:
            raise ValueError(reason or "No usable OTO body window.")
        output = apply_profile_masked(audio, sample_rate, profile, mask, wet=wet)
    else:
        output = apply_profile(audio, sample_rate, profile, wet=wet)
    return output, sample_rate, path.name
