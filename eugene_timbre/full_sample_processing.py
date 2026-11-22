"""Whole-file automatic conditioning for every WAV in a voicebank.

This path deliberately does not use OTO masks. Every WAV is read, processed
across its complete duration, and written to the mirrored output bank. Tonal
processing is family-aware; cleanup stages run per sample and report whether
they detected anything that needed correction.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

from .analysis import analyze_audio
from .auto_routing import identify_family
from .dsp import PeakingBand, TimbreProfile, apply_profile
from .io import read_wav, write_wav
from .processing_chain import _profile, load_processing_config


ANALYSIS_SIDE_CAR_SUFFIXES = {".frq", ".llsm", ".mrq", ".pmk"}


def _mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float64)
    return np.mean(data, axis=1) if data.ndim == 2 else data


def _normalised_power(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    mono = _mono(audio)
    if mono.size < 32:
        return np.array([0.0]), np.array([0.0])
    frequencies, power = welch(
        mono, fs=sample_rate,
        nperseg=min(32768, max(256, mono.size)),
        noverlap=None,
        scaling="spectrum",
    )
    return frequencies, power


def _remove_dc(audio: np.ndarray, threshold: float = 1e-7) -> tuple[np.ndarray, dict]:
    source = np.asarray(audio, dtype=np.float64)
    offsets = np.mean(source, axis=0) if source.ndim == 2 else float(np.mean(source))
    values = np.atleast_1d(np.asarray(offsets, dtype=float))
    if source.ndim == 1:
        output = source - (float(offsets) if abs(float(offsets)) > threshold else 0.0)
    else:
        correction = np.where(np.abs(values) > threshold, values, 0.0)
        output = source - correction[None, :]
    return output, {
        "applied": bool(np.any(np.abs(values) > threshold)),
        "offset_per_channel": [float(v) for v in values],
        "threshold": threshold,
    }


def _conditional_subsonic_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 22.0,
) -> tuple[np.ndarray, dict]:
    frequencies, power = _normalised_power(audio, sample_rate)
    if frequencies.size < 2 or sample_rate <= cutoff_hz * 4:
        return audio, {"applied": False, "reason": "Not enough bandwidth or sample length for subsonic analysis."}

    low = (frequencies >= 5.0) & (frequencies < cutoff_hz)
    vocal_low = (frequencies >= cutoff_hz) & (frequencies < 120.0)
    total = float(np.sum(power)) + 1e-20
    low_power = float(np.sum(power[low]))
    reference_power = float(np.sum(power[vocal_low]))
    low_share = low_power / total
    low_to_reference_db = 10.0 * np.log10((low_power + 1e-20) / (reference_power + 1e-20))

    # Only intervene when there is measurable energy below the vocal range.
    # The cutoff is kept below the fundamental of the documented Eugene layers.
    should_filter = (
        low_share >= 0.002
        and low_to_reference_db >= -24.0
        and np.sqrt(low_power) >= 1e-4
    )
    if not should_filter:
        return audio, {
            "applied": False,
            "cutoff_hz": cutoff_hz,
            "subsonic_power_share": float(low_share),
            "subsonic_vs_25_120_db": float(low_to_reference_db),
            "reason": "No meaningful subsonic rumble detected.",
        }

    sos = butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    x = np.asarray(audio, dtype=np.float64)
    min_length = 3 * (2 * len(sos) + 1)
    if x.shape[0] <= min_length:
        return audio, {
            "applied": False,
            "cutoff_hz": cutoff_hz,
            "subsonic_power_share": float(low_share),
            "subsonic_vs_25_120_db": float(low_to_reference_db),
            "reason": "Sample too short for stable zero-phase subsonic filtering.",
        }
    output = sosfiltfilt(sos, x, axis=0)
    return output, {
        "applied": True,
        "cutoff_hz": cutoff_hz,
        "order": 2,
        "subsonic_power_share": float(low_share),
        "subsonic_vs_25_120_db": float(low_to_reference_db),
        "reason": "Conditional 22 Hz rumble reduction applied to the complete sample.",
    }


def _mains_hum_notches(
    audio: np.ndarray,
    sample_rate: int,
    candidates_hz: tuple[float, ...] = (50.0, 60.0),
) -> tuple[np.ndarray, dict]:
    """Remove only unusually narrow 50/60 Hz components, never broad voice energy."""
    frequencies, power = _normalised_power(audio, sample_rate)
    detected = []
    output = np.asarray(audio, dtype=np.float64)
    if frequencies.size < 8 or output.shape[0] < sample_rate // 3:
        return audio, {"applied": False, "detected_hz": [], "reason": "Insufficient sample duration for stable hum detection."}

    resolution = float(frequencies[1] - frequencies[0])
    if resolution > 4.0:
        return audio, {"applied": False, "detected_hz": [], "reason": "Frequency resolution too coarse for narrow hum detection."}

    for target in candidates_hz:
        if target >= sample_rate / 2:
            continue
        idx = int(np.argmin(np.abs(frequencies - target)))
        if abs(float(frequencies[idx]) - target) > max(1.0, resolution):
            continue
        lo = max(0, idx - max(4, int(round(12.0 / resolution))))
        hi = min(power.size, idx + max(5, int(round(12.0 / resolution))))
        local = power[lo:hi]
        if local.size < 5:
            continue
        centre = idx - lo
        neighbours = np.concatenate((local[:max(0, centre - 2)], local[min(local.size, centre + 3):]))
        if neighbours.size == 0:
            continue
        baseline = float(np.median(neighbours)) + 1e-20
        prominence_db = 10.0 * np.log10((float(power[idx]) + 1e-20) / baseline)
        # A high prominence and narrow fixed mains bin are required; ordinary
        # voiced harmonics are not notched just because they are strong.
        if prominence_db >= 13.0:
            b, a = iirnotch(target, Q=32.0, fs=sample_rate)
            output = filtfilt(b, a, output, axis=0)
            detected.append({"frequency_hz": target, "prominence_db": float(prominence_db)})

    return output.astype(np.float32), {
        "applied": bool(detected),
        "detected_hz": detected,
        "reason": "Narrow mains hum detected and notched." if detected else "No isolated 50/60 Hz mains component detected.",
    }


def _transient_wet_curve(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a smooth 0.35..1.0 EQ wet curve, reducing EQ on sharp attacks."""
    mono = _mono(audio)
    if mono.size < int(sample_rate * 0.08):
        return np.ones(mono.size, dtype=np.float32)
    frame = max(256, int(round(sample_rate * 0.024)))
    hop = max(64, int(round(sample_rate * 0.006)))
    if mono.size < frame:
        return np.ones(mono.size, dtype=np.float32)

    starts = np.arange(0, mono.size - frame + 1, hop)
    window = np.hanning(frame)
    spectra = []
    for start in starts:
        spectra.append(np.abs(np.fft.rfft(mono[start:start + frame] * window)))
    matrix = np.asarray(spectra, dtype=np.float64)
    normalised = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    flux = np.zeros(matrix.shape[0], dtype=np.float64)
    if matrix.shape[0] > 1:
        flux[1:] = np.sum(np.maximum(normalised[1:] - normalised[:-1], 0.0) ** 2, axis=1)
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) + 1e-12
    threshold = max(float(np.percentile(flux, 88)), median + 3.5 * mad)
    attack = flux >= threshold
    frame_wet = np.ones(flux.size, dtype=np.float64)
    frame_wet[attack] = 0.35
    frame_wet = gaussian_filter1d(frame_wet, sigma=max(1.0, 0.012 * sample_rate / hop))
    centres = starts + frame // 2
    positions = np.arange(mono.size)
    wet = np.interp(positions, centres, frame_wet, left=frame_wet[0], right=frame_wet[-1])
    return np.clip(wet, 0.35, 1.0).astype(np.float32)


def _choose_pitch_profile(
    audio: np.ndarray,
    sample_rate: int,
    family: dict,
    config: dict,
) -> tuple[TimbreProfile | None, dict]:
    if family.get("family") != "pitch_layer":
        return None, {"applied": False, "reason": "No tonal pitch-layer EQ route for this sample family."}

    pitch = str(family.get("pitch", "")).upper()
    pitch_spec = (config.get("pitch_layers") or {}).get(pitch, {})
    base = _profile(pitch_spec, pitch_spec.get("name", f"{pitch}-automatic")) if pitch_spec.get("enabled") else None
    if base is None:
        base = TimbreProfile(
            name=f"{pitch}-adaptive-presence",
            description="Bounded per-sample presence shaping when measured upper-mid energy is low.",
        )

    summary = analyze_audio(audio, sample_rate)
    bands = list(base.bands)
    extra = None
    presence = float(summary["relative_band_energy"].get("2000_5000_hz", 0.0))
    centroid = float(summary.get("spectral_centroid_hz", 0.0))
    flatness = float(summary.get("spectral_flatness", 0.0))
    # This is a very small source-side presence correction, not reconstruction
    # of missing harmonics. Noise-like samples are not brightened.
    if presence < 0.005 and centroid < 1200.0 and flatness < 0.18:
        gain = float(np.clip((0.005 - presence) * 100.0, 0.15, 0.45))
        extra = PeakingBand(2800.0, gain, 0.65)
        bands.append(extra)

    profile = TimbreProfile(
        name=f"{pitch}-whole-file-voice-conditioning",
        description="Full-sample pitch-layer EQ with a transient-aware blend.",
        highpass_hz=base.highpass_hz,
        highpass_slope=base.highpass_slope,
        bands=tuple(bands),
    )
    details = {
        "applied": bool(profile.bands or profile.highpass_hz > 0),
        "pitch": pitch,
        "base_profile": base.name,
        "measured_2_5khz_energy": presence,
        "measured_spectral_centroid_hz": centroid,
        "measured_spectral_flatness": flatness,
        "adaptive_presence_band": (
            {"frequency_hz": extra.frequency_hz, "gain_db": extra.gain_db, "q": extra.q}
            if extra else None
        ),
        "profile": {
            "name": profile.name,
            "bands": [
                {"frequency_hz": band.frequency_hz, "gain_db": band.gain_db, "q": band.q}
                for band in profile.bands
            ],
        },
    }
    return profile, details


def process_full_sample(
    audio: np.ndarray,
    sample_rate: int,
    relative_path: Path | str,
    bank_root: Path,
    wet: float = 1.0,
    config: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Process the complete duration of one WAV; never uses OTO windows to skip it."""
    if not 0.0 <= wet <= 1.0 or not np.isfinite(wet):
        raise ValueError("wet must be finite and in [0, 1].")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2) or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Audio must be finite, non-empty mono or stereo data.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    cfg = config or load_processing_config()
    family = identify_family(Path(relative_path), bank_root)
    stages = {}

    x, stages["dc_offset"] = _remove_dc(x)
    x, stages["subsonic_rumble"] = _conditional_subsonic_filter(x, sample_rate)
    x, stages["mains_hum"] = _mains_hum_notches(x, sample_rate)

    profile, eq_details = _choose_pitch_profile(x, sample_rate, family, cfg)
    stages["spectral_eq"] = eq_details
    if profile and (profile.bands or profile.highpass_hz > 0) and wet > 0:
        eq_audio = apply_profile(x, sample_rate, profile, wet=1.0)
        transient_wet = _transient_wet_curve(x, sample_rate) * float(wet)
        mix = transient_wet if x.ndim == 1 else transient_wet[:, None]
        output = x.astype(np.float32) * (1.0 - mix) + eq_audio * mix
        stages["transient_preservation"] = {
            "applied": True,
            "attack_eq_wet_min": float(np.min(transient_wet)),
            "average_eq_wet": float(np.mean(transient_wet)),
            "reason": "Sharp attacks receive less tonal EQ while the full sample remains in the output chain.",
        }
    else:
        output = x.astype(np.float32)
        stages["transient_preservation"] = {
            "applied": False,
            "reason": "No tonal EQ route was active for this sample or wet was zero.",
        }

    output = np.asarray(output, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("Processing generated non-finite samples.")
    peak = float(np.max(np.abs(output)))
    if peak > 0.9995:
        output = output * (0.9995 / peak)
        stages["peak_safety"] = {"applied": True, "gain": 0.9995 / peak}
    else:
        stages["peak_safety"] = {"applied": False, "gain": 1.0}

    changed = not np.array_equal(
        np.asarray(audio, dtype=np.float32),
        output,
    )
    return output, {
        "family": family,
        "full_sample_coverage_percent": 100.0,
        "changed": bool(changed),
        "stages": stages,
        "processing_scope": "whole WAV; no OTO mask or alias coverage threshold",
    }


def _sidecar_matches(relative: Path, wav_paths: list[Path]) -> bool:
    suffix = relative.suffix.casefold()
    if suffix not in ANALYSIS_SIDE_CAR_SUFFIXES:
        return False
    sidecar_stem = relative.name.casefold()
    for wav_rel in wav_paths:
        if relative.parent != wav_rel.parent:
            continue
        candidates = {
            (wav_rel.name + suffix).casefold(),
            (wav_rel.stem + suffix).casefold(),
            (wav_rel.name + ".frq").casefold(),
        }
        if sidecar_stem in candidates:
            return True
    return False


def process_full_sample_bank(
    root: Path,
    output: Path,
    wet: float = 1.0,
    dry_run: bool = False,
    config_path: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Process every WAV over its full duration and mirror the complete bank."""
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from and not overlap the source bank.")
    if not np.isfinite(wet) or not 0.0 <= wet <= 1.0:
        raise ValueError("wet must be finite and between 0 and 1.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else load_processing_config()

    wav_paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".wav"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not wav_paths:
        raise ValueError(f"No WAV samples found in {root}.")

    reports = []
    changed_wavs: list[Path] = []
    family_counts: Counter = Counter()
    stage_counts: Counter = Counter()

    for index, source in enumerate(wav_paths, start=1):
        relative = source.relative_to(root)
        try:
            audio, sample_rate, metadata = read_wav(source)
            processed, details = process_full_sample(
                audio, sample_rate, relative, root, wet=wet, config=config
            )
            family = details["family"]
            family_counts[family["family"]] += 1
            for stage, stage_info in details["stages"].items():
                if stage_info.get("applied"):
                    stage_counts[stage] += 1
            if details["changed"]:
                changed_wavs.append(relative)

            if not dry_run:
                destination = output / relative
                write_wav(destination, processed, sample_rate, metadata["subtype"])

            reports.append({
                "source": relative.as_posix(),
                "output": relative.as_posix(),
                "sample_rate_hz": sample_rate,
                "frames": int(audio.shape[0]),
                "family": family["family"],
                "group": family["group"],
                "pitch": family.get("pitch"),
                "changed": details["changed"],
                "full_sample_coverage_percent": 100.0,
                "processing": details["stages"],
                "status": "processed_changed" if details["changed"] else "processed_unchanged",
                "error": None,
            })
        except Exception as exc:
            # A corrupt/unsupported WAV must not abort the rest of the bank.
            # It is retained byte-for-byte in the mirrored output and recorded.
            family = identify_family(relative, root)
            family_counts[family["family"]] += 1
            if not dry_run:
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            reports.append({
                "source": relative.as_posix(),
                "output": relative.as_posix(),
                "family": family["family"],
                "group": family["group"],
                "pitch": family.get("pitch"),
                "changed": False,
                "full_sample_coverage_percent": 0.0,
                "processing": {},
                "status": "fallback_copy_error",
                "error": str(exc),
            })
        if progress:
            progress(index, len(wav_paths), relative.as_posix())

    invalidated_sidecars = []
    copied_metadata = 0
    if not dry_run:
        for source in root.rglob("*"):
            if not source.is_file() or source.suffix.casefold() == ".wav":
                continue
            if output in source.resolve().parents:
                continue
            relative = source.relative_to(root)
            if _sidecar_matches(relative, changed_wavs):
                invalidated_sidecars.append(relative.as_posix())
                stale_target = output / relative
                if stale_target.exists():
                    stale_target.unlink()
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_metadata += 1

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
    changed_count = sum(1 for item in reports if item["changed"])
    fallback_count = sum(1 for item in reports if item["status"] == "fallback_copy_error")
    report = {
        "report_type": "eugene_re_full_sample_processing",
        "processing_version": "2.0",
        "source_root": str(root),
        "output_root": str(output),
        "dry_run": bool(dry_run),
        "wet": float(wet),
        "files_found": len(wav_paths),
        "files_written": 0 if dry_run else len(wav_paths),
        "processing_failures": fallback_count,
        "summary": {
            "wav_files": len(wav_paths),
            "processed_full_duration": len(wav_paths) - fallback_count,
            "changed": changed_count,
            "unchanged_after_analysis": len(wav_paths) - changed_count - fallback_count,
            "fallback_copied": fallback_count,
            "sidecars_invalidated": len(invalidated_sidecars),
            "metadata_files_copied": copied_metadata,
        },
        "family_counts": dict(family_counts),
        "processing_stage_counts": dict(stage_counts),
        "invalidated_analysis_sidecars": invalidated_sidecars,
        "metadata_policy": (
            "Non-WAV metadata copied byte-for-byte. Engine analysis sidecars associated with changed WAVs "
            "(.frq/.llsm/.mrq/.pmk) are omitted from the active output copy so the renderer can regenerate them."
        ),
        "oto_policy": "OTO files are read-only metadata; OTO timings do not gate or restrict full-sample processing.",
        "files": reports,
    }
    if not dry_run:
        (output / "processing_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report
