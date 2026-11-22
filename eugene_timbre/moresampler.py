"""Isolated A/B phone rendering through the Moresampler UTAU interface.

The runner follows the classic UTAU resampler argument contract used by
OpenUtau's ExeResampler. Each source variant is staged in its own temporary
folder so Moresampler analysis/cache files never touch the user's voicebank.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from .analysis import analyze_audio
from .io import read_wav
from .oto_analysis import OtoEntry
from .timbre_analysis import analyze_timbre


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SAFE_FLAGS = re.compile(r"^[A-Za-z0-9/+\-]*$")


def midi_note_name(note: int) -> str:
    """Convert MIDI note number to the note spelling used by OpenUtau."""
    if not 0 <= int(note) <= 127:
        raise ValueError("MIDI note must be in the range 0–127.")
    return f"{NOTE_NAMES[int(note) % 12]}{int(note) // 12 - 1}"


def encode_int12_curve(values: list[int]) -> str:
    """Encode pitch deviations using OpenUtau's Base64EncodeInt12 + RLE format."""
    encoded = []
    for value in values:
        value = int(value)
        if not -2048 <= value <= 2047:
            raise ValueError("Pitch deviations must be in the range -2048..2047 cents.")
        unsigned = value if value >= 0 else value + 4096
        encoded.append(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(unsigned >> 6) & 0x3F]
            + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[unsigned & 0x3F]
        )
    result = []
    last = ""
    repeats = 0
    for item in encoded:
        if last == item:
            repeats += 1
        elif repeats == 0:
            result.append(item)
        else:
            result.append(f"#{repeats}#")
            repeats = 0
            result.append(item)
        last = item
    if repeats:
        result.append(f"#{repeats}#")
    return "".join(result)


def build_moresampler_arguments(
    input_wav: Path | str,
    output_wav: Path | str,
    oto: OtoEntry,
    *,
    target_note: int = 60,
    duration_ms: float = 800.0,
    velocity: int = 100,
    flags: str = "",
    volume: int = 100,
    modulation: int = 0,
    tempo: float = 120.0,
    pitch_curve_cents: list[int] | None = None,
) -> list[str]:
    """Build the exact classic resampler argument vector used by OpenUtau."""
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Duration must be a positive finite number of milliseconds.")
    if not np.isfinite(tempo) or tempo <= 0:
        raise ValueError("Tempo must be a positive finite BPM value.")
    if not 0 <= velocity <= 200:
        raise ValueError("Velocity must be between 0 and 200.")
    if not 0 <= volume <= 200:
        raise ValueError("Volume must be between 0 and 200.")
    if not 0 <= modulation <= 200:
        raise ValueError("Modulation must be between 0 and 200.")
    if not SAFE_FLAGS.fullmatch(flags):
        raise ValueError("Flags may contain only letters, digits, '/', '+' and '-'; do not include spaces.")
    if not 0 <= target_note <= 127:
        raise ValueError("Target note must be a valid MIDI note number.")

    # OpenUtau rounds required phone duration up to its 50 ms render quantum.
    dur_required = max(float(duration_ms), float(oto.consonant_ms))
    dur_required = math.ceil(dur_required / 50.0 + 0.5) * 50.0

    # OpenUtau supplies a constant-zero pitch-deviation curve when the user
    # has not drawn pitch bends; samples are spaced every five ticks.
    pitch_count = max(
        1,
        int(math.ceil((tempo * 480.0 * (duration_ms + max(0.0, oto.preutter_ms))
                       / (60_000.0 * 5.0)))),
    )
    curve = pitch_curve_cents if pitch_curve_cents is not None else [0] * pitch_count
    pitch_data = encode_int12_curve(curve)

    # Argument order follows OpenUtau ExeResampler.DoResamplerReturnsFile:
    # input, output, note name, velocity, flags, offset, dur, consonant,
    # cutoff, volume, modulation, !tempo, base64 pitch deviations.
    return [
        str(Path(input_wav).resolve()),
        str(Path(output_wav).resolve()),
        midi_note_name(target_note),
        str(int(velocity)),
        flags,
        f"{oto.offset_ms:.6f}".rstrip("0").rstrip("."),
        f"{dur_required:.6f}".rstrip("0").rstrip("."),
        f"{oto.consonant_ms:.6f}".rstrip("0").rstrip("."),
        f"{oto.cutoff_ms:.6f}".rstrip("0").rstrip("."),
        str(int(volume)),
        str(int(modulation)),
        "!" + (f"{tempo:.6f}".rstrip("0").rstrip(".")),
        pitch_data,
    ]


def _moreconfig(sample_rate: int, reference_f0_hz: float) -> str:
    if not np.isfinite(reference_f0_hz) or reference_f0_hz <= 0:
        raise ValueError("A valid measured reference F0 is required for Moresampler analysis.")
    # Moresampler's voicebank configuration supports explicit per-folder F0
    # bounds. The 0.6x–1.7x range follows Kanru Hua's voicebank-author tutorial.
    return (
        "resampler-compatibility on\n"
        f"output-sampling-rate {int(sample_rate)}\n"
        "output-bit-depth 16\n"
        "analysis-f0-range-from-path off\n"
        f"analysis-f0-min {reference_f0_hz * 0.6:.3f}\n"
        f"analysis-f0-max {reference_f0_hz * 1.7:.3f}\n"
    )


def _analyse_render(path: Path) -> dict[str, Any]:
    audio, sample_rate, _ = read_wav(path)
    basic = analyze_audio(audio, sample_rate)
    timbre = analyze_timbre(audio, sample_rate)
    # Frame tracks are useful for plotting, but make the A/B JSON needlessly large.
    timbre.pop("frame_tracks", None)
    return {"basic": basic, "timbre": timbre}


def _run_one(
    executable: Path,
    input_source: Path,
    output_wav: Path,
    oto: OtoEntry,
    *,
    sample_rate: int,
    reference_f0_hz: float,
    target_note: int,
    duration_ms: float,
    velocity: int,
    flags: str,
    volume: int,
    modulation: int,
    tempo: float,
    timeout_seconds: float,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="moresampler_phone_") as temp:
        stage = Path(temp) / "pitch"
        stage.mkdir(parents=True)
        staged_input = stage / input_source.name
        # Source input is copied exactly; the processed sample is encoded with
        # the original WAV subtype immediately before this function is called.
        shutil.copy2(input_source, staged_input)
        (stage / "moreconfig.txt").write_text(
            _moreconfig(sample_rate, reference_f0_hz), encoding="utf-8"
        )
        staged_output = stage / "rendered.wav"
        args = build_moresampler_arguments(
            staged_input, staged_output, oto,
            target_note=target_note,
            duration_ms=duration_ms,
            velocity=velocity,
            flags=flags,
            volume=volume,
            modulation=modulation,
            tempo=tempo,
        )
        try:
            completed = subprocess.run(
                [str(executable), *args],
                cwd=str(executable.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Moresampler exceeded the {timeout_seconds:g}s timeout for {input_source.name}."
            ) from exc
        if not staged_output.is_file() or staged_output.stat().st_size <= 44:
            stdout = (completed.stdout or "")[-2500:]
            stderr = (completed.stderr or "")[-2500:]
            raise RuntimeError(
                f"Moresampler did not create an output WAV (exit code {completed.returncode}).\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_output, output_wav)
        return {
            "return_code": completed.returncode,
            "stdout": (completed.stdout or "")[-2500:],
            "stderr": (completed.stderr or "")[-2500:],
            "render_metrics": _analyse_render(output_wav),
        }


def render_moresampler_ab(
    source_wav: Path | str,
    restored_audio: np.ndarray,
    sample_rate: int,
    subtype: str,
    oto: OtoEntry,
    executable: Path | str,
    output_root: Path | str,
    *,
    source_bank_root: Path | str | None = None,
    pitch_layer: str = "pitch",
    reference_f0_hz: float,
    target_note: int = 60,
    duration_ms: float = 800.0,
    velocity: int = 100,
    flags: str = "",
    volume: int = 100,
    modulation: int = 0,
    tempo: float = 120.0,
    timeout_seconds: float = 180.0,
) -> dict:
    """Render original and restored audio through Moresampler with identical OTO settings."""
    source_wav = Path(source_wav).expanduser().resolve()
    executable = Path(executable).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_wav.is_file():
        raise FileNotFoundError(source_wav)
    if not executable.is_file():
        raise FileNotFoundError(f"Moresampler executable not found: {executable}")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Timeout must be positive and finite.")
    if source_bank_root is not None:
        bank_root = Path(source_bank_root).expanduser().resolve()
        if output_root == bank_root or bank_root in output_root.parents or output_root in bank_root.parents:
            raise ValueError("Moresampler A/B output must be outside the source voicebank.")
    original, original_sr, metadata = read_wav(source_wav)
    processed = np.asarray(restored_audio, dtype=np.float32)
    if sample_rate != original_sr or processed.shape != original.shape:
        raise ValueError("Restored audio must have the same sample rate, frame count and channel layout as the source.")
    if not np.all(np.isfinite(processed)):
        raise ValueError("Restored audio contains non-finite values.")
    if np.max(np.abs(processed)) > 1.0:
        raise ValueError("Restored audio would clip when written to the temporary WAV.")
    if not np.isfinite(reference_f0_hz) or reference_f0_hz <= 0:
        raise ValueError("Analyse the selected pitch layer first to obtain its median F0.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = output_root / "moresampler_ab" / pitch_layer / source_wav.stem / f"{midi_note_name(target_note)}_{int(round(duration_ms))}ms_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    original_render = destination / "original_moresampler.wav"
    restored_render = destination / "restored_moresampler.wav"

    with tempfile.TemporaryDirectory(prefix="eugene_moresampler_ab_") as temp:
        temp_root = Path(temp)
        original_dir = temp_root / "original" / pitch_layer
        restored_dir = temp_root / "restored" / pitch_layer
        original_dir.mkdir(parents=True)
        restored_dir.mkdir(parents=True)
        original_input = original_dir / source_wav.name
        restored_input = restored_dir / source_wav.name
        shutil.copy2(source_wav, original_input)
        sf.write(str(restored_input), processed, sample_rate, subtype=subtype)

        config_text = _moreconfig(sample_rate, reference_f0_hz)
        (original_dir / "moreconfig.txt").write_text(config_text, encoding="utf-8")
        (restored_dir / "moreconfig.txt").write_text(config_text, encoding="utf-8")

        args_original = build_moresampler_arguments(
            original_input, original_render, oto, target_note=target_note,
            duration_ms=duration_ms, velocity=velocity, flags=flags,
            volume=volume, modulation=modulation, tempo=tempo,
        )
        args_restored = build_moresampler_arguments(
            restored_input, restored_render, oto, target_note=target_note,
            duration_ms=duration_ms, velocity=velocity, flags=flags,
            volume=volume, modulation=modulation, tempo=tempo,
        )
        for label, args, dest in (
            ("original", args_original, original_render),
            ("restored", args_restored, restored_render),
        ):
            # Run sequentially so a single Moresampler installation is not
            # asked to write shared global config/state concurrently.
            with tempfile.TemporaryDirectory(prefix=f"moresampler_{label}_") as stage_temp:
                staged_dir = Path(stage_temp) / pitch_layer
                staged_dir.mkdir(parents=True)
                input_path = staged_dir / source_wav.name
                if label == "original":
                    shutil.copy2(source_wav, input_path)
                else:
                    sf.write(str(input_path), processed, sample_rate, subtype=subtype)
                (staged_dir / "moreconfig.txt").write_text(config_text, encoding="utf-8")
                staged_output = staged_dir / "rendered.wav"
                args = build_moresampler_arguments(
                    input_path, staged_output, oto, target_note=target_note,
                    duration_ms=duration_ms, velocity=velocity, flags=flags,
                    volume=volume, modulation=modulation, tempo=tempo,
                )
                completed = subprocess.run(
                    [str(executable), *args],
                    cwd=str(executable.parent),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    check=False,
                )
                if not staged_output.is_file() or staged_output.stat().st_size <= 44:
                    raise RuntimeError(
                        f"Moresampler failed to render the {label} sample "
                        f"(exit code {completed.returncode}).\n"
                        f"{(completed.stdout or '')[-2500:]}\n{(completed.stderr or '')[-2500:]}"
                    )
                shutil.copy2(staged_output, dest)

    original_metrics = _analyse_render(original_render)
    restored_metrics = _analyse_render(restored_render)
    report = {
        "report_type": "moresampler_phone_ab",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "resampler": str(executable),
        "source_wav": str(source_wav),
        "pitch_layer": pitch_layer,
        "oto": {
            "line_number": oto.line_number,
            "alias": oto.alias,
            "offset_ms": oto.offset_ms,
            "consonant_ms": oto.consonant_ms,
            "cutoff_ms": oto.cutoff_ms,
            "preutter_ms": oto.preutter_ms,
            "overlap_ms": oto.overlap_ms,
        },
        "render_settings": {
            "target_note_midi": int(target_note),
            "target_note_name": midi_note_name(target_note),
            "duration_ms": float(duration_ms),
            "velocity": int(velocity),
            "flags": flags,
            "volume": int(volume),
            "modulation": int(modulation),
            "tempo_bpm": float(tempo),
            "sample_rate_hz": int(sample_rate),
            "pitch_curve": "constant pitch; zero-cent deviation at five-tick intervals",
            "analysis_f0_range_hz": [round(reference_f0_hz * 0.6, 3), round(reference_f0_hz * 1.7, 3)],
            "moreconfig": config_text,
        },
        "outputs": {
            "original": original_render.name,
            "restored": restored_render.name,
        },
        "metrics": {
            "original": original_metrics,
            "restored": restored_metrics,
            "delta": {
                "rms_dbfs": restored_metrics["basic"]["rms_dbfs"] - original_metrics["basic"]["rms_dbfs"],
                "spectral_centroid_hz": restored_metrics["basic"]["spectral_centroid_hz"] - original_metrics["basic"]["spectral_centroid_hz"],
                "f0_median_hz": (
                    (restored_metrics["timbre"]["f0_hz"]["median"] or 0.0)
                    - (original_metrics["timbre"]["f0_hz"]["median"] or 0.0)
                ),
                "harmonic_to_residual_ratio_db": (
                    (restored_metrics["timbre"]["harmonic_to_residual_ratio_db"]["median"] or 0.0)
                    - (original_metrics["timbre"]["harmonic_to_residual_ratio_db"]["median"] or 0.0)
                ),
            },
        },
        "interpretation": (
            "This is a single OTO-phone render through Moresampler, not a complete phrase/wavtool render. "
            "Metrics describe acoustic changes only and do not score intelligibility or naturalness."
        ),
    }
    (destination / "moresampler_ab_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
