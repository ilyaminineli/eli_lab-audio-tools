"""OTO-aware, non-destructive Eugene:RE sample processing chain."""
from __future__ import annotations
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .auto_routing import identify_family, classify_alias, classify_event
from .dsp import TimbreProfile, PeakingBand, apply_profile, describe_profile
from .io import read_wav, write_wav
from .oto_analysis import OtoEntry, derive_timing, find_nearest_oto, match_entries, parse_oto

def load_processing_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "presets" / "eugene_re_processing_v1.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _profile(spec: dict, name: str) -> TimbreProfile:
    return TimbreProfile(
        name=name, description=spec.get("description", ""),
        highpass_hz=float(spec.get("highpass_hz", 0.0)),
        highpass_slope=int(spec.get("highpass_slope", 2)),
        bands=tuple(PeakingBand(
            frequency_hz=float(band["frequency_hz"]),
            gain_db=float(band["gain_db"]),
            q=float(band.get("q", 1.0)),
        ) for band in spec.get("bands", [])),
    )

def _mask_segment(mask: np.ndarray, start: int, end: int):
    start = max(0, min(int(start), mask.size))
    end = max(start, min(int(end), mask.size))
    if end > start:
        mask[start:end] = 1.0

def build_oto_body_mask(frame_count: int, sample_rate: int,
                        entries: list[OtoEntry], options: dict) -> tuple[np.ndarray, dict]:
    """Union stretchable windows, then protect fixed windows for every alias."""
    mask = np.zeros(frame_count, dtype=np.float32)
    protected = np.zeros(frame_count, dtype=np.float32)
    duration_ms = frame_count * 1000.0 / sample_rate
    min_stretch_ms = float(options.get("minimum_stretchable_ms", 60.0))
    marker_guard_ms = float(options.get("marker_protection_ms", 8.0))
    edge_fade_ms = float(options.get("edge_fade_ms", 12.0))
    eligible, skipped_short = 0, 0
    for entry in entries:
        timing = derive_timing(entry, duration_ms)
        regions = timing["regions_ms"]
        fixed_start, fixed_end = regions["fixed_region"]
        body_start, body_end = regions["stretchable_region"]
        _mask_segment(protected,
                      round(fixed_start * sample_rate / 1000),
                      round(fixed_end * sample_rate / 1000))
        guard = round(marker_guard_ms * sample_rate / 1000)
        for marker in (timing["markers_ms"]["preutter"], timing["markers_ms"]["overlap"]):
            center = round(marker * sample_rate / 1000)
            _mask_segment(protected, center - guard, center + guard)
        if body_end - body_start < min_stretch_ms:
            skipped_short += 1
            continue
        eligible += 1
        _mask_segment(mask, round(body_start * sample_rate / 1000),
                      round(body_end * sample_rate / 1000))
    mask *= (1.0 - protected)

    fade_samples = max(0, round(edge_fade_ms * sample_rate / 1000))
    active = mask > 0
    changes = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        fade = min(fade_samples, (end - start) // 2)
        if fade:
            mask[start:start + fade] *= np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            mask[end - fade:end] *= np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
    return mask, {
        "oto_alias_entries": len(entries),
        "eligible_stretch_regions": eligible,
        "short_stretch_regions_skipped": skipped_short,
        "mask_coverage_percent": float(mask.mean() * 100.0) if mask.size else 0.0,
        "minimum_stretchable_ms": min_stretch_ms,
        "marker_protection_ms": marker_guard_ms,
        "edge_fade_ms": edge_fade_ms,
        "fixed_region_policy": "protect the union of all aliases' fixed regions",
    }

def _match_oto(path: Path, root: Path, explicit_oto: Path | None, cache: dict):
    oto_path = explicit_oto or find_nearest_oto(path, root)
    if oto_path is None:
        return None, []
    if oto_path not in cache:
        cache[oto_path] = parse_oto(oto_path)
    parsed = cache[oto_path]
    try:
        relative = path.relative_to(oto_path.parent).as_posix()
    except ValueError:
        relative = path.name
    return oto_path, match_entries(parsed.index(), relative, path.name)

def _route(family: dict, filename: str, aliases: list[OtoEntry], config: dict):
    route = {
        **family, "aliases": [classify_alias(entry.alias) for entry in aliases],
        "enabled": False, "profile_name": "bypass", "reason": "",
        "event_class": None, "reference_pitch": family.get("reference_pitch"),
    }
    family_kind = family["family"]
    if family_kind == "pitch_layer":
        spec = (config.get("pitch_layers") or {}).get(family.get("pitch"), {})
        route.update(enabled=bool(spec.get("enabled", False)),
                     profile_name=spec.get("name", "bypass"),
                     reason=spec.get("reason", "No processing profile configured."),
                     profile_spec=spec)
    elif family_kind == "phonetic_extension":
        spec = (config.get("phonetic_extensions") or {}).get(family["group"], {})
        route.update(enabled=bool(spec.get("enabled", False)),
                     profile_name=spec.get("name", "neutral-extension"),
                     reason=spec.get("reason", "Recognised extension; held neutral pending listening tests."),
                     profile_spec=spec)
    elif family_kind == "english_extension":
        spec = (config.get("english_extensions") or {}).get(family["group"], {})
        route.update(enabled=bool(spec.get("enabled", False)),
                     profile_name=spec.get("name", "neutral-extension"),
                     reason=spec.get("reason", "English extension is routed separately."),
                     profile_spec=spec)
    elif family_kind == "consonant_release":
        route["reason"] = "Preserve attacks and aperiodic detail; no tonal-body EQ."
    elif family_kind == "phonetic_extra":
        route["event_class"] = classify_event(filename)
        route["reason"] = "Non-lexical sound event; kept outside the tonal voice chain."
    else:
        route["reason"] = "Unrecognised or root-level material; bypass until explicitly classified."
    return route

def process_auto(root: Path, output: Path, wet: float = 0.5,
                 config_path: Path | None = None, oto_file: Path | None = None,
                 dry_run: bool = False) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root == output or root in output.parents:
        raise ValueError("Output must be separate from, and outside, the source bank.")
    if not 0.0 <= wet <= 1.0:
        raise ValueError("wet must be between 0 and 1.")
    config = load_processing_config(config_path)
    mask_options = config.get("oto_mask", {})
    explicit_oto = Path(oto_file).resolve() if oto_file else None
    if explicit_oto and not explicit_oto.is_file():
        raise FileNotFoundError(explicit_oto)

    wav_paths = sorted(p for p in root.rglob("*.wav") if p.is_file() and output not in p.resolve().parents)
    oto_cache, file_reports, route_counts = {}, [], Counter()
    for path in wav_paths:
        relative = path.relative_to(root)
        family = identify_family(relative, root)
        oto_path, entries = _match_oto(path, root, explicit_oto, oto_cache)
        route = _route(family, path.name, entries, config)
        route_counts[route["family"]] += 1
        audio, sr, metadata = read_wav(path)
        mask_info = {"mask_coverage_percent": 0.0, "eligible_stretch_regions": 0}
        output_audio = audio.copy()
        status, reason, applied_profile = "bypassed", route["reason"], None

        if route["enabled"]:
            if not entries:
                reason = "Pitch route enabled, but no matching OTO aliases were found; left untouched."
            else:
                mask, mask_info = build_oto_body_mask(len(audio), sr, entries, mask_options)
                if not np.any(mask > 0):
                    reason = "No sufficiently long stretchable OTO window remained after fixed-region protection."
                else:
                    spec = route.get("profile_spec", {})
                    profile = _profile(spec, route["profile_name"])
                    if not profile.bands and profile.highpass_hz <= 0:
                        reason = "Configured profile is neutral; left untouched."
                    elif dry_run:
                        status = "dry_run_ready"
                        reason = "Would apply the configured profile only inside eligible OTO body windows."
                        applied_profile = describe_profile(profile)
                    else:
                        filtered = apply_profile(audio, sr, profile, wet=wet)
                        blend_mask = mask if audio.ndim == 1 else mask[:, None]
                        output_audio = (
                            audio.astype(np.float32) * (1.0 - blend_mask)
                            + filtered * blend_mask
                        ).astype(np.float32)
                        status = "processed"
                        applied_profile = describe_profile(profile)

        destination = output / relative
        if not dry_run and status == "processed":
            write_wav(destination, output_audio, sr, metadata["subtype"])
        elif not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        file_reports.append({
            "source": relative.as_posix(), "output": relative.as_posix(),
            "sample_rate_hz": sr, "frames": int(audio.shape[0]),
            "family": route["family"], "group": route["group"],
            "pitch": route.get("pitch"), "event_class": route.get("event_class"),
            "aliases": route["aliases"], "oto_file": str(oto_path) if oto_path else None,
            "oto_match_count": len(entries), "status": status, "reason": reason,
            "mask": mask_info, "profile": applied_profile,
        })

    if not dry_run:
        # Copy non-WAV bank files, including oto.ini, byte-for-byte.
        for source in root.rglob("*"):
            if not source.is_file() or source.suffix.lower() == ".wav":
                continue
            if output in source.resolve().parents:
                continue
            target = output / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_type": "eugene_re_auto_processing",
        "profile_version": config.get("version", "unknown"),
        "source_root": str(root), "output_root": str(output),
        "dry_run": dry_run, "wet": wet, "files_found": len(wav_paths),
        "route_counts": dict(route_counts),
        "processed_files": sum(x["status"] == "processed" for x in file_reports),
        "bypassed_files": sum(x["status"] == "bypassed" for x in file_reports),
        "dry_run_ready_files": sum(x["status"] == "dry_run_ready" for x in file_reports),
        "metadata_policy": "All non-WAV files, including oto.ini, copied byte-for-byte; source untouched.",
        "files": file_reports,
    }
    (output / "processing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
