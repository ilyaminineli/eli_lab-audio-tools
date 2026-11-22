"""OTO-aware, non-destructive Eugene:RE sample processing chain."""
from __future__ import annotations
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .auto_routing import identify_family, classify_alias, classify_event
from .dsp import TimbreProfile, PeakingBand, apply_profile_masked, describe_profile
from .io import read_wav, write_wav
from .oto_analysis import OtoEntry, derive_timing, find_nearest_oto, match_entries, parse_oto

def load_processing_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent / "presets" / "eugene_re_processing_v1.json"
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
    entry_windows, timing_warnings = [], []
    for entry in entries:
        timing = derive_timing(entry, duration_ms)
        regions = timing["regions_ms"]
        fixed_start, fixed_end = regions["fixed_region"]
        body_start, body_end = regions["stretchable_region"]
        stretch_ms = max(0.0, body_end - body_start)
        warnings = list(timing.get("warnings", []))
        if warnings:
            timing_warnings.append({
                "line_number": entry.line_number,
                "alias": entry.alias,
                "warnings": warnings,
            })
        entry_eligible = stretch_ms >= min_stretch_ms
        entry_windows.append({
            "line_number": entry.line_number,
            "alias": entry.alias,
            "fixed_region_ms": [round(fixed_start, 3), round(fixed_end, 3)],
            "stretchable_region_ms": [round(body_start, 3), round(body_end, 3)],
            "stretchable_duration_ms": round(stretch_ms, 3),
            "eligible": entry_eligible,
            "warnings": warnings,
        })
        _mask_segment(protected,
                      round(fixed_start * sample_rate / 1000),
                      round(fixed_end * sample_rate / 1000))
        guard = round(marker_guard_ms * sample_rate / 1000)
        for marker in (timing["markers_ms"]["preutter"], timing["markers_ms"]["overlap"]):
            center = round(marker * sample_rate / 1000)
            _mask_segment(protected, center - guard, center + guard)
        if not entry_eligible:
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
        "eligible_window_duration_ms_sum": round(sum(
            item["stretchable_duration_ms"] for item in entry_windows if item["eligible"]
        ), 3),
        "minimum_stretchable_ms": min_stretch_ms,
        "marker_protection_ms": marker_guard_ms,
        "edge_fade_ms": edge_fade_ms,
        "fixed_region_policy": "protect the union of all aliases' fixed regions",
        "entry_windows": entry_windows,
        "timing_warnings": timing_warnings,
    }

def _match_oto(path: Path, root: Path, explicit_oto: Path | None, cache: dict):
    oto_path = explicit_oto or find_nearest_oto(path, root)
    if oto_path is None:
        return None, [], None
    if oto_path not in cache:
        cache[oto_path] = parse_oto(oto_path)
    parsed = cache[oto_path]
    try:
        relative = path.relative_to(oto_path.parent).as_posix()
    except ValueError:
        relative = path.name
    return oto_path, match_entries(parsed.index(), relative, path.name), parsed

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

def _alias_audit(entries: list[OtoEntry]) -> dict:
    classifications = [classify_alias(entry.alias) for entry in entries]
    unknown = sorted({
        token
        for item in classifications
        for token in item.get("unclassified_tokens", [])
    })
    aliases_by_key: dict[str, list[OtoEntry]] = {}
    for entry in entries:
        aliases_by_key.setdefault(entry.alias.strip().casefold(), []).append(entry)
    duplicates = [
        {"alias": key, "line_numbers": [item.line_number for item in values]}
        for key, values in aliases_by_key.items() if len(values) > 1
    ]
    return {
        "unclassified_tokens": unknown,
        "duplicate_aliases": duplicates,
        "has_unclassified_tokens": bool(unknown),
        "has_duplicate_aliases": bool(duplicates),
    }


def _summarise_oto_files(cache: dict) -> list[dict]:
    summaries = []
    for oto_path, parsed in sorted(cache.items(), key=lambda item: str(item[0]).casefold()):
        aliases_by_key: dict[tuple[str, str], list[OtoEntry]] = {}
        orphan_entries = []
        for entry in parsed.entries:
            wav_key = entry.wav.replace("\\", "/")
            aliases_by_key.setdefault(
                (wav_key.casefold(), entry.alias.strip().casefold()), []
            ).append(entry)
            candidate = (oto_path.parent / Path(wav_key)).resolve()
            if not candidate.is_file():
                orphan_entries.append({
                    "line_number": entry.line_number,
                    "wav": entry.wav,
                    "alias": entry.alias,
                })
        duplicate_aliases = [
            {
                "wav": values[0].wav,
                "alias": values[0].alias,
                "line_numbers": [item.line_number for item in values],
            }
            for values in aliases_by_key.values() if len(values) > 1
        ]
        summaries.append({
            "path": str(oto_path),
            "encoding": parsed.encoding,
            "entries_parsed": len(parsed.entries),
            "parse_errors": parsed.errors,
            "parse_error_count": len(parsed.errors),
            "orphan_entries": orphan_entries,
            "orphan_entry_count": len(orphan_entries),
            "duplicate_aliases": duplicate_aliases,
            "duplicate_alias_count": len(duplicate_aliases),
        })
    return summaries


def process_auto(root: Path, output: Path, wet: float = 0.5,
                 config_path: Path | None = None, oto_file: Path | None = None,
                 dry_run: bool = False, allow_partial: bool = False,
                 progress=None) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from and not overlap the source bank.")
    if not np.isfinite(wet) or not 0.0 <= wet <= 1.0:
        raise ValueError("wet must be finite and between 0 and 1.")
    config = load_processing_config(config_path)
    mask_options = config.get("oto_mask", {})
    minimum_coverage = float(mask_options.get("minimum_process_coverage_percent", 5.0))
    if not np.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 100.0:
        raise ValueError("minimum_process_coverage_percent must be in [0, 100].")
    allow_partial_processing = bool(
        allow_partial or mask_options.get("allow_partial_processing", False)
    )
    context_padding_ms = float(mask_options.get("context_padding_ms", 80.0))
    if not np.isfinite(context_padding_ms) or context_padding_ms < 0:
        raise ValueError("context_padding_ms must be finite and non-negative.")

    explicit_oto = Path(oto_file).resolve() if oto_file else None
    if explicit_oto and not explicit_oto.is_file():
        raise FileNotFoundError(explicit_oto)

    wav_paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.casefold() == ".wav" and output not in p.resolve().parents
    )
    oto_cache, file_reports, route_counts = {}, [], Counter()
    for file_index, path in enumerate(wav_paths, start=1):
        relative = path.relative_to(root)
        family = identify_family(relative, root)
        oto_path, entries, parsed_oto = _match_oto(path, root, explicit_oto, oto_cache)
        route = _route(family, path.name, entries, config)
        route_counts[route["family"]] += 1
        audio, sr, metadata = read_wav(path)
        mask_info = {"mask_coverage_percent": 0.0, "eligible_stretch_regions": 0}
        output_audio = audio.copy()
        status, reason, applied_profile = "bypassed", route["reason"], None
        alias_audit = _alias_audit(entries)

        if route["enabled"]:
            if not entries:
                status = "manual_review"
                reason = "No matching OTO aliases; left untouched for manual OTO review."
            elif alias_audit["has_duplicate_aliases"]:
                status = "manual_review"
                reason = "Duplicate OTO aliases for this WAV are ambiguous; left untouched for manual review."
            elif alias_audit["has_unclassified_tokens"]:
                status = "manual_review"
                reason = (
                    "One or more OTO alias tokens could not be classified; "
                    "left untouched for phonetic review."
                )
            else:
                mask, mask_info = build_oto_body_mask(len(audio), sr, entries, mask_options)
                if mask_info.get("timing_warnings"):
                    status = "manual_review"
                    reason = "OTO timing warnings require review; this WAV was left untouched."
                elif not np.any(mask > 0):
                    status = "manual_review"
                    reason = (
                        "No sufficiently long stretchable OTO window remained after fixed-region "
                        "protection; review the per-alias timing diagnostics."
                    )
                else:
                    spec = route.get("profile_spec", {})
                    profile = _profile(spec, route["profile_name"])
                    if not profile.bands and profile.highpass_hz <= 0:
                        status = "bypassed"
                        reason = "Configured profile is neutral; left untouched."
                    else:
                        coverage = float(mask_info["mask_coverage_percent"])
                        is_partial = coverage < minimum_coverage
                        if is_partial and not allow_partial_processing:
                            status = "partial"
                            reason = (
                                f"Eligible mask coverage is {coverage:.2f}%, below the "
                                f"{minimum_coverage:.2f}% processing threshold; left untouched. "
                                "Use --allow-partial only after reviewing this sample."
                            )
                        elif dry_run:
                            status = "partial" if is_partial else "ready"
                            reason = (
                                f"Would apply the configured profile inside eligible OTO body "
                                f"windows ({coverage:.2f}% mask coverage; "
                                f"{context_padding_ms:.0f} ms filter context)."
                            )
                            applied_profile = describe_profile(profile)
                        else:
                            output_audio = apply_profile_masked(
                                audio, sr, profile, mask, wet=wet,
                                context_padding_ms=context_padding_ms,
                            )
                            status = "processed_partial" if is_partial else "processed"
                            reason = (
                                f"Applied the configured profile inside eligible OTO body "
                                f"windows ({coverage:.2f}% mask coverage; "
                                f"{context_padding_ms:.0f} ms filter context)."
                            )
                            applied_profile = describe_profile(profile)

        destination = output / relative
        if not dry_run and status in {"processed", "processed_partial"}:
            write_wav(destination, output_audio, sr, metadata["subtype"])
        elif not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        file_reports.append({
            "source": relative.as_posix(), "output": relative.as_posix(),
            "sample_rate_hz": sr, "frames": int(audio.shape[0]),
            "family": route["family"], "group": route["group"],
            "pitch": route.get("pitch"), "event_class": route.get("event_class"),
            "aliases": route["aliases"], "alias_audit": alias_audit,
            "oto_file": str(oto_path) if oto_path else None,
            "oto_match_count": len(entries),
            "oto_parse_error_count": len(parsed_oto.errors) if parsed_oto else 0,
            "status": status, "reason": reason,
            "mask": mask_info, "profile": applied_profile,
        })
        if progress:
            progress(file_index, len(wav_paths), path.relative_to(root).as_posix())

    if not dry_run:
        # Copy non-WAV bank files, including oto.ini, byte-for-byte.
        for source in root.rglob("*"):
            if not source.is_file() or source.suffix.casefold() == ".wav":
                continue
            if output in source.resolve().parents:
                continue
            target = output / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    output.mkdir(parents=True, exist_ok=True)
    oto_summaries = _summarise_oto_files(oto_cache)
    counts = Counter(item["status"] for item in file_reports)
    ready_count = counts["ready"] + counts["processed"]
    partial_count = counts["partial"] + counts["processed_partial"]
    processed_count = counts["processed"] + counts["processed_partial"]
    report = {
        "report_type": "eugene_re_auto_processing",
        "profile_version": config.get("version", "unknown"),
        "source_root": str(root), "output_root": str(output),
        "dry_run": dry_run, "wet": wet, "files_found": len(wav_paths),
        "route_counts": dict(route_counts),
        "summary": {
            "ready": ready_count,
            "partial": partial_count,
            "manual_review": counts["manual_review"],
            "bypassed": counts["bypassed"],
            "processed": processed_count,
            "processed_partial": counts["processed_partial"],
        },
        "processed_files": processed_count,
        "ready_files": ready_count,
        "partial_files": partial_count,
        "manual_review_files": counts["manual_review"],
        "bypassed_files": counts["bypassed"],
        # Retain the old field for report consumers, now counting only fully-ready dry-run files.
        "dry_run_ready_files": counts["ready"],
        "coverage_policy": {
            "minimum_process_coverage_percent": minimum_coverage,
            "allow_partial_processing": allow_partial_processing,
            "context_padding_ms": context_padding_ms,
            "partial_files_are_processed": allow_partial_processing,
        },
        "oto_files": oto_summaries,
        "metadata_policy": "All non-WAV files, including oto.ini, copied byte-for-byte; source untouched.",
        "files": file_reports,
    }
    (output / "processing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
