"""Recursive voicebank analysis with optional OTO timing alignment."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import re

import numpy as np
import soundfile as sf

from .timbre_analysis import TimbreAnalysisConfig, analyze_timbre
from .analysis import analyze_audio
from .oto_analysis import (
    OtoParseResult, analyse_oto_entry, find_nearest_oto, match_entries, parse_oto,
)

PITCH_FOLDER_RE = re.compile(r"(?<![A-Z0-9])([A-G](?:#|B)?[0-8])(?![A-Z0-9])")

def _group_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    for parent in (path.parent, *path.parent.parents):
        if parent == root.parent:
            break
        match = PITCH_FOLDER_RE.search(parent.name.upper())
        if match:
            return match.group(1)
        if parent == root:
            break
    top = relative.parts[0] if len(relative.parts) > 1 else "ROOT"
    top = re.sub(r"^\s*\d+[. _-]*", "", top)
    top = re.sub(r"[^A-Za-z0-9]+", "_", top).strip("_").upper()
    return top or "ROOT"

def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"p10": None, "median": None, "p90": None}
    return {
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }

def _aggregate(files: list[dict]) -> dict:
    keys = [
        "f0_hz.median",
        "harmonic_to_residual_ratio_db.median",
        "inharmonicity_percent.median",
        "harmonic_slope_db_per_octave.median",
        "spectral_centroid_hz_voiced_frames.median",
        "spectral_flatness_voiced_frames.median",
        "local_f0_variation_percent",
        "pitch_variation_cents_p90_minus_p10",
        "voiced_frame_fraction",
    ]
    result = {}
    for key in keys:
        values = []
        for item in files:
            node = item["timbre"]
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, (int, float)) and np.isfinite(node):
                values.append(float(node))
        result[key.replace(".", "_")] = _summary(values)
    return result

def analyze_bank(root: Path, output: Path, config: TimbreAnalysisConfig | None = None,
                 include_groups: set[str] | None = None, oto_file: Path | None = None,
                 use_oto: bool = True) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if output == root:
        raise ValueError("Output directory must differ from the voicebank root.")
    explicit_oto = Path(oto_file).resolve() if oto_file else None
    if explicit_oto and not explicit_oto.is_file():
        raise FileNotFoundError(explicit_oto)

    wav_paths = sorted(
        p for p in root.rglob("*.wav")
        if p.is_file() and output not in p.resolve().parents
    )
    groups: dict[str, list[dict]] = {}
    failures = []
    cfg = config or TimbreAnalysisConfig()
    oto_cache: dict[Path, OtoParseResult] = {}
    oto_matched_lines: dict[Path, set[int]] = {}
    audio_without_oto = []

    for path in wav_paths:
        group = _group_for(path, root)
        if include_groups and group.upper() not in {name.upper() for name in include_groups}:
            continue
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            info = sf.info(str(path))
            basic = analyze_audio(audio, sr)
            timbre = analyze_timbre(audio, sr, cfg)
            oto_report = {"status": "disabled"}
            if use_oto:
                oto_path = explicit_oto or find_nearest_oto(path, root)
                if oto_path is None:
                    oto_report = {"status": "oto_ini_not_found", "entries": []}
                    audio_without_oto.append(path.relative_to(root).as_posix())
                else:
                    if oto_path not in oto_cache:
                        oto_cache[oto_path] = parse_oto(oto_path)
                        oto_matched_lines[oto_path] = set()
                    parsed = oto_cache[oto_path]
                    try:
                        relative_to_oto = path.relative_to(oto_path.parent).as_posix()
                    except ValueError:
                        relative_to_oto = path.name
                    matches = match_entries(parsed.index(), relative_to_oto, path.name)
                    if not matches:
                        oto_report = {
                            "status": "no_matching_oto_entries",
                            "oto_ini": oto_path.relative_to(root).as_posix()
                                if root in oto_path.parents or oto_path == root else str(oto_path),
                            "encoding": parsed.encoding,
                            "entries": [],
                        }
                        audio_without_oto.append(path.relative_to(root).as_posix())
                    else:
                        oto_matched_lines[oto_path].update(entry.line_number for entry in matches)
                        oto_report = {
                            "status": "matched",
                            "oto_ini": oto_path.relative_to(root).as_posix()
                                if root in oto_path.parents or oto_path == root else str(oto_path),
                            "encoding": parsed.encoding,
                            "entry_count": len(matches),
                            "entries": [
                                analyse_oto_entry(entry, audio, sr, cfg)
                                for entry in matches
                            ],
                        }
            groups.setdefault(group, []).append({
                "relative_path": path.relative_to(root).as_posix(),
                "metadata": {
                    "sample_rate_hz": int(sr),
                    "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
                    "frames": int(audio.shape[0]),
                    "subtype": info.subtype,
                },
                "basic": basic,
                "timbre": timbre,
                "oto": oto_report,
            })
        except Exception as exc:
            failures.append({"relative_path": path.relative_to(root).as_posix(), "error": str(exc)})

    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for group, items in sorted(groups.items()):
        reports[group] = {
            "sample_count": len(items),
            "aggregate": _aggregate(items),
            "files": items,
        }
        file_name = f"{group.lower()}_deep_analysis.json"
        (output / file_name).write_text(
            json.dumps(reports[group], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    oto_sources = []
    unmatched_oto_entries = 0
    oto_parse_errors = 0
    for path, parsed in sorted(oto_cache.items(), key=lambda item: str(item[0])):
        matched_lines = oto_matched_lines.get(path, set())
        unmatched = [
            {"line_number": entry.line_number, "wav": entry.wav, "alias": entry.alias}
            for entry in parsed.entries if entry.line_number not in matched_lines
        ]
        unmatched_oto_entries += len(unmatched)
        oto_parse_errors += len(parsed.errors)
        try:
            display_path = path.relative_to(root).as_posix()
        except ValueError:
            display_path = str(path)
        oto_sources.append({
            "oto_ini": display_path,
            "encoding": parsed.encoding,
            "entry_count": len(parsed.entries),
            "matched_entry_count": len(matched_lines),
            "unmatched_entry_count": len(unmatched),
            "unmatched_entries": unmatched,
            "parse_errors": parsed.errors,
        })

    manifest = {
        "report_type": "voicebank_deep_timbre_analysis",
        "analysis_version": "1.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "config": cfg.__dict__,
        "total_wav_files_found": len(wav_paths),
        "total_wav_files_analysed": sum(item["sample_count"] for item in reports.values()),
        "failed_files": failures,
        "oto_analysis": {
            "enabled": use_oto,
            "oto_ini_files_read": len(oto_sources),
            "oto_entries_matched": sum(item["matched_entry_count"] for item in oto_sources),
            "oto_entries_unmatched": unmatched_oto_entries,
            "oto_parse_errors": oto_parse_errors,
            "audio_files_without_oto_match": audio_without_oto,
            "sources": oto_sources,
        },
        "groups": {
            name: {
                "sample_count": report["sample_count"],
                "report": f"{name.lower()}_deep_analysis.json",
                "aggregate": report["aggregate"],
            }
            for name, report in reports.items()
        },
        "safety": "Analysis only. Source WAVs and oto.ini files are read-only.",
    }
    (output / "bank_deep_analysis.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
