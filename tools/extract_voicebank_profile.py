#!/usr/bin/env python3
"""Extract a compact, structured Eugene:RE processing profile from deep JSON reports."""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PITCH = {"B2", "F3", "G4"}
PITCH_SUFFIX = re.compile(r"_(?:[A-G](?:#|B)?[0-8])$", re.I)
METRICS = (
    "f0_hz", "harmonic_to_residual_ratio_db",
    "aperiodic_residual_energy_fraction_percent", "inharmonicity_percent",
    "harmonic_slope_db_per_octave", "h1_minus_h2_db",
    "spectral_centroid_hz_voiced_frames", "spectral_flatness_voiced_frames",
    "voiced_rms_dbfs", "positive_spectral_flux", "voiced_frame_fraction",
    "local_f0_variation_percent", "pitch_variation_cents_p90_minus_p10",
)
BASIC = ("duration_seconds", "rms_dbfs", "crest_factor_db",
         "spectral_centroid_hz", "spectral_flatness")
ALIAS_METRICS = (
    "f0_hz", "harmonic_to_residual_ratio_db",
    "aperiodic_residual_energy_fraction_percent", "harmonic_slope_db_per_octave",
    "h1_minus_h2_db", "spectral_centroid_hz_voiced_frames",
    "spectral_flatness_voiced_frames", "voiced_frame_fraction",
)
VOWELS = {"a", "i", "u", "e", "o", "@a", "@i", "@u", "@e", "@o",
          "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey",
          "ih", "iy", "ow", "oy", "uh", "uw"}
PHONEME_CLASSES = {
    "nasal": {"m", "n", "ng", "ん", "ン"},
    "stop": {"b", "d", "g", "k", "p", "t", "q", "ts", "ch", "j"},
    "fricative": {"f", "h", "s", "sh", "th", "v", "z", "zh", "ふ"},
    "liquid": {"l", "r", "rr"},
    "glide": {"w", "y", "j"},
}


def num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def median_of(node):
    return num(node.get("median")) if isinstance(node, dict) else num(node)


def percentile(values):
    values = sorted(v for x in values if (v := num(x)) is not None)
    if not values:
        return {"p10": None, "median": None, "p90": None}
    def q(p):
        pos = p * (len(values) - 1)
        lo, hi = int(pos), min(len(values) - 1, int(pos) + 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"p10": q(.1), "median": q(.5), "p90": q(.9)}


def group_kind(name):
    u = name.upper().replace(" ", "_")
    if u in PITCH: return "pitch_layer"
    if "CONSONANT" in u or "RELEASE" in u: return "consonant_release"
    if "BONUS" in u or "EXTRA" in u: return "phonetic_extra"
    if "ENGLISH" in u or u.startswith("R_"): return "english_extension"
    if u in {"L", "V", "TH"}: return "phonetic_extension"
    return "other"


def alias_parts(alias):
    core = PITCH_SUFFIX.sub("", alias.strip())
    tokens = [t for t in re.split(r"[\s_]+", core) if t]
    roles = []
    for token in tokens:
        low = token.lower()
        if low in VOWELS:
            role = "vowel"
        else:
            role = next((kind for kind, members in PHONEME_CLASSES.items()
                         if low in members), "kana_or_unknown")
        roles.append(role)
    if len(tokens) >= 2:
        transition = f"{roles[0]}_to_{roles[1]}"
    elif roles:
        transition = roles[0]
    else:
        transition = "unclassified"
    onset_classes = [r for r in roles if r in PHONEME_CLASSES]
    return core, tokens, roles, transition, onset_classes


def collect_metric_values(target, prefix, node, keys):
    for key in keys:
        value = median_of(node.get(key)) if isinstance(node, dict) else None
        if value is not None:
            target[f"{prefix}{key}"].append(value)


def compact_region(region, aggregate):
    if not isinstance(region, dict):
        return
    status = region.get("deep_analysis_status", "not_reported")
    aggregate["status"][status] += 1
    duration = num(region.get("duration_ms"))
    if duration is not None:
        aggregate["duration_ms"].append(duration)
    timbre = region.get("timbre") or {}
    collect_metric_values(aggregate["metrics"], "", timbre, METRICS)


def finish_region_aggregate(aggregate):
    return {
        "analysis_status_counts": dict(aggregate["status"]),
        "duration_ms": percentile(aggregate["duration_ms"]),
        "metrics": {k: percentile(v) for k, v in sorted(aggregate["metrics"].items())},
    }


def summarize_group(name, samples, source):
    value_lists = defaultdict(list)
    band_lists = defaultdict(list)
    env_rows = []
    harmonic_values = defaultdict(list)
    alias_acc = {}
    sample_names = []
    event_classes = Counter()
    oto_status = Counter()
    group_sample_count = 0

    for sample in samples:
        group_sample_count += 1
        path = sample.get("relative_path") or sample.get("file") or ""
        sample_id = str(path).replace("\\", "/").split("/")[-1]
        basic = sample.get("basic") or sample.get("analysis") or {}
        timbre = sample.get("timbre") or {}
        file_metrics = {}
        for key in BASIC:
            value = num(basic.get(key))
            if value is not None:
                file_metrics[key] = value
                value_lists[key].append(value)
        for key in METRICS:
            value = median_of(timbre.get(key))
            if value is not None:
                file_metrics[key] = value
                value_lists[key].append(value)

        # Legacy files mislabeled linear peak amplitude as peak_dbfs; never interpret it as dBFS.
        peak = num(basic.get("peak_amplitude"))
        if peak is not None:
            file_metrics["peak_amplitude"] = peak
        else:
            legacy_peak = num(basic.get("peak_dbfs"))
            if legacy_peak is not None:
                if legacy_peak <= 0:
                    file_metrics["peak_dbfs"] = legacy_peak
                else:
                    file_metrics["legacy_peak_field_linear_amplitude"] = legacy_peak

        for key, value in (basic.get("relative_band_energy") or {}).items():
            if num(value) is not None:
                band_lists[key].append(float(value))
        envelope = timbre.get("spectral_envelope") or {}
        if envelope.get("median_relative_db"):
            env_rows.append(envelope)
        for row in timbre.get("harmonic_profile_relative_to_h1_db", []) or []:
            harmonic, value = row.get("harmonic"), num(row.get("median_db"))
            if harmonic and value is not None:
                harmonic_values[int(harmonic)].append(value)

        oto = sample.get("oto") or {}
        status = oto.get("status", "not_reported")
        oto_status[status] += 1
        entries = oto.get("entries", []) or []
        sample_names.append(sample_id)
        if group_kind(name) == "phonetic_extra":
            label = sample_id.lower()
            event = next((key for words, key in (
                (("cough",), "cough"), (("gasp", "breath"), "gasp_or_breath"),
                (("laugh", "giggle"), "laughter"), (("sob", "cry"), "sob_or_cry"),
                (("throat", "growl"), "throat_or_growl"),
                (("sigh",), "sigh"),
            ) if any(word in label for word in words)), "unclassified_extra")
            event_classes[event] += 1

        for entry in entries:
            alias = str(entry.get("alias", ""))
            core, tokens, roles, transition, consonant_roles = alias_parts(alias)
            accumulator = alias_acc.get(core)
            if accumulator is None:
                accumulator = {
                    "canonical_alias": core, "tokens": tokens, "token_roles": roles,
                    "transition_class": transition, "consonant_classes": consonant_roles,
                    "aliases_seen": set(), "samples": set(), "line_count": 0,
                    "timing": defaultdict(list), "stretchable_duration_ms": [],
                    "region": {
                        key: {"status": Counter(), "duration_ms": [],
                              "metrics": defaultdict(list)}
                        for key in ("fixed_region", "stretchable_region")
                    },
                    "warnings": Counter(),
                }
                alias_acc[core] = accumulator
            accumulator["aliases_seen"].add(alias)
            accumulator["samples"].add(sample_id)
            accumulator["line_count"] += 1
            timing = entry.get("timing") or {}
            for key, value in (timing.get("oto_values_ms") or {}).items():
                if num(value) is not None:
                    accumulator["timing"][key].append(float(value))
            stretch_duration = num(timing.get("stretchable_duration_ms"))
            if stretch_duration is not None:
                accumulator["stretchable_duration_ms"].append(stretch_duration)
            for warning in timing.get("warnings", []) or []:
                accumulator["warnings"][str(warning)] += 1
            regions = entry.get("region_analysis") or {}
            for region_name in accumulator["region"]:
                compact_region(regions.get(region_name), accumulator["region"][region_name])

    envelope_summary = None
    if env_rows:
        import numpy as np
        width = min(len(row.get("median_relative_db", [])) for row in env_rows)
        matrix = np.asarray([row["median_relative_db"][:width] for row in env_rows], dtype=float)
        envelope_summary = {
            "sample_count": len(env_rows),
            "frequency_hz": env_rows[0].get("frequency_hz", [])[:width],
            "median_of_sample_medians_db": np.median(matrix, axis=0).tolist(),
            "p10_of_sample_medians_db": np.percentile(matrix, 10, axis=0).tolist(),
            "p90_of_sample_medians_db": np.percentile(matrix, 90, axis=0).tolist(),
        }

    aliases = {}
    for core, acc in sorted(alias_acc.items()):
        aliases[core] = {
            "tokens": acc["tokens"],
            "token_roles": acc["token_roles"],
            "transition_class": acc["transition_class"],
            "consonant_classes": acc["consonant_classes"],
            "aliases_seen": sorted(acc["aliases_seen"])[:3],
            "sample_count": len(acc["samples"]),
            "entry_count": acc["line_count"],
            "oto_timing_ms": {
                key: percentile(values) for key, values in sorted(acc["timing"].items())
            },
            "stretchable_duration_ms": percentile(acc["stretchable_duration_ms"]),
            "region_features": {
                key: finish_region_aggregate(value)
                for key, value in acc["region"].items()
            },
            "warnings": dict(acc["warnings"]),
        }

    return {
        "sample_count": group_sample_count,
        "group_kind": group_kind(name),
        "metrics": {k: percentile(v) for k, v in sorted(value_lists.items())},
        "relative_band_energy": {k: percentile(v) for k, v in sorted(band_lists.items())},
        "spectral_envelope": envelope_summary,
        "harmonic_profile": [
            {"harmonic": h, **percentile(v)} for h, v in sorted(harmonic_values.items())
        ],
        "oto_coverage": dict(oto_status),
        "alias_count": len(aliases),
        "alias_profiles": aliases,
        "sample_name_examples": sorted(set(sample_names))[:12],
        "phonetic_event_classes": dict(event_classes) if event_classes else {},
        "source_report": source,
    }


def add_deep_report(path, target, allow=None, skip=()):
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for name, group in (data.get("groups") or {}).items():
        if name.upper() in {x.upper() for x in skip}:
            continue
        if allow is not None and name.upper() not in {x.upper() for x in allow}:
            continue
        target[name] = summarize_group(name, group.get("files", []), path.name)
        counts[name] = target[name]["sample_count"]
    return counts


def add_legacy_report(path, name, target):
    if name in target:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    target[name] = summarize_group(name, data.get("files", []), path.name)
    return target[name]["sample_count"]


def build(pitch_reports, full_bank, supplemental):
    groups, sources = {}, []
    for path in pitch_reports:
        counts = add_deep_report(path, groups, allow=PITCH)
        sources.append({"file": path.name, "bytes": path.stat().st_size, "groups": counts})
    counts = add_deep_report(full_bank, groups, skip=PITCH)
    sources.append({
        "file": full_bank.name, "bytes": full_bank.stat().st_size,
        "purpose": "non-pitch groups; pitch layers excluded to avoid duplicate weighting",
        "groups": counts,
    })
    for path in supplemental:
        name = path.stem.replace("_analysis", "").replace(" ", "_").upper()
        count = add_legacy_report(path, name, groups)
        if count is not None:
            sources.append({
                "file": path.name, "bytes": path.stat().st_size,
                "purpose": "legacy fallback for groups absent from full-bank report",
                "samples": count,
            })

    cross_pitch = defaultdict(dict)
    for name, group in groups.items():
        if name.upper() not in PITCH:
            continue
        for canonical, profile in group["alias_profiles"].items():
            cross_pitch[canonical][name] = {
                "entry_count": profile["entry_count"],
                "sample_count": profile["sample_count"],
                "stretchable_duration_ms": profile["region_features"]["stretchable_region"]["duration_ms"],
                "region_features": profile["region_features"]["stretchable_region"]["metrics"],
            }

    return {
        "schema_version": "2.0",
        "profile_type": "eugene_re_compact_processing_features",
        "source_reports": sources,
        "processing_families": groups,
        "cross_pitch_aliases": {
            key: banks for key, banks in sorted(cross_pitch.items()) if len(banks) > 1
        },
        "interpretation_limits": [
            "This compact profile is derived from the submitted reports; audition remains necessary.",
            "Harmonic/residual and inharmonicity measurements are proxies, not naturalness scores.",
            "Pitch reports supply B2/F3/G4; the full-bank report supplies non-pitch groups to avoid double-counting.",
            "Positive legacy peak_dbfs values were linear amplitudes and are excluded from dBFS comparisons.",
            "One WAV may have multiple OTO aliases; alias features are aggregated per canonical alias and keep original timing statistics.",
            "Phoneme labels are heuristic from alias tokens; ambiguous kana tokens are retained as kana_or_unknown instead of guessed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch-reports", nargs=3, type=Path, required=True)
    parser.add_argument("--full-bank", type=Path, required=True)
    parser.add_argument("--supplemental", nargs="*", type=Path, default=[])
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.pitch_reports, args.full_bank, args.supplemental)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Compact profile written: {args.output}; "
        f"groups={len(result['processing_families'])}; "
        f"cross-pitch aliases={len(result['cross_pitch_aliases'])}"
    )


if __name__ == "__main__":
    main()
