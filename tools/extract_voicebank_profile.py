#!/usr/bin/env python3
"""Compact large Eugene:RE analysis reports into processing-oriented features."""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import defaultdict, Counter
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


def num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def median_of(node):
    return num(node.get("median")) if isinstance(node, dict) else num(node)


def pct(values):
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
    return core, [t for t in re.split(r"[\s_]+", core) if t]


def compact_region(region):
    if not isinstance(region, dict): return None
    timbre = region.get("timbre") or {}
    result = {k: median_of(timbre.get(k)) for k in METRICS
              if median_of(timbre.get(k)) is not None}
    result.update({
        "duration_ms": num(region.get("duration_ms")),
        "status": region.get("deep_analysis_status"),
        "reason": region.get("reason"),
    })
    env = timbre.get("spectral_envelope") or {}
    if env:
        result["spectral_envelope"] = {
            "frequency_hz": env.get("frequency_hz"),
            "median_relative_db": env.get("median_relative_db"),
        }
    if timbre.get("harmonic_profile_relative_to_h1_db"):
        result["harmonic_profile"] = timbre["harmonic_profile_relative_to_h1_db"]
    return result


def summarise_group(name, samples, source):
    values, bands = defaultdict(list), defaultdict(list)
    envelopes, harmonic_values = [], defaultdict(list)
    aliases, sample_rows, oto_status = [], [], Counter()
    for sample in samples:
        path = sample.get("relative_path") or sample.get("file") or ""
        sample_id = str(path).replace("\\", "/").split("/")[-1]
        basic = sample.get("basic") or sample.get("analysis") or {}
        timbre = sample.get("timbre") or {}
        features = {}
        for key in BASIC:
            value = num(basic.get(key))
            if value is not None:
                features[key] = value
                values[key].append(value)
        for key in METRICS:
            value = median_of(timbre.get(key))
            if value is not None:
                features[key] = value
                values[key].append(value)
        # Old reports mislabeled a linear amplitude as peak_dbfs; don't treat it as dBFS.
        peak = num(basic.get("peak_amplitude"))
        if peak is not None:
            features["peak_amplitude"] = peak
        elif num(basic.get("peak_dbfs")) is not None:
            peak = num(basic["peak_dbfs"])
            features["peak_dbfs"] = peak if peak <= 0 else None
            if peak > 0:
                features["legacy_peak_field_linear_amplitude"] = peak
        for key, value in (basic.get("relative_band_energy") or {}).items():
            if num(value) is not None:
                bands[key].append(float(value))
        env = timbre.get("spectral_envelope") or {}
        if env.get("median_relative_db"):
            envelopes.append(env)
        for row in timbre.get("harmonic_profile_relative_to_h1_db", []) or []:
            h, value = row.get("harmonic"), num(row.get("median_db"))
            if h and value is not None:
                harmonic_values[int(h)].append(value)

        oto = sample.get("oto") or {}
        status = oto.get("status", "not_reported")
        oto_status[status] += 1
        entries = oto.get("entries", []) or []
        sample_rows.append({
            "sample": sample_id, "metrics": features,
            "oto_status": status, "alias_count": len(entries),
        })
        for entry in entries:
            alias = str(entry.get("alias", ""))
            core, tokens = alias_parts(alias)
            timing = entry.get("timing") or {}
            regions = entry.get("region_analysis") or {}
            aliases.append({
                "group": name, "group_kind": group_kind(name), "sample": sample_id,
                "alias": alias, "canonical_alias": core, "tokens": tokens,
                "oto_values_ms": timing.get("oto_values_ms", {}),
                "markers_ms": timing.get("markers_ms", {}),
                "regions_ms": timing.get("regions_ms", {}),
                "stretchable_duration_ms": num(timing.get("stretchable_duration_ms")),
                "warnings": timing.get("warnings", []),
                "region_features": {
                    region: compact_region(regions.get(region))
                    for region in ("source_window", "fixed_region", "stretchable_region")
                },
            })

    envelope_summary = None
    if envelopes:
        import numpy as np
        rows = [e for e in envelopes if e.get("median_relative_db")]
        width = min(len(e["median_relative_db"]) for e in rows)
        matrix = np.asarray([e["median_relative_db"][:width] for e in rows], dtype=float)
        envelope_summary = {
            "sample_count": len(rows),
            "frequency_hz": rows[0].get("frequency_hz", [])[:width],
            "median_of_sample_medians_db": np.median(matrix, axis=0).tolist(),
            "p10_of_sample_medians_db": np.percentile(matrix, 10, axis=0).tolist(),
            "p90_of_sample_medians_db": np.percentile(matrix, 90, axis=0).tolist(),
        }

    return {
        "sample_count": len(sample_rows), "group_kind": group_kind(name),
        "metrics": {k: pct(v) for k, v in sorted(values.items())},
        "relative_band_energy": {k: pct(v) for k, v in sorted(bands.items())},
        "spectral_envelope": envelope_summary,
        "harmonic_profile": [
            {"harmonic": h, **pct(v)} for h, v in sorted(harmonic_values.items())
        ],
        "oto_coverage": dict(oto_status),
        "samples": sample_rows, "aliases": aliases, "source_report": source,
    }


def add_deep_report(path, target, allow=None, skip=()):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, group in (data.get("groups") or {}).items():
        if name.upper() in {x.upper() for x in skip}:
            continue
        if allow is not None and name.upper() not in {x.upper() for x in allow}:
            continue
        target[name] = summarise_group(name, group.get("files", []), path.name)
        result[name] = target[name]["sample_count"]
    return result


def add_legacy_report(path, name, target):
    if name in target:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    target[name] = summarise_group(name, data.get("files", []), path.name)
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
                "purpose": "legacy fallback for group not present in full-bank report",
                "samples": count,
            })

    cross_pitch = defaultdict(lambda: defaultdict(list))
    for name, group in groups.items():
        if name.upper() not in PITCH:
            continue
        for entry in group["aliases"]:
            region = entry["region_features"].get("stretchable_region") or {}
            cross_pitch[entry["canonical_alias"]][name].append({
                "alias": entry["alias"], "sample": entry["sample"],
                "stretchable_duration_ms": entry["stretchable_duration_ms"],
                "metrics": {k: region[k] for k in METRICS if k in region},
            })

    return {
        "schema_version": "1.0",
        "profile_type": "eugene_re_compact_processing_features",
        "source_reports": sources,
        "processing_families": groups,
        "cross_pitch_aliases": {
            key: dict(banks) for key, banks in sorted(cross_pitch.items())
            if len(banks) > 1
        },
        "interpretation_limits": [
            "Compact reduction of submitted reports; listening tests remain necessary.",
            "Harmonic/residual ratio and inharmonicity are analysis proxies, not naturalness scores.",
            "Dedicated B2/F3/G4 reports are used; the full-bank report supplies non-pitch groups to avoid duplicate weighting.",
            "Legacy peak_dbfs values above 0 are linear amplitudes and are excluded from loudness decisions.",
            "OTO windows are diagnostic; one WAV may have multiple aliases with different time regions.",
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
