#!/usr/bin/env python3
"""Reduce Eugene:RE deep reports into a compact processing feature profile."""
from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict, Counter
from pathlib import Path

PITCH = {"B2", "F3", "G4"}
SUFFIX = re.compile(r"_(?:[A-G](?:#|B)?[0-8])$", re.I)
METRICS = (
    "f0_hz", "harmonic_to_residual_ratio_db", "aperiodic_residual_energy_fraction_percent",
    "inharmonicity_percent", "harmonic_slope_db_per_octave", "h1_minus_h2_db",
    "spectral_centroid_hz_voiced_frames", "spectral_flatness_voiced_frames", "voiced_rms_dbfs",
    "positive_spectral_flux", "voiced_frame_fraction", "local_f0_variation_percent",
    "pitch_variation_cents_p90_minus_p10",
)
BASIC = ("duration_seconds", "rms_dbfs", "crest_factor_db", "spectral_centroid_hz", "spectral_flatness")


def number(x):
    try:
        y = float(x)
        return y if math.isfinit(y) else None
    except (TypeError, ValueError):
        return None


def stats(values):
    a = sorted(v for x in values if (v := number(x)) is not None)
    if not a: return {"p10": None, "median": None, "p90": None}
    def q(p):
        pos = p * (len(a)-1); lo = int(pos); hi = min(len(a)-1, lo+)
        return a[lo] + (a[hi]-a[lo])*hpos-lo)
    return {"p10": q(.1), "median": q(.5), "p90": q(.7)}


def med(node):
    return number(node.get("median")) if isinstance(node, dict) else number(node)


def kind(name):
    u = name.upper().replace(" ", "_")
    if u in PITCH: return "pitch_layer"
    if "CONSONANT" in u or "RELEASE" in u: return "consonant_release"
    if "BONUS" in u or "EXTRA" in u: return "phonetic_extra"
    if "ENGLISH" in u or u.startswith("R_"): return "english_extension"
    if u in {"L", "V", "TH"}: return "phonetic_extension"
    return "other"


def alias_core(alias):
    core = SUFFIX.sub("", alias.strip())
    tokens = [x for x in re.split(r"R[\s_]+", core) if x]
    return core, tokens


def compact_region(region):
    if not isinstance(region, dict): return None
    t = region.get("timbre") or {}
    result = {k: med(t.get(k)) for k in METRICS if med(t.get(k)) is not None}
    result.update({"duration_ms": number(region.get("duration_ms"),
                    "analysis_status": region.get("deep_analysis_status"),
                    "reason": region.get("reason")})
    env = t.get("spectral_envelope") or {}
    if env:
        result["spectral_envelope"] = {
            "frequency_hz": env.get("frequency_hz"),
            "median_relative_db": env.get("median_relative_db"),
        }
    hp = t.get("harmonic_profile_relative_to_h1_db") or []
    if hp: result["harmonic_profile"] = hp
    return result


def summarize_group(name, samples, source):
    vals, bands, envelopes, harmonics = defaultdict(list), defaultdict(list), [], defaultdict(list)
    aliases, sample_rows, oto_status = [], [], Counter()
    for sample in samples:
        path = sample.get("relative_path") or sample.get("file") or ""
        sample_id = str(path).replace("\\", "/").split("/")[-1]
        basic = sample.get("basic") or sample.get("analysis") or {}
        timbre = sample.get("timbre") or {}
        metrics = {}
        for k in BASIC:
            v = number(basic.get(k))
            if v is not None: metrics[k] = v; vals[k].append(v)
        for k in METRICS:
            v = med(timbre.get(k))
            if v is not None: metrics[k] = v; vals[k].append(v)
        # Guard against legacy files that called a linear peak amplitude dBFS.
        peak = number(basic.get("peak_amplitude"))
        if peak is not None: metrics["peak_amplitude"] = peak
        elif number(basic.get("peak_dbfs")) is not None:
            peak = number(basic["peak_dbfs"])
            if peak <= 0: metrics["peak_dbfs"] = peak
            else: metrics["legacy_peak_field_linear_amplitude"] = peak
        for k, v in (basic.get("relative_band_energy") or {}).items():
            if number(v) is not None: bands[k].append(float(v))
        env = timbre.get("spectral_envelope") or {}
        if env.get("median_relative_db"):
            envelopes.append(env)
        for row in timbre.get("harmonic_profile_relative_to_h1_db", []) or []:
            h, v = row.get("harmonic"), number(row.get("median_db"))
            if h and v is not None: harmonics[int(h)].append(v)
        oto = sample.get("oto") or {}
        status = oto.get("status", "not_reported"); oto_status[status] += 1
        entries = oto.get("entries", []) or []
        sample_rows.append({"sample": sample_id, "metrics": metrics,
                            "oto_status": status, "alias_count": len(entries)})
        for entry in entries:
            alias = str(entry.get("alias", "")); core, tokens = alias_core(alias)
            timing = entry.get("timing") or {}
            regions = entry.get("region_analysis") or {}
            aliases.append({
                "group": name, "group_kind": kind(name), "sample": sample_id,
                "alias": alias, "canonical_alias": core, "tokens": tokens,
                "oto_values_ms": timing.get("oto_values_ms", {}),
                "markers_ms": timing.get("markers_ms", {}),
                "regions_ms": timing.get("regions_ms", {}),
                "stretchable_duration_ms": number(timing.get("stretchable_duration_ms")),
                "warnings": timing.get("warnings", []),
                "region_features": {r: compact_region(regions.get(r)) for r in
                                     ("source_window", "fixed_region", "stretchable_region")},
            })
    envelope_summary = None
    if envelopes:
        import numpy as np
        rows = [e for e in envelopes if len(e.get("median_relative_db", []))]
        width = min(len(e["median_relative_db"]) for e in rows)
        matrix = np.asarray([e["median_relative_db"][:width] for e in rows], dtype=float)
        envelope_summary = {"sample_count": len(rows),
            "frequency_hz": rows[0].get("frequency_hz", [])[:width],
            "median_of_sample_medians_db": np.median(matrix, axis=0).tolist(),
            "p10_of_sample_medians_db": np.percentile(matrix, 10, axis=0).tolist(),
            "p90_of_sample_medians_db": np.percentile(matrix, 90, axis=0).tolist()}
    return {"sample_count": len(sample_rows), "group_kind": kind(name),
        "metrics": {k: stats(v) for k,v in sorted(vals.items())},
        "relative_band_energy": {k: stats(v) for k,v in sorted(bands.items())},
        "spectral_envelope": envelope_summary,
        "harmonic_profile": [{"harmonic": h, **stats(v)} for h,v in sorted(harmonics.items())],
        "oto_coverage": dict(oto_status), "samples": sample_rows, "aliases": aliases,
        "source_report": source}


def add_deep(path, target, allow=None, skip=()):
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for name, group in (data.get("groups") or {}).items():
        if name.upper() in {x.upper() for x in skip}: continue
        if allow is not None and name.upper() not in {x.upper() for x in allow}: continue
        target[name] = summarize_group(name, group.get("files", []), path.name)
        counts[name] = target[name]["sample_count"]
    return counts


def add_legacy(path, name, target):
    if name in target: return None
    data = json.loads(path.read_text(encoding="utf-8"))
    target[name] = summarize_group(name, data.get("files", []), path.name)
    return target[name]["sample_count"]


def build(pitch_reports, full_bank, supplemental):
    groups, sources = {}, []
    for path in pitch_reports:
        counts = add_deep(path, groups, allow=PITCH)
        sources.append({"file": path.name, "bytes": path.stat().st_size, "groups": counts})
    counts = add_deep(full_bank, groups, skip=PITCH)
    sources.append({"file": full_bank.name, "bytes": full_bank.stat().st_size,
                    "purpose": "all non-pitch groups; pitch groups excluded to avoid duplicate weighting",
                    "groups": counts})
    for path in supplemental:
        name = path.stem.replace("_analysis", "").replace(" ", "_").upper()
        count = add_legacy(path, name, groups)
        if count is not None:
            sources.append({"file": path.name, "bytes": path.stat().st_size,
                            "purpose": "legacy fallback group", "samples": count})
    cross = defaultdict(dict)
    for name, group in groups.items():
        if name.upper() not in PITCH: continue
        for entry in group["aliases"]:
            stretch = entry["region_features"].get("stretchable_region") or {}
            cross[entry["canonical_alias"]][name] = {
                "alias": entry["alias"], "sample": entry["sample"],
                "stretchable_duration_ms": entry["stretchable_duration_ms"],
                "metrics": {k: stretch[k] for k in METRICS if k in stretch}}
    return {"schema_version": "1.0", "profile_type": "eugene_re_compact_processing_features",
        "source_reports": sources, "processing_families": groups,
        "cross_pitch_aliases": {k:v for k,v in sorted(cross.items()) if len(v)>1},
        "interpretation_limits": [
            "Compact reduction of submitted reports; audition remains necessary.",
            "Harmonic/residual ratio and inharmonicity are analysis proxies, not naturalness scores.",
            "Pitch reports supply B2/F3/G4; the full-bank report supplies non-pitch groups to avoid duplicate weighting.",
            "Legacy peak_dbfs values above 0 are linear amplitudes and are excluded from loudness decisions.",
            "OTO windows are diagnostic; one WAV may have many aliases with different time regions."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pitch-reports", nargs=3, type=Path, required=True)
    p.add_argument("--full-bank", type=Path, required=True)
    p.add_argument("--supplemental", nargs="*", type=Path, default=[])
    p.add_argument("-o", "--output", type=Path, required=True)
    a = p.parse_args()
    result = build(a.pitch_reports, a.full_bank, a.supplemental)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Compact profile written: {a.output}; groups={len(result['processing_families'])}; cross-pitch aliases={len(result['cross_pitch_aliases'])}")

if __name__ == "__main__": main()
