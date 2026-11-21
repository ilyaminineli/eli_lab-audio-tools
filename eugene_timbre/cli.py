"""CLI for OTO-aware voicebank timbre analysis and tonal processing."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from .analysis import analyze_audio
from .bank_analysis import analyze_bank
from .compare import compare_files
from .dsp import (
    WARM_UNCANNY_B2, F3_GENTLE, G4_NEUTRAL, NEUTRAL,
    apply_profile, describe_profile,
)
from .io import read_wav, write_wav
from .oto_analysis import analyse_oto_entry, find_nearest_oto, match_entries, parse_oto
from .timbre_analysis import TimbreAnalysisConfig, analyze_timbre

PROFILES = {
    "neutral": NEUTRAL,
    "warm-uncanny-b2": WARM_UNCANNY_B2,
    "f3-gentle": F3_GENTLE,
    "g4-neutral": G4_NEUTRAL,
}

def wavs(src: Path):
    if src.is_file():
        if src.suffix.lower() != ".wav":
            raise ValueError("Input must be WAV.")
        return [src]
    if not src.is_dir():
        raise FileNotFoundError(src)
    return sorted(p for p in src.rglob("*.wav") if p.is_file())

def make_config(args):
    return TimbreAnalysisConfig(
        frame_ms=args.frame_ms, hop_ms=args.hop_ms,
        f0_min_hz=args.f0_min, f0_max_hz=args.f0_max,
        max_harmonics=args.max_harmonics,
    )

def add_deep_options(parser):
    parser.add_argument("--frame-ms", type=float, default=85.0)
    parser.add_argument("--hop-ms", type=float, default=20.0)
    parser.add_argument("--f0-min", type=float, default=65.0)
    parser.add_argument("--f0-max", type=float, default=850.0)
    parser.add_argument("--max-harmonics", type=int, default=20)

def _sample_report(input_path: Path, config: TimbreAnalysisConfig,
                   oto_file: Path | None = None, alias: str | None = None,
                   use_oto: bool = True) -> dict:
    input_path = input_path.resolve()
    audio, sr, meta = read_wav(input_path)
    oto_report = {"status": "disabled", "entries": []}
    if use_oto:
        oto_path = Path(oto_file).resolve() if oto_file else find_nearest_oto(input_path, input_path.parent)
        if oto_path is None:
            oto_report = {"status": "oto_ini_not_found", "entries": []}
        else:
            parsed = parse_oto(oto_path)
            try:
                relative = input_path.relative_to(oto_path.parent).as_posix()
            except ValueError:
                relative = input_path.name
            matches = match_entries(parsed.index(), relative, input_path.name)
            if alias:
                matches = [entry for entry in matches if entry.alias == alias]
            oto_report = {
                "status": "matched" if matches else "no_matching_oto_entries",
                "oto_ini": str(oto_path),
                "encoding": parsed.encoding,
                "parse_errors": parsed.errors,
                "entry_count": len(matches),
                "entries": [analyse_oto_entry(entry, audio, sr, config) for entry in matches],
            }
    return {
        "file": str(input_path),
        "metadata": meta,
        "basic": analyze_audio(audio, sr),
        "timbre": analyze_timbre(audio, sr, config),
        "oto": oto_report,
    }

def analyze_basic(args):
    items = []
    for path in wavs(Path(args.input)):
        audio, sr, meta = read_wav(path)
        items.append({"file": str(path), "metadata": meta, "analysis": analyze_audio(audio, sr)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"files": items}, indent=2), encoding="utf-8")
    print(f"Basic analysis written: {args.output}")
    return 0

def analyze_sample_deep(args):
    report = _sample_report(
        Path(args.input), make_config(args),
        oto_file=Path(args.oto_file) if args.oto_file else None,
        alias=args.alias,
        use_oto=not args.no_oto,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Deep sample report written: {args.output}")
    return 0

def analyze_directory_json(args):
    root = Path(args.input).resolve()
    if root.is_file():
        report = _sample_report(
            root, make_config(args),
            oto_file=Path(args.oto_file) if args.oto_file else None,
            use_oto=not args.no_oto,
        )
        combined = {"report_type": "single_sample_deep_timbre_analysis", "files": [report]}
    else:
        with tempfile.TemporaryDirectory(prefix="eugene_timbre_") as temp:
            manifest = analyze_bank(
                root, Path(temp), make_config(args),
                oto_file=Path(args.oto_file) if args.oto_file else None,
                use_oto=not args.no_oto,
            )
            group_reports = {}
            for group_name, group_info in manifest["groups"].items():
                report_path = Path(temp) / group_info["report"]
                group_reports[group_name] = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            combined = {
                "report_type": "voicebank_deep_timbre_analysis",
                "manifest": manifest,
                "groups": group_reports,
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    file_count = (
        sum(group["sample_count"] for group in combined["groups"].values())
        if "groups" in combined else len(combined["files"])
    )
    print(f"Deep analysis written: {output} ({file_count} WAV(s)).")
    return 0

def analyze_bank_deep(args):
    groups = set(args.groups) if args.groups else None
    report = analyze_bank(
        Path(args.input), Path(args.output), make_config(args), groups,
        oto_file=Path(args.oto_file) if args.oto_file else None,
        use_oto=not args.no_oto,
    )
    print(
        f"Analysed {report['total_wav_files_analysed']} of "
        f"{report['total_wav_files_found']} WAV(s)."
    )
    print(f"Deep reports written to: {args.output}")
    oto = report["oto_analysis"]
    if oto["enabled"]:
        print(
            f"OTO: {oto['oto_ini_files_read']} file(s), "
            f"{oto['oto_entries_matched']} matched alias entries, "
            f"{oto['oto_entries_unmatched']} unmatched entries."
        )
    if report["failed_files"]:
        print(f"Failed WAVs: {len(report['failed_files'])}; see bank_deep_analysis.json")
    return 0

def compare(args):
    report = compare_files([Path(path) for path in args.analysis_files])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote subbank comparison: {args.output}")
    return 0

def process(args):
    src, dst = Path(args.input), Path(args.output)
    if src.resolve() == dst.resolve():
        raise ValueError("Output must not be the input folder.")
    profile = PROFILES[args.profile]
    items = []
    for path in wavs(src):
        audio, sr, meta = read_wav(path)
        restored = apply_profile(audio, sr, profile, args.wet)
        rel = Path(path.name) if src.is_file() else path.relative_to(src)
        out = dst / rel
        write_wav(out, restored, sr, meta["subtype"])
        items.append({
            "input": str(path), "output": str(out),
            "before": analyze_audio(audio, sr), "after": analyze_audio(restored, sr),
        })
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "restoration_report.json").write_text(json.dumps({
        "profile": describe_profile(profile), "wet": args.wet,
        "processed_count": len(items), "files": items,
        "notice": "Experimental tonal-balance test. Listen and render in target resamplers before using in a release bank.",
    }, indent=2), encoding="utf-8")
    print(f"Processed {len(items)} WAV(s) to {dst}. Originals untouched.")
    return 0

def main():
    parser = argparse.ArgumentParser(prog="eugene-timbre")
    sub = parser.add_subparsers(dest="cmd", required=True)

    deep = sub.add_parser(
        "analyze",
        help="Deep-analyse a WAV or voicebank folder and write one JSON file."
    )
    deep.add_argument("input", help="A WAV or voicebank/pitch-layer folder.")
    deep.add_argument("-o", "--output", default="timbre_analysis.json")
    deep.add_argument("--oto-file", help="Optional explicit oto.ini; otherwise auto-discovered.")
    deep.add_argument("--no-oto", action="store_true", help="Disable OTO timing analysis.")
    add_deep_options(deep)
    deep.set_defaults(func=analyze_directory_json)

    basic = sub.add_parser("analyze-basic", help="Run the earlier whole-file spectral summary only.")
    basic.add_argument("input")
    basic.add_argument("-o", "--output", default="timbre_analysis_basic.json")
    basic.set_defaults(func=analyze_basic)

    sample = sub.add_parser(
        "analyze-deep-sample",
        help="Deep harmonic, spectral-envelope and optional OTO analysis of one WAV."
    )
    sample.add_argument("input")
    sample.add_argument("-o", "--output", default="deep_timbre_analysis.json")
    sample.add_argument("--oto-file", help="Optional explicit oto.ini; otherwise auto-discovered beside the WAV.")
    sample.add_argument("--alias", help="Optional exact OTO alias to analyse.")
    sample.add_argument("--no-oto", action="store_true")
    add_deep_options(sample)
    sample.set_defaults(func=analyze_sample_deep)

    bank = sub.add_parser(
        "analyze-deep-bank",
        help="Recursively analyse a voicebank into per-folder JSON reports."
    )
    bank.add_argument("input", help="Voicebank folder; a pitch-layer folder such as '5. G4' is accepted.")
    bank.add_argument("-o", "--output", required=True, help="Separate report directory.")
    bank.add_argument("--groups", nargs="*", help="Optional group filter, e.g. B2 F3 G4.")
    bank.add_argument("--oto-file", help="Optional explicit oto.ini; otherwise each WAV uses the nearest oto.ini.")
    bank.add_argument("--no-oto", action="store_true", help="Disable OTO parsing.")
    add_deep_options(bank)
    bank.set_defaults(func=analyze_bank_deep)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("analysis_files", nargs="+", help="Subbank analysis JSON files.")
    compare_parser.add_argument("-o", "--output", default="subbank_comparison.json")
    compare_parser.set_defaults(func=compare)

    proc = sub.add_parser("process")
    proc.add_argument("input")
    proc.add_argument("-o", "--output", required=True)
    proc.add_argument("--profile", choices=sorted(PROFILES), default="warm-uncanny-b2")
    proc.add_argument("--wet", type=float, default=1.0)
    proc.set_defaults(func=process)

    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
