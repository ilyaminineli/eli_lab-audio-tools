"CLI for bank analysis, detailed timbre measurement and tonal processing."""
import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze_audio
from .bank_analysis import analyze_bank
from .compare import compare_files
from .dsp import (
    WARM_UNCANNY_B2, F3_GENTLE, G4_NEUTRAL, NEUTRAL,
    apply_profile, describe_profile,
)
from .io import read_wav, write_wav
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

def analyze(args):
    items = []
    for path in wavs(Path(args.input)):
        audio, sr, meta = read_wav(path)
        items.append({"file": str(path), "metadata": meta, "analysis": analyze_audio(audio, sr)})
    Path(args.output).write_text(json.dumps({"files": items}, indent=2), encoding="utf-8")
    print(f"Analysed {len(items)} file(s): {args.output}")
    return 0

def make_config(args):
    return TimbreAnalysisConfig(
        frame_ms=args.frame_ms, hop_ms=args.hop_ms,
        f0_min_hz=args.f0_min, f0_max_hz=args.f0_max,
        max_harmonics=args.max_harmonics,
    )

def analyze_sample_deep(args):
    audio, sr, meta = read_wav(Path(args.input))
    report = {
        "file": str(Path(args.input).resolve()),
        "metadata": meta,
        "basic": analyze_audio(audio, sr),
        "timbre": analyze_timbre(audio, sr, make_config(args)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Deep sample report written: {args.output}")
    return 0

def analyze_bank_deep(args):
    groups = set(args.groups) if args.groups else None
    report = analyze_bank(Path(args.input), Path(args.output), make_config(args), groups)
    print(f"Analysed {report['total_wav_files_analysed']} of {report['total_wav_files_found']} WAV(s).")
    print(f"Deep reports written to: {args.output}")
    if report["failed_files"]:
        print(f"Failed files: {len(report['failed_files'])}; see bank_deep_analysis.json")
    return 0

def compare(args):
    report = compare_files([Path(path) for path in args.analysis_files])
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

def add_deep_options(parser):
    parser.add_argument("--frame-ms", type=float, default=85.0)
    parser.add_argument("--hop-ms", type=float, default=20.0)
    parser.add_argument("--f0-min", type=float, default=65.0)
    parser.add_argument("--f0-max", type=float, default=850.0)
    parser.add_argument("--max-harmonics", type=int, default=20)

def main():
    parser = argparse.ArgumentParser(prog="eugene-timbre")
    sub = parser.add_subparsers(dest="cmd", required=True)

    basic = sub.add_parser("analyze")
    basic.add_argument("input")
    basic.add_argument("-o", "--output", default="timbre_analysis.json")
    basic.set_defaults(func=analyze)

    sample = sub.add_parser("analyze-deep-sample", help="Deep harmonic and spectral-envelope analysis of one WAV.")
    sample.add_argument("input")
    sample.add_argument("-o", "--output", default="deep_timbre_analysis.json")
    add_deep_options(sample)
    sample.set_defaults(func=analyze_sample_deep)

    bank = sub.add_parser("analyze-deep-bank", help="Recursively analyse a voicebank into per-folder JSON reports.")
    bank.add_argument("input")
    bank.add_argument("-o", "--output", required=True, help="Separate report directory.")
    bank.add_argument("--groups", nargs="*", help="Optional group filter, e.g. B2 F3 G4.")
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
