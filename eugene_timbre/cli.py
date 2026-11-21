"""CLI for batch WAV analysis, comparison and conservative timbre processing."""
import argparse, json, sys
from pathlib import Path
from .analysis import analyze_audio
from .compare import compare_files
from .dsp import WARM_UNCANNY_B2, F3_GENTLE, G4_NEUTRAL, NEUTRAL, apply_profile, describe_profile
from .io import read_wav, write_wav

PROFILES = {
    "neutral": NEUTRAL,
    "warm-uncanny-b2": WARM_UNCANNY_B2,
    "f3-gentle": F3_GENTLE,
    "g4-neutral": G4_NEUTRAL,
}

def wavs(src: Path):
    if src.is_file():
        if src.suffix.lower() != ".wav": raise ValueError("Input must be WAV.")
        return [src]
    if not src.is_dir(): raise FileNotFoundError(src)
    return sorted(p for p in src.rglob("*.wav") if p.is_file())

def analyze(args):
    items=[]
    for p in wavs(Path(args.input)):
        audio,sr,meta=read_wav(p)
        items.append({"file":str(p),"metadata":meta,"analysis":analyze_audio(audio,sr)})
    Path(args.output).write_text(json.dumps({"files":items},indent=2),encoding="utf-8")
    print(f"Analysed {len(items)} file(s): {args.output}")
    return 0

def compare(args):
    report=compare_files([Path(p) for p in args.analysis_files])
    Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Wrote subbank comparison: {args.output}")
    return 0

def process(args):
    src,dst=Path(args.input),Path(args.output)
    if src.resolve()==dst.resolve(): raise ValueError("Output must not be the input folder.")
    profile=PROFILES[args.profile]
    items=[]
    for p in wavs(src):
        audio,sr,meta=read_wav(p)
        restored=apply_profile(audio,sr,profile,args.wet)
        rel=Path(p.name) if src.is_file() else p.relative_to(src)
        out=dst/rel
        write_wav(out,restored,sr,meta["subtype"])
        items.append({"input":str(p),"output":str(out),
                      "before":analyze_audio(audio,sr),"after":analyze_audio(restored,sr)})
    dst.mkdir(parents=True,exist_ok=True)
    (dst/"restoration_report.json").write_text(json.dumps({
        "profile":describe_profile(profile),"wet":args.wet,
        "processed_count":len(items),"files":items,
        "notice":"Experimental tonal-balance test. Listen and render in target resamplers before using in a release bank.",
        "cache_warning":"When replacing WAVs in a duplicate voicebank, refresh the engine analyses (.frq, .llsm, and any TIPS cache) for the changed samples. This tool does not alter caches or oto.ini."
    },indent=2),encoding="utf-8")
    print(f"Processed {len(items)} WAV(s) to {dst}. Originals and sidecar files untouched.")
    return 0

def main():
    p=argparse.ArgumentParser(prog="eugene-timbre")
    s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("analyze"); a.add_argument("input"); a.add_argument("-o","--output",default="timbre_analysis.json"); a.set_defaults(func=analyze)
    c=s.add_parser("compare"); c.add_argument("analysis_files",nargs="+",help="B2/F3/G4 analysis JSON files"); c.add_argument("-o","--output",default="subbank_comparison.json"); c.set_defaults(func=compare)
    b=s.add_parser("process"); b.add_argument("input"); b.add_argument("-o","--output",required=True)
    b.add_argument("--profile",choices=sorted(PROFILES),default="warm-uncanny-b2")
    b.add_argument("--wet",type=float,default=1.0); b.set_defaults(func=process)
    args=p.parse_args()
    try: return args.func(args)
    except Exception as e:
        print(f"Error: {e}",file=sys.stderr); return 1

if __name__=="__main__": raise SystemExit(main())
