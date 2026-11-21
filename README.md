# eli_lab-audio-tools

Free ELI LAB audio utilities. This repository includes **Eugene:RE Timbre Restoration**, a non-destructive analysis and tonal-processing utility for existing RVC-derived UTAU source WAVs.

## Scope

The tool is currently focused on Eugene:RE. It analyses source timbre and offers experimental EQ profiles; it does not rebuild or retime the voicebank. It does not edit OTO settings, aliases, offsets, phoneme boundaries, filenames, sample duration or pitch. Analysis-only commands never modify the input WAVs. Processing writes to a separate output directory.

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Deep timbre analysis

Analyse the entire voicebank and write a separate JSON report for each pitch layer / folder group:

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE" --output "D:\AudioAnalysis\EugeneRE_deep"
```

Analyse only B2:

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE" --groups B2 --output "D:\AudioAnalysis\EugeneRE_B2_deep"
```

Analyse one sample in detail:

```powershell
eugene-timbre analyze-deep-sample "D:\Voicebanks\EugeneRE\6. B2\-_a.wav" -o "D:\AudioAnalysis\B2_a_deep.json"
```

The deep analyser measures F0 trajectories, harmonic amplitudes and spectral slope, harmonic/residual energy, inharmonicity, cepstral spectral envelopes, voiced-frame statistics and temporal variation. Reports include per-frame tracks and bank-level summaries. See [docs/deep-analysis.md](docs/deep-analysis.md) for the metrics and interpretation limits.

## Compare the earlier broad subbank analyses

```powershell
eugene-timbre compare B2_analysis.json F3_analysis.json G4_analysis.json -o subbank_comparison.json
```

## Analyse broad WAV statistics

```powershell
eugene-timbre analyze "D:\Voicebanks\EugeneRE\6. B2" -o "B2_analysis_v2.json"
```

## Experimental tonal processing

```powershell
eugene-timbre process "D:\Voicebanks\EugeneRE\6. B2\test_samples" --profile warm-uncanny-b2 --wet 1.0 --output "D:\Voicebanks\Eugene_RE_Restored_TEST\6. B2"
```

Profiles currently include a conservative B2 test, a gentler F3 test, neutral G4 and bypass reference. These are listening-test profiles, not automatic de-synthesis. Match loudness when comparing and render identical UST notes through the target resamplers. Static EQ cannot restore missing aperiodic detail or fix conversion phase artifacts.

## Python API

```python
import soundfile as sf
from eugene_timbre.timbre_analysis import analyze_timbre

audio, sr = sf.read("sample.wav", dtype="float32")
report = analyze_timbre(audio, sr)
print(report["harmonic_to_residual_ratio_db"])
print(report["spectral_envelope"]["median_relative_db"])
```

Dependencies: NumPy, SciPy and SoundFile. No cloud service, subscription or AI inference is required.
