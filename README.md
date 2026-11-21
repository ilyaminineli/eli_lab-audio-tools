# eli_lab-audio-tools

Free ELI LAB audio utilities. The current project focuses on Eugene:RE timbre analysis and non-destructive restoration experiments.

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Main analysis command: one JSON per selected pitch layer

This keeps the same input/output style as the earlier analyser. It now performs deep harmonic, spectral-envelope and OTO-timing analysis, automatically discovering the nearest `oto.ini`.

```powershell
eugene-timbre analyze "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE\5. G4" -o "G4_analysis.json"
```

For B2, change the input folder to `6. B2` and output file to `B2_analysis.json`. Use a destination outside the input folder when you want to keep the voicebank directory unchanged.

Analyse the entire voicebank root in one JSON:

```powershell
eugene-timbre analyze "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE" -o "EugeneRE_deep_analysis.json"
```

Useful options:

- `--oto-file "D:\\path\\oto.ini"` — use one explicit OTO file instead of auto-discovery.
- `--no-oto` — disable OTO alignment.
- `--f0-min 65 --f0-max 850` — adjust the fundamental-frequency search range.
- `--frame-ms 85 --hop-ms 20` — adjust analysis window and hop.

## Alternative: separate report files by group

```powershell
eugene-timbre analyze-deep-bank "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE" --output "C:\AudioAnalysis\EugeneRE_deep"
```

This writes a manifest plus separate JSON files for B2, F3, G4 and other folders. Add `--groups B2 F3 G4` to select only those groups.

## Analyse one WAV

```powershell
eugene-timbre analyze-deep-sample "D:\Voicebanks\EugeneRE\6. B2\a_a_i.wav" -o "D:\AudioAnalysis\a_a_i_deep.json"
```

Add `--alias "a あ_B2"` to select one exact alias.

## Earlier broad analysis only

The original whole-file energy summary is still available as:

```powershell
eugene-timbre analyze-basic "D:\Voicebanks\EugeneRE\6. B2" -o "B2_basic_analysis.json"
```

The deep analyser measures F0 trajectories, harmonic amplitudes, harmonic/residual energy, inharmonicity, cepstral spectral envelopes, frame-level variation and OTO-aligned source/fixed/stretchable regions. It preserves all original OTO timing values and never modifies the voicebank. See [docs/deep-analysis.md](docs/deep-analysis.md) and [docs/oto-timing-analysis.md](docs/oto-timing-analysis.md).

## Experimental tonal processing

```powershell
eugene-timbre process "D:\Voicebanks\EugeneRE\6. B2\test_samples" --profile warm-uncanny-b2 --wet 1.0 --output "D:\Voicebanks\Eugene_RE_Restored_TEST\6. B2"
```

These profiles remain listening-test starting points, not automatic de-synthesis. Compare at matched loudness and render identical UST notes through the target resamplers.

Dependencies: NumPy, SciPy and SoundFile. No cloud service, subscription or AI inference is required.
