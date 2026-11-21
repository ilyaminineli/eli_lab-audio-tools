# eli_lab-audio-tools

Free ELI LAB audio utilities. The current project focuses on Eugene:RE timbre analysis and non-destructive restoration experiments.

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## OTO-aware deep timbre analysis

Analyse one entire pitch-layer folder. The analyser will discover the nearest `oto.ini` automatically and attach each matching alias and timing window to the sample's JSON analysis.

```powershell
eugene-timbre analyze-deep-bank "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE\5. G4" --output "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE\G4_deep_analysis"
```

For B2, change the input to `6. B2` and use a separate output folder. The output path must not be the input folder.

Analyse the entire voicebank root, with each subfolder's nearest `oto.ini`:

```powershell
eugene-timbre analyze-deep-bank "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE" --output "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE_deep"
```

Optional flags:

- `--groups B2 F3 G4` — analyse only selected groups.
- `--oto-file "D:\\path\\oto.ini"` — use one explicit OTO file.
- `--no-oto` — run the deep waveform analysis without OTO alignment.

For one sample:

```powershell
eugene-timbre analyze-deep-sample "D:\Voicebanks\EugeneRE\6. B2\a_a_i.wav" --output "D:\AudioAnalysis\a_a_i_deep.json"
```

Add `--alias "a あ_B2"` to select a single exact alias.

The analyser reports F0 trajectories, harmonic amplitudes, harmonic/residual energy, inharmonicity, cepstral spectral envelopes, frame-level variation, and OTO-aligned source/fixed/stretchable regions. It preserves all original OTO timing values and never modifies the voicebank. See [docs/deep-analysis.md](docs/deep-analysis.md) and [docs/oto-timing-analysis.md](docs/oto-timing-analysis.md).

## Experimental tonal processing

```powershell
eugene-timbre process "D:\Voicebanks\EugeneRE\6. B2\test_samples" --profile warm-uncanny-b2 --wet 1.0 --output "D:\Voicebanks\Eugene_RE_Restored_TEST\6. B2"
```

These profiles remain listening-test starting points, not automatic de-synthesis. Compare at matched loudness and render identical UST notes through the target resamplers.

Dependencies: NumPy, SciPy and SoundFile. No cloud service, subscription or AI inference is required.
