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


## Eugene:RE Pitch Restoration Studio

The pitch-folder-first workflow is the recommended place to begin new restoration work. It does not scan or automatically classify the entire bank: you select individual pitch folders (for example, B2, F3 or G4), analyse them, tune a separate profile for each pitch, audition a source sample against a processed preview, and export selected pitch folders to a separate output directory.

Install the optional graphical interface on Windows:

```powershell
pip install -e ".[ui]"
eugene-timbre studio
```

You can also launch it with `eugene-timbre-ui`. The UI requires PySide6; the core CLI and DSP do not.

Workflow:

1. Choose the voicebank root. The scanner lists immediate subfolders containing a pitch token, such as `6. B2`, `4. F3` and `5. G4`.
2. Nothing is preselected for export. Use **Select current**, check individual folders, or explicitly use **Select all** when intended.
3. Select a layer and run **Analyse checked**. The report includes median F0, spectral centroid, harmonic/residual proxy, relative band energy and a median cepstral spectral envelope.
4. Set that pitch's five-band EQ profile. Each pitch stores its own settings.
5. Optionally choose an analysed reference pitch and generate a bounded five-band EQ suggestion from the difference between the measured cepstral spectral envelopes. If either envelope is unavailable, it falls back to broad relative-band-energy matching. This is an approximation, not a formant model or a naturalness score.
6. Select a sample and audition the original and processed preview at the chosen wet blend.
7. Export checked pitch folders to a separate output root. The source bank is never modified. Non-WAV files, including `oto.ini`, are copied byte-for-byte.

### Moresampler phone A/B testing

The studio can render one selected WAV/OTO alias through a local `moresampler.exe`:

1. Select the Moresampler executable.
2. Choose the WAV and matching OTO alias.
3. The target note is initially suggested from the measured sample F0; adjust it when testing a different destination note.
4. Set the requested phone duration and, optionally, Moresampler flags.
5. Click **Render original + restored through Moresampler**.

Both variants use the same classic UTAU resampler arguments: note name, velocity, flags, OTO offset, duration, consonant, cutoff, volume, modulation, tempo and a constant-pitch curve. The test runs a temporary copy of `moresampler.exe` and its adjacent DLLs, with an isolated global `moreconfig.txt`; your installed executable folder and existing configuration are not edited. Global output settings enable resampler compatibility and match the source sample rate. Each temporary pitch folder gets a separate voicebank `moreconfig.txt` that preserves existing analysis/synthesis options and sets the F0-analysis bounds to 0.6×–1.7× the measured sample F0.

Outputs are written beneath `moresampler_ab/<pitch>/<sample>/<note>_<duration>ms_<timestamp>/`:
- `original_moresampler.wav`
- `restored_moresampler.wav`
- `moresampler_ab_report.json`

This is a **single phone/alias render**, not a full UST phrase or wavtool render. It is useful for checking whether a restoration survives Moresampler's own source analysis and resynthesis, but it cannot yet evaluate cross-phoneme transitions, note-to-note continuity or full-song phrasing. Listen to both WAVs at matched loudness; the report's spectral metrics are diagnostics, not intelligibility or naturalness scores.

**OTO protection is optional.** When enabled, the studio uses the matching OTO entries to protect fixed regions and skips files without a usable stretchable body region. When disabled, the selected pitch-folder WAVs are processed as complete recordings. This mode deliberately has no whole-file 5% coverage threshold and does not run phonetic alias classification across unrelated folders.

Every processed pitch folder receives a `pitch_restoration_report.json` with before/after loudness, spectral centroid, band-energy changes, peak sample difference, changed-sample fraction and per-file status. A file marked changed has a measurable sample delta; this does not prove that it sounds more natural.

The envelope-match suggestion is only a starting point: it approximates a measured cepstral-envelope difference with five broad EQ bands. It does not infer the ideal vocal-tract envelope, separate individual formants, or guarantee greater naturalness. Choose the target resampler, render identical short phrases, loudness-match the renders, and listen for vowel identity, consonant intelligibility, transitions, and artifacts before adopting a profile.

## Legacy automatic whole-bank processing chain (experimental v1)

The analyser also includes a conservative auto-routing stage for pitch layers, consonant releases, L/V/TH extensions, R English, bonus phonetic events and unknown/root-level files.

Preview routing first:

\`\`\`powershell
eugene-timbre process-auto "C:\\Users\\photo\\OneDrive\\Документы\\UtauV\\Data\\Singers\\EugeneRE" -o "D:\\EugeneRE_Restoration_DRYRUN" --dry-run
\`\`\`

Run the initial non-destructive test chain into a separate voicebank folder:

\`\`\`powershell
eugene-timbre process-auto "C:\\Users\\photo\\OneDrive\\Документы\\UtauV\\Data\\Singers\\EugeneRE" -o "D:\\EugeneRE_Restoration_TEST" --wet 0.5
\`\`\`

Only B2 and F3 currently receive tiny pitch-body presence tests. G4, consonant releases, L/V/TH, R English and bonus events have their own routes and remain neutral/bypassed pending listening tests. Processing is restricted to eligible OTO stretchable regions. If any alias marks part of a shared WAV as fixed, the chain protects that area.

Before a first wet run, inspect the dry-run report's per-alias OTO windows, timing warnings, unmatched/orphan OTO entries, duplicate aliases and alias-token audit. Samples with less than 5% eligible mask coverage are marked `partial` and copied unchanged by default; samples with missing/ambiguous aliases or invalid OTO timing are marked `manual_review`. Only use `--allow-partial` after reviewing the low-coverage samples. Filtered segments receive 80 ms of surrounding source context, while only the eligible OTO mask is blended into the output. OTO files and other non-WAV bank files are copied unchanged to the output tree.

This is a first comparison chain, not a final restoration preset. Do not overwrite the original bank. Render matching UST phrases in the target resamplers and compare the dry-run report, originals and processed output at matched loudness. See [docs/processing-chain-design.md](docs/processing-chain-design.md).
