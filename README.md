# eli_lab-audio-tools

Free ELI LAB audio utilities. The Eugene:RE workflow is designed to prepare existing artificial voicebank recordings for better UTAU resampling, with a simple scan → preview → export workflow. The source bank is never edited in place.

## Primary design goal: natural resampling

The programme is not intended to make the dry source WAVs generically brighter or cleaner. It should prepare a separate voicebank copy that gives Moresampler, TIPS and Worldline-family renderers useful source material while preserving Eugene's harmonic character, breath/aperiodic texture, vowel resonance, articulation and time-varying expression.

The existing pitch EQ recommendations are exploratory listening-test starting points, not validated naturalness corrections. Do not treat them as the final automatic restoration system. The resampler-aware design and the next analysis stages are documented in [docs/resampler-aware-naturalness-design.md](docs/resampler-aware-naturalness-design.md).

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Quick Restore — recommended workflow

Launch the simple interface:

```powershell
pip install -e ".[ui]"
eugene-timbre studio
```

1. Point **Input bank** at the existing Eugene:RE folder. The supplied path is prefilled.
2. Point **Output copy** at a separate folder. The supplied path is prefilled.
3. Click **Scan bank and prepare preview**. The app scans the WAV inventory and OTO-safe regions, then selects a representative pitch-layer sample.
4. Listen to **Play original** and **Play processed preview**. The preview uses the exact same OTO-aware routing/profile logic as export.
5. Click **Create processed voicebank copy**. The app exports the full directory tree to the output folder, retaining filenames, folder layout, OTO metadata and unchanged samples. The source bank is never modified.

The first automatic recipe is deliberately conservative and is still an experimental baseline:
- B2: small +0.6 dB body/presence lift around 1.45 kHz, only in eligible OTO stretchable regions.
- F3: small +0.3 dB lift around 1.45 kHz, only in eligible OTO stretchable regions.
- G4, L/V/TH, R English, consonant releases, bonus phonetic extras and root material are copied unchanged until a suitable, tested recipe exists.
- Samples with duplicate/unknown OTO aliases, timing warnings or insufficient safe body coverage are copied unchanged and listed in `processing_report.json`.

The output is a complete mirrored bank, not just a folder of changed WAVs. Back up the original before replacing it. Review `processing_report.json` to see exactly which files changed and which were copied unchanged.

The former multi-panel editing interface remains available as:

```powershell
eugene-timbre studio-advanced
```

The executable shortcut `eugene-timbre-ui` launches Quick Restore.

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

The pitch-folder-first workflow is the recommended place to begin new restoration work. The studio now indexes the complete voicebank hierarchy first, including pitch folders, L/V/TH extensions, R English, consonant releases, bonus phonetic extras, setup tools, documentation, illustrations, and root-level files. The BANK STRUCTURE tree displays all folders and file names plus WAV/FRQ/LLSM/OTO counts. Only the actual pitch layers (for example, B2, F3 or G4) are currently enabled for this pitch-specific EQ export; add-on folders are inventoried separately and are not silently processed with a pitch EQ.

Install the optional graphical interface on Windows:

```powershell
pip install -e ".[ui]"
eugene-timbre studio
```

You can also launch it with `eugene-timbre-ui`. The UI requires PySide6; the core CLI and DSP do not.

Workflow:

1. The default paths are prefilled for the supplied Eugene:RE setup. Choose the voicebank root if it differs, then click **Scan whole bank**. The scanner indexes every folder and file recursively and shows the hierarchy in **BANK STRUCTURE**. WAV, FRQ, LLSM and `oto.ini` counts are included.
2. Nothing is preselected for export. Use **Select current**, check individual folders, or explicitly use **Select all** when intended.
3. Select a pitch layer and run **Analyse checked**. The report includes median F0, spectral centroid, harmonic/residual proxy, relative band energy and a median cepstral spectral envelope.
4. After analysis, the studio automatically generates four conservative source-EQ starting points for each pitch layer: **Clarity / presence**, **Warm / body**, **Air / detail**, and **Soft / dark**, plus a neutral control. Their gains are informed by that layer's measured centroid, relative-band energy, H/R proxy and spectral flatness. They are bounded to ±1.5 dB and keep high-pass OFF; they are listening-test suggestions, not an objective correction.
5. Choose an approach and click **Apply to current** or **Apply to checked** to load the resulting pitch-specific gains into the visible EQ controls. **Generate + process checked using selected approach** runs that recommendation and exports the checked pitch folders in one confirmed action. Alternatively, audition first and use **Process checked pitch folders** after manual tuning. The source bank is never modified.
6. The graph overlays the measured spectral envelope and live EQ response. The preview reports the measured peak delta, changed-sample fraction and delta RMS. A neutral profile (all gains at 0 dB, high-pass OFF) makes no audible or numerical change.
7. Optionally choose another analysed pitch as a reference and use **Match reference pitch**. This compares the measured cepstral envelopes and fits a bounded five-band approximation; it is not a formant model or naturalness score.
8. Save/load profiles as JSON. Each pitch layer keeps its own separate profile.
5. Optionally choose an analysed reference pitch and generate a bounded five-band EQ suggestion from the difference between the measured cepstral spectral envelopes. If either envelope is unavailable, it falls back to broad relative-band-energy matching. This is an approximation, not a formant model or a naturalness score.
6. Select a sample and audition the original and processed preview at the chosen wet blend. The preview reports the measured peak delta, changed-sample fraction and delta RMS. A neutral profile (all gains at 0 dB, high-pass OFF) makes no audible or numerical change, so the app warns before export or Moresampler A/B.
9. Exported pitch folders are written to a separate output root. The source bank is never modified. Non-WAV files, including `oto.ini`, FRQ and LLSM data, are copied byte-for-byte.

### Moresampler phone A/B testing

Source-WAV EQ and Moresampler renderer flags are separate controls. The **Moresampler style template** dropdown provides small starter flag sets (Intimate/breathier, Tense/projected, Open vowels, Formant emphasis, Drier, Rough/growl). These flags change Moresampler's synthesis model; they do not rewrite the source WAV. Enable **A/B test selected Moresampler flags against neutral flags** to compare the chosen flag set with the same original sample rendered using no flags.

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


### Eugene:RE paths prefilled in the studio

The UI starts with these editable defaults:

- Input: `C:/Users/photo/OneDrive/Документы/UtauV/Data/Singers/EugeneRE`
- Output: `C:/EugeneRE_Restoration`
- Moresampler: `C:/Users/photo/OneDrive/Документы/UtauV/Data/Resamplers/moresampler.exe`

The app scans automatically on launch only when the default input folder exists on the computer. The initial pitch EQ remains neutral; use **Suggest envelope match** or adjust the visible bands/high-pass controls to create an explicit filter profile. The OTO body mask is OFF by default and is clearly described in the processing status.


## UTAU rendering and restoration design

See [docs/utau-rendering-and-profile-approaches.md](docs/utau-rendering-and-profile-approaches.md) for the documented phonemizer → resampler → wavtool pipeline, OTO timing, Moresampler's analysis/synthesis settings and flags, and the boundaries between source EQ, renderer behaviour and final phrase mixing.

The current studio's automatic recommendations are deliberately conservative. It does not guess a universal ideal vocal tone, retune OTO entries, or apply reverb/compression to the voicebank. Renderer flags are only meaningful when the selected resampler supports them. Compare identical phrases at matched loudness before adopting a recipe.
