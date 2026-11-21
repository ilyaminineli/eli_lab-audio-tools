# OTO-aware timbre analysis

The deep bank analyser reads `oto.ini` files as read-only timing metadata. It does not alter aliases or timing values.

## Run against one pitch-layer folder

The selected folder can be a pitch layer such as `5. G4`. The analyser automatically finds the nearest `oto.ini` in that folder or its parent folders.

```powershell
eugene-timbre analyze-deep-bank "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE\5. G4" --output "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE\G4_deep_analysis"
```

For B2, use the same command with `6. B2` as the input folder and a separate B2 output directory.

Use an explicit OTO file when necessary:

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE\6. B2" --oto-file "D:\Voicebanks\EugeneRE\6. B2\oto.ini" --output "D:\AudioAnalysis\B2_deep"
```

To analyse one WAV and all of its OTO aliases:

```powershell
eugene-timbre analyze-deep-sample "D:\Voicebanks\EugeneRE\6. B2\a_a_i.wav" --output "D:\AudioAnalysis\a_a_i_deep.json"
```

Add `--alias "a あ_B2"` to restrict the single-sample report to one exact alias.

## OTO fields and derived regions

The parser reads the standard line structure:

`wav=alias,offset,consonant,cutoff,preutter,overlap`

All timing values are retained in milliseconds. It also calculates absolute marker positions relative to the WAV timeline:

- `offset`: start of the selected source window.
- `fixed_region_end = offset + consonant`.
- `preutter_marker = offset + preutter`.
- `overlap_marker = offset + overlap`.
- `usable_end`: for non-negative cutoff, `duration - cutoff`; for negative cutoff, `offset - cutoff`.

The negative-cutoff coordinate follows OpenUtau's OTO display implementation. The output divides the selected source window into:

- `fixed_region`: offset to the fixed-region end (bounded by usable end).
- `stretchable_region`: fixed-region end to usable end.
- `source_window`: the full OTO-selected span.
- `before_entry_window` and `after_entry_window`: material excluded from this OTO entry, not assumed to be silence.

For each alias, the report includes OTO line number, decoded alias, original timing values, calculated markers, warnings, and separate analysis of the source, fixed and stretchable regions. Regions shorter than 60 ms receive a basic spectral summary where possible but are skipped for F0/harmonic-envelope analysis, because they are too short for a reliable voiced-frame analysis.

## Output

- `bank_deep_analysis.json`: bank manifest, OTO file list, parse errors, unmatched aliases and group summaries.
- `b2_deep_analysis.json`, `f3_deep_analysis.json`, `g4_deep_analysis.json`: per-sample analysis with a list of OTO alias entries for each WAV.

A WAV may have multiple OTO aliases. The analyser therefore keeps all matching entries rather than assuming one WAV equals one alias.

The OTO parser supports UTF-8 and CP932/Shift-JIS, preserving Japanese aliases in JSON. It only reads `oto.ini`; it does not rewrite it.

## Notes on interpretation

The fixed and stretchable regions are useful analysis windows, not a replacement for listening or for the commissioned timing decisions. Preutterance and overlap are reported as markers, not interpreted as waveform edits. A region that contains an attack or an unvoiced consonant may not produce a meaningful F0 estimate; that is expected.
