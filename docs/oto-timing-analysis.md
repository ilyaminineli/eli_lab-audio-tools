# OTO-aware timbre analysis

The deep analyser reads `oto.ini` files as read-only timing metadata. It does not alter aliases or timing values.

## Run with the same one-file output style as the earlier analyser

Choose the pitch-layer directory as the input and a JSON file as the output:

```powershell
eugene-timbre analyze "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE\5. G4" -o "G4_analysis.json"
```

For B2, change the input to `6. B2` and output file to `B2_analysis.json`. The command automatically discovers `oto.ini` in the selected folder, or the nearest parent folder. It writes one JSON document containing the manifest, group summaries, per-file metrics and all OTO alias matches.

Analyse the entire voicebank root in one JSON:

```powershell
eugene-timbre analyze "C:\Users\photo\OneDrive\Документы\UtauV\Data\Singers\EugeneRE" -o "EugeneRE_deep_analysis.json"
```

The CLI also supports `--oto-file "D:\\path\\oto.ini"` to specify one OTO file explicitly, and `--no-oto` to disable OTO alignment.

## Alternative: separate reports by folder

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE" --output "D:\AudioAnalysis\EugeneRE_deep"
```

This creates `bank_deep_analysis.json` plus one JSON file for each discovered group.

## Analyse one sample

```powershell
eugene-timbre analyze-deep-sample "D:\Voicebanks\EugeneRE\6. B2\a_a_i.wav" -o "D:\AudioAnalysis\a_a_i_deep.json"
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

## Output structure

- One-file output: top-level `manifest` and `groups` objects, with one group report containing per-WAV records and OTO alias entries.
- Separate-report output: `bank_deep_analysis.json` plus `b2_deep_analysis.json`, `f3_deep_analysis.json`, `g4_deep_analysis.json`, and reports for other folders.
- Each WAV can have multiple OTO aliases. The analyser keeps all matching entries rather than assuming one WAV equals one alias.

The OTO parser supports UTF-8 and CP932/Shift-JIS, preserving Japanese aliases in JSON. It only reads `oto.ini`; it does not rewrite it.

## Notes on interpretation

The fixed and stretchable regions are useful analysis windows, not a replacement for listening or for the commissioned timing decisions. Preutterance and overlap are reported as markers, not interpreted as waveform edits. A region that contains an attack or an unvoiced consonant may not produce a meaningful F0 estimate; that is expected.
