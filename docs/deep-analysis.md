# Eugene:RE Deep Timbre Analysis

This is a measurement-only stage for studying existing voicebank samples before designing timbre-processing profiles. It does not change WAVs, OTO settings or the source folder.

## Analyse the whole voicebank

Choose an output directory separate from the source bank:

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE" --output "D:\AudioAnalysis\EugeneRE_deep"
```

The scanner groups folders by pitch labels such as B2, F3 and G4. Other folders are reported under their folder names (for example consonant releases, L, V, TH, English and bonus extras). Output contains:

- `bank_deep_analysis.json` — manifest with group summaries and failure list.
- `b2_deep_analysis.json`, `f3_deep_analysis.json`, `g4_deep_analysis.json` — detailed pitch-layer reports.
- One report for every other discovered group.

Analyse one pitch layer:

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE" --groups B2 --output "D:\AudioAnalysis\EugeneRE_B2_deep"
```

Analyse one representative WAV:

```powershell
eugene-timbre analyze-deep-sample "D:\Voicebanks\EugeneRE\6. B2\-_a.wav" -o "D:\AudioAnalysis\B2_a_deep.json"
```

## Measurements

- **F0 trajectory:** autocorrelation-based fundamental-frequency estimate, voiced-frame fraction, confidence, pitch percentiles, pitch range and local frame-to-frame variation.
- **Harmonic profile:** relative amplitudes of H1 through the configured harmonic count, harmonic slope, H1-H2 and H2-H4 relationships.
- **Harmonic/residual ratio:** FFT-bin energy proxy for how much energy is concentrated near the tracked harmonic series. It is not a laboratory HNR measurement.
- **Inharmonicity:** amplitude-weighted deviation of local spectral peaks from integer multiples of tracked F0.
- **Spectral envelope:** real-cepstrum-smoothed magnitude envelope, normalised against the median level from 250–1,500 Hz, with p10/median/p90 curves and broad envelope peaks.
- **Temporal behaviour:** per-frame F0, RMS, spectral centroid, flatness, positive spectral flux, harmonic/residual ratio, inharmonicity and harmonic slope.

## Default analysis settings

Defaults: 85 ms frame, 20 ms hop, 65–850 Hz F0 range and 20 harmonics. Adjust for a voicebank outside this range:

```powershell
eugene-timbre analyze-deep-bank "D:\Voicebanks\EugeneRE" --groups G4 --f0-min 120 --f0-max 1000 --output "D:\AudioAnalysis\EugeneRE_G4_deep"
```

## Interpretation limits

1. Whole-file summaries mix vowels, transitions and consonants. Compare equivalent aliases between pitch layers and inspect frame tracks.
2. Broad envelope peaks are descriptive spectral resonances, not guaranteed vocal-tract formants.
3. A low local-F0-variation value is not automatically a defect. A deliberately steady synthetic performance may be correct.
4. Harmonic/residual and inharmonicity values are analysis proxies, not standalone naturalness scores.
5. Do not infer EQ cuts from whole-file averages alone. Compare envelope curves from equivalent sustained vowels and audition the result through the target resamplers.

The JSON structure is designed for a later GUI workflow: select a bank folder, choose a subbank, preview one or two samples, inspect analysis and compare proposed processing before exporting into a separate output folder.
