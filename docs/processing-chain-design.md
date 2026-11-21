# Eugene:RE analysis findings and processing chain v1

## Compact feature findings

The compact profile uses B2, F3 and G4 dedicated reports, plus non-pitch groups from the full-bank report. The pitch groups are not double-counted.

| Group | Samples | Median F0 | Median RMS | Centroid | H/R proxy | Residual fraction |
|---|---:|---:|---:|---:|---:|---:|
| B2 | 93 | 174 Hz | -12.89 dBFS | 443 Hz | 27.0 dB | 0.198% |
| F3 | 93 | 195 Hz | -14.45 dBFS | 416 Hz | 27.8 dB | 0.165% |
| G4 | 93 | 392 Hz | -12.50 dBFS | 567 Hz | 30.5 dB | 0.090% |
| Consonant releases | 33 | 195 Hz | -16.59 dBFS | 495 Hz | 23.5 dB | 0.446% |
| L | 6 | 196 Hz | -13.78 dBFS | 450 Hz | 27.7 dB | 0.171% |
| V | 3 | 195 Hz | -13.57 dBFS | 457 Hz | 28.0 dB | 0.157% |
| TH | 3 | 196 Hz | -14.47 dBFS | 449 Hz | 23.0 dB | 0.510% |
| R English | 3 | 195 Hz | -13.09 dBFS | 440 Hz | 27.6 dB | 0.175% |
| Bonus extras | 8 | mixed | -29.27 dBFS | 1,522 Hz | 1.1 dB | 44.7% |

H/R and residual-energy measurements are analyzer proxies, not laboratory HNR or naturalness scores.

## Pitch-layer interpretation

- **B2 and F3** share a broad envelope rise around 300–400 Hz, around +2.3 dB in the median relative envelope. Their 2–5 kHz energy is low (about 0.27–0.29%).
- **G4** is distinct: higher centroid, about 0.76% energy in 2–5 kHz, and a different H1-H2 relationship. Do not reuse B2/F3 EQ.
- Harmonic slopes are broadly similar (roughly -12 to -14 dB/octave). The measurements do not justify generic brightening above 5 kHz.
- Pitch variation differs, but is not treated as a defect or used to inject modulation.

## Additional phonetic families

- **Consonant releases (33):** higher flatness and flux than the pitch banks, higher centroid and a larger residual-energy fraction. Preserve attacks and aperiodic detail; no general denoising or vowel-body EQ.
- **L / V / R English:** median F0 is around 195–196 Hz, close to F3, but each remains independently routable and neutral by default.
- **TH:** a small set with fricative aliases and a larger residual fraction. It does not inherit pitch-body EQ.
- **Bonus extras:** cough, gasp/breath, laughter, sob/cry and throat/growl are recognised by filename and bypass the tonal singing chain.
- **Root-level sample:** doremi_solfege.wav is not assumed to be a pitch-layer phoneme and is left for manual review.

## Processing chain v1

1. Automatically recognise pitch layers, consonant releases, phonetic extensions, English extensions, phonetic extras and unknown/root material from directory names.
2. Parse OTO aliases into token sequences and heuristic consonant classes.
3. Apply only small B2 (+0.6 dB at 1.45 kHz) and F3 (+0.3 dB at 1.45 kHz) presence tests. G4 remains neutral.
4. Apply those edits only inside eligible stretchable OTO windows. If any alias sharing the same WAV marks a region as fixed, protect that region for the entire WAV. Protect preutter/overlap markers and fade mask edges.
5. Filter each contiguous eligible region with 80 ms of source context on both sides, then blend only the OTO mask into the original. Samples outside the mask remain unchanged.
6. Require at least 5% mask coverage for automatic processing. Lower-coverage samples are marked partial and copied unchanged unless `--allow-partial` is explicitly supplied. Missing matches, duplicate aliases, unclassified pitch-layer alias tokens, timing warnings and empty masks are marked for manual review.
7. Audit every discovered oto.ini for parse errors, missing WAV references and duplicate aliases, and retain per-alias stretchable-window diagnostics in the processing report.
8. Keep consonant releases, L/V/TH, R English and non-lexical extras on separate routes. They are not forced through vowel processing.
9. Export to a separate tree and copy non-WAV files (including oto.ini) byte-for-byte.

The v1 chain intentionally avoids denoising, saturation, noise injection, time stretching, pitch shifting, loudness normalization and OTO edits. Those stages need matched UTAU renders and listening comparisons before tuning.
