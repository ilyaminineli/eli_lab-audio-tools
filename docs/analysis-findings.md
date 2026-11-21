# Eugene:RE analysis findings

Source: `B2_analysis.json`, `F3_analysis.json`, `G4_analysis.json`, `CONSONANT_analysis.json`, `BONUS_analysis.json`, `L_analysis.json`, `V_analysis.json`, `TH_analysis.json`, and `R ENGLISH_analysis.json` on the repository main branch, inspected 2026-11-21.

## Main pitch-bank comparison

Medians across the complete files in each analysis set:

| Metric | B2 (93 files) | F3 (93 files) | G4 (93 files) |
|---|---:|---:|---:|
| Spectral centroid | 443 Hz | 416 Hz | 567 Hz |
| Spectral flatness | 0.000045 | 0.000044 | 0.000064 |
| RMS | -12.89 dBFS | -14.45 dBFS | -12.50 dBFS |
| Relative energy 150-500 Hz | 53.75% | 79.35% | 59.49% |
| Relative energy 500-2,000 Hz | 45.88% | 20.30% | 39.78% |
| Relative energy 2,000-5,000 Hz | 0.273% | 0.294% | 0.763% |
| Relative energy 5,000-10,000 Hz | 0.011% | 0.014% | 0.034% |

The key point is that B2/F3/G4 are not tonally interchangeable. G4 has about 2.8 times the median 2-5 kHz energy of B2. Avoid applying one strong global EQ profile to all three banks.

## Why the first EQ was barely audible

The earlier B2 preset attenuated 3.2 kHz by 1.2 dB and 5.2 kHz by 0.5 dB. In the B2 analysis, those regions contain very little of the total energy: median energy from 2-5 kHz is about 0.27%, and 5-10 kHz is about 0.011%. A gentle cut there cannot substantially change the overall tone.

The revised B2 test instead uses a wide +1.0 dB bell at 1.45 kHz, Q 0.7, where the analysed samples contain substantially more energy. The B2 high-pass is disabled to preserve low fundamentals. This is still only a broad tonal experiment; it cannot remove phase instability, missing aperiodic texture, or RVC conversion artifacts.

## Limits of the submitted analysis data

The JSON contains whole-file band-energy summaries, not time-resolved vowel spectra or formant tracks. It is sufficient for broad subbank comparisons but not for designing narrow resonant notches. Spectral flatness is very low in the pitch banks, indicating highly concentrated tonal energy; this is compatible with a harmonic/periodic source, but is not by itself proof of a defect.

A legacy field issue was found: in the already-uploaded JSONs, values under `peak_dbfs` (typically 0.4-0.9) appear to be linear peak amplitudes, not dBFS. The v0.3 analyzer now reports both `peak_amplitude` and correctly calculated `peak_dbfs`. Do not use the old `peak_dbfs` values to judge headroom.

## Sample identity notes

The supplied `- e.wav` name corresponds to the F3 naming convention in the tree; B2 uses `-_e.wav`. `a_a_i.wav` and `a_n_a.wav` appear in both F3 and B2, so identify them by parent subbank folder, not filename alone. The attached `a_a_i.wav` measurements match the F3 entry; the attached `a_n_a.wav` measurements match the B2 entry.

## Resampler cache safety

The tree contains `.frq` pitch-analysis sidecars and some `.wav.llsm` Moresampler analysis sidecars. If replacing WAVs in a duplicate bank, refresh the corresponding analyses for the changed samples. Moresampler documents `auto-update-llsm-mrq on` to detect a newer WAV and update its LLSM/MRQ analysis. This tool intentionally never edits `oto.ini`, `.frq`, `.llsm`, or TIPS cache files.
