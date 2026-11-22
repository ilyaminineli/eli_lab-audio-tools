# Full-sample Eugene:RE conditioning v2

## Scope

Quick Restore v2 processes the **complete duration of every WAV** in the selected bank. It does not use OTO body windows, alias confidence, or mask coverage as a reason to skip a sample.

Each readable WAV receives the same ordered pipeline, with conditional decisions made from that sample's signal:
1. Per-channel DC-offset correction.
2. Subsonic rumble measurement and conditional 22 Hz high-pass.
3. Narrow 50/60 Hz mains-hum measurement and conditional notch.
4. Family-aware, full-file spectral EQ for pitch-layer material.
5. Transient-aware EQ blending so sharp attacks receive less tonal shaping.
6. Peak safety if the output would exceed the permitted peak.

Consonant releases, phonetic extensions, English material, non-lexical events and root-level WAVs are all included. They receive DC/subsonic/hum analysis and full-sample output; the programme does not impose vowel EQ on these different kinds of sound.

## Current tonal strategy

- B2 uses the existing gentle 1.45 kHz presence profile.
- F3 uses the smaller existing 1.45 kHz presence profile.
- Other pitch layers begin with a neutral profile, with an adaptive +0.15 to +0.45 dB broad 2.8 kHz presence band only when the file's measured 2–5 kHz energy and centroid are very low and its spectral flatness does not indicate noisy material.
- This is source-side tonal conditioning. It does not invent new harmonics or guarantee a more natural resampled voice.

The transient curve is derived from positive spectral flux over short overlapping frames. Sharp changes reduce the EQ wet amount to 35% locally; sustained material receives the configured wet amount. This is a protective blend, not a consonant classifier.

## Conditional cleanup criteria

### DC offset
Each channel mean is measured independently. Offsets larger than 1e-7 are removed. No high-pass is used as a substitute for DC correction.

### Subsonic rumble
Power below 22 Hz is compared with the useful 22–120 Hz region. A second-order 22 Hz high-pass is enabled only when subsonic energy is measurably strong (at least 0.2% of total spectral power, at least -24 dB relative to the 22–120 Hz reference band, and above a small absolute floor).

### Mains hum
50 and 60 Hz are examined with Welch spectral analysis. A notch is applied only if an isolated peak exceeds the surrounding local median by at least 13 dB. Broad low-frequency energy and ordinary vowel harmonics are not enough to trigger a notch.

## Files, sidecars and output

- The source bank is never modified.
- Every readable WAV is written to the same relative path in the output copy, even when the numerical correction is zero.
- Unsupported/corrupt WAVs are copied byte-for-byte and recorded as **fallback_copy_error**; they do not abort the bank export.
- oto.ini and other non-WAV metadata are copied byte-for-byte.
- When a WAV changes, matching .frq, .llsm, .mrq, and .pmk analysis sidecars are removed from the active output copy so the corresponding engine regenerates analysis for the new audio. Folder-level Moresampler desc.mrq is invalidated if a WAV in that folder changes.
- Original sidecars remain intact in the source bank.
- processing_report.json lists every WAV, its family, processing stages, changed flag, errors, and invalidated sidecars.

## Preview

The selected WAV is processed by the same process_full_sample function used by the bank export. The UI displays:
- Original waveform.
- Processed waveform, sharing the same amplitude scale and timeline.
- Difference waveform (processed minus original), scaled independently so small corrections remain visible.
- Peak difference and difference RMS.
- A list of stages that were actually activated.

The difference view is diagnostic. A visually larger difference does not itself mean a better voice.

## Deliberately not enabled yet

A genuine reference-guided spectral-envelope correction needs matching real and synthetic phoneme material. The two short da/di/du examples are useful calibration samples but do not define trustworthy targets for every vowel, consonant, pitch and English extension in the bank. The automatic whole-bank pipeline therefore does not pretend to recover missing harmonics or force every sample toward one reference spectrum.

Broad vowel-formant cuts, de-essing, broadband denoising, artificial breath injection, saturation and neural harmonic reconstruction are also not enabled as universal automatic operations. They require a reliable artefact criterion or matching reference material and must be validated through the target resampler after source processing.

## Validation

The automated tests cover:
- DC-offset correction.
- Conditional subsonic rumble filtering.
- Narrow mains-hum detection.
- Full-duration processing across pitch layers, extensions, releases, bonus events and root-level WAVs.
- Processing despite missing/ambiguous OTO data.
- Output sidecar invalidation, including folder-level desc.mrq.
- Fallback copying for corrupt WAVs without aborting the bank.
- Full-file preview behaviour and waveform UI syntax.
