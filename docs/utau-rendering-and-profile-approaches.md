# UTAU rendering and Eugene:RE restoration profiles

This document explains where the current restoration tool operates in the UTAU signal path, and what the profile templates mean.

## 1. How a classic UTAU voice is rendered

A UTAU voicebank is a concatenative sample library, not a single recording that the synthesizer simply plays back.

1. The phonemizer converts lyrics and notes into a sequence of phonetic units.
2. For each unit, the renderer chooses a matching sample/alias from the voicebank and reads its OTO timing.
3. The **resampler** changes that recording's pitch and duration and applies renderer-supported flags.
4. The **wavtool** joins the rendered slices and handles their timing/overlap to produce the phrase.

OpenUtau documents the resampler as the component that transforms sample pitch/duration and applies flags, while the wavtool concatenates the resampler output. OTO entries include offset, fixed/consonant area, cutoff, preutterance, and overlap. These are not cosmetic labels: they determine which source region is used, where the note aligns, which portion can be stretched/looped, and how notes overlap.

References:
- [OpenUtau: Voicebank Development](https://github.com/openutau/OpenUtau/wiki/Voicebank-Development)
- [OpenUtau: Resamplers and Wavtools](https://github.com/openutau/OpenUtau/wiki/Resamplers-and-Wavtools)
- [UTAU.forum: How to Create Your First Voicebank — OTO parameters](https://utau.forum/resources/how-to-create-your-first-voicebank-part-1.563/)

## 2. Where Moresampler fits

Moresampler is unusual because it can operate as a resampler, a wavtool, and an automatic voicebank configurator. OpenUtau's compatibility notes describe configuring it either as a resampler or as both resampler and wavtool.

It analyses input WAVs into its own model/cache data (notably LLSM, with pitch information such as MRQ/desc.mrq in documented versions), then synthesizes the requested phone. Therefore a clean source WAV is not the only consideration: analysis settings, pitch estimation, duration extension, OTO segmentation and the selected renderer flags all affect the result.

The current Eugene:RE A/B runner deliberately stages original and restored samples in separate temporary folders so Moresampler performs fresh analysis for each version. It currently runs a **single OTO phone in resampler mode**. It does not yet build a complete UST phrase, perform all inter-note overlaps, or reproduce a complete wavtool render. A short-phone result is useful for inspecting a source/flag change, but it cannot validate transitions or phrase-level flow.

References:
- [Moresampler readme / version history](https://github.com/hungrierr/Moresampler/blob/main/readme.txt)
- [OpenUtau: Moresampler compatibility notes](https://github.com/openutau/OpenUtau/wiki/Resamplers-and-Wavtools)

## 3. Keep four kinds of processing separate

### A. Source-WAV processing (this app's EQ)

This is offline DSP applied to the voicebank's recorded WAV copy before resampling. It can change the source spectrum but does not retune pitch, duration, phoneme alignment, or OTO timing.

The current tool provides five broad peaking bands (centres at 110, 300, 950, 3000, and 7000 Hz) and an optional high-pass filter. The generated templates keep the high-pass OFF by default: the low band may contain the voicebank's actual fundamental and useful body, especially in lower pitch layers.

When OTO protection is enabled, the DSP is limited to matched stretchable body regions; when disabled, it processes the entire selected WAV. Neither mode changes OTO values or rewrites FRQ/LLSM data. Non-WAV files are copied unchanged.

### B. OTO / phonetic structure

OTO tuning is a separate layer of the synthesis system. Offset, fixed area, cutoff, preutterance and overlap influence consonant preservation, vowel sustain, timing and crossfades. EQ cannot repair a bad alias, a badly placed preutterance, or a broken transition. The application inventories the complete Eugene:RE hierarchy, but pitch EQ is currently limited to the actual F3/G4/B2 pitch layers. R English, releases, L/V/TH and bonus phonetic extras need phonetic-aware policies before they are processed.

### C. Resampler flags (Moresampler templates)

These values change the synthesis model rather than filtering the saved WAV:

| Flag | Documented role | What the template is for |
|---|---|---|
| `Mt` | Tension / vocal-fold tenseness | More projected or tighter tone |
| `Mb` | Breathiness | Intimate / airy tone |
| `Mo` | Openness | More open vowel colour |
| `Md` | Dryness | Small change to breath-noise modulation; not a reverb control |
| `ME` | Formant emphasis | A little more formant prominence |
| `MC` | Coarseness | Rougher texture |
| `MG` | Growl | Growled character |
| `MD` | Distortion | More distorted character |

The starter presets intentionally use small values. These are audition ideas, not verified optimal Eugene:RE settings. The same flag can sound different on different samples and engines. The flag comparison mode renders A with neutral Moresampler flags and B with the chosen template, while keeping the source WAV the same.

Reference: [Moresampler readme — MEF and growling flags](https://github.com/hungrierr/Moresampler/blob/main/readme.txt).

### D. Moresampler analysis and synthesis configuration

These settings live in `moreconfig.txt`, not in the source-WAV EQ and not in the OTO file. They influence how Moresampler analyses the sample or performs synthesis.

| Setting | Documented behaviour | Practical caution |
|---|---|---|
| `analysis-noise-reduction` | Reduces noise during LLSM analysis; the documentation says it works better with longer recordings. | Compare on/off for noisy samples, but do not assume it is desirable for breathy or textured samples. |
| `analysis-anti-distortion` | Attempts to correct analysis inaccuracies caused by noise distortion or low volume/quantization. | The readme warns it may slightly blur speech. Use it as a diagnostic A/B, not a universal default. |
| `analysis-biased-f0-estimation` | Emphasizes voicing probability; can reduce voiced-as-unvoiced false negatives but increases false positives. | Described as experimental and potentially harmful to clean/smooth speech; avoid enabling it blindly. |
| `analysis-suppress-subharmonics` | Removes subharmonics during analysis. | The readme suggests it may help screamy voices but slightly degrades breathy voices. |
| `synthesis-duration-extension-method` | Selects `stretch`, `loop`, or `auto` for extending the duration of a speech segment. | This is important for held vowels and long notes; judge it on identical phrases, not a short isolated phone only. |
| `synthesis-utau-style-normalization` | Adaptive segment gain; `full` adjusts voiced and unvoiced parts, `voiced` adjusts only voiced parts, and `off` disables it. | This is per-segment gain management, not a conventional compressor or tonal EQ. |
| `synthesis-loudness-preservation` | Attempts to retain perceived loudness after modification. | Compare loudness-matched renders; do not mistake a louder render for a better timbre. |

The official readme documents these switches and explicitly warns about trade-offs for anti-distortion, biased F0 estimation and subharmonic suppression. The current UI does not edit these global/per-bank config options yet; it safely copies the installed Moresampler runtime/config into an isolated temporary directory for A/B renders and keeps the installed configuration unchanged. This configuration layer is a logical next addition after the source-EQ and renderer-flag templates have been auditioned.

Reference: [Moresampler readme — analysis and synthesis options](https://github.com/hungrierr/Moresampler/blob/main/readme.txt).

### E. Final phrase / mix processing

Compression, de-essing, saturation, stereo placement, delay and reverb are normally best judged on the assembled phrase or song, not indiscriminately baked into every source-bank WAV. A reverberated or compressed source bank can make UTAU's pitch/time processing harder to control and can be especially problematic when the same sample is reused in multiple phonetic contexts.

The current application does **not** add reverb, chorus, delay, mastering compression, or a global output mix chain. Those belong in a later full-phrase render/mix stage.

## 4. What analysis can and cannot recommend

The analysis measures median F0, spectral centroid, spectral-flatness and harmonic/residual proxies, relative band energy and a median cepstral spectral envelope.

These measurements can support:
- A bounded match between two pitch layers using their measured spectral envelopes.
- A subtle clarity, warmth, air, or darker-colour starting point, moderated by the layer's measured spectral balance.
- Warnings when a high-frequency boost may mostly emphasize noisy/aperiodic content.

They cannot identify an objectively correct vocal tract, guarantee a more natural result, or decide the artistic target. A low centroid is not automatically a flaw; a high residual/noise component may be part of the singer's texture. This is why the tool generates alternatives rather than silently applying one "best" curve.

## 5. Suggested comparison protocol

1. Analyse all pitch layers that should be compared.
2. Generate the EQ suggestions for each layer.
3. Choose one artistic approach, apply it to the checked layers, and inspect the actual gains produced per pitch.
4. Audition original and processed samples at matched loudness.
5. Render identical short phrases through the real target renderer. Compare vowels, consonant clarity, transitions, pitch-shift artifacts and note-length behaviour.
6. Only after phrase-level listening, process the selected source folders into a separate output bank.
7. Test Moresampler flag profiles separately from source EQ, then combine only the settings that survive listening.

A neutral render is always the control. Keep a copy of the original bank.
