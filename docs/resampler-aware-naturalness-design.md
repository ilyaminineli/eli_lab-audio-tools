# Eugene:RE — resampler-aware naturalness and voicebank preparation

## Primary goal

The primary goal of this programme is to make Eugene:RE sound more alive and less artificial **after UTAU resampling**, not to make its source WAVs generically brighter, cleaner, louder, or more polished.

The source voicebank is the material each renderer analyses. A change to the source WAV can alter the pitch estimate, harmonic model, aperiodic/noise model, spectral envelope, and the result after pitch or duration transformation. Therefore, source preprocessing must be judged by the rendered result, not by whether the edited source WAV looks or sounds more attractive in isolation.

Moresampler is the first reference renderer because it is the user's principal engine and exposes useful analysis/synthesis controls. The design must remain renderer-aware so the same source-bank improvements can be tested with TIPS and Worldline variants too.

## 1. A voice is more than its EQ curve

A useful simplified description of voiced sound is source plus filter:

- **Source / phonation:** vocal-fold vibration creates a periodic harmonic structure. Airflow turbulence at the glottis can add aspiration and breathiness. The source has its own spectral shape, pitch trajectory and cycle-to-cycle behaviour.
- **Vocal-tract filter:** the changing configuration of the mouth, tongue, jaw, lips and nasal coupling shapes that source. Its resonances (formants) move as vowels and consonants change.
- **Articulation / mixed sources:** consonants such as /s/, /ʃ/, /f/, /tʃ/ contain turbulence noise; sounds such as /v/ and /z/ combine voicing with turbulence. This aperiodic energy is part of the phoneme, not automatically unwanted noise.
- **Time:** natural expression also contains changing F0, amplitude, spectral tilt, phonation, vowel formants and transitions. A static EQ cannot recreate those temporal behaviours.

The source-filter account is documented in Oxford phonetics teaching material and Oxford Academic's source-filter reference. It also makes clear why one broad frequency curve cannot stand in for phonation, articulation and time-varying resonance.

References:
- https://www.phon.ox.ac.uk/jcoleman/source_filter
- https://academic.oup.com/edited-volume/61882/chapter-abstract/547687311
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4847936/

## 2. What can make a concatenative voicebank sound artificial

These are hypotheses to test against rendered phrases, not defects that can be diagnosed from one average spectrum:

1. **Overly uniform or flattened source character.** Vowels may have too little variation in harmonic spectral tilt, breathiness, phonation or amplitude.
2. **Incorrect voiced/aperiodic separation.** A resampler may mistake breath, frication or inharmonic material for voiced signal, or remove the noise component that carries consonant and breath character.
3. **Pitch and timing transformation artifacts.** Large pitch shifts, long duration extension, poor F0 tracking, or inappropriate stretch/loop choices can smear or metallically colour a phoneme.
4. **Static or mismatched resonance.** A vowel's formant envelope changes with articulation. If only pitch changes while the expected vowel/resonance movement is weak or misestimated, it can sound synthetic.
5. **Consonant / vowel boundary problems.** Incorrect OTO offset, fixed area, cutoff, preutterance or overlap can truncate consonant transients, stretch the wrong region, or create abrupt joins.
6. **Over-normalization.** Per-segment gain normalization may erase intended level differences or make short/quiet phones too prominent.
7. **Repeated samples and limited recorded variation.** Reusing the same recording across many pitches and phonetic contexts can expose the sample seams. EQ cannot manufacture new articulatory performances.

The programme must identify which of these is present before suggesting a correction.

## 3. What the existing Eugene:RE measurements actually say

The repository's current bank summary reports:

| Pitch layer | Median F0 | Spectral centroid | H/R proxy | Residual fraction |
|---|---:|---:|---:|---:|
| B2 | 174 Hz | 443 Hz | 27.0 dB | 0.198% |
| F3 | 195 Hz | 416 Hz | 27.8 dB | 0.165% |
| G4 | 392 Hz | 567 Hz | 30.5 dB | 0.090% |

Other findings already recorded:
- B2 and F3 have a broad median envelope rise around 300–400 Hz and low 2–5 kHz relative energy (about 0.27–0.29%).
- G4 has higher centroid, more 2–5 kHz energy (about 0.76%), and a different H1–H2 relationship.
- Median harmonic slopes are broadly similar, around -12 to -14 dB/octave.
- Consonant releases have higher flatness/flux and a larger residual-energy fraction than the pitch layers.
- Bonus phonetic extras have much more aperiodic/noisy material and should not be forced through the vowel EQ path.

Important qualifications:
- H/R and residual values above are from this project's FFT-bin heuristic. They are not laboratory HNR and cannot by themselves prove that the recording is too periodic, too clean, or artificial.
- The project report explicitly says whole-file measurements mix vowels, transitions and consonants. It recommends comparing equivalent aliases and inspecting frame tracks.
- The low upper-mid energy in B2/F3 is not an instruction to boost 1.45 kHz or 3 kHz automatically. It is a reason to inspect sustained vowels and compare the same phonemes in renderer output.
- The current analysis does not yet quantify all the features needed to assess expressive human voice quality.

Existing evidence: [processing-chain-design.md](processing-chain-design.md) and [deep-analysis.md](deep-analysis.md).

## 4. How the target renderers should be treated

### Moresampler — primary reference

Moresampler creates analysis data for input samples (including its `.llsm` and pitch map `desc.mrq` data), estimates pitch/voicing, then synthesizes modified phones. Its history mentions sinusoidal analysis, separate noise-component handling, phase alignment, improved aperiodic extraction for inharmonic input, and work on stability for breathy voices.

Relevant documented controls include:
- `analysis-noise-reduction`: changes how input noise is treated during LLSM analysis; the readme says it works better on longer recordings.
- `analysis-biased-f0-estimation`: can reduce voiced-as-unvoiced false negatives but raises false positives and may degrade clean/smooth speech.
- `analysis-suppress-subharmonics`: may help screamy voices but slightly degrades breathy voices.
- `analysis-anti-distortion`: attempts to address analysis errors caused by noise distortion or low/quantized level, but may slightly blur speech.
- `synthesis-duration-extension-method`: stretch, loop or auto; duration extension should be tested on held vowels and phrase contexts.
- `synthesis-utau-style-normalization` and `synthesis-loudness-preservation`: affect segment gain and perceived loudness.

The readme also documents high-frequency phase reconstruction/alignment, click-artifact correction and smoothing drastic phase changes around phoneme transitions. These are directly relevant to perceived naturalness, so the programme should expose and compare supported engine settings rather than assume source EQ is the only lever.

References:
- https://github.com/hungrierr/Moresampler/blob/main/readme.txt
- https://openutau.com/Resamplers-and-Wavtools/

### TIPS

TIPS is a separate third-party resampler. Its frequency-map sidecar is `.pmk`, not Moresampler's `.mrq` or the built-in resampler's `.frq`. Public documentation located for TIPS is limited: community listings describe it as working well with soft voicebanks, while another UTAU reference notes that it can bring out background noise and may suit lower voices. These are qualitative reports, not an algorithm specification.

Therefore:
- Inventory `.pmk` files and keep the originals available for comparison.
- Do not convert, overwrite or silently reuse a `.pmk` file after its source WAV has been modified; the prepared bank needs a TIPS-generated map for the prepared WAV.
- Test TIPS with and without the relevant `.pmk` map when building the comparison set.
- Do not assume Moresampler flags or settings transfer to TIPS.

References:
- https://utau.info/resources/resamplers/
- https://utau.us/resample.html
- https://openutau.com/Resamplers-and-Wavtools/

### Worldline and WORLDLINE-R are not interchangeable labels

OpenUtau exposes a classic `worldline` resampler and integrated WORLDLINE-R renderers. WORLDLINE-R is its own phrase renderer/wavtool path, not simply an external `worldline.exe` called through the classic interface.

The OpenUtau source currently distinguishes:
- WORLDLINE-R: WORLD-based synthesis.
- WORLDLINE-R1.1: uses HNSEP features and continuous-noise synthesis.
- WORLDLINE-R2: extracts WORLD-style features and passes mel features to a neural vocoder.

The renderer source also samples time-varying F0, gender, tension, breathiness and voicing curves, synthesizes complete phrases, and applies phrase dynamics. This provides a separate, valuable test of whether the voicebank's phonation and source material respond well to dynamic parameter changes.

References:
- https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Core/Classic/WorldlineRenderer.cs
- https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Core/Classic/WorldlineResampler.cs
- https://openutau.com/Resamplers-and-Wavtools/
- https://openutau.com/wiki/Tutorials/

## 5. Analysis the programme needs before it recommends preprocessing

The existing analyser already measures F0 tracks, pitch confidence, harmonic amplitudes, spectral slope, H1–H2, H2–H4, an H/R proxy, inharmonicity, spectral envelope, flatness, flux, RMS and OTO-region summaries. This is a useful start, but it is not yet a complete voice-quality assessment.

Recommended additions:

### A. Phonation and source quality
- Stable-vowel jitter and shimmer, with reliability/voicing checks.
- Cepstral peak prominence (CPP/CPPS), carefully separated from the current H/R proxy.
- H1–H2, H2–H4 and higher harmonic-band slopes over equivalent vowels.
- Periodic/aperiodic energy profiles by frequency and over time.
- Voiced-frame and voicing-transition reliability, including octave-error detection.

### B. Vowel and resonance behaviour
- Classify or select known vowel aliases, then compare the same vowel between B2/F3/G4.
- Track F1/F2/F3 candidates and bandwidths across the stable vowel and its transitions. Treat them as estimates; a cepstral envelope peak is not automatically a formant.
- Measure formant/envelope movement and stability, not only a median envelope.
- Compare pitch-layer consistency after accounting for the different expected F0 and vowel source.

### C. Expression and articulation
- F0 contours: onset, settling, drift, vibrato rate/depth where present, and note-release behaviour.
- RMS / amplitude envelopes and intra-sample dynamics.
- Consonant transients, aspiration and frication spectra, positive spectral flux and release timing.
- OTO-aware boundary regions: attack, fixed/consonant section, preutterance/overlap, stretchable vowel body and cutoff.

Do not automatically inject jitter, vibrato, breath noise or saturation. First determine whether the recordings contain those features and whether their absence in the rendered output is caused by the source, the resampler, or the UST/OTO timing.

Praat's voice-analysis framework is a useful reference for pulse-based jitter, shimmer and harmonicity measurements. Its manual also cautions that the results depend on pitch and voicing settings. H1–H2 and CPP are useful complementary measures for pressed/breathy phonation, but should be interpreted together rather than as one universal quality score.

References:
- https://praat.org/manual/Voice.html
- https://praat.org/manual/Harmonicity.html
- https://pubmed.ncbi.nlm.nih.gov/40021420/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4847936/

## 6. Correct processing architecture

Keep these stages separate and report them separately:

1. **Source diagnosis:** read-only per-sample/per-alias acoustic and OTO analysis. This establishes the baseline and does not edit the bank.
2. **Source preparation:** optional, conservative EQ or source correction on a separate copy. It must be per pitch and, when evidence requires it, per phoneme/alias class. Its purpose is to improve the renderer's input analysis and reconstruction, not simply to make the dry WAV brighter.
3. **Engine analysis metadata:** renderer-specific FRQ/PMK/MRQ/LLSM and configuration. Preserve the original maps as source-bank reference material, but never assume they remain valid after a WAV changes. The prepared output must be given fresh engine-specific analysis maps (FRQ for the appropriate classic engines, PMK for TIPS, MRQ/LLSM for Moresampler) or have stale maps removed from the active rendering path until the engine regenerates them. Record the engine/version and regeneration state.
4. **Renderer configuration:** Moresampler `moreconfig.txt`, Moresampler flags, TIPS-compatible options, or WORLDLINE-R expressions. These are engine controls, not source EQ.
5. **Phrase rendering:** render the same UST/USTX phrase through the target engine. This is where pitch shifting, duration extension, OTO timing and inter-phoneme transitions can be judged.
6. **Final mix:** optional DAW effects after the phrase exists. Reverb, delay, bus compression and mastering should not be indiscriminately baked into every voicebank WAV.

## 7. A/B test protocol

For each selected test phrase, retain a neutral baseline and generate controlled variants:

- A: original bank, original renderer configuration.
- B: original bank, alternative renderer-analysis configuration only.
- C: source-prepared bank, same renderer configuration as A.
- D: source-prepared bank, alternative renderer configuration.
- E: renderer expression/flag change only.

Keep the UST/USTX, phonemizer output, note pitches, timing, OTO, wavtool, renderer version, sample rate, loudness target and all non-tested settings identical. Change one factor at a time. When comparing source preprocessing, ensure the renderer re-analyses the changed WAV rather than reusing stale LLSM/FRQ/PMK/MRQ caches.

Evaluate:
- vowel identity and formant stability after pitch shifts;
- breathiness and aperiodic consonants;
- metallic/robotic artifacts and phase/click problems;
- consonant clarity and timing;
- transitions, held vowels and note releases;
- expressive variation and phrase-level coherence;
- loudness-matched listening preference.

A profile is accepted only if the rendered result improves the target quality without damaging the other checks. A positive EQ change, higher centroid or higher H/R value is not itself evidence of improvement.

## 8. Product direction and UI language

The main screen should communicate this actual goal:

**Voicebank preparation for natural resampling**
- Analyse source voice and OTO regions.
- Compare pitch layers and equivalent phonemes.
- Prepare separate copies for each target resampler.
- Inspect renderer configuration and metadata.
- Render phrase test sets and compare controlled A/B results.
- Export a tested bank copy.

"Clarity", "warmth", "air" and "dark" can remain optional creative starting points, but they should not be the primary automatic restoration system. The primary recommendation system should say what specific measured issue it is trying to correct, which renderer it targets, and which phrase tests show that it helps.
