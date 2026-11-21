# eli_lab-audio-tools

Free ELI LAB audio utilities. This repository includes **Eugene:RE Timbre Restoration**, a non-destructive analysis and tonal-processing utility for existing RVC-derived UTAU source WAVs.

## Scope

The tool is for timbre experiments only. It does not rebuild or retime the voicebank. It never edits `oto.ini`, aliases, offsets, consonant/cutoff/preutterance/overlap values, filenames, sample duration, pitch, `.frq`, `.llsm`, or TIPS cache files. Output always goes to a separate directory.

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Compare the analysed subbanks

```powershell
eugene-timbre compare B2_analysis.json F3_analysis.json G4_analysis.json -o subbank_comparison.json
```

The comparison reports broad medians; it does not infer narrow EQ notches. See [docs/analysis-findings.md](docs/analysis-findings.md).

## Analyse fresh WAVs

```powershell
eugene-timbre analyze "D:\Voicebanks\Eugene_RE\6. B2" -o "B2_analysis_v2.json"
```

The v0.3 analyzer reports `peak_amplitude` and a correctly calculated `peak_dbfs`. Earlier uploaded analysis JSONs used the name `peak_dbfs` for linear peak amplitude; see the analysis findings note.

## Process a small B2 test set

```powershell
eugene-timbre process "D:\Voicebanks\Eugene_RE\6. B2\test_samples" --profile warm-uncanny-b2 --wet 1.0 --output "D:\Voicebanks\Eugene_RE_Restored_TEST\6. B2"
```

Profiles:

- `warm-uncanny-b2`: HPF off; +1.0 dB bell at 1.45 kHz, Q 0.7.
- `f3-gentle`: HPF off; +0.5 dB bell at 1.45 kHz, Q 0.7.
- `g4-neutral`: bypass reference.
- `neutral`: bypass reference.

These are listening-test profiles, not automatic de-synthesis. Match loudness when comparing, then render identical UST notes through your target engines. If the processed version sounds more forward but not more natural, revert it; static EQ cannot restore missing aperiodic detail or fix conversion phase artifacts.

## Resampler cache warning

The supplied bank tree includes `.frq` and `.wav.llsm` sidecars. After replacing source WAVs in a duplicate bank, refresh the corresponding engine analyses. Moresampler's documentation describes `auto-update-llsm-mrq on` for updating LLSM/MRQ data when a WAV is newer than its analysis. Do not delete or overwrite caches in the original release bank.

## Python API

```python
import soundfile as sf
from eugene_timbre.dsp import apply_profile, WARM_UNCANNY_B2

audio, sr = sf.read("sample.wav", dtype="float32")
out = apply_profile(audio, sr, WARM_UNCANNY_B2, wet=1.0)
sf.write("sample_restored.wav", out, sr, subtype="PCM_16")
```

Dependencies: NumPy, SciPy and SoundFile. No cloud service, subscription or AI inference is required.
