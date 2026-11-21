# eli_lab-audio-tools

Free ELI LAB audio utilities. This repository includes **Eugene:RE Timbre Restoration**, a non-destructive tool for refining existing RVC-derived UTAU source WAVs.

## Scope

This tool addresses broad spectral balance only. It does **not** rebuild or retime the voicebank. It never edits oto.ini, aliases, offsets, consonant/cutoff/preutterance/overlap values, filenames, sample duration, pitch, or the original inputs.

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Analyse before processing

```powershell
eugene-timbre analyze "D:\Voicebanks\Eugene_RE\_B2" -o "B2_analysis.json"
```

## Process a small B2 test set

```powershell
eugene-timbre process "D:\Voicebanks\Eugene_RE\_B2\test_samples" --profile warm-uncanny-b2 --wet 0.75 --output "D:\Voicebanks\Eugene_RE_Restored_TEST\_B2"
```

Use `--wet 0.5` for a subtler blend, or `--profile neutral` for a bypass/reference export. Results go to a separate folder and include `restoration_report.json`.

## Initial RX-style B2 starting values

| Filter | Frequency | Gain | Q / slope |
|---|---:|---:|---:|
| HPF | 35 Hz | — | 12 dB/oct |
| Bell | 3,200 Hz | -1.2 dB | Q 1.0 |
| Bell | 5,200 Hz | -0.5 dB | Q 0.85 |

These are deliberately restrained listening-test values, **not a measured final correction**. The screenshot shows a steep low-frequency roll-off already active; this utility intentionally does not reproduce it. The 35 Hz, 12 dB/oct filter only removes subsonic rumble and should be bypassed if it thins the B2 samples.

Do not blindly apply this profile to _F3 or _G4. Develop separate profiles from corresponding vowel measurements and listening.

## Validation

1. Select representative B2 samples: sustained vowel, CV, VC and nasal transition.
2. Compare original and processed at matched playback loudness.
3. Render identical notes using Moresampler, TIPS and WORLDLINE-R.
4. If the processed WAV seems smoother solo but loses character or performs worse after resampling, revert it.
5. Keep Moresampler flags neutral for the baseline comparison.

## Python API

```python
import soundfile as sf
from eugene_timbre.dsp import apply_profile, WARM_UNCANNY_B2

audio, sr = sf.read("sample.wav", dtype="float32")
out = apply_profile(audio, sr, WARM_UNCANNY_B2, wet=0.75)
sf.write("sample_restored.wav", out, sr, subtype="PCM_16")
```

Dependencies are NumPy, SciPy and SoundFile. No cloud service, subscription, or AI inference is required.
