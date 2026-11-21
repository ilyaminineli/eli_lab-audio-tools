# Resampler experiments for Eugene:RE

Use the same short UST phrase, note velocities, pitch bends, wavtool and output loudness for each test. Change one flag at a time and record the renderer and sample-bank copy used.

## Moresampler

Start from the current baseline (all extension flags neutral), then compare:

| Test | Flags | Purpose |
|---|---|---|
| A | `Mm100` | Default/new model baseline |
| B | `Mm75` | Blend toward the older model |
| C | `Mm50` | Equal model interpolation |
| D | `Mm100 Mb5` | Very small breathiness addition |
| E | `Mm100 Mb10` | Slightly stronger breathiness |

Moresampler documents `Mm` as interpolation between its classical speech model and its newer model (default 100), and positive `Mb` as adding breathiness. The direction that sounds more natural is bank- and phrase-dependent; audition these rather than assuming lower `Mm` is better. Avoid `Mr+` for the first test because positive `Mr` adds a singer's formant around 3 kHz, while the B2 analysis has little energy in that region. Avoid `MD` and `MG` while evaluating naturalness because they add distortion/growl character.

## Cache refresh

The current bank tree includes `.frq` files and selected `.wav.llsm` files. When WAVs are changed in a duplicate test bank, ensure the resampler re-analyses the changed source. Moresampler's `auto-update-llsm-mrq on` setting checks whether a WAV is newer than its LLSM/MRQ analysis and updates the analysis. Do not remove sidecars from the original release bank.
