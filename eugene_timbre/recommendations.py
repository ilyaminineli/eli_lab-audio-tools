"""Analysis-driven DSP suggestions and separately-scoped Moresampler style presets.

The EQ profiles are conservative listening-test starting points. They are not
claims of objective correction or naturalness. Renderer flags are separate:
they affect Moresampler synthesis and do not alter source WAVs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

BAND_LAYOUT = (
    ("Low body", 110.0),
    ("Body", 300.0),
    ("Core", 950.0),
    ("Presence", 3000.0),
    ("Air", 7000.0),
)

EQ_APPROACHES = (
    ("neutral", "Neutral reference"),
    ("clarity", "Clarity / presence"),
    ("warmth", "Warm / body"),
    ("air", "Air / detail"),
    ("soft", "Soft / dark"),
    ("reference_match", "Match reference pitch"),
)

# These values are deliberately small. Dynamic gains are reduced further when
# the measured spectrum already contains substantial energy in the target band.
MORESAMPLER_FLAG_TEMPLATES = (
    {
        "key": "neutral",
        "label": "Neutral / original character",
        "flags": "",
        "description": "No Moresampler timbre flags. Use as the A/B reference.",
        "caution": "Reference setting.",
    },
    {
        "key": "intimate",
        "label": "Intimate / breathier",
        "flags": "Mb+8Mt-4",
        "description": "A small breathiness increase with slightly relaxed vocal-fold tension.",
        "caution": "More breath noise can reduce clarity, especially in quiet consonants.",
    },
    {
        "key": "projected",
        "label": "Tense / projected",
        "flags": "Mt+8ME+4",
        "description": "A restrained tension increase with a little formant emphasis.",
        "caution": "Can make the source sound harder or sharper; audition at matched loudness.",
    },
    {
        "key": "open",
        "label": "Open vowels",
        "flags": "Mo+8",
        "description": "A subtle Moresampler openness change, intended as an expressive alternative.",
        "caution": "Can change vowel colour; check Japanese and English vowel identity.",
    },
    {
        "key": "formant",
        "label": "Formant emphasis",
        "flags": "ME+8",
        "description": "A subtle emphasis of the modelled formants.",
        "caution": "This is a renderer-side spectral/model change, not a parametric EQ band.",
    },
    {
        "key": "dry",
        "label": "Drier / less breath modulation",
        "flags": "Md+8",
        "description": "A small dryness change in Moresampler's breath-noise model.",
        "caution": "The documented effect is subtle; it is not a reverb/dry-wet control.",
    },
    {
        "key": "rough",
        "label": "Rough / growling",
        "flags": "MC+8MG+5",
        "description": "A restrained coarseness and growl experiment.",
        "caution": "Character effect only. It can sound distorted and is not a clean-up setting.",
    },
)


def moresampler_flag_templates() -> list[dict[str, str]]:
    """Return copies of known Moresampler flag starting points."""
    return deepcopy(list(MORESAMPLER_FLAG_TEMPLATES))


def _number(mapping: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = mapping.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _relative_energy(analysis: dict, key: str, default: float = 0.0) -> float:
    values = analysis.get("summary", {}).get("relative_band_energy_median", {})
    value = _number(values, key)
    return max(0.0, value) if value is not None else default


def _make_profile(
    analysis: dict,
    approach: str,
    gains: tuple[float, float, float, float, float],
    description: str,
    reasons: list[str],
) -> dict:
    bands = []
    for (name, frequency), gain in zip(BAND_LAYOUT, gains):
        bands.append({
            "name": name,
            "frequency_hz": frequency,
            "gain_db": round(_clamp(gain, -1.5, 1.5), 2),
            "q": 0.7,
        })
    return {
        "name": f"{analysis.get('pitch', 'pitch')}-{approach}",
        "description": description,
        "highpass_hz": 0.0,
        "highpass_slope": 2,
        "bands": bands,
        "approach": approach,
        "generated_from_analysis": True,
        "analysis_pitch": analysis.get("pitch"),
        "recommendation_reasons": list(reasons),
        "recommendation_limit": (
            "Bounded creative starting point, not a measured correction target or naturalness score."
        ),
    }


def recommend_eq_profile(analysis: dict, approach: str) -> dict:
    """Create a bounded, analysis-informed EQ starting point for one pitch layer.

    Supported approaches:
      neutral: bypass/reference;
      clarity: gentle presence with less gain when presence is already strong;
      warmth: a small body emphasis, reduced when the layer is already dark;
      air: a small high-band lift only when measured high-band energy and noise
           proxies do not suggest that this would mainly amplify noise;
      soft: an intentional darker character curve;
      reference_match: requires a separate reference analysis and is provided
           by recommend_profiles rather than this function.
    """
    if analysis.get("report_type") != "eugene_pitch_layer_analysis":
        raise ValueError("Expected a completed Eugene pitch-layer analysis report.")
    if approach not in {key for key, _ in EQ_APPROACHES if key != "reference_match"}:
        raise ValueError(f"Unsupported EQ approach: {approach}")

    summary = analysis.get("summary", {})
    centroid = _number(summary, "spectral_centroid_median_hz")
    hnr = _number(summary, "harmonic_to_residual_median_db")
    flatness = _number(summary, "spectral_flatness_median")
    low = _relative_energy(analysis, "20_150_hz")
    body = _relative_energy(analysis, "150_500_hz")
    core = _relative_energy(analysis, "500_2000_hz")
    presence = _relative_energy(analysis, "2000_5000_hz")
    air = _relative_energy(analysis, "5000_10000_hz")
    reasons: list[str] = []

    # Noise-sensitive high-band boosts are backed off when the measured
    # harmonic/residual and flatness proxies suggest a noisy source.
    quality = 0.55
    if hnr is not None:
        quality *= _clamp((hnr + 3.0) / 15.0, 0.2, 1.0)
    if flatness is not None:
        quality *= _clamp((0.38 - flatness) / 0.30, 0.2, 1.0)
    quality = _clamp(quality, 0.15, 0.75)

    centroid_hz = centroid if centroid is not None else 2400.0
    total_low = low + body
    if approach == "neutral":
        gains = (0.0, 0.0, 0.0, 0.0, 0.0)
        reasons.append("Neutral bypass profile; it is the unchanged A/B reference.")
        description = "No EQ or high-pass filter. Use this profile as the control render."
    elif approach == "clarity":
        # Presence gain responds to the measured centroid and already-strong
        # 2–5 kHz energy. A small low-mid trim is only proposed when this
        # band pair dominates the layer's own energy distribution.
        centroid_factor = _clamp((3300.0 - centroid_hz) / 1900.0, 0.15, 1.0)
        presence_factor = _clamp(1.0 - max(0.0, presence - 0.24) * 1.8, 0.25, 1.0)
        presence_gain = 1.0 * centroid_factor * presence_factor
        body_cut = -_clamp(max(0.0, total_low - 0.46) * 1.4, 0.0, 0.55)
        air_gain = 0.25 * quality * _clamp((0.16 - air) / 0.16, 0.0, 1.0)
        gains = (0.0, body_cut, 0.0, presence_gain, air_gain)
        reasons.append(f"Median spectral centroid: {centroid_hz:.0f} Hz.")
        reasons.append(f"Measured 2–5 kHz relative energy: {presence:.3f}; presence gain is reduced when this band is already strong.")
        if body_cut:
            reasons.append("Low-mid energy is prominent, so a small 300 Hz trim is included.")
        if hnr is not None or flatness is not None:
            reasons.append("The optional air lift is limited by the harmonic/residual and spectral-flatness proxies.")
        description = "A restrained presence-focused starting point, scaled using this pitch layer's measured spectrum."
    elif approach == "warmth":
        # Add some body only if the centroid is not already low and the
        # 150–500 Hz band is not dominant.
        centroid_factor = _clamp((centroid_hz - 900.0) / 2200.0, 0.12, 1.0)
        body_factor = _clamp(1.0 - max(0.0, body - 0.28) * 2.2, 0.2, 1.0)
        body_gain = 0.9 * centroid_factor * body_factor
        presence_cut = -0.25 if presence > 0.24 and centroid_hz > 2500 else 0.0
        gains = (0.0, body_gain, 0.0, presence_cut, 0.0)
        reasons.append(f"Median spectral centroid: {centroid_hz:.0f} Hz.")
        reasons.append(f"Measured 150–500 Hz relative energy: {body:.3f}; the body lift is reduced when that region is already strong.")
        description = "A small 300 Hz body emphasis intended as a warmer colour, not a universal correction."
    elif approach == "air":
        air_headroom = _clamp((0.16 - air) / 0.16, 0.0, 1.0)
        air_gain = 0.9 * air_headroom * quality
        presence_gain = 0.25 * _clamp((3000.0 - centroid_hz) / 1600.0, 0.0, 1.0) * quality
        gains = (0.0, 0.0, 0.0, presence_gain, air_gain)
        reasons.append(f"Measured 5–10 kHz relative energy: {air:.3f}; air lift is reduced when high-band energy is already present.")
        if hnr is not None:
            reasons.append(f"Median harmonic/residual proxy: {hnr:.1f} dB; used only to limit a noise-sensitive high-band boost.")
        if flatness is not None:
            reasons.append(f"Median spectral flatness proxy: {flatness:.3f}; higher values reduce the high-band suggestion.")
        description = "A subtle high-frequency detail option. It is deliberately conservative because air boosts can expose noise."
    elif approach == "soft":
        gains = (0.0, 0.15, 0.0, -0.75, -0.9)
        reasons.append("Intentional darker/softer colour curve; these cuts are an artistic option, not a repair.")
        reasons.append(f"Analysed median centroid was {centroid_hz:.0f} Hz; audition at matched loudness.")
        description = "A darker, softer presentation with restrained upper-mid and air reduction."
    else:
        raise ValueError(f"Unsupported EQ approach: {approach}")

    return _make_profile(
        analysis,
        approach,
        gains,
        description,
        reasons,
    )


def recommend_profiles(analysis: dict, reference_analysis: dict | None = None) -> list[dict]:
    """Generate the available EQ variants after one pitch layer is analysed."""
    profiles = []
    for key, label in EQ_APPROACHES:
        if key == "reference_match":
            if reference_analysis is None:
                profiles.append({
                    "key": key,
                    "label": label,
                    "available": False,
                    "reason": "Analyse another pitch layer and select it as the reference to enable this option.",
                })
                continue
            if analysis.get("pitch") == reference_analysis.get("pitch"):
                profiles.append({
                    "key": key,
                    "label": label,
                    "available": False,
                    "reason": "Choose a different pitch layer as the reference.",
                })
                continue
            from .pitch_studio import suggest_reference_match
            profile = suggest_reference_match(analysis, reference_analysis, strength=0.6, max_gain_db=2.0)
            profile["approach"] = "reference_match"
            profile["generated_from_analysis"] = True
            profile["recommendation_reasons"] = [
                f"Compares the measured cepstral spectral envelope of {analysis.get('pitch')} with {reference_analysis.get('pitch')}.",
                "The result is a bounded five-band EQ approximation, not a formant model.",
            ]
            profiles.append({
                "key": key, "label": label, "available": True,
                "profile": profile, "reason": " " .join(profile["recommendation_reasons"]),
            })
        else:
            profile = recommend_eq_profile(analysis, key)
            profiles.append({
                "key": key, "label": label, "available": True,
                "profile": profile,
                "reason": " ".join(profile["recommendation_reasons"]),
            })
    return profiles


def get_eq_approach_labels() -> list[tuple[str, str]]:
    return list(EQ_APPROACHES)
