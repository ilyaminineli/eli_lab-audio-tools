"""Small testable helpers for the one-click Eugene:RE restore workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .auto_routing import identify_family
from .dsp import apply_profile_masked
from .io import read_wav
from .processing_chain import (
    _alias_audit,
    _match_oto,
    _profile,
    _route,
    build_oto_body_mask,
    load_processing_config,
)


DEFAULT_WET = 0.5


def prepare_preview(root: Path, relative_path: str, wet: float = DEFAULT_WET) -> dict:
    """Create an OTO-aware before/after preview for exactly one source WAV."""
    root = root.expanduser().resolve()
    source = (root / Path(relative_path)).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError("Preview sample is not inside the selected voicebank.")

    audio, sample_rate, metadata = read_wav(source)
    config = load_processing_config()
    family = identify_family(Path(relative_path), root)
    oto_path, entries, parsed = _match_oto(source, root, None, {})
    route = _route(family, source.name, entries, config)

    unchanged = audio.astype(np.float32, copy=True)
    output = unchanged
    status = "bypassed"
    reason = route.get("reason", "No processing route is configured.")
    mask_info = {"mask_coverage_percent": 0.0, "eligible_stretch_regions": 0}
    profile_description = None

    if route.get("enabled"):
        audit = _alias_audit(entries)
        if not entries:
            status, reason = "manual_review", "No matching OTO alias; this sample is not modified."
        elif audit["has_duplicate_aliases"]:
            status, reason = "manual_review", "Duplicate OTO aliases; this sample is not modified."
        elif audit["has_unclassified_tokens"]:
            status, reason = "manual_review", "Unclassified OTO alias tokens; this sample is not modified."
        else:
            mask_options = config.get("oto_mask", {})
            mask, mask_info = build_oto_body_mask(
                len(audio), sample_rate, entries, mask_options
            )
            minimum_coverage = float(mask_options.get("minimum_process_coverage_percent", 5.0))
            if mask_info.get("timing_warnings"):
                status, reason = "manual_review", "OTO timing warnings; this sample is not modified."
            elif not np.any(mask > 0):
                status, reason = "manual_review", "No safe stretchable OTO body region; sample is not modified."
            elif mask_info["mask_coverage_percent"] < minimum_coverage:
                status, reason = "partial", (
                    f"Only {mask_info['mask_coverage_percent']:.2f}% of the WAV is eligible; "
                    "below the safe processing threshold."
                )
            else:
                spec = route.get("profile_spec", {})
                profile = _profile(spec, route.get("profile_name", "profile"))
                if profile.bands or profile.highpass_hz > 0:
                    output = apply_profile_masked(
                        audio, sample_rate, profile, mask,
                        wet=wet,
                        context_padding_ms=float(mask_options.get("context_padding_ms", 80.0)),
                    )
                    status = "preview_ready"
                    reason = (
                        f"Preview uses {route.get('profile_name')} inside "
                        f"{mask_info['mask_coverage_percent']:.1f}% OTO-approved body regions."
                    )
                    profile_description = {
                        "name": profile.name,
                        "bands": [
                            {"frequency_hz": b.frequency_hz, "gain_db": b.gain_db, "q": b.q}
                            for b in profile.bands
                        ],
                        "highpass_hz": profile.highpass_hz,
                    }
                else:
                    status, reason = "bypassed", "This family is configured for neutral pass-through."

    return {
        "source": str(source),
        "relative_path": relative_path,
        "sample_rate": sample_rate,
        "subtype": metadata["subtype"],
        "original": unchanged,
        "processed": np.asarray(output, dtype=np.float32),
        "status": status,
        "reason": reason,
        "profile": profile_description,
        "mask": mask_info,
        "oto_path": str(oto_path) if oto_path else None,
        "oto_parse_errors": len(parsed.errors) if parsed else 0,
        "same_audio": np.array_equal(unchanged, output),
    }
