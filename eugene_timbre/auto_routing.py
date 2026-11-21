"""Automatic recognition of Eugene:RE pitch layers and phonetic families."""
from __future__ import annotations
import re
from pathlib import Path

PITCH_RE = re.compile(r"(?<![A-Z0-9])([A-G](?:#|B)?[0-8])(?![A-Z0-9])", re.I)
PREFIX_RE = re.compile(r"^\s*\d+\s*[._ -]*")
VOWELS = {"a", "i", "u", "e", "o", "@a", "@i", "@u", "@e", "@o",
          "あ", "い", "う", "え", "お", "ア", "イ", "ウ", "エ", "オ"}
PHONEME_CLASSES = {
    "nasal": {"m", "n", "ng", "ん", "ン"},
    "stop": {"b", "d", "g", "k", "p", "t", "q", "ts", "ch", "j"},
    "fricative": {"f", "h", "s", "sh", "th", "v", "z", "zh"},
    "liquid": {"l", "r", "rr"},
    "glide": {"w", "y"},
}

def normalize_folder(value: str) -> str:
    value = PREFIX_RE.sub("", value.strip())
    return re.sub(r"[^A-Za-z0-9#]+", "_", value).strip("_").upper()

def classify_alias(alias: str) -> dict:
    core = re.sub(r"_(?:[A-G](?:#|B)?[0-8])$", "", alias.strip(), flags=re.I)
    tokens = [item for item in re.split(r"[\s_]+", core) if item]
    roles = []
    for token in tokens:
        low = token.lower()
        if low in VOWELS:
            role = "vowel"
        else:
            role = next((kind for kind, values in PHONEME_CLASSES.items()
                         if low in values), "kana_or_unknown")
        roles.append(role)
    classes = sorted({role for role in roles if role in PHONEME_CLASSES})
    transition = f"{roles[0]}_to_{roles[1]}" if len(roles) >= 2 else (
        roles[0] if roles else "unclassified"
    )
    return {
        "alias": alias, "canonical_alias": core, "tokens": tokens,
        "token_roles": roles, "consonant_classes": classes,
        "transition_class": transition,
    }

def classify_event(filename: str) -> str:
    name = Path(filename).name.lower()
    rules = (
        (("cough",), "cough"),
        (("gasp", "breath"), "gasp_or_breath"),
        (("laugh", "giggle"), "laughter"),
        (("sob", "cry"), "sob_or_cry"),
        (("throat", "growl"), "throat_or_growl"),
        (("sigh",), "sigh"),
    )
    return next((label for words, label in rules if any(word in name for word in words)),
                "unclassified_extra")

def identify_family(relative_path: Path, root: Path | None = None) -> dict:
    """Identify a route from the selected bank's structure."""
    candidates = [relative_path.parent.name]
    if root is not None:
        candidates.append(root.name)
    candidates.extend(relative_path.parts[:-1])
    normalized = [normalize_folder(item) for item in candidates if item]
    for folder in normalized:
        pitch_match = PITCH_RE.search(folder)
        if pitch_match:
            return {"group": pitch_match.group(1).upper(), "family": "pitch_layer",
                    "pitch": pitch_match.group(1).upper()}
    for folder in normalized:
        if "CONSONANT" in folder or "RELEASE" in folder:
            return {"group": "CONSONANT_RELEASES", "family": "consonant_release"}
        if "BONUS" in folder or "EXTRA" in folder:
            return {"group": "BONUS_PHONETIC_EXTRAS", "family": "phonetic_extra"}
        if "ENGLISH" in folder or folder.startswith("R_"):
            return {"group": "R_ENGLISH", "family": "english_extension",
                    "reference_pitch": "F3"}
        if folder in {"L", "V", "TH"}:
            return {"group": folder, "family": "phonetic_extension",
                    "reference_pitch": "F3"}
        if "APPEND" in folder or "EXTENSION" in folder:
            return {"group": folder, "family": "phonetic_append"}
    return {"group": normalized[0] if normalized else "ROOT", "family": "other"}
