"""Safe WAV I/O helpers."""
from pathlib import Path
import numpy as np
import soundfile as sf

def read_wav(path: Path):
    info = sf.info(str(path))
    if info.format != "WAV":
        raise ValueError(f"{path} is not a WAV.")
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, int(sr), {"subtype": info.subtype, "frames": info.frames}

def write_wav(path: Path, audio: np.ndarray, sr: int, subtype: str):
    x = np.asarray(audio, dtype=np.float32)
    if np.max(np.abs(x)) > 1.0:
        raise ValueError("Output would clip; refusing to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    allowed = {"PCM_16","PCM_24","PCM_32","FLOAT","DOUBLE"}
    sf.write(str(path), x, sr, subtype=subtype if subtype in allowed else "PCM_16")
