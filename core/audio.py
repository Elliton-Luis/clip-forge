"""audio.py — energia de áudio por janela (ffmpeg volumedetect).

SRP: só áudio. KISS: usa ffmpeg já obrigatório, sem librosa.
"""
import re
import shutil
import statistics
import subprocess
from .config import AUDIO_ENERGY_TIMEOUT, CLIPPER_FFMPEG_THREADS


def measure(video_path: str, start: float, end: float) -> str:
    duration = max(0.1, end - start)
    cmd = [
        "ffmpeg", "-hide_banner", "-threads", str(CLIPPER_FFMPEG_THREADS),
        "-ss", str(max(0, start)), "-t", str(duration),
        "-i", video_path, "-filter:a", "volumedetect", "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=AUDIO_ENERGY_TIMEOUT)
        stderr = r.stderr or ""
        mean_m = re.search(r"mean_volume:\s*([-\d\.]+)\s*dB", stderr)
        max_m = re.search(r"max_volume:\s*([-\d\.]+)\s*dB", stderr)
        mean_vol = float(mean_m.group(1)) if mean_m else None
        max_vol = float(max_m.group(1)) if max_m else None
        vol = max_vol if max_vol is not None else mean_vol
        if vol is None:
            return "media"
        if vol > -7:
            return "alta"
        if vol > -18:
            return "media"
        return "baixa"
    except subprocess.TimeoutExpired:
        return "media"
    except Exception:
        return "media"


def annotate(video_path: str, candidates: list, enable: bool = True) -> list:
    if not enable:
        print("   -> áudio desativado (--no-audio-features)")
        return candidates
    if not shutil.which("ffmpeg"):
        print("   ! ffmpeg ausente — pulando energia (media)")
        return candidates
    print(f"   -> medindo energia de {len(candidates)} candidatos...")
    for c in candidates:
        dur = c.duration if c.duration > 0 else 1
        c.speech_rate = round(len(c.words) / dur, 2)
        c.energy = measure(video_path, c.start, c.end)
    from collections import Counter
    dist = Counter(c.energy for c in candidates)
    avg = statistics.mean([c.speech_rate for c in candidates]) if candidates else 0
    print(f"      distribuição: {dict(dist)} | speech_rate médio: {avg:.2f} w/s")
    return candidates
