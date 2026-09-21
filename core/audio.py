"""audio.py — energia de áudio por janela (ffmpeg volumedetect).

SRP: só áudio. KISS: usa ffmpeg já obrigatório, sem librosa.
"""
import re
import shutil
import statistics
import subprocess
from .config import AUDIO_ENERGY_TIMEOUT, CLIPPER_FFMPEG_THREADS


def _note(progress, text: str) -> None:
    if progress is not None:
        progress.note(text)
    else:
        print(f"   -> {text}")


def _warn(progress, text: str) -> None:
    if progress is not None:
        progress.warn(text)
    else:
        print(f"   ! {text}")


def _detail(progress, text: str) -> None:
    if progress is not None:
        progress.detail(text)
    else:
        print(f"   -> {text}")


def measure(video_path: str, start: float, end: float) -> str:
    duration = max(0.1, end - start)
    # Só áudio: -vn + -map 0:a:0 impedem o decode do vídeo (auditoria
    # 2026-09-20: sem isso, cada janela decodificava AV1/H264 à toa e batia
    # no timeout de 8 s, retornando "media" sem medir nada).
    cmd = [
        "ffmpeg", "-hide_banner", "-threads", str(CLIPPER_FFMPEG_THREADS),
        "-ss", str(max(0, start)), "-t", str(duration),
        "-i", video_path, "-vn", "-map", "0:a:0",
        "-filter:a", "volumedetect", "-f", "null", "-",
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


def annotate(video_path: str, candidates: list, enable: bool = True,
             progress=None) -> list:
    if not enable:
        _note(progress, "áudio desativado (--no-audio-features)")
        return candidates
    if not shutil.which("ffmpeg"):
        _warn(progress, "ffmpeg ausente — pulando energia (media)")
        return candidates
    _detail(progress, f"medindo energia de {len(candidates)} candidatos...")
    st = progress.stage("audio", "Energia", total=len(candidates),
                        unit="candidatos") if progress is not None else None
    for c in candidates:
        dur = c.duration if c.duration > 0 else 1
        c.speech_rate = round(len(c.words) / dur, 2)
        c.energy = measure(video_path, c.start, c.end)
        if st is not None:
            progress.adv("audio", 1)
    from collections import Counter
    dist = Counter(c.energy for c in candidates)
    avg = statistics.mean([c.speech_rate for c in candidates]) if candidates else 0
    summary = f"energia {dict(dist)} | speech_rate médio: {avg:.2f} w/s"
    if st is not None:
        progress.done("audio", summary)
    else:
        print(f"      distribuição: {dict(dist)} | speech_rate médio: {avg:.2f} w/s")
    return candidates
