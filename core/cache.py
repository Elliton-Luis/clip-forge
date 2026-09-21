"""cache.py — fingerprint + persistência de transcrição/scores.

SRP: só cache. YAGNI: fingerprint por tamanho+mtime (barato, sem hash de conteúdo).
"""
import hashlib
import json
import os
from pathlib import Path
from .models import Word, Segment
from .config import (WHISPER_MODEL_SIZE, WHISPER_LANGUAGE,
                      TRANSCRIPT_PIPELINE_VERSION, AUDIO_ENERGY_VERSION)


def fingerprint(video_path: str, model_size: str | None = None) -> str:
    st = os.stat(video_path)
    lang = WHISPER_LANGUAGE or "auto"
    model = model_size or WHISPER_MODEL_SIZE
    raw = (f"{Path(video_path).resolve()}|{st.st_size}|{int(st.st_mtime)}"
           f"|{model}|{lang}|pipev{TRANSCRIPT_PIPELINE_VERSION}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _say(progress, text: str) -> None:
    if progress is not None:
        progress.note(text)
    else:
        print(text)


def _warn(progress, text: str) -> None:
    if progress is not None:
        progress.warn(text)
    else:
        print(text)


def _detail(progress, text: str) -> None:
    if progress is not None:
        progress.detail(text)
    else:
        print(text)


def load_transcript(cache_dir: Path, fp: str, progress=None):
    p = cache_dir / f"{fp}.transcript.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        segs = []
        for s in data.get("segments", []):
            words = []
            for w in s.get("words", []):
                ww = Word(w["text"], w["start"], w["end"],
                          confidence=w.get("confidence"),
                          timestamp_source=w.get("timestamp_source", "whisper") or "whisper")
                words.append(ww)
            segs.append(Segment(text=s["text"], start=s["start"], end=s["end"], words=words,
                                timestamp_source=s.get("timestamp_source", "whisper") or "whisper"))
        _detail(progress, f"cache hit: transcrição {p} ({len(segs)} segmentos)")
        return segs
    except Exception as e:
        _warn(progress, f"   ! cache corrompido ({p}): {e} — retranscrevendo")
        return None


def save_transcript(cache_dir: Path, fp: str, segments: list, progress=None) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{fp}.transcript.json"
        data = {
            "fingerprint": fp,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end,
                 "timestamp_source": getattr(s, "timestamp_source", "whisper"),
                 "words": [{"text": w.text, "start": w.start, "end": w.end,
                            "confidence": getattr(w, "confidence", None),
                            "timestamp_source": getattr(w, "timestamp_source", "whisper")}
                           for w in s.words]}
                for s in segments
            ],
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _detail(progress, f"transcrição em cache: {p}")
    except Exception as e:
        _warn(progress, f"   ! falha ao salvar cache transcrição: {e}")


def load_scores(cache_dir: Path, fp: str, candidates: list, progress=None) -> bool:
    p = cache_dir / f"{fp}.scores.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("energy_version", 0) != AUDIO_ENERGY_VERSION:
            _warn(progress, f"   ! cache scores com medição de energia antiga "
                  f"(v{data.get('energy_version', 0)} vs v{AUDIO_ENERGY_VERSION}) — re-pontuando")
            return False
        scores = data.get("scores", [])
        if len(scores) != len(candidates):
            _warn(progress, f"   ! cache scores tamanho divergente ({len(scores)} vs {len(candidates)}) — re-pontuando")
            return False
        for c, s in zip(candidates, scores):
            c.score = float(s.get("score", 0))
            c.reason = s.get("reason", "")
            c.title = s.get("title", "")
            c.hashtags = s.get("hashtags", "")
            c.failed = bool(s.get("failed", False))
            c.energy = s.get("energy", "media")
            c.speech_rate = float(s.get("speech_rate", 0))
            # Peak (modo experimental): arquivos antigos não têm as chaves.
            c.window_start = s.get("window_start")
            c.window_end = s.get("window_end")
            c.peak_start = s.get("peak_start")
            c.peak_end = s.get("peak_end")
            c.peak_score = float(s.get("peak_score", 0) or 0)
            c.peak_source = s.get("peak_source", "none") or "none"
            c.peak_reason = s.get("peak_reason", "")
            c.title_source = s.get("title_source", "window") or "window"
        _detail(progress, f"cache hit: scores {p}")
        return True
    except Exception as e:
        _warn(progress, f"   ! cache scores corrompido ({p}): {e} — re-pontuando")
        return False


def save_scores(cache_dir: Path, fp: str, candidates: list, progress=None) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{fp}.scores.json"
        data = {
            "fingerprint": fp,
            "energy_version": AUDIO_ENERGY_VERSION,
            "scores": [
                {"score": c.score, "reason": c.reason, "title": c.title,
                 "hashtags": c.hashtags, "failed": c.failed,
                 "energy": c.energy, "speech_rate": c.speech_rate,
                 "window_start": getattr(c, "window_start", None),
                 "window_end": getattr(c, "window_end", None),
                 "peak_start": getattr(c, "peak_start", None),
                 "peak_end": getattr(c, "peak_end", None),
                 "peak_score": getattr(c, "peak_score", 0.0),
                 "peak_source": getattr(c, "peak_source", "none"),
                 "peak_reason": getattr(c, "peak_reason", ""),
                 "title_source": getattr(c, "title_source", "window")}
                for c in candidates
            ],
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _detail(progress, f"scores em cache: {p}")
    except Exception as e:
        _warn(progress, f"   ! falha ao salvar cache scores: {e}")
