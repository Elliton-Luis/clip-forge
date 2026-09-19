"""cache.py — fingerprint + persistência de transcrição/scores.

SRP: só cache. YAGNI: fingerprint por tamanho+mtime (barato, sem hash de conteúdo).
"""
import hashlib
import json
import os
from pathlib import Path
from .models import Word, Segment
from .config import WHISPER_MODEL_SIZE, WHISPER_LANGUAGE


def fingerprint(video_path: str, model_size: str | None = None) -> str:
    st = os.stat(video_path)
    lang = WHISPER_LANGUAGE or "auto"
    model = model_size or WHISPER_MODEL_SIZE
    raw = f"{Path(video_path).resolve()}|{st.st_size}|{int(st.st_mtime)}|{model}|{lang}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_transcript(cache_dir: Path, fp: str):
    p = cache_dir / f"{fp}.transcript.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        segs = []
        for s in data.get("segments", []):
            words = [Word(w["text"], w["start"], w["end"]) for w in s.get("words", [])]
            segs.append(Segment(text=s["text"], start=s["start"], end=s["end"], words=words))
        print(f"   -> cache hit: transcrição {p} ({len(segs)} segmentos)")
        return segs
    except Exception as e:
        print(f"   ! cache corrompido ({p}): {e} — retranscrevendo")
        return None


def save_transcript(cache_dir: Path, fp: str, segments: list) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{fp}.transcript.json"
        data = {
            "fingerprint": fp,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end,
                 "words": [{"text": w.text, "start": w.start, "end": w.end} for w in s.words]}
                for s in segments
            ],
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"   -> transcrição em cache: {p}")
    except Exception as e:
        print(f"   ! falha ao salvar cache transcrição: {e}")


def load_scores(cache_dir: Path, fp: str, candidates: list) -> bool:
    p = cache_dir / f"{fp}.scores.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        scores = data.get("scores", [])
        if len(scores) != len(candidates):
            print(f"   ! cache scores tamanho divergente ({len(scores)} vs {len(candidates)}) — re-pontuando")
            return False
        for c, s in zip(candidates, scores):
            c.score = float(s.get("score", 0))
            c.reason = s.get("reason", "")
            c.title = s.get("title", "")
            c.hashtags = s.get("hashtags", "")
            c.failed = bool(s.get("failed", False))
            c.energy = s.get("energy", "media")
            c.speech_rate = float(s.get("speech_rate", 0))
        print(f"   -> cache hit: scores {p}")
        return True
    except Exception as e:
        print(f"   ! cache scores corrompido ({p}): {e} — re-pontuando")
        return False


def save_scores(cache_dir: Path, fp: str, candidates: list) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{fp}.scores.json"
        data = {
            "fingerprint": fp,
            "scores": [
                {"score": c.score, "reason": c.reason, "title": c.title,
                 "hashtags": c.hashtags, "failed": c.failed,
                 "energy": c.energy, "speech_rate": c.speech_rate}
                for c in candidates
            ],
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   -> scores em cache: {p}")
    except Exception as e:
        print(f"   ! falha ao salvar cache scores: {e}")
