"""candidates.py — janelas candidatas + snap para fronteira de palavra.

SRP: só construção/ajuste de candidatos. Sem I/O nem rede.
"""
from .models import Candidate
from .config import (
    MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, WINDOW_STEP_SECONDS,
    SNAP_TOLERANCE_SECONDS, DEFAULT_PAD_SECONDS,
)


def build(segments: list) -> list[Candidate]:
    if not segments:
        return []
    cands: list[Candidate] = []
    n = len(segments)
    i = 0
    cursor = segments[0].start

    while i < n and cursor < segments[-1].end:
        start_idx = i
        start_time = max(cursor, segments[start_idx].start)
        j = start_idx
        end_time = segments[j].end
        while j < n - 1 and (segments[j + 1].end - start_time) <= MAX_CLIP_SECONDS:
            j += 1
            end_time = segments[j].end

        if end_time - start_time >= MIN_CLIP_SECONDS:
            words, texts = [], []
            for k in range(start_idx, j + 1):
                texts.append(segments[k].text)
                words.extend(segments[k].words)
            dur = end_time - start_time
            sr = len(words) / dur if dur > 0 else 0
            cands.append(Candidate(
                start=start_time, end=end_time,
                text=" ".join(texts).strip(), words=words, speech_rate=round(sr, 2),
            ))

        cursor = start_time + WINDOW_STEP_SECONDS
        while i < n and segments[i].start < cursor:
            i += 1

    print(f"[2/5] {len(cands)} janelas candidatas geradas.")
    return cands


def _snap_one(c: Candidate, pad: float = DEFAULT_PAD_SECONDS) -> Candidate:
    if not c.words:
        return c
    words = sorted(c.words, key=lambda w: w.start)
    orig_start, orig_end = c.start, c.end

    def snap_value(target: float) -> float:
        best, best_dist = target, SNAP_TOLERANCE_SECONDS + 1
        for w in words:
            for v in (w.start, w.end):
                d = abs(v - target)
                if d <= SNAP_TOLERANCE_SECONDS and d < best_dist:
                    best, best_dist = v, d
        return best

    best_start = snap_value(orig_start)
    best_end = snap_value(orig_end)
    snapped_start = max(0, best_start - pad)
    snapped_end = best_end + pad
    if snapped_end <= snapped_start:
        snapped_end = snapped_start + 1.0

    if abs(snapped_start - orig_start) > 0.05 or abs(snapped_end - orig_end) > 0.05:
        c.original_start, c.original_end = orig_start, orig_end
        c.start, c.end = snapped_start, snapped_end
        c.snapped = True
        dur = c.duration
        c.speech_rate = round(len(c.words) / dur, 2) if dur > 0 else 0
    return c


def snap_all(candidates: list[Candidate], pad: float) -> list[Candidate]:
    for c in candidates:
        _snap_one(c, pad)
    n = sum(1 for c in candidates if c.snapped)
    print(f"   -> {n}/{len(candidates)} candidatos com snap (pad={pad}s, tol=±{SNAP_TOLERANCE_SECONDS}s)")
    return candidates
