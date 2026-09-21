"""candidates.py — janelas candidatas + snap para fronteira de palavra.

SRP: só construção/ajuste de candidatos. Sem I/O nem rede.
"""
from .models import Candidate
from .config import (
    MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, WINDOW_STEP_SECONDS,
    SNAP_TOLERANCE_SECONDS, DEFAULT_PAD_SECONDS,
)


def _note(progress, text: str) -> None:
    if progress is not None:
        progress.note(text)
    else:
        print(text)


def build(segments: list, min_dur: float = MIN_CLIP_SECONDS,
          max_dur: float = MAX_CLIP_SECONDS,
          media_end: float | None = None, progress=None) -> list[Candidate]:
    """Janelas deslizantes de [min_dur, max_dur]. MAX é teto real: nenhuma
    janela nasce maior que max_dur (o --pad também nunca estoura, ver _snap_one).
    media_end (duração real da mídia, quando conhecida): janelas nunca passam
    dele — bounds de segmento do Whisper podem extrapolar o áudio (decoder
    overshoot), mas clip não pode existir fora do vídeo.
    """
    if not (0 < min_dur <= max_dur):
        raise ValueError(f"range inválido: min={min_dur} max={max_dur} (exige 0 < min <= max)")
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
        while j < n - 1 and (segments[j + 1].end - start_time) <= max_dur:
            j += 1
            end_time = segments[j].end
        if media_end is not None:
            end_time = min(end_time, media_end)

        if end_time - start_time >= min_dur:
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

    _note(progress, f"[2/5] {len(cands)} janelas candidatas geradas.")
    return cands


def _snap_one(c: Candidate, pad: float = DEFAULT_PAD_SECONDS,
              min_dur: float = MIN_CLIP_SECONDS,
              max_dur: float = MAX_CLIP_SECONDS,
              media_end: float | None = None) -> Candidate:
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
    # Fora da mídia não existe clip: fim nunca passa de media_end (bounds de
    # segmento do Whisper podem extrapolar o áudio real — overshoot do decoder).
    if media_end is not None:
        snapped_start = min(snapped_start, media_end)
        snapped_end = min(snapped_end, media_end)
    # --pad nunca quebra MAX: corta o excesso do fim (início já é snap+pad).
    if snapped_end - snapped_start > max_dur:
        snapped_end = snapped_start + max_dur
    # Abaixo do MIN: expande contexto (início p/ trás até 0, resto p/ frente
    # até media_end). Se nem assim alcança (mídia curta), mantém curto —
    # nunca estoura MAX para compensar.
    if snapped_end - snapped_start < min_dur:
        need = min_dur - (snapped_end - snapped_start)
        back = min(need, snapped_start)
        snapped_start -= back
        need -= back
        if need > 0:
            fwd_end = media_end if media_end is not None else snapped_end + need
            snapped_end = min(snapped_end + need, fwd_end,
                              snapped_start + max_dur)

    if abs(snapped_start - orig_start) > 0.05 or abs(snapped_end - orig_end) > 0.05:
        c.original_start, c.original_end = orig_start, orig_end
        c.start, c.end = snapped_start, snapped_end
        c.snapped = True
        dur = c.duration
        c.speech_rate = round(len(c.words) / dur, 2) if dur > 0 else 0
    return c


def snap_all(candidates: list[Candidate], pad: float,
             min_dur: float = MIN_CLIP_SECONDS,
             max_dur: float = MAX_CLIP_SECONDS,
             media_end: float | None = None, progress=None) -> list[Candidate]:
    for c in candidates:
        _snap_one(c, pad, min_dur, max_dur, media_end)
    n = sum(1 for c in candidates if c.snapped)
    _note(progress,
          f"   -> {n}/{len(candidates)} candidatos com snap (pad={pad}s, tol=±{SNAP_TOLERANCE_SECONDS}s, range=[{min_dur},{max_dur}]s)")
    return candidates
