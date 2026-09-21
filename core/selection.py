"""selection.py — NMS com decaimento + diversidade temporal.

SRP: só seleção. Sem I/O.
"""
from .config import DEFAULT_MIN_SCORE, DEFAULT_MAX_PER_10MIN, NMS_DECAY_STRENGTH


def _warn(progress, text: str) -> None:
    if progress is not None:
        progress.warn(text)
    else:
        print(text)


def _note(progress, text: str) -> None:
    if progress is not None:
        progress.note(text)
    else:
        print(text)


def select(candidates: list, top_n: int,
           min_score: float = DEFAULT_MIN_SCORE,
           max_per_10min: int = DEFAULT_MAX_PER_10MIN,
           progress=None) -> list:
    pool = [c for c in candidates if not c.failed and c.score >= min_score]
    if not pool:
        _warn(progress,
              f"   ! nenhum candidato >= {min_score} (de {len(candidates)}). Tente --min-score menor.")
        return []

    pool.sort(key=lambda c: c.score, reverse=True)
    selected: list = []
    buckets: dict[int, int] = {}

    for c in pool:
        bucket = int(c.start // 600)
        if buckets.get(bucket, 0) >= max_per_10min:
            continue
        max_overlap = max((c.overlap_ratio(s) for s in selected), default=0.0)
        penalized = c.score * (1 - NMS_DECAY_STRENGTH * max_overlap) if selected else c.score
        if penalized < min_score:
            continue
        c._penalized_score = penalized  # type: ignore
        selected.append(c)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if len(selected) >= top_n:
            break

    # segunda passada mantém limite de diversidade
    if len(selected) < top_n:
        remaining = [c for c in pool if c not in selected]
        for c in remaining:
            bucket = int(c.start // 600)
            if buckets.get(bucket, 0) >= max_per_10min:
                continue
            max_overlap = max((c.overlap_ratio(s) for s in selected), default=0.0)
            penalized = c.score * (1 - NMS_DECAY_STRENGTH * max_overlap)
            if penalized < min_score:
                continue
            c._penalized_score = penalized  # type: ignore
            selected.append(c)
            buckets[bucket] = buckets.get(bucket, 0) + 1
            if len(selected) >= top_n:
                break

    selected.sort(key=lambda c: c.start)
    _note(progress,
          f"[4/5] {len(selected)} clipes selecionados (de {len(pool)} elegíveis, min_score={min_score}, max_per_10min={max_per_10min})")
    if selected:
        _note(progress,
              "      scores (orig→penalizado): " +
              ", ".join(f"{c.score:.1f}→{getattr(c,'_penalized_score',c.score):.1f}" for c in selected))
    return selected
